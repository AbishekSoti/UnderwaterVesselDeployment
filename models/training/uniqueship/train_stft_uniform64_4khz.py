#!/usr/bin/env python3
"""UniqueShip Fold 1: 64 uniform spectral-power bands, 0–4 kHz, four fixed models.

Train on the authors' training split, report validation only. After model
selection, use --test-only to load saved checkpoints and evaluate the test split
WITHOUT retraining. Old 128-bin log-STFT caches/checkpoints are never modified.

Requires the existing corrected train_stft_classical.py in the same directory,
plus numpy, pandas, librosa, soundfile, scikit-learn, and joblib.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import warnings
from pathlib import Path

import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

# Reuse only proven metadata parsing, path resolution, and leakage checks.
# No historical feature extraction or model loading from this module.
import train_stft_classical as base

SR = 16_000
SECONDS = 5
NFFT = 4096
HOP = 160
MAX_FREQ_HZ = 4000
N_BANDS = 64
FFT_BINS_PER_BAND = 16
T_FRAMES = ((SR * SECONDS - NFFT) // HOP) + 1  # 475
FEATURE_DIM = 2 * (N_BANDS + T_FRAMES)         # 1078
VERSION = "uniqueship_stft_uniform_power64_0to4khz_v1"
MODEL_KEYS = ("logreg", "rbf_svm", "mlp", "histgb")
CACHE_COMMIT_INTERVAL = 250


def extract_features(path: Path) -> np.ndarray:
    """Replicate the previous audio/STFT settings; change only the frequency map.

    Average 16 linearly-spaced native FFT *power* bins into each of 64
    non-overlapping 62.5-Hz bands, then apply the same ref=max dB scaling.
    DC is excluded; bin 1024 corresponds to 4,000 Hz and is included.
    """
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav[:, 0]  # match the existing implementation
    if wav.ndim != 1:
        raise ValueError(f"Unsupported audio dimensions {wav.shape}: {path}")
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    required = SR * SECONDS
    if len(wav) < required:
        wav = np.pad(wav, (0, required - len(wav)))
    elif len(wav) > required:
        wav = wav[:required]

    spec = librosa.stft(wav, n_fft=NFFT, hop_length=HOP, center=False)
    power = (np.abs(spec) ** 2).astype(np.float32, copy=False)
    # FFT bins 1..1024: 3.90625 .. 4000 Hz, 1024 bins total.
    band_power = power[1:1025, :].reshape(N_BANDS, FFT_BINS_PER_BAND, T_FRAMES).mean(axis=1)
    # Keep EXACT previous dB reference and librosa defaults, including top_db=80.
    db = librosa.power_to_db(band_power, ref=np.max).astype(np.float32, copy=False)
    x = np.concatenate((db.mean(axis=1), db.std(axis=1),
                        db.mean(axis=0), db.std(axis=0))).astype(np.float32)
    if x.shape != (FEATURE_DIM,) or not np.isfinite(x).all():
        raise ValueError(f"Invalid {x.shape} feature vector from {path}")
    return x


def cache_paths(frame: pd.DataFrame, split: str, fold: int, pcol: str, lcol: str,
                root: Path, cache_dir: Path) -> tuple[Path, Path, Path]:
    h = hashlib.sha256()
    h.update(VERSION.encode("utf-8"))
    h.update(str(root).encode("utf-8"))
    h.update(str(fold).encode("ascii"))
    for p, label in frame[[pcol, lcol]].itertuples(index=False, name=None):
        h.update(str(p).encode("utf-8")); h.update(b"\x00")
        h.update(str(label).encode("utf-8")); h.update(b"\n")
    basename = f"fold{fold}_{split}_{h.hexdigest()[:14]}"
    return (cache_dir / (basename + "_X.npy"),
            cache_dir / (basename + "_partial.npy"),
            cache_dir / (basename + "_progress.json"))


def write_json_atomically(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_or_extract(frame: pd.DataFrame, split: str, fold: int,
                    root: Path, pcol: str, lcol: str, label_map: dict,
                    cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    output, partial, progress_file = cache_paths(frame, split, fold, pcol, lcol, root, cache_dir)
    n = len(frame)
    expected = (n, FEATURE_DIM)
    y = np.asarray([label_map[str(k)] for k in frame[lcol]], dtype=np.int16)

    if output.exists():
        existing = np.load(output, mmap_mode="r", allow_pickle=False)
        if existing.shape != expected or existing.dtype != np.float32:
            raise ValueError(f"Old/mismatched cache: {output}, got {existing.shape}/{existing.dtype}")
        print(f"{split}: reusing complete feature cache {output.name} {expected}", flush=True)
        return existing, y

    start_row = 0
    if partial.exists() and progress_file.exists():
        progress = json.loads(progress_file.read_text(encoding="utf-8"))
        x = np.load(partial, mmap_mode="r+", allow_pickle=False)
        if progress.get("version") != VERSION or x.shape != expected or x.dtype != np.float32:
            raise ValueError("Partial cache does not match this configuration; inspect before deleting it.")
        start_row = int(progress["completed_rows"])
        if not 0 <= start_row <= n:
            raise ValueError("Invalid resume progress count")
        print(f"{split}: resuming extraction at row {start_row:,}/{n:,}", flush=True)
    else:
        if partial.exists() or progress_file.exists():
            raise ValueError("Found orphaned partial cache/progress file. Inspect before restarting.")
        x = np.lib.format.open_memmap(partial, mode="w+", dtype=np.float32, shape=expected)
        write_json_atomically(progress_file, {"version": VERSION, "completed_rows": 0, "rows": n})
        print(f"{split}: extracting {n:,} five-second audio clips", flush=True)

    tic = time.perf_counter()
    try:
        for i in range(start_row, n):
            x[i] = extract_features(base.resolve_audio(root, str(frame.iloc[i][pcol])))
            if (i + 1) % CACHE_COMMIT_INTERVAL == 0 or i + 1 == n:
                x.flush()
                write_json_atomically(progress_file, {
                    "version": VERSION, "completed_rows": i + 1, "rows": n
                })
                print(f"{split}: {i+1:,}/{n:,} ({time.perf_counter()-tic:.0f}s this session)", flush=True)
    except BaseException:
        x.flush()
        print(f"{split}: interrupted, partial features retained for resume.", flush=True)
        raise
    x.flush()
    del x
    os.replace(partial, output)
    progress_file.unlink(missing_ok=True)
    return np.load(output, mmap_mode="r", allow_pickle=False), y


def classifiers() -> dict:
    """Same families and fixed hyperparameters as the completed baseline run."""
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1200)),
        "rbf_svm": make_pipeline(
            StandardScaler(),
            Nystroem(kernel="rbf", gamma=1.0 / FEATURE_DIM,
                     n_components=512, random_state=42),
            LinearSVC(C=1.0, random_state=42, max_iter=3000),
        ),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu", early_stopping=False,
            max_iter=150, random_state=42,
        )),
        "histgb": HistGradientBoostingClassifier(
            max_iter=100, max_leaf_nodes=15, learning_rate=0.08,
            l2_regularization=1.0, random_state=42, early_stopping=False,
        ),
    }


def score_and_save(model_key: str, estimator, frame: pd.DataFrame, x: np.ndarray, y: np.ndarray,
                   class_names: list[str], pcol: str, split: str, fold: int,
                   result_dir: Path) -> dict:
    t0 = time.perf_counter()
    pred = estimator.predict(x)
    elapsed = time.perf_counter() - t0
    acc = float(accuracy_score(y, pred))
    macro = float(f1_score(y, pred, average="macro", zero_division=0))
    weighted = float(f1_score(y, pred, average="weighted", zero_division=0))
    print(f"  {split}: accuracy={acc:.4%}, macro-F1={macro:.4%}, weighted-F1={weighted:.4%}", flush=True)
    pd.DataFrame(confusion_matrix(y, pred, labels=list(range(len(class_names)))),
                 index=class_names, columns=class_names).to_csv(
                     result_dir / f"fold{fold}_{model_key}_{split}_confusion.csv")
    (result_dir / f"fold{fold}_{model_key}_{split}_classification_report.json").write_text(
        json.dumps(classification_report(y, pred, labels=list(range(len(class_names))),
                 target_names=class_names, output_dict=True, zero_division=0), indent=2),
        encoding="utf-8")
    if split == "test":
        pd.DataFrame({"path": frame[pcol].astype(str).to_numpy(),
                      "true_class": [class_names[int(k)] for k in y],
                      "predicted_class": [class_names[int(k)] for k in pred],
                      "correct": pred == y}).to_csv(
                          result_dir / f"fold{fold}_{model_key}_test_predictions.csv", index=False)
    return {"model": model_key, "fold": fold, "split": split,
            "samples": len(y), "accuracy": acc, "macro_f1": macro,
            "weighted_f1": weighted, "predict_seconds_total": elapsed,
            "feature_dim": FEATURE_DIM, "frontend": VERSION}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_ROOT)
    ap.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    ap.add_argument("--test-only", action="store_true",
                    help="Loads previously saved models; never fits, only tests")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--artifact-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    args = ap.parse_args()
    if args.fold not in range(1, 6):
        ap.error("--fold must be 1 through 5")
    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        ap.error(f"Dataset root missing: {root}")
    parent = root.parent.parent
    cache_dir = args.cache_dir or parent / "feature_cache" / VERSION
    artifact_dir = args.artifact_dir or parent / "model_artifacts" / VERSION
    result_dir = args.results_dir or base.REPO_ROOT / "benchmarks/results/uniqueship_stft" / VERSION
    result_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    print(f"Frontend: {VERSION}; 16 kHz, FFT 4096, hop 160, {N_BANDS} x {T_FRAMES} -> {FEATURE_DIM}D")
    print("64 uniform bands: native FFT bins 1..1024, ~3.91 to 4000 Hz, 62.5 Hz/band")
    raw, pcol, lcol, label_map = base.load_csvs(root, args.fold, None, None)
    base.validate_folds(raw, pcol, lcol)
    class_names = [k for k,v in sorted(label_map.items(), key=lambda item: item[1])]
    models = classifiers()
    rows = []

    if args.test_only:
        # Check ALL checkpoints BEFORE spending time extracting test data.
        saved = {}
        for key in dict.fromkeys(args.models):
            f = artifact_dir / f"fold{args.fold}_{key}_full.joblib"
            if not f.is_file():
                raise FileNotFoundError(f"Missing model checkpoint: {f}")
            payload = joblib.load(f)
            for field, correct in {"version": VERSION, "fold": args.fold,
                                    "feature_dim": FEATURE_DIM, "label_map": label_map}.items():
                if payload.get(field) != correct:
                    raise ValueError(f"Checkpoint {f} has incompatible {field}")
            saved[key] = payload["estimator"]
        frame = raw["test"].reset_index(drop=True)
        x, y = load_or_extract(frame, "test", args.fold, root, pcol, lcol, label_map, cache_dir)
        for key, estimator in saved.items():
            rows.append(score_and_save(key, estimator, frame, x, y, class_names,
                                       pcol, "test", args.fold, result_dir))
        output = result_dir / f"fold{args.fold}_test_metrics.csv"
    else:
        tr = raw["train"].reset_index(drop=True)
        va = raw["validation"].reset_index(drop=True)
        x_train, y_train = load_or_extract(tr, "train", args.fold, root, pcol, lcol, label_map, cache_dir)
        x_val, y_val = load_or_extract(va, "validation", args.fold, root, pcol, lcol, label_map, cache_dir)
        for key in dict.fromkeys(args.models):
            f = artifact_dir / f"fold{args.fold}_{key}_full.joblib"
            if f.exists():
                payload = joblib.load(f)
                if (payload.get("version") != VERSION or payload.get("fold") != args.fold
                    or payload.get("feature_dim") != FEATURE_DIM or payload.get("label_map") != label_map):
                    raise ValueError(f"Existing checkpoint has incompatible configuration: {f}")
                estimator = payload["estimator"]
                fit_seconds = payload.get("fit_seconds", float("nan"))
                print(f"{key}: using existing checkpoint (no retraining)", flush=True)
            else:
                estimator = clone(models[key])
                print(f"{key}: training on {len(y_train):,} cached feature vectors", flush=True)
                t0 = time.perf_counter()
                estimator.fit(x_train, y_train)
                fit_seconds = time.perf_counter() - t0
                # Save checkpoint before reports: individual models survive interruptions.
                partial = f.with_suffix(".tmp")
                joblib.dump({"estimator": estimator, "label_map": label_map,
                             "version": VERSION, "feature_dim": FEATURE_DIM,
                             "fold": args.fold, "fit_seconds": fit_seconds}, partial, compress=3)
                os.replace(partial, f)
                print(f"  checkpoint saved: {f}", flush=True)
            record = score_and_save(key, estimator, va, x_val, y_val, class_names,
                                    pcol, "validation", args.fold, result_dir)
            record["fit_seconds"] = fit_seconds
            rows.append(record)
            # Incremental saved metrics after each completed model.
            pd.DataFrame(rows).to_csv(result_dir / f"fold{args.fold}_validation_metrics.csv", index=False)
        output = result_dir / f"fold{args.fold}_validation_metrics.csv"
    pd.DataFrame(rows).to_csv(output, index=False)
    print("\nResults:\n", pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved: {output}")
    if not args.test_only:
        print("Test partition was NOT used. Later use --test-only for final evaluation.")


if __name__ == "__main__":
    main()
