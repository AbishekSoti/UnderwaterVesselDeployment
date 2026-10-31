#!/usr/bin/env python3
"""UniqueShip 5-Class Balanced: STFT + separate frequency/time-axis statistics.

Place in UnderwaterVesselDeployment/models/training/uniqueship/train_stft_classical.py

Reads the authors' predefined fold CSVs; never creates a random train/test split.
The historical ShipsEar frontend is reproduced after explicit resampling to 16 kHz.
Four classifiers: logistic regression, RBF-SVM (exact/approximated), MLP and
histogram gradient boosting. See README_UNIQUESHIP_STFT.md.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
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
from sklearn.svm import LinearSVC, SVC

SR = 16_000
SECONDS = 5
NFFT = 4096
HOP = 160
F_BINS = 128
N_SAMPLES = SR * SECONDS
T_FRAMES = (N_SAMPLES - NFFT) // HOP + 1
FEATURE_DIM = 2 * (F_BINS + T_FRAMES)
SEED = 42
VERSION = "uniqueship_shipsear_stft_axis_stats_v1"
DEFAULT_ROOT = Path.home() / "Local Documents/Datasets/UniqueShip/5class_balanced/audio/main5_balanced"
REPO_ROOT = next((p for p in Path(__file__).resolve().parents if (p / ".git").is_dir()), Path.cwd())


def norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def pick_column(df: pd.DataFrame, *, kind: str, override: str | None = None) -> str:
    if override:
        if override not in df.columns:
            raise ValueError(f"Column {override!r} not present. Available: {list(df.columns)}")
        return override
    names = {norm(c): c for c in df.columns}
    if kind == "path":
        candidates = ["audiofilepath", "audiopath", "relpath", "relativepath", "filepath",
                      "wavfilepath", "wavpath", "file", "filename", "path", "audioloc"]
        for candidate in candidates:
            if candidate in names:
                return names[candidate]
        for c in df.columns:
            v = df[c].dropna().astype(str).head(20)
            if not v.empty and v.str.contains(r"\.(wav|wv)(?:$|\?)", case=False, regex=True).mean() > 0.6:
                return c
    else:
        candidates = ["class", "classlabel", "shipclass", "vesselclass", "vesseltype",
                      "shiptype", "label", "target", "category", "shipcategory"]
        for candidate in candidates:
            if candidate in names:
                return names[candidate]
    raise ValueError(f"Cannot identify {kind} column automatically. CSV columns: {list(df.columns)}. "
                     f"Pass --{'path' if kind == 'path' else 'label'}-column COLUMN_NAME.")


def csv_file(root: Path, fold: int, split: str) -> Path:
    return root / f"fold_{fold}" / f"metadata_5s_{split}.csv"


def load_csvs(root: Path, fold: int, path_col: str | None, label_col: str | None):
    raw = {}
    for split in ("train", "validation", "test"):
        path = csv_file(root, fold, split)
        if not path.is_file():
            raise FileNotFoundError(f"Missing {path}")
        # dtype=str prevents the MMSI identifier from being represented as a float.
        raw[split] = pd.read_csv(path, dtype=str, low_memory=False)
        if raw[split].empty:
            raise ValueError(f"No rows in {path}")
        print(f"{split:10} {len(raw[split]):>7,} samples  columns={list(raw[split].columns)}")
    pcol = pick_column(raw["train"], kind="path", override=path_col)
    lcol = pick_column(raw["train"], kind="label", override=label_col)
    for split, df in raw.items():
        if pcol not in df or lcol not in df:
            raise ValueError(f"Columns {pcol!r}/{lcol!r} not present in {split} CSV")
        if df[[pcol, lcol]].isna().any().any():
            raise ValueError(f"Missing path/label values in {split} CSV")
    print(f"Using CSV path column {pcol!r}; label column {lcol!r}.")
    labels = sorted(set().union(*(df[lcol].unique().tolist() for df in raw.values())))
    label_map = {str(lab): ix for ix, lab in enumerate(labels)}
    print("Class mapping:", label_map)
    return raw, pcol, lcol, label_map


def validate_folds(raw: dict[str, pd.DataFrame], pcol: str, lcol: str):
    sets = {k: set(df[pcol].astype(str)) for k, df in raw.items()}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        overlap = sets[a] & sets[b]
        if overlap:
            raise ValueError(f"Same audio path in {a} and {b}: {next(iter(overlap))}")
    cols = {norm(c): c for c in raw["train"].columns}
    mmsi_col = cols.get("mmsi") or cols.get("vesselmmsi")
    if mmsi_col:
        def ids(df):
            # Background clips do not represent a vessel; the authors encode
            # their missing MMSI using sentinels such as "0.0". Never treat
            # these as real vessel IDs during the leakage audit.
            labels = df[lcol].fillna("").astype(str).str.strip().str.casefold()
            background = labels.isin({"background", "ambient", "no_ship", "no ship"})
            s = df.loc[~background, mmsi_col].fillna("").astype(str).str.strip()
            invalid = {"", "0", "0.0", "-1", "-1.0", "nan", "none", "null", "<na>"}
            bad = s.str.casefold().isin(invalid)
            if bad.any():
                print(f"WARNING: {int(bad.sum())} vessel-labelled clips lack a usable MMSI; "
                      "confirm author metadata before claiming complete vessel-ID verification.")
            return set(s[~bad])
        groups = {k: ids(v) for k, v in raw.items()}
        for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
            overlap = groups[a] & groups[b]
            if overlap:
                raise ValueError(f"MMSI leakage {a}/{b}: {list(overlap)[:5]}")
        print("MMSI grouping: PASS (distinct across splits)")
    else:
        print("Warning: no obvious MMSI column; predefined folds used but MMSI grouping not independently checked.")
    # Diagnostic of author-provided background partitions. UniqueShip documents
    # day-grouped background; some calendar dates can appear in more than one
    # official partition. Report that discrepancy rather than misidentifying it
    # as proof of duplicate audio or silently asserting leak-free grouping.
    bg_entries = {}
    for split, df in raw.items():
        bg_mask = df[lcol].astype(str).str.contains(
            r"background|ambient|no[ _-]?ship", case=False, regex=True
        )
        hydro_col = next((c for c in df.columns if norm(c) == "hydrophone"), None)
        date_col = next((c for c in df.columns if norm(c) == "date"), None)
        rows = []
        for _, row in df.loc[bg_mask].iterrows():
            audio_path = str(row[pcol])
            match = re.search(r"(?<![0-9])(?:19|20)[0-9]{6}", audio_path)
            if not match and date_col is not None:
                match = re.search(r"(?<![0-9])(?:19|20)[0-9]{6}",
                                  str(row[date_col]).replace("-", ""))
            day = match.group(0) if match else None
            hydro = str(row[hydro_col]).strip() if hydro_col else "unknown"
            rows.append((day, hydro, audio_path))
        bg_entries[split] = rows

    if not any(bg_entries.values()):
        print("Background-day grouping: NOT CHECKED (no background labels found)")
    else:
        days = {s: {day for day, _, _ in rows if day} for s, rows in bg_entries.items()}
        hydro_days = {
            s: {(hydro, day) for day, hydro, _ in rows if day}
            for s, rows in bg_entries.items()
        }
        pair_names = (("train", "validation"), ("train", "test"),
                      ("validation", "test"))
        for a, b in pair_names:
            shared_days = days[a] & days[b]
            shared_hydro_days = hydro_days[a] & hydro_days[b]
            if not shared_days:
                print(f"Background dates {a}/{b}: no shared dates detected")
                continue
            print(f"WARNING: Background dates {a}/{b}: {len(shared_days)} shared "
                  f"calendar date(s), e.g. {sorted(shared_days)[:5]}; "
                  f"{len(shared_hydro_days)} shared hydrophone+date group(s).")
            for d in sorted(shared_days)[:2]:
                for split in (a, b):
                    examples = [(h, p) for day, h, p in bg_entries[split] if day == d]
                    if examples:
                        print(f"  {split} {d}: {len(examples)} background clips; "
                              f"hydrophones={sorted({h for h, _ in examples})[:5]}; "
                              f"sample={examples[0][1]}")
        print("Background-day grouping: AUDIT ONLY. These shared dates do not "
              "establish duplicate audio, but day-disjointness is not independently "
              "verified. Keep the authors' CSV partitions and investigate any "
              "warnings before publishing leak-free claims.")
    print("Exact audio paths across partitions: PASS")


def select_quick(df: pd.DataFrame, label_col: str, limit: int) -> pd.DataFrame:
    if limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    # Stratified deterministic sampling *within* the author's existing partition.
    from sklearn.model_selection import train_test_split
    try:
        subset, _ = train_test_split(df, train_size=limit, stratify=df[label_col], random_state=SEED)
    except ValueError:
        subset = df.sample(n=limit, random_state=SEED)
    return subset.sort_index().reset_index(drop=True)


def resolve_audio(root: Path, csv_rel_path: str) -> Path:
    name = str(csv_rel_path).strip().replace("\\", "/")
    rel = Path(name)
    if rel.is_absolute() and rel.is_file():
        return rel
    stripped = name.lstrip("./")
    probes = [root / stripped, root / "audio" / stripped, root.parent / stripped]
    if stripped.startswith(root.name + "/"):
        probes.insert(0, root.parent / stripped)
    if stripped.startswith("audio/"):
        probes.insert(0, root / stripped)
    seen = set()
    for p in probes:
        if str(p) not in seen and p.is_file():
            return p
        seen.add(str(p))
    raise FileNotFoundError(f"CSV audio path {csv_rel_path!r} could not be resolved. "
                            f"Tried: {', '.join(str(x) for x in probes)}. "
                            "Inspect the CSV and use the appropriate root/path column.")


# Cached interpolation weights: preserves the old ShipsEar log-frequency remap,
# without running Python's np.interp for every individual frame.
def logfreq_indexes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.fft.rfftfreq(NFFT, d=1.0 / SR)[1:]
    targets = np.geomspace(bins[0], bins[-1], F_BINS)
    right = np.searchsorted(bins, targets, side="left")
    right = np.clip(right, 1, len(bins) - 1)
    left = right - 1
    weight = ((targets - bins[left]) / (bins[right] - bins[left])).astype(np.float32)
    return left, right, weight


LEFT, RIGHT, WEIGHT = logfreq_indexes()


def extract_one(path: Path) -> np.ndarray:
    wav, sr = sf.read(path, dtype="float32", always_2d=False)
    if wav.ndim == 2:
        wav = wav[:, 0]  # match prior ShipsEar 'channel 0' implementation
    if wav.ndim != 1:
        raise ValueError(f"Unexpected waveform shape {wav.shape}: {path}")
    if sr != SR:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=SR)
    if len(wav) < N_SAMPLES:
        wav = np.pad(wav, (0, N_SAMPLES - len(wav)))
    elif len(wav) > N_SAMPLES:
        wav = wav[:N_SAMPLES]
    spec = librosa.stft(wav, n_fft=NFFT, hop_length=HOP, center=False)
    power = (np.abs(spec) ** 2).astype(np.float32, copy=False)[1:, :]
    remapped = power[LEFT, :] * (1.0 - WEIGHT[:, None]) + power[RIGHT, :] * WEIGHT[:, None]
    db = librosa.power_to_db(remapped, ref=np.max).astype(np.float32, copy=False)
    if db.shape != (F_BINS, T_FRAMES):
        raise RuntimeError(f"Unexpected STFT size {db.shape}, expected {(F_BINS, T_FRAMES)}")
    # 128 mean/std across time, and 475 mean/std across frequency.
    vec = np.concatenate([db.mean(axis=1), db.std(axis=1),
                          db.mean(axis=0), db.std(axis=0)]).astype(np.float32)
    if vec.shape != (FEATURE_DIM,) or not np.isfinite(vec).all():
        raise ValueError(f"Invalid feature vector for {path}")
    return vec


def cache_tag(df: pd.DataFrame, pcol: str, lcol: str, split: str, fold: int) -> str:
    # Hash the ordered path/label pairs so modified CSV or --quick cannot reuse a wrong cache.
    h = hashlib.sha256()
    h.update(VERSION.encode())
    for p, lab in df[[pcol, lcol]].itertuples(index=False, name=None):
        h.update(str(p).encode()); h.update(b"\x00")
        h.update(str(lab).encode()); h.update(b"\n")
    return f"fold{fold}_{split}_{h.hexdigest()[:14]}"


def extract_split(df: pd.DataFrame, root: Path, pcol: str, lcol: str,
                  label_map: dict[str, int], cache: Path, split: str, fold: int,
                  force: bool) -> tuple[np.ndarray, np.ndarray]:
    cache.mkdir(parents=True, exist_ok=True)
    tag = cache_tag(df, pcol, lcol, split, fold)
    x_path = cache / f"{tag}_X.npy"
    if x_path.is_file() and not force:
        x = np.load(x_path, mmap_mode="r")
        if x.shape == (len(df), FEATURE_DIM):
            print(f"{split}: using cached {x.shape} from {x_path}")
            y = np.array([label_map[str(v)] for v in df[lcol]], dtype=np.int16)
            return x, y
    partial_path = x_path.with_name(x_path.stem + ".inprogress.npy")
    x = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32,
                                  shape=(len(df), FEATURE_DIM))
    print(f"{split}: extracting {len(df):,} clips to {x_path}")
    start = time.perf_counter()
    try:
        for i, p in enumerate(df[pcol].astype(str)):
            x[i] = extract_one(resolve_audio(root, p))
            if (i + 1) % 500 == 0 or i + 1 == len(df):
                x.flush()
                elapsed = time.perf_counter() - start
                print(f"    {i+1:>7,}/{len(df):,}  {elapsed:.0f}s elapsed", flush=True)
    except BaseException:
        del x
        partial_path.unlink(missing_ok=True)  # incomplete cache is never considered valid
        raise
    x.flush()
    del x
    partial_path.replace(x_path)  # only complete extracted matrices become reusable caches
    x = np.load(x_path, mmap_mode="r")
    y = np.array([label_map[str(v)] for v in df[lcol]], dtype=np.int16)
    return x, y


def build_models(rbf_mode: str) -> dict:
    models = {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1200, C=1.0)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64),
                              activation="relu", early_stopping=False,
                              max_iter=150, random_state=SEED)),
        "histgb": HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                          learning_rate=0.08, l2_regularization=1.0,
                          random_state=SEED, early_stopping=False),
    }
    if rbf_mode == "exact":
        models["rbf_svm"] = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10,
                                                      gamma="scale", cache_size=1000))
    else:
        # A nonlinear RBF-kernel approximation that scales far better than libsvm.
        models["rbf_svm"] = make_pipeline(StandardScaler(),
                    Nystroem(kernel="rbf", gamma=1.0 / FEATURE_DIM,
                             n_components=512, random_state=SEED),
                    LinearSVC(C=1.0, random_state=SEED, max_iter=3000))
    return models


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data-root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--path-column", default=None)
    ap.add_argument("--label-column", default=None)
    ap.add_argument("--inspect-only", action="store_true", help="Show CSV columns/examples and exit")
    ap.add_argument("--quick", action="store_true", help="Use up to 2000 train / 500 val / 500 test rows")
    ap.add_argument("--models", nargs="+", default=["logreg", "rbf_svm", "mlp", "histgb"],
                    choices=["logreg", "rbf_svm", "mlp", "histgb"])
    ap.add_argument("--rbf-mode", choices=["exact", "approx"], default="approx",
                    help="Exact SVC is generally not feasible for full UniqueShip")
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--results-dir", type=Path, default=None)
    ap.add_argument("--rebuild-cache", action="store_true")
    ap.add_argument("--evaluate-test", action="store_true", help="Evaluate held-out test set (final run)")
    args = ap.parse_args()

    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        ap.error(f"Data root does not exist: {root}")
    if args.fold not in range(1, 6):
        ap.error("--fold must be 1..5")
    cache_dir = args.cache_dir or (root.parent.parent / "feature_cache" / "stft_axis_stats_v1")
    results_dir = args.results_dir or (REPO_ROOT / "benchmarks/results/uniqueship_stft")

    print(f"Dataset root: {root}\nSTFT: {F_BINS} x {T_FRAMES}; axis-statistics = {FEATURE_DIM}-D")
    raw, pcol, lcol, label_map = load_csvs(root, args.fold, args.path_column, args.label_column)
    validate_folds(raw, pcol, lcol)
    for key, df in raw.items():
        print(f"\n{key} examples:\n{df[[pcol,lcol]].head(3).to_string(index=False)}")
    if args.inspect_only:
        for split, df in raw.items():
            for sample in df[pcol].head(2).astype(str):
                resolved = resolve_audio(root, sample)
                print(f"  Resolved {split}: {resolved}")
        print("Data root and sample audio paths: PASS")
        return
    # A quick run is a debugging/screening run, not a publishable full benchmark.
    limits = {"train": 2000, "validation": 500, "test": 500} if args.quick else {}
    data = {s: select_quick(df, lcol, limits.get(s, 0)) for s, df in raw.items()}
    cache_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    x_sets, y_sets = {}, {}
    for split in ("train", "validation", "test"):
        # Skip test feature extraction entirely until explicitly requested.
        if split == "test" and not args.evaluate_test:
            continue
        x_sets[split], y_sets[split] = extract_split(data[split], root, pcol, lcol, label_map,
                        cache_dir, split, args.fold, args.rebuild_cache)
    xtr, ytr = x_sets["train"], y_sets["train"]
    xval, yval = x_sets["validation"], y_sets["validation"]
    models = build_models(args.rbf_mode)
    if not args.quick and args.rbf_mode == "exact" and "rbf_svm" in args.models:
        print("WARNING: Exact RBF SVC may take extremely long on >100k clips; Ctrl+C to cancel.")
    records = []
    for key in args.models:
        print(f"\nTraining {key} (rbf_mode={args.rbf_mode} if applicable) ...", flush=True)
        estimator = clone(models[key])
        tic = time.perf_counter()
        estimator.fit(xtr, ytr)
        train_sec = time.perf_counter() - tic
        yp = estimator.predict(xval)
        acc = accuracy_score(yval, yp)
        f1 = f1_score(yval, yp, average="macro", zero_division=0)
        result = {"model": key, "rbf_mode": args.rbf_mode if key == "rbf_svm" else None,
                  "fold": args.fold, "quick": bool(args.quick), "train_n": len(ytr),
                  "validation_n": len(yval), "train_seconds": round(train_sec, 3),
                  "validation_accuracy": float(acc), "validation_macro_f1": float(f1)}
        print(f"  validation accuracy={acc:.4f}, macro-F1={f1:.4f}; fit={train_sec:.1f}s")
        if args.evaluate_test:
            ytest = y_sets["test"]
            test_pred = estimator.predict(x_sets["test"])
            result["test_accuracy"] = float(accuracy_score(ytest, test_pred))
            result["test_macro_f1"] = float(f1_score(ytest, test_pred, average="macro", zero_division=0))
            print("  test accuracy:", result["test_accuracy"], "macro-F1:", result["test_macro_f1"])
            (results_dir / f"fold{args.fold}_{key}_{args.rbf_mode if key == 'rbf_svm' else 'standard'}_confusion.csv").write_text(
                pd.DataFrame(confusion_matrix(ytest, test_pred, labels=range(len(label_map))),
                    index=list(label_map), columns=list(label_map)).to_csv(), encoding="utf8")
        # Checkpoints outside Git: feature_cache/model_artifacts are in Local Documents.
        artifact_dir = root.parent.parent / "model_artifacts" / "stft_axis_stats_v1"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        suffix = "_quick" if args.quick else "_full"
        mode = "_" + args.rbf_mode if key == "rbf_svm" else ""
        joblib.dump({"estimator": estimator, "label_map": label_map, "config_version": VERSION,
                     "feature_dim": FEATURE_DIM, "fold": args.fold},
                    artifact_dir / f"fold{args.fold}_{key}{mode}{suffix}.joblib", compress=3)
        records.append(result)
    print("\nSummary:\n", pd.DataFrame(records).to_string(index=False))
    outfile = results_dir / (f"fold{args.fold}_{'quick' if args.quick else 'full'}_{args.rbf_mode}_"
                             f"{'test' if args.evaluate_test else 'validation'}_metrics.csv")
    pd.DataFrame(records).to_csv(outfile, index=False)
    (results_dir / f"fold{args.fold}_class_map.json").write_text(json.dumps(label_map, indent=2))
    print("Saved metrics:", outfile)


if __name__ == "__main__":
    main()
