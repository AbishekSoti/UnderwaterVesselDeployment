#!/usr/bin/env python3
"""Resumable, validation-only UniqueShip STFT hyperparameter sweep.

Place next to train_stft_classical.py. Reuses full Fold 1 .npy feature caches;
NEVER extracts WAVs, opens test-feature files, or evaluates the test set.
Scores by validation macro-F1, stores best models per family outside Git.

Typical: caffeinate -i python -u models/training/uniqueship/sweep_stft_models.py --hours 9
Resume: run the same command again. See --help for controls.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
import random
import sys
import time
import traceback
import warnings
from datetime import datetime
from pathlib import Path

# Set thread env vars before importing NumPy/scikit-learn. Other thread pools
# are also capped with threadpoolctl to avoid memory/CPU oversubscription.
for _key in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[_key] = "2"

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesClassifier, HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import train_stft_classical as base


def json_text(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def candidate_id(family, params):
    return family + "_" + hashlib.sha256(json_text(params).encode()).hexdigest()[:12]


def create_plan(seed: int):
    """Deterministic, diverse 72-job randomized shortlist; interleaved families."""
    rng = random.Random(seed)
    mlp = [dict(hidden_layer_sizes=list(arch), alpha=alpha, learning_rate_init=lr,
                batch_size=batch, max_iter=220)
           for arch, alpha, lr, batch in itertools.product(
               [(256, 128), (256, 128, 64), (512, 256), (512, 256, 128), (1024, 512, 256)],
               [1e-4, 1e-3, 1e-2], [3e-4, 1e-3], [256, 512])]
    hist = [dict(max_iter=n, max_leaf_nodes=leaves, learning_rate=lr,
                 l2_regularization=l2, min_samples_leaf=20)
            for n, leaves, lr, l2 in itertools.product(
                [150, 300, 450], [15, 31, 63, 127], [0.04, 0.08, 0.12], [0.1, 1.0])]
    extra = [dict(n_estimators=n, max_leaf_nodes=leaves, max_features=feat,
                  min_samples_leaf=minimum)
             for n, leaves, feat, minimum in itertools.product(
                 [300, 600], [63, 127], ["sqrt", 0.3], [2, 5])]
    forest = [dict(n_estimators=n, max_leaf_nodes=leaves, max_features=feat,
                   min_samples_leaf=minimum)
              for n, leaves, feat, minimum in itertools.product(
                  [200, 400], [63, 127], ["sqrt", 0.3], [2, 5])]
    groups = {}
    for family, candidates, count in (("mlp", mlp, 30), ("histgb", hist, 24),
                                       ("extratrees", extra, 10), ("randomforest", forest, 8)):
        rng.shuffle(candidates)
        groups[family] = [{"id": candidate_id(family, p), "family": family, "params": p}
                          for p in candidates[:count]]
    # Round-robin for broad coverage even when the user runs only one night.
    result = []
    for i in range(max(map(len, groups.values()))):
        for family in ("mlp", "histgb", "extratrees", "randomforest"):
            if i < len(groups[family]):
                result.append(groups[family][i])
    return result


def append_csv_atomic(path: Path, records: list[dict]):
    temp = path.with_name(path.name + ".tmp")
    columns = ["id", "family", "status", "params", "validation_macro_f1",
               "validation_accuracy", "fit_seconds", "validation_seconds", "train_n",
               "validation_n", "started_at", "error"]
    with temp.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def load_rows(path: Path):
    if not path.exists():
        return []
    return pd.read_csv(path, dtype=str, keep_default_na=False).to_dict("records")


def build_scaler_caches(train_x, val_x, tag, scratch_dir):
    """Scale once across MLP experiments, in chunks; only training fits scaler."""
    scratch_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"fold_scaled_{tag}"
    train_path = scratch_dir / f"{prefix}_train.npy"
    val_path = scratch_dir / f"{prefix}_validation.npy"
    scaler_path = scratch_dir / f"{prefix}_scaler.joblib"
    if all(p.exists() for p in (train_path, val_path, scaler_path)):
        tx = np.load(train_path, mmap_mode="r")
        vx = np.load(val_path, mmap_mode="r")
        if tx.shape == train_x.shape and vx.shape == val_x.shape:
            print("Reusing standardized training/validation caches", flush=True)
            return tx, vx, joblib.load(scaler_path)
    print("Scaling cached training data ONCE for all MLP candidates...", flush=True)
    scaler = StandardScaler().fit(train_x)
    for src, dst in ((train_x, train_path), (val_x, val_path)):
        temp = dst.with_name(dst.stem + ".inprogress.npy")
        mmap = np.lib.format.open_memmap(temp, mode="w+", dtype=np.float32, shape=src.shape)
        for start in range(0, len(src), 4096):
            stop = min(start + 4096, len(src))
            mmap[start:stop] = scaler.transform(src[start:stop]).astype(np.float32, copy=False)
        mmap.flush()
        del mmap
        temp.replace(dst)
    joblib.dump(scaler, scaler_path)
    return (np.load(train_path, mmap_mode="r"),
            np.load(val_path, mmap_mode="r"), scaler)


def build_model(family, params, seed, cores):
    p = dict(params)
    if family == "mlp":
        p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
        return MLPClassifier(activation="relu", solver="adam", early_stopping=False,
                             n_iter_no_change=18, tol=1e-4, random_state=seed, **p)
    if family == "histgb":
        return HistGradientBoostingClassifier(early_stopping=False,
                                              random_state=seed, **p)
    if family == "extratrees":
        return ExtraTreesClassifier(n_jobs=cores, random_state=seed, **p)
    if family == "randomforest":
        return RandomForestClassifier(n_jobs=cores, random_state=seed, bootstrap=True, **p)
    raise ValueError(f"Unknown family: {family}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--data-root", type=Path, default=base.DEFAULT_ROOT)
    ap.add_argument("--cache-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--artifact-dir", type=Path, default=None)
    ap.add_argument("--hours", type=float, default=9.0,
                    help="Stop starting new jobs after this many hours (does not interrupt a fit)")
    ap.add_argument("--max-runs", type=int, default=0,
                    help="Maximum new fits for this invocation; 0 means all")
    ap.add_argument("--cores", type=int, default=2,
                    help="Parallel workers for tree ensembles (1-4 recommended on laptops)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--families", nargs="+", default=["mlp", "histgb", "extratrees", "randomforest"],
                    choices=["mlp", "histgb", "extratrees", "randomforest"])
    ap.add_argument("--list-only", action="store_true", help="Print plan and exit")
    args = ap.parse_args()
    if not 1 <= args.fold <= 5:
        ap.error("--fold must be 1 through 5")
    if args.hours <= 0:
        ap.error("--hours must be positive")
    if args.max_runs < 0:
        ap.error("--max-runs cannot be negative")
    if not 1 <= args.cores <= 16:
        ap.error("--cores must be between 1 and 16")

    root = args.data_root.expanduser().resolve()
    cache_dir = (args.cache_dir or root.parent.parent / "feature_cache" / "stft_axis_stats_v1").expanduser()
    out_dir = (args.out_dir or base.REPO_ROOT / "benchmarks" / "results" /
               "uniqueship_stft" / f"fold{args.fold}_overnight_sweep").expanduser()
    artifact_dir = (args.artifact_dir or root.parent.parent / "model_artifacts" /
                    "stft_axis_stats_v1" / "sweeps").expanduser()
    if not root.is_dir():
        ap.error(f"Dataset root does not exist: {root}")
    plan = [p for p in create_plan(args.seed) if p["family"] in args.families]
    if args.list_only:
        print(f"Planned {len(plan)} candidates ({args.families}):")
        for i, j in enumerate(plan, 1):
            print(f"{i:3}. {j['family']:<12} {json_text(j['params'])}")
        return

    print(f"Starting Fold {args.fold}, budget {args.hours:g}h, {len(plan)} candidates, "
          f"cores={args.cores}, seed={args.seed}", flush=True)
    raw, path_col, label_col, label_map = base.load_csvs(root, args.fold, None, None)
    # Run original metadata leakage checks once; warnings are not silently ignored.
    base.validate_folds(raw, path_col, label_col)
    split_paths = {}
    for split in ("train", "validation"):
        df = raw[split]
        tag = base.cache_tag(df, path_col, label_col, split, args.fold)
        p = cache_dir / f"{tag}_X.npy"
        if not p.exists():
            raise FileNotFoundError(f"Required FULL cached features missing: {p}\n"
                                    "Run existing full train_stft_classical.py first. "
                                    "This script deliberately DOES NOT re-extract audio.")
        split_paths[split] = (p, tag)
    xtr = np.load(split_paths["train"][0], mmap_mode="r")
    xv = np.load(split_paths["validation"][0], mmap_mode="r")
    if xtr.shape != (len(raw["train"]), base.FEATURE_DIM) or xv.shape != (len(raw["validation"]), base.FEATURE_DIM):
        raise ValueError(f"Wrong cache shapes: training={xtr.shape}, validation={xv.shape}")
    ytr = np.asarray([label_map[str(x)] for x in raw["train"][label_col]], dtype=np.int32)
    yv = np.asarray([label_map[str(x)] for x in raw["validation"][label_col]], dtype=np.int32)
    print(f"Loaded cached features: train {xtr.shape}, validation {xv.shape}. "
          "TEST REMAINS UNTOUCHED.", flush=True)

    out_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "sweep_results.csv"
    plan_path = out_dir / "sweep_plan.json"
    fingerprint = hashlib.sha256(json_text({"train": split_paths["train"][1],
                                             "validation": split_paths["validation"][1],
                                             "root": str(root), "version": base.VERSION,
                                             "fold": args.fold, "seed": args.seed}).encode()).hexdigest()
    if plan_path.exists():
        old = json.loads(plan_path.read_text())
        if old["fingerprint"] != fingerprint:
            raise ValueError(f"Dataset/caches/seed have changed; refusing to reuse {results_path}. "
                             "Choose a new --out-dir for a different dataset/sweep.")
    else:
        plan_path.write_text(json.dumps({"fingerprint": fingerprint,
                                         "dataset_root": str(root),
                                         "feature_version": base.VERSION,
                                         "candidates": plan}, indent=2), encoding="utf-8")
    rows = load_rows(results_path)
    done = {r["id"] for r in rows if r.get("status") == "ok"}
    # Also skip completed models whose training failed last time; -- retry by removing
    # the corresponding row from results CSV if desired.
    done |= {r["id"] for r in rows if r.get("status") == "failed"}
    print(f"Previously completed: {len(done)}; remaining: {len([p for p in plan if p['id'] not in done])}",
          flush=True)

    scaled_xtr = scaled_xv = scaler = None
    scaled_tag = hashlib.sha256((split_paths["train"][1] + split_paths["validation"][1]).encode()).hexdigest()[:14]
    beginning = time.monotonic()
    fits = 0
    with threadpool_limits(limits=2):
        for job in plan:
            if job["id"] in done:
                continue
            if time.monotonic() - beginning >= args.hours * 3600:
                print("Time budget met; stopping before next candidate.", flush=True)
                break
            if args.max_runs and fits >= args.max_runs:
                break
            family, params = job["family"], job["params"]
            print(f"\n[{fits+1} new] {job['id']} {json_text(params)}", flush=True)
            started_at = datetime.now().astimezone().isoformat(timespec="seconds")
            tic = time.perf_counter()
            outcome = {"id": job["id"], "family": family, "params": json_text(params),
                       "train_n": len(ytr), "validation_n": len(yv), "started_at": started_at,
                       "status": "failed", "validation_macro_f1": "", "validation_accuracy": "",
                       "fit_seconds": "", "validation_seconds": "", "error": ""}
            try:
                if family == "mlp":
                    if scaled_xtr is None:
                        scaled_xtr, scaled_xv, scaler = build_scaler_caches(
                            xtr, xv, scaled_tag, cache_dir / "sweep_scaled")
                    train_data, val_data = scaled_xtr, scaled_xv
                else:
                    train_data, val_data = xtr, xv
                est = build_model(family, params, args.seed, args.cores)
                with warnings.catch_warnings(record=True) as warn_list:
                    warnings.simplefilter("always")
                    fit_tic = time.perf_counter()
                    est.fit(train_data, ytr)
                    outcome["fit_seconds"] = round(time.perf_counter() - fit_tic, 3)
                pred_tic = time.perf_counter()
                pred = est.predict(val_data)
                outcome["validation_seconds"] = round(time.perf_counter() - pred_tic, 3)
                outcome["validation_macro_f1"] = float(f1_score(yv, pred, average="macro", zero_division=0))
                outcome["validation_accuracy"] = float(accuracy_score(yv, pred))
                outcome["status"] = "ok"
                if warn_list:
                    # Nonconvergence is common during sweeps; record it, don't hide it.
                    print("Training warnings:", "; ".join(str(w.message)[:180] for w in warn_list[:2]),
                          flush=True)
                print(f"Finished in {time.perf_counter()-tic:.1f}s: "
                      f"validation F1={outcome['validation_macro_f1']:.4%}, "
                      f"accuracy={outcome['validation_accuracy']:.4%}", flush=True)
                family_prior = [float(r["validation_macro_f1"]) for r in rows
                                if r.get("family") == family and r.get("status") == "ok"]
                if not family_prior or outcome["validation_macro_f1"] > max(family_prior):
                    # For MLP, add the already fitted training-only scaler back so the
                    # saved estimator expects RAW 1206-D features, just like original.
                    fitted = Pipeline([("scaler", scaler), ("mlp", est)]) if family == "mlp" else est
                    payload = {"estimator": fitted, "label_map": label_map,
                               "config_version": base.VERSION, "feature_dim": base.FEATURE_DIM,
                               "fold": args.fold, "family": family, "params": params,
                               "validation_macro_f1": outcome["validation_macro_f1"],
                               "dataset_root": str(root), "split_fingerprint": fingerprint,
                               "selected_on": "validation_only_not_test"}
                    checkpoint = artifact_dir / f"fold{args.fold}_{family}_sweep_best.joblib"
                    tmp = checkpoint.with_name(checkpoint.name + ".tmp")
                    joblib.dump(payload, tmp, compress=3)
                    tmp.replace(checkpoint)
                    print(f"NEW BEST {family}: saved {checkpoint}", flush=True)
                del est
            except KeyboardInterrupt:
                print("Interrupted: prior checkpoints and results are safe. Rerun to resume.", flush=True)
                raise
            except Exception as exc:
                outcome["error"] = f"{type(exc).__name__}: {exc}"[:450]
                print("FAILED:", outcome["error"], flush=True)
                traceback.print_exc(limit=5)
            rows.append(outcome)
            done.add(job["id"])
            append_csv_atomic(results_path, rows)
            completed = pd.DataFrame([r for r in rows if r.get("status") == "ok"])
            if not completed.empty:
                completed["validation_macro_f1"] = pd.to_numeric(completed["validation_macro_f1"])
                completed["validation_accuracy"] = pd.to_numeric(completed["validation_accuracy"])
                completed.sort_values(["validation_macro_f1", "validation_accuracy"],
                                      ascending=False).to_csv(out_dir / "leaderboard.csv", index=False)
            fits += 1
            print(f"Saved progress ({fits} new, {len(done)}/{len(plan)} planned); "
                  f"elapsed {(time.monotonic()-beginning)/3600:.2f} hours", flush=True)

    print("\nSweep finished/paused. No test evaluations were performed.", flush=True)
    print(f"Results: {results_path}\nLeaderboard: {out_dir / 'leaderboard.csv'}\n"
          f"Best per-family models: {artifact_dir}", flush=True)

if __name__ == "__main__":
    main()
