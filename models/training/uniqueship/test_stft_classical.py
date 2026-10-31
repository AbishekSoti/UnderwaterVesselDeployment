#!/usr/bin/env python3
"""Test-only evaluation of already-trained UniqueShip STFT classical models.

Put this file NEXT TO train_stft_classical.py at:
  UnderwaterVesselDeployment/models/training/uniqueship/test_stft_classical.py

IMPORTANT: This script NEVER fits or retrains a model. It loads the saved
full-fold Joblib checkpoints and extracts only the test split's features.

Default: UniqueShip 5-Class Balanced, official fold 1, all four classifiers.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Import the exact same path resolution, STFT frontend, caching and split audit
# as the original training run. Keep the two scripts in the same folder.
import train_stft_classical as base

MODEL_FILES = {
    "logreg": "fold{fold}_logreg_full.joblib",
    "rbf_svm": "fold{fold}_rbf_svm_approx_full.joblib",
    "mlp": "fold{fold}_mlp_full.joblib",
    "histgb": "fold{fold}_histgb_full.joblib",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_ROOT)
    parser.add_argument("--models", nargs="+", choices=list(MODEL_FILES), default=list(MODEL_FILES))
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--artifact-dir", type=Path, default=None)
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--rebuild-test-cache", action="store_true")
    args = parser.parse_args()

    if args.fold not in range(1, 6):
        parser.error("--fold must be 1..5")

    root = args.data_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Missing dataset directory: {root}")

    artifact_dir = args.artifact_dir or root.parent.parent / "model_artifacts" / "stft_axis_stats_v1"
    cache_dir = args.cache_dir or root.parent.parent / "feature_cache" / "stft_axis_stats_v1"
    results_dir = args.results_dir or base.REPO_ROOT / "benchmarks" / "results" / "uniqueship_stft"

    print("TEST ONLY: loading existing trained models; no fitting or retraining.", flush=True)
    print(f"Fold: {args.fold}; STFT: {base.F_BINS}x{base.T_FRAMES}; features: {base.FEATURE_DIM}")

    # Preflight: refuse to waste time extracting audio when any requested model
    # is missing or was saved from a different fold / frontend / label mapping.
    raw, path_col, label_col, label_map = base.load_csvs(root, args.fold, None, None)
    base.validate_folds(raw, path_col, label_col)

    estimators = {}
    for key in dict.fromkeys(args.models):
        checkpoint_path = artifact_dir / MODEL_FILES[key].format(fold=args.fold)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Saved FULL model not found: {checkpoint_path}\n"
                "Run the full training script first. This script will not retrain it."
            )
        payload = joblib.load(checkpoint_path)
        if not isinstance(payload, dict) or "estimator" not in payload:
            raise ValueError(f"Unexpected checkpoint format: {checkpoint_path}")
        expected = {
            "fold": args.fold,
            "feature_dim": base.FEATURE_DIM,
            "config_version": base.VERSION,
            "label_map": label_map,
        }
        for field, expected_value in expected.items():
            if payload.get(field) != expected_value:
                raise ValueError(
                    f"{checkpoint_path.name}: {field} mismatch: "
                    f"checkpoint={payload.get(field)!r}, expected={expected_value!r}"
                )
        estimators[key] = payload["estimator"]
        print(f"Loaded {key}: {checkpoint_path.name}", flush=True)

    # Test only: no extraction of training or validation features.
    test_df = raw["test"].reset_index(drop=True)
    x_test, y_test = base.extract_split(
        test_df, root, path_col, label_col, label_map,
        cache_dir, "test", args.fold, args.rebuild_test_cache
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    class_names = [name for name, _ in sorted(label_map.items(), key=lambda item: item[1])]
    label_indices = list(range(len(class_names)))

    summary = []
    for key, estimator in estimators.items():
        t0 = time.perf_counter()
        preds = np.asarray(estimator.predict(x_test), dtype=np.int64)
        predict_seconds = time.perf_counter() - t0
        if preds.shape != y_test.shape:
            raise ValueError(f"{key}: predicted shape {preds.shape} != labels shape {y_test.shape}")

        accuracy = float(accuracy_score(y_test, preds))
        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))

        summary.append({
            "model": key,
            "rbf_mode": "approx" if key == "rbf_svm" else "",
            "fold": args.fold,
            "test_n": len(y_test),
            "test_accuracy": accuracy,
            "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1,
            "predict_seconds_total": round(predict_seconds, 4),
        })

        # Per-recording predictions for future Daniel Bourke-style EDA and plots.
        per_clip = pd.DataFrame({
            "path": test_df[path_col].astype(str).to_numpy(),
            "true_class": [class_names[int(k)] for k in y_test],
            "predicted_class": [class_names[int(k)] for k in preds],
            "true_label": y_test.astype(int),
            "predicted_label": preds,
            "correct": preds == y_test,
        })
        for metadata_col in ("MMSI", "date", "hydrophone"):
            if metadata_col in test_df.columns:
                per_clip[metadata_col] = test_df[metadata_col].to_numpy()
        per_clip_path = results_dir / f"fold{args.fold}_{key}_full_test_predictions.csv"
        per_clip.to_csv(per_clip_path, index=False)

        matrix = confusion_matrix(y_test, preds, labels=label_indices)
        counts = pd.DataFrame(matrix, index=class_names, columns=class_names)
        counts.index.name = "actual\\predicted"
        counts.to_csv(results_dir / f"fold{args.fold}_{key}_full_test_confusion.csv")
        row_sums = matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
        pd.DataFrame(norm_matrix, index=class_names, columns=class_names).to_csv(
            results_dir / f"fold{args.fold}_{key}_full_test_confusion_normalized.csv"
        )
        report = classification_report(
            y_test, preds, labels=label_indices, target_names=class_names,
            output_dict=True, zero_division=0,
        )
        (results_dir / f"fold{args.fold}_{key}_full_test_classification_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(f"{key:9} | accuracy={accuracy:.4%} | macro-F1={macro_f1:.4%} "
              f"| weighted-F1={weighted_f1:.4%} | predict={predict_seconds:.2f}s", flush=True)

    summary_df = pd.DataFrame(summary)
    metrics_path = results_dir / f"fold{args.fold}_full_test_only_metrics.csv"
    summary_df.to_csv(metrics_path, index=False)
    print("\nFinal held-out test results (no retraining):")
    print(summary_df.to_string(index=False, float_format=lambda number: f"{number:.4f}"))
    print(f"\nSaved summary: {metrics_path}")
    print(f"Saved per-clip predictions, confusion matrices, and class reports in: {results_dir}")
    print("NOTE: background-day partition warnings are still unresolved; audit separately.")


if __name__ == "__main__":
    main()
