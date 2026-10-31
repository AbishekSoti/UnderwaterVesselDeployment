# UniqueShip 5-class — STFT axis statistics + four scikit-learn classifiers

**Repository location:** `UnderwaterVesselDeployment/models/training/uniqueship/train_stft_classical.py`  
**Small metrics:** `UnderwaterVesselDeployment/benchmarks/results/uniqueship_stft/`  
**Audio and feature caches (outside Git):** `~/Local Documents/Datasets/UniqueShip/5class_balanced/`

## Intended experiment

UniqueShip supplies 20-kHz 5-second clips. Explicitly resample to **16 kHz**, apply the historical ShipsEar STFT frontend (`n_fft=4096`, `hop=160`, `center=False`, **128 log-frequency-remapped bins**, log-power dB), and get **128 × 475** spectrograms. This reproduces the *frontend convention* from earlier experiments, not the original ShipsEar classifier performance.

On the `(128,475)` map `S`:

```python
frequency_mean = S.mean(axis=1)  # 128, averages over time
frequency_std  = S.std(axis=1)   # 128, varies over time
time_mean      = S.mean(axis=0)  # 475, averages over frequency
time_std       = S.std(axis=0)   # 475, varies over frequency
features = np.concatenate([frequency_mean, frequency_std, time_mean, time_std])
# shape: (1206,)
```

**Important limitation:** this preserves a separate marginal trace across frequency and time, but *does not retain all local time–frequency correlations*. A CNN on the unpooled map would preserve more two-dimensional structure.

## Four models

1. `LogisticRegression` + train-only `StandardScaler`: fast, regularised linear baseline.
2. RBF-SVM: **exact** `SVC(kernel='rbf')` for quick runs, or **approximate** `Nystroem(RBF) + LinearSVC` for large-sample experiments. Always identify which method was used; they are not the same trained model.
3. `MLPClassifier` + train-only `StandardScaler`: dense neural nonlinear baseline, `(128,64)` hidden layers.
4. `HistGradientBoostingClassifier`: nonlinear decision trees, distinct inductive bias; scales better than ordinary gradient boosting to many samples.

**Scalability:** Exact RBF `SVC` is at least quadratic in sample count and may be impractical for ~120k training clips. Train all models on the same small subset for a screening comparison, and report the full-dataset Nystroem result as an **approximation**, not an exact RBF-SVM reproduction.

## Install (in the repository's Python environment)

```bash
python -m pip install numpy pandas scipy scikit-learn librosa soundfile joblib
```

## Run from the repository root

```bash
python models/training/uniqueship/train_stft_classical.py --inspect-only
```

`--inspect-only` prints the actual metadata columns, sample labels, and verifies disjoint paths/MMSI (when the CSV has an identifiable MMSI column). If column detection cannot find a path or class label, specify `--path-column NAME --label-column NAME`. The author-defined `fold_1` CSVs are used; no random partition is created.

**First small screening run** (all four; avoids generating the full dataset's features):

```bash
python models/training/uniqueship/train_stft_classical.py \
  --fold 1 --quick --rbf-mode exact
```

This caps samples at 2,000 training, 500 validation and 500 test, but **only train + validation are extracted/evaluated** unless `--evaluate-test` is included. The subsample is stratified *within* the author's split partitions. The numbers from a quick run **must not be reported as a complete benchmark**.

**Full 5-class fold (with scalable nonlinear kernel approximation):**

```bash
python models/training/uniqueship/train_stft_classical.py \
  --fold 1 --rbf-mode approx
```

**After selecting the method/settings using only validation**, evaluate the original held-out test partition explicitly:

```bash
python models/training/uniqueship/train_stft_classical.py \
  --fold 1 --rbf-mode approx --evaluate-test
```

This last command retrains all models under the chosen configuration. Feature caches are reused. Do not tune hyperparameters against repeated test performance.

Then evaluate all five author-provided folds (`--fold 1` through `--fold 5`), without mixing their train/test sample identities.

## Output and protection of raw data

- Feature matrices cache as `.npy` in `~/Local Documents/Datasets/UniqueShip/5class_balanced/feature_cache/stft_axis_stats_v1/`. The cache writes atomically: an interrupted `.inprogress.npy` file is not reused as complete.
- Models are saved in `~/Local Documents/Datasets/UniqueShip/5class_balanced/model_artifacts/stft_axis_stats_v1/`.
- Small CSV metrics and confusion matrices go in `benchmarks/results/uniqueship_stft/` inside the repo. Decide whether to commit metrics; do not commit large model artifacts or audio.
- A model fitted on `fold_1` cannot be reused as a `fold_2` model.
- `--rebuild-cache` forces a recalc. The cache key includes ordered path/label pairs and the fixed frontend version, but not raw file checksums. If source audio is changed, use `--rebuild-cache`.
- The code checks path and (when present) MMSI separation. The background *day* partitioning should also be audited using the release metadata before publication; the original authors' splits should be retained.

## Status and provenance

This implementation was smoke-tested against a **small synthetic CSV + WAV dataset**, not against the user's local files (which are on the user's Mac). Do not call it a completed UniqueShip result until its CSV schema and audio paths have been checked with `--inspect-only` and the real runs complete.
