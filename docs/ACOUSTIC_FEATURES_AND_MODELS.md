# Underwater Vessel Classification — Acoustic Features & Model Reference

**Maintained reference for `UnderwaterVesselDeployment`**  
**Recommended repository location:** `docs/ACOUSTIC_FEATURES_AND_MODELS.md`  
**Compiled:** 26 October 2026  
**Status:** Historical configurations recovered from the two paper drafts and accompanying experiment code. **Not** a claim that all implementations or UniqueShip experiments are reproduced or complete.

## 0. What to implement first on UniqueShip

**Stage 1: STFT + four classical models.** Use the *already downloaded* **UniqueShip 5-Class Balanced** release (213 h, five vessel-disjoint split repetitions). Do **not** use a random clip-level split.

| Order | Classifier | Frontend input | Classifier input | Historical comparison (ShipsEar) |
|---:|---|---|---:|---|
| 1 | Logistic Regression | STFT, time mean+std, train-only scaling | **256-D** | Pi 5 ShipsEar: 87.87% accuracy, 88.03% macro-F1 |
| 2 | Linear SVM | Same | **256-D** | Pi 5 ShipsEar: 90.34% accuracy, 89.70% macro-F1 |
| 3 | RBF-SVM | Same | **256-D** | Pi 5 ShipsEar: 98.88% accuracy, 98.83% macro-F1 |
| 4 | MLP | Same | **256-D** | Pi 5 ShipsEar: 96.40% accuracy, 96.49% macro-F1 |

**Model identity matters:** Linear SVM and RBF-SVM are two **separate** models; together with logistic regression and MLP they are the **four classical classifiers** in the ISCAS paper. The two-layer CNN and Patch Transformer are the additional two, making six deployed models. A 4-layer CNN is also documented for DeepShip, but is not one of those four initial STFT classifiers.

**Do not transplant scores:** these are historical ShipsEar fixed-test-set results, **not UniqueShip results**. ShipsEar recording-level independence was not reconstructable. UniqueShip's fold CSVs should define membership and prevent MMSI leakage.

## 1. Feature-size glossary

Notation:
- `F × T` = frequency/cochlear channels × temporal frames for **one** 5-s clip.
- `1 × F × T` = CNN/Transformer tensor with a singleton input channel.
- `2F` = classical classifier feature vector after concatenating **temporal mean and temporal standard deviation**, one of each for every frequency/channel.
- Shapes below are **historical 16-kHz ShipsEar implementations**, unless marked otherwise. A different dataset, sampling rate, padding, hop, or frontend version can change `T` and/or `F`.

| Representation | Prior feature map (`F × T`) | Mean+std vector | Historical configuration | Status / caution |
|---|---:|---:|---|---|
| **STFT (log-spaced remap)** | **128 × 475** | **256-D** | FFT 4096, hop 160, `center=False`, remove DC, interpolate power spectrum to 128 log-spaced frequencies, convert to dB | **Canonical stage-1 frontend**; this is not simply the first 128 FFT bins |
| **Mel spectrogram** | **256 × 494** | **512-D** | FFT 1024, hop 160, 256 Mel bands, power/dB, `center=False` | ShipsEar exploratory configuration; DeepShip paper also used **128 × 494 / 256-D** |
| **MFCC** | **63 × 488** | **126-D** | FFT 2048, hop 160, 256 Mel bands → 64 coefficients, drop coefficient 0, `center=False` | ShipsEar / DeepShip historic 63-coefficient configuration |
| **CQT** | **252 × 501** | **504-D** | hop 160, 36 bins/octave, 252 bins, minimum frequency 50 Hz | Historical high-resolution CQT; other CQT sweeps used fewer bins, e.g. 126 |
| **Gammatone envelope** | **252 × 500** *calculated for a particular prototype* | **504-D** *if mean+std applied* | 252 ERB-spaced bands, 50–8000 Hz, 4th-order gammatone FIR impulse response ~0.05 s, Hilbert envelope, block-average at 160 samples, log1p | **Experimental/code setting, not a validated canonical paper result** |
| **GFCC** | **TBD** | **2F once channel count is verified** | Gammatone-frequency cepstral coefficients | Mentioned in DeepShip feature experiments; **do not invent** coefficient count/shape |
| **CARFAC BM** | **100 × 490** | **200-D** | Basilar-membrane signal, historical CARFAC feature map | Earlier experiment/cache shape; **not yet run on UniqueShip** |
| **CARFAC IHC** | **100 × 490** | **200-D** | Inner hair-cell signal, historical CARFAC feature map | Earlier experiment/cache shape; **not yet run on UniqueShip** |
| **CARFAC LI** | **100 × 490** | **200-D** | Historical LI auditory stage | Used in ShipsEar classical experiments; **not in the corrected DeepShip CNN4 BM/IHC benchmark** |
| **CARFAC SPK** | **100 × 490** | **200-D** | Historical spiking stage | Used in ShipsEar classical experiments; **not in the corrected DeepShip CNN4 BM/IHC benchmark** |

**Other variant:** concatenating pooled BM + IHC + LI + SPK features gives `4 × 200 = 800-D` in the recorded ShipsEar `ALL_CONCAT` CARFAC experiment. That is a separate ablation, **not** a single feature channel.

**Important source distinction:** the ISCAS paper focuses on STFT, pooling, CNN and Transformer deployment; the larger feature inventory and most non-STFT sizes come from the earlier *Towards Deployable...* study and saved experiment specifications. Some variants (e.g., Mel 64/128/256, CQT 126/252) differ across model sweeps and are **not interchangeable**.

## 2. Canonical STFT pipeline (historical ShipsEar)

1. Load 5 seconds of audio as mono, **16,000 Hz** (`N = 80,000` samples).
2. STFT: `n_fft = 4096`, `hop_length = 160`, `center = False`; default analysis window in the recovered librosa code (confirm pinned librosa version).
3. Compute **power**: `abs(STFT) ** 2`.
4. Drop DC; use `np.geomspace` and per-frame interpolation to **128 log-spaced frequency positions** spanning the available non-DC band.
5. Convert to **dB**, `librosa.power_to_db(..., ref=np.max)`.
6. Output shape: **`(128, 475)`**, since `floor((80000-4096)/160)+1 = 475`.
7. For classical classifiers: take mean and standard deviation **across time** for every row, concatenate to `(256,)`. Fit any `StandardScaler`/normalization **on training samples only**, then transform validation/test.
8. For CNN and Transformer: retain **`(1, 128, 475)`** feature map. The CNN does learned temporal pooling later; the Transformer tokenises the map.

**Contrast with unpooled STFT:** `128 × 475 = 60,800` flattened numbers. Mean+std pooling reduces this to **256**, a `237.5×` reduction. CNN pooled head size is different (see §4).

### Transferring this to UniqueShip (20 kHz native)

UniqueShip provides 5-second clips at **20 kHz**. The historical dimension `(128,475)` applies **only** after resampling to 16 kHz (with the previous 4096/160/`center=False` settings). For native 20 kHz audio and unchanged FFT/hop the estimated map is:

- `100,000` samples → **`128 × 600`** (calculated with `floor((100000-4096)/160)+1`).
- Temporal mean+std still gives **256-D** with 128 output channels.
- The physical FFT window/hop durations change with sample rate; native-20-kHz and resampled-16-kHz settings should be registered as **distinct frontends**, never silently mixed.

**Recommended first controlled run:** 20 kHz source → explicit, documented resample to **16 kHz** → exact historical STFT 128×475 → mean+std 256-D → four classifiers. Later compare a native-20-kHz frontend as a separate experiment. Do not use saved ShipsEar scalers, models or class mappings with UniqueShip.

## 3. Historical model inventory

| Model | Original experiment | Input | Architecture / parameter record | Notes |
|---|---|---|---|---|
| Logistic Regression | ShipsEar / Pi 5 | pooled STFT **256-D** | scikit-learn linear classifier; fitted model is data dependent | In initial UniqueShip quartet |
| Linear SVM | ShipsEar / Pi 5 | pooled STFT **256-D** | scikit-learn linear margin classifier | In initial quartet |
| RBF-SVM | ShipsEar / Pi 5 / Pi Zero | pooled STFT **256-D** | SVC with RBF kernel; an earlier saved baseline used `C=10`, `gamma='scale'` — **verify final training config** | In initial quartet; strongest historic classical baseline |
| MLP | ShipsEar / Pi 5 | pooled STFT **256-D** | scikit-learn MLP; **exact hidden layers should be copied from the final saved model/config**, not inferred from total file size | In initial quartet |
| CNN2 | ShipsEar | **1 × 128 × 475** STFT | 2 conv layers, historical channels **24 → 80**, kernels **5×15 / 9×9**; pooling then mean+std; **183,237 params** in selected STFT version | Later phase; different setup from the DeepShip CNN4 |
| Patch Transformer | ShipsEar | **1 × 128 × 475** STFT | full-frequency **128×10** patches; **47 tokens**, **303,621 params** in selected exploratory configuration | Later phase; compare with 128×20 and 8×20 ablations |
| CNN4 (TinyVGG-like) | DeepShip | STFT/Mel/MFCC/CQT/BM/IHC feature maps | **32 → 32 → 64 → 64** conv channels; kernels **5×15, 9×9, 3×3, 3×3**; 2×2 pools; temporal mean+std; ~149K–173K params depending on frontend | Separate four-class DeepShip experiment; later UniqueShip comparison possible |
| ResNet18 | DeepShip exploratory | spectrogram input | ~**11.17M** params | Larger diagnostic comparator; not initial stage |
| AST teacher / KD student | DeepShip exploratory | separate AST preprocessing | teacher ~**86.2M** params; *student size varied among exploratory drafts* | Research direction, **not** part of initial four models and no demonstrated successful UniqueShip distillation |

**CNN temporal-pooling dimensionality:** for the published ShipsEar CNN2 with learned map `80 × 32 × 118`, the final flattened classifier input is `2 × 80 × 32 = **5,120-D**`; without temporal pooling it is `80 × 32 × 118 = **302,080-D**`. This is **not** the same as the classical model's 256-D pooled input.

## 4. Historical selected results (not UniqueShip results)

### ShipsEar, frozen 1,778-train / 445-test split

| Model / representation | Historical metric | Historical value |
|---|---|---:|
| CNN2, STFT | One selected run, test macro-F1 | **0.9918** |
| RBF-SVM, STFT | Test macro-F1 | **0.9883** |
| CNN2, CQT | Selected test macro-F1 | **0.9839** |
| CNN2, Mel | Selected test macro-F1 | **0.9789** |
| RBF-SVM, LI | Selected test macro-F1 | **0.9712** |
| RBF-SVM, MFCC | Selected test macro-F1 | **0.9677** |
| CNN2, IHC | Selected test macro-F1 | **0.9640** |
| CNN2, BM | Selected test macro-F1 | **0.9606** |
| Patch Transformer, STFT | Exploratory test macro-F1 | **0.9711** |

Separate five-seed runs on the fixed ShipsEar split previously gave approximate macro-F1 means: logistic regression 88.03%, linear SVM 89.70%, RBF-SVM 98.83%, MLP **96.08% ± 0.71**, CNN2 **99.14% ± 0.32**, Patch Transformer **96.48% ± 0.63**. Some classical estimators were deterministic under the fixed split. **Do not treat five random seeds as five independent vessels-disjoint folds**. The Pi 5 deployed model checkpoint/test values in §0 are a different reporting slice.

### DeepShip, recording-level split, four-class CNN4

| Feature | Historical four-layer CNN parameters | Val macro-F1 | Test macro-F1 |
|---|---:|---:|---:|
| Mel | ~157K | 0.6519 | **0.7226** |
| STFT | ~157K | 0.6351 | 0.7129 |
| CQT | ~173K | 0.6408 | 0.6902 |
| MFCC | ~149K | 0.6162 | 0.6735 |
| CARFAC BM | ~154K | 0.6766 | 0.6694 |
| CARFAC IHC | ~154K | 0.6562 | 0.6336 |

These are different datasets/class definitions/evaluation splits; do **not** rank them directly against ShipsEar or UniqueShip test scores.

## 5. UniqueShip experiment contract

**Data:** `Local Documents/Datasets/UniqueShip/5class_balanced/audio/main5_balanced/` on the Mac. Preserve the exact directory and fold CSV paths; the CSVs are under `fold_1/` ... `fold_5/` and audio under `audio/`.

**For each of the five split repetitions:**

- Resolve CSV relative filepaths to audio files; inspect labels and class-ID mapping.
- Assert **no MMSI overlap** between train, validation and test ship samples; verify background-day separation according to supplied metadata. Do **not** randomly resplit five-second clips.
- Train on the fold's training partition only; fit any normalization, feature standardization and hyperparameter selection using train/validation, never test.
- Use one canonical STFT extractor and a deterministic preprocessing configuration across the **four stage-1 classifiers**.
- Report sample counts, MMSI counts, class balance, per-class precision/recall/F1, macro-F1, accuracy, confusion matrix, trained feature shape, wall time, peak memory and total training/inference time.
- Report **five-split mean ± sample standard deviation**, and also raw fold metrics; don't conflate fold variation with training-seed variation.
- **Memory management:** 153k+ raw STFT maps occupy tens of GB as float32; process WAVs in batches/chunks, and cache only compact pooled vectors or use on-disk arrays. Do not preload the entire spectrogram corpus into RAM.
- Keep class counts explicit. UniqueShip 5-class = four vessels + background; UniqueShip 12-class 5-hour release (not currently downloaded) is a *separate* future benchmark.

### Roadmap

- **Now:** feature `STFT-128` + Logistic Regression, Linear SVM, RBF-SVM, MLP (256-D pooled inputs).
- **Next:** CNN2 and Patch Transformer on raw 1×128×475 STFT maps; compare against the four baselines.
- **Then:** Mel 256/128 variants, MFCC 63, CQT 252/126, and possibly Gammatone/GFCC as *explicit distinct configurations*.
- **CARFAC soon, but not part of stage 1:** BM/IHC, then LI/SPK where code and labels are verified. These may cost considerably more feature-generation time; cache per fold with provenance.
- **Later:** four-layer DeepShip-style CNN, embedding/distillation experiments, and edge timing on representative hardware.

## 6. Provenance, versions and unresolved details

**Primary papers**

1. *Towards Deployable Underwater Vessel Classification* (Soti et al., 2026), arXiv:2609.29179: <https://arxiv.org/abs/2609.29179> — ShipsEar/DeepShip, multiple representations, CNN2/CNN4, pooling, recording-level evaluation caveat.
2. *Lightweight Underwater Acoustic Vessel Classification for Embedded Deployment* (Soti et al., ISCAS 2027 manuscript, Oct 2026) — four classical classifiers + CNN2 + Patch Transformer, STFT128×475, embedded Pi benchmarking. Source was the user's 20 Oct paper-text draft, not a verified journal publication.

**Additional source for specific frontend settings:** historical `Audioclassification` experiment notebook code recovered from project conversation/library, including the exact `librosa` frontend constants listed in §1–2. A DeepShip CNN4 benchmark script lists `stft`, `mel`, `mfcc`, `cqt`, `gammatone`, `gfcc`, `bm`, `ihc`; earlier ShipsEar experiments additionally include `li` and `spk`.

**Known unresolved items (verify against actual saved code/checkpoints before calling them canonical):**

- Final MLP hidden-layer layout, activation, optimiser, and regularisation.
- Final SVM `C`, `gamma` and exact preprocessing settings for each trained artifact; `C=10` comes from an older baseline, not a fully reconstructed final specification.
- Exact GFCC frontend dimensionality and feature shape, plus whether Gammatone/GFCC caches were successfully generated in the final DeepShip split.
- Full CARFAC filterbank/channel-count/time-resolution parameter sets; the shapes given are historical cache shapes.
- Mel and CQT variant names to distinguish 64/128/256 Mel and 126/252 CQT experiments.
- UniqueShip 20→16 kHz resampling filter implementation and exact final fold mapping.

**Do not overwrite source experiments.** This file is a *reference registry*, not executable configuration. When new runs establish final settings, write a versioned experiment config and update this document with the config filename/commit, rather than silently changing historical measurements.
