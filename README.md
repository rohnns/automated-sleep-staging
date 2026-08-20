# Automated Sleep Staging

## Overview

An automated 30-second-epoch sleep staging pipeline for the **Sleep-EDF Expanded** dataset
(PhysioNet). It loads PSG (EDF) recordings with their hypnogram annotations via MNE, preprocesses
the continuous signal, maps Rechtschaffen & Kales (R&K) labels to the five-class **AASM** scheme
(`W`, `N1`, `N2`, `N3`, `REM`), encodes each 30 s epoch into one of three representations (raw
waveform, band-power, STFT time-frequency image), and trains one baseline model per representation
plus a classical logistic-regression baseline.

The headline experiment is an **external-generalization protocol**: models are trained and
model-selected entirely on the **Sleep Cassette (SC)** cohort, then evaluated exactly once, without
further tuning, on the entire **Sleep Telemetry (ST)** cohort. Final SC→ST results for all four
models are committed here (see [Results](#results)); a read-only Streamlit dashboard visualizes the
exported predictions.

## Dataset & Experimental Protocol

The pipeline does **not** download Sleep-EDF Expanded; point it at a local copy via `SLEEP_EDF_ROOT`
(takes precedence) or `acquisition.data_root`.

| Cohort | Full name | Recordings | Subjects | Population | Role |
| --- | --- | ---: | ---: | --- | --- |
| **SC** | Sleep Cassette | 153 | 78 unique (55 train + 12 validation used) | Healthy, studied at home | Train + validation |
| **ST** | Sleep Telemetry | 44 | 22 | Mild difficulty falling asleep, studied in hospital | External test |

**Protocol** (`training/sc_to_st.py::run_primary_experiment`). SC subjects are split
**subject-wise**, never recording-wise, 70/15/15 with `seed=42`; only the train (55) and validation
(12) partitions are used, and checkpoint selection plus early stopping use **SC validation macro-F1
only**. The full ST cohort is then evaluated **exactly once** after training — a genuine
cross-cohort test, not a within-SC holdout. Subject-disjointness is enforced in code
(`assert_no_subject_leakage`, `SubjectSplit.__post_init__`), and split grouping keys match
`EncodedDataset.subject_id` so a split cannot silently match zero recordings. Verified against the
committed `inventory.json`: **55 / 12 / 22** subjects, **41,148** supervised test epochs.

**Labels.** Fixed R&K→AASM table (`StageMapper`): `W→W`, `1→N1`, `2→N2`, `3→N3`, `4→N3`, `R→REM`;
`Movement time` and `Sleep stage ?` → `IGNORE`. R&K stages 3 and 4 merge into `N3` per the AASM
2007 manual. `IGNORE` epochs are **never dropped** — they keep epoch indexing aligned with the
signal and are excluded from loss and all metrics via `ignore_index=-100`. The pipeline builder
refuses `unmapped_policy="drop"`, which would desynchronize signal/label indexing. A separate
**legacy** experiment reuses the same SC split but tests on the leftover 11 SC subjects — see
[Results](#results).

## Pipeline

Fixed, order-sensitive transform chain from `preprocessing/pipeline.py::build_default_pipeline`,
applied after `SleepEDFLoader` reads the EDF + hypnogram pair. Every step except steps 1, 3, and 9
is toggleable in `configs/default.yaml`. All steps are non-destructive with respect to epoch count:
rejection marks epochs `IGNORE` rather than deleting them.

| # | Transform | Configuration / status |
| ---: | --- | --- |
| 1 | `AnnotationUnroller` | Unrolls the hypnogram's variable-length bouts onto a fixed 30 s grid. |
| 2 | `SleepBoundaryDetector` | Finds sleep onset/offset; does not itself crop. |
| 3 | `ChannelSelector` | `Fpz-Cz`, `Pz-Oz` (EEG) + `horizontal` (EOG). EMG is opt-in and excluded by default — SC's submental EMG is a 1 Hz preprocessed envelope, not a waveform. |
| 4 | `BadChannelDetector` | **Partial — flagging only.** Flags flat / high-NaN / saturated / high-variance / extreme peak-to-peak channels into `raw.info['bads']`. **No interpolation and no exclusion from encoding or model training** — flagged samples still reach the model. Only `ICATransform` reads the flags. |
| 5 | `ReferenceTransform` | CAR implemented, **not applied** (`reference.mode: original`). |
| 6 | `SignalFilter` | Global notch (50 Hz, auto-skipped if ≥ Nyquist), then per-type band-pass: EEG 0.5–30 Hz, EOG 0.5–15 Hz, EMG 10–30 Hz. |
| 7 | `ICATransform` | MNE `ICA`, enabled, EEG-only, EOG-guided exclusion via `find_bads_eog`. Skipped if fewer than 2 usable EEG channels remain. See the audit below. |
| 8 | `WakeCropper` | Crops to the scored sleep period ± 30 min, snapped outward to the epoch grid so signal stays phase-aligned with labels. |
| 9 | `StageMapper` | R&K → AASM; unmapped → `IGNORE`. |
| 10 | `AmplitudeEpochRejector` | Per-epoch peak-to-peak against per-type thresholds (EEG 5×10⁻⁴ V, EOG/EMG 1×10⁻³ V); non-finite samples also reject. Marks `IGNORE`, never drops. |
| 11 | `RecordingNormalizer` | Per-recording, per-channel `zscore` (also `robust` / `center`), applied last. Per-recording only — no dataset-wide normalization. |

**Preprocessing verification reports.** Three committed analyses. The first two are
**rejection-rate analyses** measuring the shipped thresholds in place, and they sample **different
points in the pipeline**, so their totals are not directly comparable:

| Report | Type | Measured at | Headline |
| --- | --- | --- | --- |
| `outputs/amp-rej-verification/` | Amplitude-rejection verification | **After** wake cropping, production pipeline, 197 recordings | 238,317 epochs, 2,090 rejected (**0.88%**); per-stage 0.23% (N2) to 1.48% (W); dominant reason `eog:horizontal:peak_to_peak` |
| `scripts/utilities/class_distribution_with_rejection.md` | Class-distribution + rejection-rate analysis | **Before** wake cropping, SC cohort | 459,198 epochs, 9,760 rejected (**2.13%**); full per-stage counts pre/post |
| `scripts/utilities/verify_acquisition.py` | Acquisition verification | Acquisition | Channel names and types, sampling rate, recording duration, reference scheme |

The two rejection rates differ because of wake cropping, not disagreement: N1/N2/N3/REM counts are
identical across both reports, and only W changes (289,856 → 70,140). Uncropped wake carries most of
the artifact, so including it roughly doubles the measured rejection rate.

The class distribution is severely imbalanced (W and N2 dominate; N1 and N3 are small), which is why
training uses `class_weighting: balanced` and why macro-F1 is the primary metric.

## Representations & Models

All encoders implement a common `BaseEncoder` interface; `IGNORE` epochs are preserved through encoding.

| Representation | Encoder / shape | Key parameters | Model |
| --- | --- | --- | --- |
| **Raw** | `RawSignalEncoder` → `(N, C, 3000)` | 30 s @ 100 Hz, `float32`; no encoder-level scaling | `RawCNN1D` — 3× Conv1d → ReLU → MaxPool1d, AdaptiveAvgPool1d, Linear(64→5) |
| **Band-power** | `BandPowerEncoder` → `(N, C, 10)` | 5 bands (δ, θ, α, σ, β) × log-absolute + relative power; Welch `nperseg=400`, `noverlap=200`, `nfft=400`, Hamming, median averaging; `log(power + 1e-10)` | `BandPowerMLP` — Flatten → Linear(64) → ReLU → Dropout(0.2) → Linear(5) |
| **Time-frequency** | `TimeFrequencyEncoder` (STFT) → `(N, C, F, T)` | `n_fft=win_length=256`, `hop_length=100`, Hann window, 0.5–30 Hz, log-power | `STFTCNN2D` — 3× Conv2d → ReLU → MaxPool2d, AdaptiveAvgPool2d, Linear(64→5) |
| **Classical** | Flattened `(C × 10)` band-power features | `max_iter=1000` | scikit-learn `LogisticRegression` |

The band-power branch uses an MLP rather than a CNN because the features are compact, non-sequential spectral summaries. The classical baseline is a linear model for the same representation.

**Unavailable alternatives:** the CWT backend is a placeholder whose `transform()` raises `EncoderNotImplementedError`; band-power ratio features (`include_ratios`) are also unimplemented.

## Training & Evaluation

From `training/trainer.py::train_baseline` and `configs/default.yaml`:

| Setting | Value |
| --- | --- |
| Framework / device | PyTorch; CUDA → MPS → CPU, with bf16 autocast on CUDA |
| Loss | `CrossEntropyLoss(ignore_index=-100)` with balanced class weights from the training split |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=0.0` |
| Batch / epochs | 32 / `max_epochs=20`; early stopping after 5 epochs without val macro-F1 improvement |
| Model selection | Best **SC validation macro-F1** checkpoint is saved and reloaded before test evaluation |
| Sampling | `LocalityAwareSampler` (`block_size=4`) to reduce page-cache thrashing on memory-mapped data |
| Seed | 42 (`numpy` + `torch`, including CUDA) |
| Test evaluation | Run once after best-checkpoint reload; never used for model selection |

**Metrics:** accuracy, macro-F1, Cohen's kappa, per-class precision/recall/F1, and raw + row-normalized confusion matrices, computed on supervised epochs only. Metrics are pooled across the cohort; no per-subject breakdown is implemented.

Primary SC validation history exists during training but was not persisted for the authoritative run. Final ST test metrics are committed in `artifacts/reports/sc_to_st/inventory.json` and `artifacts/reports/sc_to_st/classical/classical_metrics.json`.
## Results

### Primary: SC → ST external test

Evaluated once on the full ST cohort — 22 subjects and 41,148 supervised epochs — after training and model selection on SC only. Results are committed in `artifacts/reports/sc_to_st/inventory.json`.

| Model | Accuracy | Macro-F1 | Cohen's κ |
| --- | ---: | ---: | ---: |
| Classical | 0.7633 | 0.6853 | 0.6601 |
| Raw CNN-1D | 0.7505 | 0.6782 | 0.6247 |
| STFT CNN-2D | 0.7198 | 0.6609 | 0.5994 |
| BandPower MLP | 0.6781 | 0.5957 | 0.5364 |

**Macro-F1 is the primary comparison metric** because the stage distribution is strongly imbalanced. Accuracy and Cohen's κ are reported as complementary metrics.

The classical band-power logistic regression is the **strongest model in the committed SC→ST run**, leading on accuracy, macro-F1, and κ. The Raw CNN-1D is the strongest neural model by macro-F1 (0.6782). The result shows that a linear model on 10 hand-designed spectral features can outperform the learned representations under this cross-cohort protocol.

**Comparison to human inter-rater agreement.** The task specification gives **κ ≈ 0.76–0.83** as the reference agreement between human scorers and requires model comparison against that range. All four SC→ST models fall below it:

| Model | Cohen's κ | Comparison with 0.76–0.83 |
| --- | ---: | --- |
| Classical | 0.6601 | Below |
| Raw CNN-1D | 0.6247 | Below |
| STFT CNN-2D | 0.5994 | Below |
| BandPower MLP | 0.5364 | Below |

The range is reproduced from the task specification; this repository does not independently verify its underlying source. None of the SC→ST models reaches the inter-rater range specified in the brief.

For context, the legacy within-SC run reaches κ = 0.7669, illustrating the performance difference between within-cohort and external cross-cohort evaluation.

The results highlight the difficulty of cross-cohort generalization: models are selected on SC and evaluated once on ST without target-cohort adaptation. **N1 has the lowest per-class F1 in all four models** (0.3301–0.4970).

> **Training dynamics:** loss-curve and per-epoch-history export is implemented for future runs. The committed run predates this feature, so its historical curves are unavailable and no curves are fabricated.

### Legacy: internal-SC baseline

This is a separate experiment using an 11-subject held-out SC test set, not the external ST cohort. It is retained for historical comparison and must not be interpreted as an SC→ST result.

| Model | Accuracy | Macro-F1 | Cohen's κ |
| --- | ---: | ---: | ---: |
| Raw CNN-1D | 0.8219 | 0.7754 | 0.7669 |
| STFT CNN-2D | 0.7948 | 0.7339 | 0.7236 |
| BandPower MLP | 0.7928 | 0.6964 | 0.7275 |

Performance is higher on this within-SC holdout than on the external SC→ST test, illustrating the difficulty of cross-cohort generalization.

## Dashboard


app.py is a read-only Streamlit dashboard.

- Selection: recording/subject and representation.
- Hypnogram: predicted vs. actual.
- Metrics: confusion matrix and per-class precision/recall/F1.
- Sleep statistics: TST, efficiency, sleep/REM latency, and time in each stage.
- Raw EEG: 30 s epoch browser.
- PSD: selected-epoch Welch PSD and selected-stage mean PSD across scored epochs.
- Spatial power: discrete band-power markers for the recorded bipolar EEG derivations; no interpolation.
- Montage: illustrative derivation schematic only, not a true MNE montage or topomap.

## Installation & Usage

The repository requires **Python ≥ 3.12** (`pyproject.toml`). On Windows a bare `python` often
resolves to an older interpreter still on `PATH`, so prefer the `py` launcher with an explicit
version (list what is available with `py -0p`):

```
cd sleep-staging-pipeline
py -3.13 -m pip install -e ".[dev]"
```

Pinned: `mne==1.12.1`, `matplotlib==3.10.3`, `numpy==2.2.6`, `pyyaml==6.0.2`, `scikit-learn==1.8.0`,
`scipy==1.15.3`, `streamlit==1.56.0`, `torch==2.11.0`; dev extra `pytest==9.1.1`,
`pytest-cov==7.1.0`. `torch` is pinned without a CUDA suffix — install a matching CUDA build for GPU
training, or the CPU wheel of the same version. CPU and CUDA builds are **not** expected to produce
identical numbers (see [Reproducibility](#reproducibility)).

First set `SLEEP_EDF_ROOT` to a local Sleep-EDF Expanded copy (preferred over editing
`acquisition.data_root`; the tracked config may hold a development-local path). Then:

```
py -3.13 main.py --model all                                  # train on SC, evaluate once on ST
py -3.13 scripts/evaluation/export_whole_night_outputs.py     # legacy whole-night exports
py -3.13 -m streamlit run app.py                              # dashboard
```

`--model raw|bandpower|time_frequency|classical` runs a single model; `--smoke` runs a reduced
path; `--max-sc-recordings` / `--max-st-recordings` cap recordings processed. On macOS/Linux, or in
an activated 3.12+ virtual environment, substitute `python` for `py -3.13`.

Per-recording encodings are cached (`training/sc_to_st_cache.py`) and keyed by an encoder
fingerprint, so only the first run pays full preprocessing cost per representation; the cache is
multi-GB at full scale and lives outside the repository by default (`SLEEP_CACHE_ROOT`).

**Artifacts.** `outputs/` holds committed human-facing deliverables (legacy exports, results
reports, and the amplitude-rejection verification report). `artifacts/` and `models/` are generated
and gitignored — **regenerable**, not
bit-identical on rerun. The two committed exceptions, `artifacts/reports/sc_to_st/inventory.json`
and `.../classical/classical_metrics.json`, are the authoritative record of the SC→ST results and
should not be overwritten by a rerun.


## Reproducibility

The pipeline is **deterministic in data splitting** but not bit-for-bit deterministic in CUDA training.

- **Split:** `seed=42` fixes the 55/12/22 subject split across runs.
- **Neural training:** `seed=42` is set, but CUDA/cuDNN may still introduce small metric differences across reruns.
- **Classical baseline:** Logistic Regression is deterministic and reproduces its results to four decimals.
- **Authoritative results:** The committed SC→ST metrics and report files come from one full run and should be treated as the final results.
- **Rerun:** Delete generated `models/` and non-committed `artifacts/`, then run `py -3.13 main.py --model all`.


## Citation

This repository includes no CITATION file of its own. It targets the **Sleep-EDF Expanded** dataset
(PhysioNet), documented at <https://physionet.org/content/sleep-edfx/>, which carries the standard
PhysioNet citation requirement (Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet,"
*Circulation* 101(23), 2000) in addition to citing the dataset itself. Refer to the PhysioNet page
for the exact current citation text rather than this README, since that guidance can change
independently of this repository.
