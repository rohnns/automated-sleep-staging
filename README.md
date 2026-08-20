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

All encoders implement a common `BaseEncoder` interface; `IGNORE` epochs survive encoding.

| Representation | Encoder → shape | Key parameters | Model |
| --- | --- | --- | --- |
| **Raw** | `RawSignalEncoder` → `(N, C, 3000)` | Pass-through 30 s @ 100 Hz, `float32`; no encoder-level scaling | `RawCNN1D` — 3× (Conv1d → ReLU → MaxPool1d), AdaptiveAvgPool1d, Linear(64→5) |
| **Band-power** | `BandPowerEncoder` → `(N, C, 10)` | 5 bands (δ, θ, α, σ, β) × (log-absolute + relative power). Welch: `nperseg=400`, `noverlap=200`, `nfft=400`, hamming, `average=median`. `log(power + 1e-10)`. No gamma (signal is ≤30 Hz upstream) | `BandPowerMLP` — Flatten → Linear(64) → ReLU → Dropout(0.2) → Linear(5) |
| **Time-frequency** | `TimeFrequencyEncoder` (STFT backend) → `(N, C, F, T)` | `scipy.signal.stft`, `n_fft=win_length=256` (Δf ≈ 0.39 Hz), `hop_length=100`, Hann, `fmin=0.5`, `fmax=30.0`, log-power | `STFTCNN2D` — 3× (Conv2d → ReLU → MaxPool2d), AdaptiveAvgPool2d, Linear(64→5) |
| **Classical (band-power preproc)** | Flattened band-power features | `max_iter=1000` | scikit-learn `LogisticRegression` |

The band-power branch is an MLP (and the classical baseline a linear model) by design: band-power
features are a compact, non-sequential per-band summary with no spatial or temporal locality for a
convolution to exploit. **Not usable:** the CWT backend implements its geometry methods but
`transform()` raises `EncoderNotImplementedError`, so the `method: stft | cwt` config switch fails
at encode time if set to `cwt`; band-power **ratio** features (`include_ratios`) likewise raise
`EncodingError`.

## Training & Evaluation

From `training/trainer.py::train_baseline` and `configs/default.yaml`:

| Setting | Value |
| --- | --- |
| Framework / device | PyTorch; auto-selected CUDA → MPS → CPU, bf16 autocast on CUDA only |
| Loss | `CrossEntropyLoss(ignore_index=-100)`, `class_weighting: balanced` computed from the train split only |
| Optimizer | Adam, `lr=1e-3`, `weight_decay=0.0` |
| Batch size / epochs | 32 / `max_epochs=20`, early stopping after 5 epochs without val macro-F1 improvement |
| Model selection | Best **SC validation macro-F1** checkpoint is saved to disk *and* reloaded (`load_state_dict(best_state)`) before any test evaluation |
| Sampling | `LocalityAwareSampler` (`block_size=4`) instead of a global shuffle — an I/O tuning knob to avoid page-cache thrashing on memory-mapped data, not a scientific choice |
| Seed | 42 (`set_seed()`: numpy + torch, including CUDA) |
| Test evaluation | Run **once**, after best-checkpoint reload; never used for selection or early stopping |

**Metrics** (`training/metrics.py`), computed on supervised (non-`IGNORE`) epochs only: accuracy,
macro-F1, **Cohen's kappa** (implemented directly, not from an external library), per-class
precision/recall/F1, and raw + row-normalized confusion matrices. Metrics are **pooled across the
cohort** — no per-subject breakdown is implemented. SC validation metrics exist per epoch in
`TrainResult.history` but are not committed for the primary run; ST test metrics are committed in
`artifacts/reports/sc_to_st/inventory.json` and `.../classical/classical_metrics.json`.

## Results

### Primary: SC → ST external test

Evaluated once on the full ST cohort — 22 subjects, 41,148 supervised epochs — after training and
model selection on SC only. Committed in `artifacts/reports/sc_to_st/inventory.json`.

| Model | Accuracy | Macro-F1 | Cohen's κ |
| --- | ---: | ---: | ---: |
| Classical (LogisticRegression on bandpower) | 0.7633 | 0.6853 | 0.6601 |
| Raw CNN-1D | 0.7505 | 0.6782 | 0.6247 |
| STFT CNN-2D | 0.7198 | 0.6609 | 0.5994 |
| BandPower MLP | 0.6781 | 0.5957 | 0.5364 |

**Macro-F1 is the primary comparison metric**, because the stage distribution is strongly
imbalanced (N1 is a small minority, N2 dominates) and macro-F1 weights every stage equally.
Accuracy and κ are complementary: accuracy is the most interpretable but is inflated by majority
classes, and κ corrects for chance agreement.

The classical band-power logistic regression is the **strongest model in the committed SC→ST run**,
leading on all three metrics; the Raw CNN-1D is the strongest *neural* model by macro-F1 (0.6782)
but beats the classical baseline on none of the three. That a linear model on 10 hand-designed
spectral features outperforms three learned representations is a meaningful result on this
cross-cohort protocol.

**Comparison to human inter-rater agreement.** The task specification gives published inter-rater
agreement between expert human scorers as **κ ≈ 0.76–0.83** and asks for the model κ to be compared
against it. Every model here falls below that band:

| Model | Cohen's κ | vs. human band (0.76–0.83) |
| --- | ---: | --- |
| Classical | 0.6601 | below by 0.10–0.17 |
| Raw CNN-1D | 0.6247 | below by 0.14–0.21 |
| STFT CNN-2D | 0.5994 | below by 0.16–0.23 |
| BandPower MLP | 0.5364 | below by 0.22–0.29 |

None of these models should be read as matching expert-level agreement. The 0.76–0.83 range is
quoted from the task specification, which states it without a primary reference; it is reproduced
here because the comparison is a stated requirement, not because this repository independently
verified the figure. For context, the legacy within-SC run below reaches κ = 0.7669 on the
easier same-cohort protocol, which does land inside the band — the gap between the two is the size
of the cross-cohort generalization penalty.

The results highlight the difficulty of cross-cohort external generalization: every model was
selected on SC (healthy, at home) and tested on ST (hospital, different population) with no
adaptation. **N1 is the lowest-F1 class in all four models** (0.3301–0.4970).

**Training-dynamics artifacts.** Each PyTorch run automatically writes loss curves with the
checkpoint epoch marked (`artifacts/figures/sc_to_st/<rep>/<rep>_loss_curves.png`) and per-epoch
history JSON including a signed `loss_gap_train_minus_val` overfitting signal
(`artifacts/reports/sc_to_st/<rep>/<rep>_training_history.json`). The classical baseline has no
epoch loop and no curves. Persistence was added *after* the committed run, whose in-memory history
is unrecoverable — **nothing has been fabricated**; curves generate on the next run.

### Legacy: internal-SC baseline

**A different experiment** — the test set is an 11-subject held-out partition of SC, not ST.
Reported only because it is a committed artifact (`outputs/model_outputs_report.md`); never an
SC→ST number. Every metric exceeds its SC→ST counterpart, and the Raw CNN-1D's κ = 0.7669 lands
inside the 0.76–0.83 human band that no SC→ST model reaches — consistent with an easier
within-cohort protocol rather than genuine external generalization.

| Model | Accuracy | Macro-F1 | Cohen's κ |
| --- | ---: | ---: | ---: |
| Raw CNN-1D | 0.8219 | 0.7754 | 0.7669 |
| STFT CNN-2D | 0.7948 | 0.7339 | 0.7236 |
| BandPower MLP | 0.7928 | 0.6964 | 0.7275 |

## Dashboard

`app.py` is a **read-only** Streamlit app — it never trains or refits, only reads artifacts already
written by `main.py` (`artifacts/predictions/sc_to_st/`, falling back to `outputs/model_outputs/`).

- **Selection**: recording/subject dropdown from discovered `manifest.json` files; representation
  radio (`raw`, `bandpower`, `time_frequency`).
- **Predicted vs. actual hypnogram**, **confusion matrix** (raw + row-normalized), and **per-class
  precision/recall/F1**.
- **Sleep statistics**: total sleep time, efficiency, sleep-onset and REM latency, per-stage
  time-in-stage — expert vs. predicted.
- **Raw EEG browser**: slider-selected 30 s epoch, channels offset and per-channel normalized.
- **PSD (epoch)**: Welch PSD of the selected epoch, in dB.
- **PSD (stage)**: choose W/N1/N2/N3/REM; plots the **mean of the per-epoch Welch PSDs** in dB
  across every expert-scored epoch of that stage, with aggregation method and epoch count labelled.
  Capped at 200 epochs per stage for responsiveness; empty stages show a message, not a blank plot.
- **Illustrative derivation schematic**: the recorded bipolar derivations (Fpz-Cz, Pz-Oz, horizontal
  EOG) on a simplified midline outline. **Not an MNE scalp montage and not a topographic
  representation** — drawn for orientation only, not digitized coordinates.
- **Spatial power distribution (selected stage)**: band power (δ/θ/α/σ/β) at each recorded EEG
  derivation, drawn as **discrete markers on a head outline — nothing is interpolated between
  them**. Each marker sits at the midpoint of its electrode pair as a placement convention and
  represents the whole bipolar derivation, not a point measurement. Computed from the same mean
  stage PSD as the panel above.
- **Interpolated topographic map**: **deliberately not rendered.** A scalp topomap interpolates
  power across many electrodes at known 10-20 positions; 2 bipolar derivations cannot support that
  without inventing coordinates and fabricating spatial structure. The dashboard explains this
  in-app. The discrete map above shows the spatial information that genuinely exists.

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


## Limitations & Task Compliance

| Area | Status | Notes |
| --- | --- | --- |
| Acquisition, hypnogram loading, 30 s epoching | **Complete** | MNE-based, fixed 30 s alignment. |
| R&K → AASM mapping and `IGNORE` handling | **Complete** | Unmapped/movement/? epochs stay aligned, excluded from loss and metrics. |
| Filtering | **Complete** | Notch + per-channel-type band-pass. |
| Channel selection | **Complete** | Configured EEG/EOG set. |
| Normalization | **Complete** | Per-recording, per-channel z-score. |
| **Bad-channel detection** | **Partial — flagging only** | No interpolation and no exclusion from encoding or model training; flagged samples still reach the model. Only ICA reads the flags. |
| Re-referencing | **Implemented, not applied** | CAR exists; results use `reference.mode: original`. |
| ICA | **Implemented and enabled** | Audited sample (8 SC, 4 ST) showed no material artifact removal; the audit was not exhaustive. |
| Raw / band-power / STFT encodings | **Complete** | All three implemented and evaluated. |
| CWT backend | **Optional / incomplete** | Interface exists; `transform()` intentionally unimplemented. |
| Subject-wise SC split + external ST test | **Complete** | Leakage checks in code; ST evaluated once after model selection. |
| Best-checkpoint selection / early stopping | **Complete** | Best SC-val macro-F1 checkpoint restored before ST evaluation. |
| Final ST metrics | **Complete** | Accuracy, macro-F1, κ, per-class metrics, confusion matrices committed. |
| Train/validation loss curves | **Implemented capability** | Historical curves unavailable because the authoritative run predates persistence. Nothing fabricated; curves generate on the next run. |
| Stage-level PSD | **Complete** | Selected-epoch and stage-level (mean of per-epoch Welch PSDs) both in the dashboard. |
| Class distribution reporting | **Complete** | Pre/post-rejection per-stage counts in `scripts/utilities/class_distribution_with_rejection.md`; channel/sfreq/duration/reference reporting via `scripts/utilities/verify_acquisition.py`. |
| Amplitude rejection + rejection-rate analysis | **Complete** | Per-epoch peak-to-peak rejection implemented, and amplitude-rejection verification measures its impact across all 197 recordings (0.88% rejected, no stage disproportionately affected). |
| Threshold-choice validation | **Not done** | The brief asks to "validate threshold choice." The verification above measures the shipped setting in place; no threshold-sensitivity analysis compares it against alternative cutoffs, so the specific values are not evidenced as preferable. Not clinically derived or clinically validated. |
| Spatial power map for selected stage | **Implemented — discrete, not interpolated** | Per-derivation band power on a head outline, markers only. Satisfies the spatial-visualization intent without inventing a continuous scalp field. |
| Interpolated topographic map | **Dataset-limited — not faked** | Two *bipolar* EEG derivations + one EOG; a bipolar derivation has no single scalp coordinate, so nothing valid can be interpolated. Explained in-app rather than fabricated. |
| Montage / 10-20 visualization | **Dataset-limited — illustrative** | Illustrative derivation schematic; not an MNE scalp montage, not a topographic representation, no digitized coordinates. |
| Dashboard | **Complete with documented limitation** | All viewers implemented; no real scalp topomap, per above. |
| **LOSO** | **Optional — not run** | The brief makes this conditional ("if compute allows"). No LOSO result is claimed anywhere in this README. |
| Per-subject metrics, band-power ratio features | **Not implemented** | Metrics are pooled across the cohort; `include_ratios` raises `EncodingError`. |
| **Demo** | **Walkthrough documented; artifact not included** | Supply a demo artifact separately if the brief requires one. |
| Legacy internal-SC results | **Historical** | Kept for traceability; not the primary external-generalization result. |

**Open follow-up.** The classical logistic regression runs with `max_iter=1000` and has historically
emitted an `lbfgs` convergence warning; the committed metrics were produced under that configuration
and are reported as-is. This is a solver-convergence follow-up, not a submission blocker, since the
brief does not require a converged classical solver. Other limitations: filter and
amplitude-rejection thresholds are engineering baselines that have not been through
threshold-choice validation (see [Pipeline](#pipeline)), and the three PyTorch models were trained
on CUDA (`"device": "cuda"` in `inventory.json`) while the classical baseline is CPU/scikit-learn.

## Reproducibility

The distinction that matters is **deterministic** (bit-for-bit identical on every run) versus
**regenerable** (a rerun reproduces the artifacts and closely comparable numbers, not necessarily
identical ones). This pipeline is deterministic in data handling and regenerable — not
bit-for-bit deterministic — in neural training.

- **Subject split — deterministic.** Fully determined by `seed=42`; cohort membership (55/12/22) is
  identical on every run and across machines.
- **CUDA training — not bit-for-bit deterministic.** `seed=42` is applied via `set_seed()` (numpy +
  torch, including CUDA), but **seeding alone does not make neural results bit-for-bit
  reproducible**: the repository does *not* set `torch.backends.cudnn.deterministic=True` or call
  `torch.use_deterministic_algorithms()`, so cuDNN may select non-deterministic kernels.
- **Reruns can produce small metric differences.** Observed drift is in the third decimal and can be
  large enough to reorder closely-spaced models; hardware, driver, and backend build shift results
  by a similar margin. The classical baseline *is* exactly reproducible (deterministic solver, fixed
  `random_state`, CPU) to four decimals.
- **The committed SC→ST metrics are the authoritative results for this submission** — the values in
  [Results](#results) and the two committed report files come from one full run. A rerun landing on
  slightly different values has not contradicted them.
- **Rerun from scratch**: delete `models/`, `artifacts/predictions/`, `artifacts/figures/`, and
  non-committed `artifacts/reports/` content, then rerun `py -3.13 main.py --model all`. Config is
  one YAML file (`configs/default.yaml`) loaded into typed dataclasses.

## Citation

This repository includes no CITATION file of its own. It targets the **Sleep-EDF Expanded** dataset
(PhysioNet), documented at <https://physionet.org/content/sleep-edfx/>, which carries the standard
PhysioNet citation requirement (Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet,"
*Circulation* 101(23), 2000) in addition to citing the dataset itself. Refer to the PhysioNet page
for the exact current citation text rather than this README, since that guidance can change
independently of this repository.
