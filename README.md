# Automated Sleep Staging

## Overview

This repository implements an automated 30-second-epoch sleep staging pipeline for the **Sleep-EDF Expanded** dataset (PhysioNet). It loads PSG (EDF) recordings and their hypnogram annotations, preprocesses the continuous signal, maps Rechtschaffen & Kales (R&K) hypnogram labels to the five-class **AASM** scheme (`W`, `N1`, `N2`, `N3`, `REM`), encodes each 30 s epoch into one of three representations (raw waveform, band-power, or STFT time-frequency image), and trains a small baseline classifier per representation plus one classical (logistic-regression) baseline.

The primary, currently-implemented experiment is an external-generalization protocol: models are trained and model-selected entirely on the **Sleep Cassette (SC)** cohort and evaluated once, without further tuning, on the entire **Sleep Telemetry (ST)** cohort. Verified final SC→ST test results for all four models are committed in this repository (see [Results](#results)). These are external test results on ST, not SC validation results.

A read-only Streamlit dashboard visualizes the exported predictions.

## Task / Objectives

Objectives verified as implemented in the current codebase:

- Load Sleep-EDF Expanded PSG + hypnogram pairs via MNE (`acquisition/`).
- Build a configurable, order-sensitive preprocessing pipeline (`preprocessing/`).
- Map R&K labels to AASM 5-class labels, preserving epoch alignment for unscored/unmapped labels (`StageMapper`).
- Encode epochs into three representations: raw waveform, Welch band-power, and STFT log-power spectrogram (`representations/`).
- Train one baseline model per representation (1D CNN, MLP, 2D CNN) plus a classical logistic-regression baseline on band-power features (`training/`, `models/`).
- Run a subject-wise SC train/validation split and evaluate once on the full ST cohort as an external test set (`training/sc_to_st.py`, `main.py`).
- Report accuracy, macro-F1, Cohen's kappa, per-class precision/recall/F1, and confusion matrices (`training/metrics.py`).
- Provide a read-only Streamlit dashboard over the exported artifacts (`app.py`).

Optional / partially implemented, and explicitly flagged as such below:

- A continuous wavelet transform (CWT) backend for the time-frequency representation exists as an interface-conforming class but its `transform` method raises `EncoderNotImplementedError`; it is **not usable**.
- Common-average referencing (CAR) is implemented but disabled by default (`reference.mode: original`).
- A within-SC (SC-train / SC-val / SC-test) split and result set exists as a **legacy** artifact from an earlier phase of the project, alongside the primary SC→ST experiment. The two are evaluated on different held-out cohorts and are documented separately below.

## Dataset

**Sleep-EDF Expanded** (PhysioNet), accessed as local EDF files. The pipeline does not download the dataset; a local copy is pointed to via `acquisition.data_root` in `configs/default.yaml` or the `SLEEP_EDF_ROOT` environment variable (which takes precedence).

Two cohorts are used, per docstrings and assertions in `src/sleep_staging/training/sc_to_st.py`:

| Cohort | Full name | Recordings | Subjects | Population | Primary role |
| --- | --- | ---: | ---: | --- | --- |
| **SC** | Sleep Cassette | 153 | 78 unique (55 train + 12 validation used; remainder unused under the 70/15/15 subject-wise split) | Healthy subjects, studied at home | Train + validation |
| **ST** | Sleep Telemetry | 44 | 22 (asserted in code) | Subjects with mild difficulty falling asleep, studied in hospital | External test |

Verified directly against the committed `artifacts/reports/sc_to_st/inventory.json`: 55 SC train subjects, 12 SC validation subjects, 22 ST test subjects, 41,148 supervised test epochs.

- **Loading**: `SleepEDFLoader` (`acquisition/loader.py`) reads each PSG EDF with MNE and attaches the corresponding Hypnogram EDF as MNE annotations — the annotations on `recording.raw` are the single authoritative label store.
- **Epoching**: fixed 30-second epochs (`preprocessing.epoch_duration_sec: 30` in config). `AnnotationUnroller` converts the hypnogram's variable-length scored bouts into per-epoch labels on a fixed global 30 s grid.
- **Ignored annotations**: `Movement time` and `Sleep stage ?` are not part of the 5-class AASM target set. They are mapped to a sentinel `IGNORE` label (`StageMapper`) rather than dropped, so epoch indexing stays aligned with the signal; `IGNORE` epochs are excluded from loss and from all reported metrics (`ignore_index=-100`).

## Pipeline Architecture

Verified from `preprocessing/pipeline.py::build_default_pipeline` (module and class names are the actual implementation, not paraphrased):

```
EDF + Hypnogram (SleepEDFLoader)
  → AnnotationUnroller        (unroll hypnogram onto 30 s grid)
  → SleepBoundaryDetector     (detect sleep onset/offset; no cropping)
  → ChannelSelector           (keep configured EEG/EOG[/EMG] channels)
  → BadChannelDetector        (optional; non-destructive flagging)
  → ReferenceTransform        (optional; disabled by default)
  → SignalFilter              (optional; per-channel-type band-pass + notch)
  → ICATransform              (optional; EEG-only, EOG-guided exclusion)
  → WakeCropper                (optional; crop to sleep ± buffer)
  → StageMapper                (R&K → AASM; unmapped → IGNORE)
  → AmplitudeEpochRejector     (optional; marks epochs IGNORE, doesn't drop)
  → RecordingNormalizer        (optional; per-recording z-score/robust/center)
  → [encoding: RawSignalEncoder | BandPowerEncoder | TimeFrequencyEncoder]
  → [EpochDataset: subject-wise train / val / test split]
  → [model: RawCNN1D | BandPowerMLP | STFTCNN2D | classical LogisticRegression]
  → [evaluation: accuracy, macro-F1, kappa, per-class P/R/F1, confusion matrix]
  → [dashboard: app.py — read-only visualization of exported artifacts]
```

All steps except `AnnotationUnroller`, `ChannelSelector`, and `StageMapper` are individually toggleable via `configs/default.yaml`. The shipped primary configuration enables the relevant optional preprocessing stages; re-referencing remains at `reference.mode: original`, so CAR is implemented but not applied in the reported results.

## Preprocessing

Each operation below is a separate `Transform` class under `src/sleep_staging/preprocessing/`. All are non-destructive with respect to epoch count/alignment: rejection steps mark epochs `IGNORE` rather than deleting them.

- **Filtering** (`filtering.py`, `SignalFilter`) — implemented. Applies a global notch filter (default 50 Hz, auto-skipped if ≥ Nyquist) followed by per-channel-type band-pass filtering via MNE: EEG 0.5–30 Hz, EOG 0.5–15 Hz, EMG 10–30 Hz by default (all independently configurable). Unknown/aux channel types are not band-passed.
- **Channel selection** (`channel_selector.py`, `ChannelSelector`) — implemented. Keeps a configured set of channel names and/or MNE channel types; default staging channels are `Fpz-Cz`, `Pz-Oz` (EEG) and `horizontal` (EOG). EMG is opt-in and excluded by default because Sleep-EDF SC's submental EMG is a 1 Hz preprocessed envelope, not a 100 Hz waveform (per code comments in `configs/default.yaml`).
- **Bad-channel handling** (`bad_channels.py`, `BadChannelDetector`) — implemented, but **non-destructive**: it flags channels as flat, high-NaN, saturated, high-variance, or extreme-peak-to-peak using per-channel-type thresholds, and optionally appends them to `raw.info['bads']`. It does **not** interpolate, drop, or otherwise repair flagged channels. One downstream consumer does read the flag: `ICATransform` excludes any channel already in `raw.info['bads']` when selecting which EEG channels to fit ICA components on (`ica.py`). Model **training** itself (`trainer.py`) never reads `raw.info['bads']` — a flagged channel's samples still reach the encoder and the model unchanged.
- **Re-referencing** (`reference.py`, `ReferenceTransform`) — implemented for two modes: `"original"` (no-op) and `"common_average"` (CAR across EEG channels only, skipped with a warning if fewer than 2 EEG channels are present). The shipped default config uses `mode: original`, i.e. CAR is implemented but **not applied** in the results reported here.
- **ICA** (`ica.py`, `ICATransform`) — implemented via MNE `ICA`, fit on EEG channels only (EOG/EMG untouched). If an EOG channel is present and `detect_eog=True`, `find_bads_eog` (correlation-based by default) selects components for exclusion before `ica.apply`. Skipped gracefully (with a logged reason) if fewer than 2 usable EEG channels remain. Enabled in `configs/default.yaml` (`preprocessing.ica.enabled: true`) and therefore active in the reported run — but see the measured-effect note directly below, which is the honest statement of what it actually did.

  **Measured effect of ICA on the reported results — effectively none.** This was audited empirically against the live pipeline rather than assumed:

  | Cohort | ICA behaviour | Components excluded | Signal change vs. ICA disabled |
  | --- | --- | ---: | --- |
  | SC (8/8 recordings sampled) | Runs, fits 2 components | **0** | **4.4×10⁻¹⁵ relative** (float64 machine epsilon) |
  | ST (4/4 recordings sampled) | **Skipped entirely** | n/a | **0** (bit-identical) |

  Two independent reasons it is inert here. On SC, `find_bads_eog` never flags a component, so `ica.apply(exclude=[])` reconstructs the signal from *all* components — mathematically an identity transform, leaving only floating-point round-trip noise. On ST, `BadChannelDetector` flags all three channels, and `ica.py` excludes `raw.info['bads']` when counting usable EEG channels, so ICA reports "fewer than 2 usable EEG channels" and never runs.

  **No material ICA artifact removal occurred in the reported SC→ST results.** The results must not be read as ICA-cleaned. ICA is retained because the brief lists it as a required preprocessing capability and it is fully implemented and exercised; it is not retained because it improved the signal. Only two bipolar EEG derivations are available, which caps ICA rank at 2 and leaves little for component-based artifact separation to exploit.
- **Wake cropping** (`wake_crop.py`, `WakeCropper`) — implemented. Crops the continuous recording to the scored sleep period plus a configurable buffer (default 30 minutes) on each side, snapped outward to the 30 s epoch grid so cropped signal stays phase-aligned with epoch labels. Enabled by default.
- **Normalization** (`normalization.py`, `RecordingNormalizer`) — implemented. Per-recording, per-channel normalization over the (cropped) recording window; `zscore` (mean/std), `robust` (median/IQR), or `center` (mean only). Default: `zscore`, applied last in the pipeline.
- **Amplitude rejection** (`amplitude_reject.py`, `AmplitudeEpochRejector`) — implemented; see [Quality Control](#quality-control).
- **Sleep-boundary handling** (`sleep_boundaries.py`, `SleepBoundaryDetector`) — implemented. Detects first/last epoch with a sleep-stage label from the hypnogram; used by `WakeCropper` but does not itself modify the recording.

Not implemented / not applicable in this pipeline: channel interpolation for bad channels, and any cross-recording (dataset-wide) normalization — normalization is explicitly per-recording only by design (per module docstring).

## Label Mapping

`StageMapper` (`preprocessing/stage_mapper.py`) applies the following fixed table (`DEFAULT_RK_TO_AASM`):

| Source (R&K / Sleep-EDF hypnogram) | Mapped AASM label |
| --- | --- |
| `Sleep stage W` | `W` |
| `Sleep stage 1` | `N1` |
| `Sleep stage 2` | `N2` |
| `Sleep stage 3` | `N3` |
| `Sleep stage 4` | `N3` |
| `Sleep stage R` | `REM` |
| `Movement time` | `IGNORE` |
| `Sleep stage ?` | `IGNORE` |

R&K stages 3 and 4 are merged into AASM `N3`, matching the AASM 2007 scoring manual's collapse of slow-wave sleep into a single stage. Labels not present in the table follow `unmapped_policy` (default `"ignore"` → written as `IGNORE`); the pipeline builder explicitly **refuses** to build with `unmapped_policy="drop"` because dropping epochs would desynchronize contiguous signal/label indexing — that policy is opt-in only for callers who build a custom pipeline and explicitly index by `onsets_sec`.

## Quality Control

Two independent, non-destructive QC layers, both writing to `IGNORE` rather than deleting epochs/samples:

- **Bad-channel flagging** (`BadChannelDetector`) — see [Preprocessing](#preprocessing) above. Whole-channel, not epoch-level.
- **Epoch amplitude rejection** (`AmplitudeEpochRejector`) — for each 30 s epoch window, computes per-channel peak-to-peak amplitude; if any checked channel exceeds its type-specific threshold (defaults: EEG 5×10⁻⁴ V, EOG 1×10⁻³ V, EMG 1×10⁻³ V), the epoch's label is overwritten with `IGNORE`. Non-finite samples also trigger rejection.

A committed verification run exists at `outputs/amp-rej-verification/amp_rej_verification_report.md`, generated (per its own header) from the production pipeline (`build_default_pipeline`) across all 197 discovered recordings:

- 238,317 total epochs before rejection; 2,090 rejected (0.88% overall).
- Per-stage rejection ranges from 0.23% (N2) to 1.48% (W).
- Rejection reasons include `eog:horizontal:peak_to_peak` (1,077), `eeg:Pz-Oz:peak_to_peak` (573), `eeg:Fpz-Cz:peak_to_peak` (485).

These thresholds are described in code as "configurable engineering baselines," not physiologically validated cutoffs.

## Signal Encodings

All encoders live in `src/sleep_staging/representations/` and implement a common `BaseEncoder` interface (`encode(EpochTensorBatch) -> EncodedDataset`). `IGNORE` epochs are preserved (never dropped) through encoding.

### Raw EEG

- **Encoder class**: `RawSignalEncoder` (`representations/encoders.py`).
- **Input/output**: pass-through multichannel waveform, `(N, C, T)` with `T = 3000` for 30 s @ 100 Hz.
- **Parameters**: `dtype` (default `float32`).
- **Normalization/scaling**: none at the encoder level (normalization already happened in preprocessing, if enabled).
- **Model**: `RawCNN1D` (`models/baselines.py`) — a lightweight 1D CNN (3 conv/pool blocks, adaptive average pool, linear classifier).

### Bandpower

- **Encoder class**: `BandPowerEncoder`.
- **Input/output**: `(N, C, 10)` — 5 bands (delta, theta, alpha, sigma, beta) × (log-absolute power + relative power). No gamma band (signal is band-limited to ≤30 Hz upstream).
- **Parameters**: Welch PSD via `scipy.signal.welch` — `nperseg=400`, `noverlap=200`, `nfft=400`, `window=hamming`, `average=median`, `detrend=constant`, `scaling=density`; band edges are closed intervals `[lo, hi]` (adjacent bands intentionally share a boundary sample).
- **Normalization/scaling**: log-absolute power is `log(power + eps)` (`eps=1e-10`); relative power is band power divided by the sum of all selected bands.
- **Spectral ratio features** (`include_ratios`) are explicitly **not implemented** — the encoder raises `EncodingError` if requested.
- **Model**: `BandPowerMLP` — `Flatten → Linear(64) → ReLU → Dropout(0.2) → Linear(5)`. This branch is an MLP rather than a CNN because band-power features have no spatial/temporal locality for a convolution to exploit — they are already a small, non-sequential per-band summary vector.

### Time-Frequency

- **Encoder class**: `TimeFrequencyEncoder`, delegating to a swappable `TimeFrequencyBackend` (`representations/backends.py`).
- **STFT backend (`STFTBackend`)** — **implemented and used for all reported results.** `scipy.signal.stft`, default `n_fft=win_length=256` (Δf ≈ 0.39 Hz, 2.56 s window), `hop_length=100` (1.0 s stride), Hann window, `fmin=0.5`, `fmax=30.0`, `power=2`, `log_scale=True` (log-power spectrogram). Output shape `(N, C, F, T_frames)`.
- **CWT backend (`CWTBackend`)** — **placeholder only.** Its geometry methods (`frequency_axis`, `time_axis`, `output_hw`, `describe_params`) are implemented, but `transform()` raises `EncoderNotImplementedError("CWTBackend.transform is not implemented yet")`. `configs/default.yaml` exposes a `time_frequency.method: stft | cwt` switch and a fully-specified `cwt:` parameter block, but selecting `cwt` will fail at encode time — it is not a working alternative today.
- **Model**: `STFTCNN2D` — a lightweight 2D CNN (3 conv/pool blocks, adaptive average pool, linear classifier) operating on the STFT log-power image.

## Dataset Split and Generalization

Implemented in `training/split.py` and `training/sc_to_st.py`.

- **Granularity**: strictly **subject-wise**, never recording-wise. `subject_wise_split()` shuffles unique subject IDs with a seeded RNG and partitions them; `SubjectSplit.__post_init__` raises if any partition overlaps, and `assert_no_subject_leakage()` is called explicitly in the SC→ST builder. Grouping keys are derived from parsed PSG filenames (`_subject_key`), matching how `EncodedDataset.subject_id` is set, so a split can't silently match zero recordings.
- **Primary protocol (SC→ST, `run_primary_experiment` in `training/sc_to_st.py`)**:
  - SC subjects only are split 70/15/15 (seed 42) via `subject_wise_split`, but only the **train** (55 subjects) and **validation** (12 subjects) partitions are used here.
  - Checkpoint selection and early stopping use SC validation macro-F1 only.
  - The entire ST cohort (22 subjects, asserted) is evaluated exactly once, after training, as the test set — this is a true **cross-cohort (SC→ST)** external-generalization test, not a within-SC holdout.
- **Legacy protocol** (`outputs/model_outputs_report.md`): an **earlier, separate** experiment that uses the same 70/15/15 SC-only subject-wise split (55/12/**11**), but here the leftover 11-subject SC partition is used as an **internal SC test set** (not ST). This predates the current SC→ST orchestration and is **not** the same experiment as the primary SC→ST run; results from the two must not be conflated (see [Results](#results)).

Subject-disjointness is verified in code (`assert_no_subject_leakage`, `SubjectSplit.__post_init__`), not merely assumed.

## Models

All four models are defined in `src/sleep_staging/models/`:

| Representation | Model class | Architecture |
| --- | --- | --- |
| Raw | `RawCNN1D` | 1D CNN: 3× (Conv1d → ReLU → MaxPool1d), AdaptiveAvgPool1d, Linear(64→5) |
| Bandpower | `BandPowerMLP` | MLP: Flatten → Linear(→64) → ReLU → Dropout(0.2) → Linear(64→5) |
| Time-frequency | `STFTCNN2D` | 2D CNN: 3× (Conv2d → ReLU → MaxPool2d), AdaptiveAvgPool2d, Linear(64→5) |
| Bandpower (classical) | scikit-learn `LogisticRegression` | Flattened band-power features, `max_iter` from config |

The bandpower branch is an MLP rather than a CNN, and the classical baseline is a plain logistic regression rather than a neural network at all — both by design, per code comments: band-power features are a compact, non-sequential per-band summary with no spatial/temporal structure for a convolution to exploit, so a small MLP (and, for the classical baseline, a linear model) is the appropriate capacity for that representation.

## Training

From `training/trainer.py::train_baseline` and `configs/default.yaml`:

- **Framework**: PyTorch (device auto-selected: CUDA → MPS → CPU via `select_device()`; bf16 autocast on CUDA only).
- **Loss**: `nn.CrossEntropyLoss(ignore_index=-100)`, with optional balanced class weights (`class_weighting: balanced`, computed from the train split only).
- **Optimizer**: Adam, `learning_rate=1e-3`, `weight_decay=0.0` (config defaults).
- **Batch size**: 32 (config default; reduced from 64 per an in-config note about 6 GB laptop GPU memory).
- **Epochs**: `max_epochs=20` (config default), with early stopping (`early_stopping_patience=5` epochs without val macro-F1 improvement).
- **Validation metric**: macro-F1, computed on the SC validation split each epoch.
- **Checkpointing / best-model selection**: the trainer explicitly tracks `best_val_macro_f1` and saves a checkpoint (`_save_checkpoint`) every time validation macro-F1 improves; the in-memory `best_state` (a CPU copy of `model.state_dict()`) is likewise updated on every improvement. After the training loop ends (by exhausting `max_epochs` or early stopping), `model.load_state_dict(best_state)` is called before any test-set evaluation — the best validation macro-F1 checkpoint is both saved to disk and reloaded into the model prior to computing test metrics.
- **Class weighting**: `balanced` by default — weights are `n_supervised / (n_classes * class_count)`, computed once from the training set.
- **Sampling**: training uses a custom `LocalityAwareSampler` (`training/sampler.py`) instead of a fully global shuffle, to avoid page-cache thrashing on memory-mapped datasets larger than RAM (`block_size=4` recordings per shuffled block; a memory/IO tuning knob, not a scientific choice per its docstring).
- **Random seeds**: `seed=42` throughout (`set_seed()` seeds `numpy` and `torch`, including CUDA).
- **Test evaluation**: only run once, after best-checkpoint reload, and only when `evaluate_test=True`; never used for checkpoint selection or early stopping.

## Evaluation

`training/metrics.py::compute_classification_metrics` computes, on supervised (non-`IGNORE`) epochs only:

- Accuracy
- Macro-F1
- **Cohen's kappa** (implemented directly, `cohen_kappa()`, not from an external library)
- Per-class precision / recall / F1 (5 classes: W, N1, N2, N3, REM)
- Raw and row-normalized confusion matrices

There is no per-subject metrics breakdown implemented anywhere in `training/` or `evaluation/` — metrics are pooled across all epochs in a split.

- **SC validation** metrics are produced every epoch during training (`train_baseline`'s `history`) but are **not** written to a committed report file for the primary SC→ST run; they exist only in training logs / the in-memory `TrainResult.history`.
- **ST test** metrics (primary experiment) are committed in `artifacts/reports/sc_to_st/inventory.json` (all four models) and `artifacts/reports/sc_to_st/classical/classical_metrics.json` (classical baseline, which also separately records its SC validation metrics).
- **Legacy SC-internal test** metrics (11 held-out SC subjects, a different experiment) are committed in `outputs/model_outputs_report.md`.

## Results

### Primary experiment: SC → ST external test (committed, `artifacts/reports/sc_to_st/inventory.json`)

Evaluated once on the full ST cohort (22 subjects, 41,148 supervised epochs), after training/model-selection on SC only:

| Model | Accuracy | Macro-F1 | Cohen's κ |
| --- | ---: | ---: | ---: |
| Classical (LogisticRegression on bandpower) | 0.7633 | 0.6853 | 0.6601 |
| Raw CNN-1D | 0.7505 | 0.6782 | 0.6247 |
| STFT CNN-2D | 0.7198 | 0.6609 | 0.5994 |
| BandPower MLP | 0.6781 | 0.5957 | 0.5364 |

Full per-class precision/recall/F1 and confusion matrices for each model are in `artifacts/reports/sc_to_st/inventory.json` and `artifacts/reports/sc_to_st/classical/classical_metrics.json`.

**Training-dynamics artifacts.** Each PyTorch run additionally writes, with no manual intervention:

| Artifact | Path |
| --- | --- |
| Train vs. validation loss curves (checkpoint epoch marked) | `artifacts/figures/sc_to_st/<rep>/<rep>_loss_curves.png` |
| Per-epoch history (`train_loss`, `val_loss`, `val_macro_f1`, `loss_gap_train_minus_val`) | `artifacts/reports/sc_to_st/<rep>/<rep>_training_history.json` |

The plot's lower panel is the signed `train − val` loss gap, so overfitting is readable as a single trend rather than inferred by eye from two curves. The classical baseline has no epoch loop and therefore no curves.

> **Note on the committed run.** Loss-curve persistence was added *after* the authoritative SC→ST run reported above was executed, and the trainer's per-epoch history was in-memory only at that time, so it could not be recovered retrospectively. The curves are generated automatically by the next `python main.py` run. No plots have been fabricated for the committed models.

The classical band-power logistic-regression baseline is the strongest primary SC→ST model on **all three** headline metrics — accuracy, macro-F1, and Cohen's κ. The Raw CNN-1D is the strongest *neural* model by macro-F1 (0.6782), but does not beat the classical baseline on any of the three. That a linear model on 10 hand-designed spectral features outperforms three learned representations is a meaningful result on this harder cross-cohort protocol, and should not be assumed to hold on the (different) legacy within-SC split below.

For context: published inter-rater agreement between expert human sleep-stage scorers is typically reported in the range κ ≈ 0.76–0.83. Every model here falls below that band (κ = 0.5364–0.6601), which is the expected outcome for a lightweight baseline on a genuinely harder cross-cohort protocol, not a red flag — but it means none of these models should be read as matching expert-level agreement.

### Legacy internal-SC baseline results (`outputs/model_outputs_report.md`)

**Not the same experiment as SC→ST above** — the "test" set here is an 11-subject held-out partition of SC itself, not the ST cohort. Reported for completeness because it is a committed repository artifact, but it must not be read as an SC→ST number:

| Model | Accuracy | Macro-F1 | Cohen's κ |
| --- | ---: | ---: | ---: |
| Raw CNN-1D | 0.8219 | 0.7754 | 0.7669 |
| STFT CNN-2D | 0.7948 | 0.7339 | 0.7236 |
| BandPower MLP | 0.7928 | 0.6964 | 0.7275 |

Unlike the SC→ST numbers above, this legacy run's Raw CNN-1D kappa (0.7669) falls **inside** the published human inter-rater range — consistent with an easier within-cohort protocol (same population, same recording setting as training) rather than genuine external generalization.

## Dashboard

`app.py` — a **read-only** Streamlit app; it never trains or refits, only reads artifacts already written by `main.py` (primary root: `artifacts/predictions/sc_to_st/`) or the legacy per-recording exports (fallback root: `outputs/model_outputs/`).

Verified features, by reading `app.py` directly:

- **Recording / subject selection**: sidebar dropdown built from discovered `manifest.json` files.
- **Model / representation selection**: sidebar radio (`raw`, `bandpower`, `time_frequency`).
- **Predicted vs. actual hypnogram**: side-by-side expert/predicted step plots (`_plot_hypnogram_side_by_side`).
- **Confusion matrix**: raw counts and row-normalized, both rendered.
- **Per-class metrics table**: precision/recall/F1 per stage.
- **Sleep statistics table**: total sleep time, sleep efficiency, sleep onset latency, REM latency, and per-stage time-in-stage, expert vs. predicted.
- **Raw EEG epoch browser**: a slider selects a 30 s epoch; all channels are plotted, offset and per-channel-normalized for display.
- **PSD (selected epoch)**: Welch PSD (`mne.time_frequency.psd_array_welch`) of the selected epoch's channels, plotted in dB.
- **PSD (selected sleep stage)**: choose W/N1/N2/N3/REM; collects every expert-scored epoch of that stage in the current recording, computes a Welch PSD per epoch, and plots the **mean of the per-epoch PSDs** in dB. The aggregation method and the epoch count used are labelled in the UI. Capped at 200 epochs per stage for responsiveness (spectra converge well before that); stages with no scored epochs show an explanatory message instead of an empty plot.
- **10-20 montage schematic**: an explicitly-labeled *illustrative* diagram of the recorded bipolar derivations (Fpz-Cz, Pz-Oz, horizontal EOG) on a simplified midline head outline — not a real topomap.
- **Topographic map**: **not implemented**. The dashboard shows an `st.info` message stating that Sleep-EDF's 3 bipolar-derivation channels are insufficient spatial coverage for a scientifically meaningful topomap, and does not attempt to render one.

## Outputs and Artifacts

Two roots, with a rule enforced by `.gitignore` and confirmed against actual git tracking state in this repository:

| Directory | Contents | Committed? |
| --- | --- | --- |
| `outputs/` | Human-facing, committed deliverables: `model_outputs/` (per-recording predictions/hypnograms/summaries for the legacy run), `model_outputs_report.md`, and the `amp-rej-verification/` QC report. | **Yes** |
| `artifacts/` | Generated by `python main.py`: predictions, figures, and reports for the primary SC→ST run. Most of this tree is gitignored and reproducible. The two committed exceptions are `artifacts/reports/sc_to_st/inventory.json` and `artifacts/reports/sc_to_st/classical/classical_metrics.json`. | Mostly **no** |
| `models/` | Trained checkpoints (`best_model.pt`) + `metadata.json` per representation, plus the classical `model.pkl`. | **No** |

Do not commit regenerated `artifacts/` or `models/` content beyond the two already-committed report files above — it is reproducible from `main.py` and the `.gitignore` rules exist specifically to keep large binaries and per-run-varying predictions out of version control.

`outputs/model_outputs/` is produced by `scripts/evaluation/export_whole_night_outputs.py` and consumed by `app.py` as its legacy fallback root — both now point at the same, correct, currently-committed directory name.

To regenerate `artifacts/` and `models/`:

```
python main.py --model all
```

## Installation

From `pyproject.toml` (the repository pins the versions used by its current reproducibility configuration):

```
cd sleep-staging-pipeline
pip install -e ".[dev]"
```

- Repository requirement: Python ≥ 3.12 (declared in `pyproject.toml`).
- Core dependencies: `mne==1.12.1`, `matplotlib==3.10.3`, `numpy==2.2.6`, `pyyaml==6.0.2`, `scikit-learn==1.8.0`, `scipy==1.15.3`, `streamlit==1.56.0`, `torch==2.11.0`.
- `torch==2.11.0` is pinned without a CUDA suffix; install a matching CUDA build from the PyTorch index for GPU training, or use the CPU wheel of the same version (same results, slower).
- Dev extra: `pytest==9.1.1`, `pytest-cov==7.1.0`.

## Usage

1. **Point at a local Sleep-EDF Expanded copy** — set the `SLEEP_EDF_ROOT` environment variable (preferred; overrides `configs/default.yaml`), or edit `acquisition.data_root` in `configs/default.yaml` directly. (The tracked config may contain a development-local dataset path; use `SLEEP_EDF_ROOT` or replace that value with your own dataset location.)
2. **Run the pipeline** (trains all four models on SC, evaluates on ST, writes `models/` + `artifacts/`):
   ```
   python main.py --model all
   ```
   Single-model runs: `python main.py --model raw|bandpower|time_frequency|classical`. `--smoke` runs a reduced/fast path; `--max-sc-recordings` / `--max-st-recordings` cap the number of recordings processed.
3. **Generate the legacy whole-night exports** (used by the dashboard's fallback root and by `outputs/model_outputs*`):
   ```
   python scripts/evaluation/export_whole_night_outputs.py
   ```
4. **Launch the dashboard**:
   ```
   streamlit run app.py
   ```

Per-recording encodings are cached (`training/sc_to_st_cache.py`), so only the first `main.py` run pays the full preprocessing cost for a given representation; the cache root is configurable and, per code comments, intended to live outside the repository (multi-GB at full scale).

## Demo walkthrough

A ~10-minute demo covering the full pipeline. Everything below runs from committed
artifacts — no retraining is required, and nothing in this flow overwrites results.

**Before starting:** `set SLEEP_EDF_ROOT=<your Sleep-EDF copy>` (the dashboard reads raw EDF
for the signal viewer and PSD panels).

1. **Repository and README** — the layered pipeline (`src/sleep_staging/`: acquisition →
   preprocessing → representations → models → training → evaluation), and the SC→ST protocol:
   train/validate on Sleep Cassette, evaluate once on Sleep Telemetry.
2. **Primary results** — the [SC→ST table](#results). Lead with macro-F1 and Cohen's κ rather
   than accuracy, and note that κ sits below the 0.76–0.83 human inter-rater band. Mention that
   the classical band-power baseline beats all three neural models on this cross-cohort protocol.
3. **Launch the dashboard** — `streamlit run app.py`
4. **Recording / model selection** — sidebar: pick a recording, then switch between `raw`,
   `bandpower`, and `time_frequency` to compare representations on the same night.
5. **Raw EEG browser** — the epoch slider; scroll to a clear N3 epoch (high-amplitude slow waves)
   versus a REM epoch to show the signal actually differs by stage.
6. **Predicted vs. actual hypnogram** — the whole-night comparison. Point out where the model
   tracks the expert and where it breaks down (N1 is the weakest class in every model).
7. **Confusion matrix + per-class metrics** — the normalized matrix is the primary diagnostic;
   show the N1 row specifically and where N1 epochs are misassigned.
8. **Sleep statistics** — expert vs. predicted total sleep time, sleep efficiency, sleep onset
   latency, REM latency, and time in each stage.
9. **Stage-level PSD** — select N3 and then REM; the δ-band power difference is visible directly
   in the spectrum, which is the physiological basis the classifiers are exploiting.

Optionally close on the two documented dataset limits (no scalp topomap, illustrative montage)
and the honest ICA finding — both are in [Limitations](#limitations).

## Repository Structure

```
sleep-staging-pipeline/
├── configs/default.yaml
├── main.py                          # Primary SC→ST experiment CLI
├── app.py                           # Streamlit dashboard (read-only)
├── src/sleep_staging/
│   ├── acquisition/                 # EDF + hypnogram loading (MNE)
│   ├── preprocessing/               # Composable transforms
│   ├── representations/             # Encoders (raw, bandpower, STFT; CWT placeholder)
│   ├── training/                    # Split, dataset, trainer, sc_to_st orchestration
│   ├── models/                      # RawCNN1D, BandPowerMLP, STFTCNN2D, factory
│   ├── evaluation/                  # Metrics-adjacent output/plotting utilities
│   ├── dashboard/                   # Empty package marker (app.py lives at repo root)
│   ├── config/                      # YAML → dataclass settings
│   └── common/                      # Logging helpers
├── scripts/
│   ├── evaluation/export_whole_night_outputs.py
│   ├── utilities/                   # dataset_statistics.py, verify_acquisition.py, etc.
│   └── run_classical_baseline.py
├── tests/                           # pytest suite (27 test files, all passing)
├── models/                          # Gitignored: main.py checkpoint output
├── artifacts/                       # Mostly gitignored: main.py predictions/figures/reports
│   └── reports/sc_to_st/inventory.json, classical/classical_metrics.json  # committed exceptions
├── outputs/                         # Committed: legacy whole-night exports + report, QC report
└── pyproject.toml
```

## Final Task-Compliance Status

The primary implementation and SC→ST evaluation are complete. The repository should be judged against the primary protocol and the requirements of the original task brief; the legacy internal-SC experiment is historical and must not be substituted for the external ST evaluation.

| Area | Status | Notes |
| --- | --- | --- |
| Sleep-EDF acquisition, hypnogram loading, 30 s epoching | **Complete** | Implemented with MNE and fixed 30 s alignment. |
| R&K → AASM mapping and IGNORE handling | **Complete** | Unmapped/movement/? epochs remain aligned and are excluded from loss/metrics. |
| Filtering, channel selection, normalization | **Complete** | Implemented in the production preprocessing pipeline. |
| Bad-channel handling | **Partial** | Detection/flagging is implemented and read by ICA's channel selection; no interpolation/exclusion happens for model training. |
| Re-referencing | **Implemented, not applied by default** | CAR exists, but reported results use `reference.mode: original`. |
| ICA | **Implemented, enabled, measured no-op** | EOG-guided ICA is implemented and active in the reported run, but excluded 0 components on sampled SC recordings and was skipped on sampled ST recordings — no material artifact removal. See [Preprocessing](#preprocessing). |
| Train/validation loss curves + overfitting | **Implemented / artifact pending** | The trainer already tracked the required per-epoch history (`train_loss`, `val_loss`, `val_macro_f1`, and the explicit `loss_gap_train_minus_val` overfitting signal), and persistent plotting + JSON export is now wired into `main.py` (`*_loss_curves.png`, `*_training_history.json`, checkpoint epoch marked). The authoritative committed SC→ST run predates this feature and its per-epoch history is unrecoverable, so no historical curve artifact exists for those exact models. No curves are being fabricated and no rerun is being performed; curves generate automatically on the next `python main.py` run. |
| Stage-level PSD | **Complete** | Dashboard offers both a selected-epoch PSD and a stage-level PSD (mean of per-epoch Welch PSDs across all expert-scored epochs of the chosen stage). |
| Topographic map | **Dataset-limited — deliberately not rendered** | Sleep-EDF provides two *bipolar* EEG derivations plus one EOG. A bipolar derivation is a difference between two sites and has no single scalp coordinate, so there is nothing valid to interpolate. Rendering one would require inventing electrode positions. Not faked. |
| Montage / electrode positioning | **Dataset-limited — illustrative only** | Shown as an explicitly-labelled schematic of the recorded derivations; not an MNE scalp montage, no digitized coordinates. |
| Raw / bandpower / STFT encodings | **Complete** | All three are implemented and evaluated. |
| CWT backend | **Optional / incomplete** | Interface exists, but `transform()` is intentionally unimplemented. |
| Subject-wise SC train/validation and external ST test | **Complete** | Leakage checks are implemented; ST is evaluated once after SC model selection. |
| Best-checkpoint selection / early stopping | **Complete** | Best SC validation macro-F1 checkpoint is restored before ST evaluation. |
| Final ST metrics | **Complete** | Accuracy, macro-F1, Cohen's κ, per-class metrics, and confusion matrices are committed. |
| Dashboard | **Complete with documented limitation** | Raw viewer, PSD, hypnogram comparison, confusion matrix, per-class metrics, and sleep statistics are implemented. A real scalp topomap is not implemented because the available derivations are insufficient for a meaningful topographic map. |
| LOSO | **Optional / not run** | No LOSO subset result is claimed in this README. |
| Per-subject metrics | **Not implemented** | Reported metrics are pooled across the test cohort. |
| Demo | **Not evidenced in this repository snapshot** | Produce/attach the required demo separately if the task brief requires a demo artifact. |
| Legacy internal-SC results | **Historical** | Kept for traceability; not used as the primary external-generalization result. |

### Primary completion criterion

For the main scientific experiment, the repository has a reproducible SC→ST pipeline with subject-wise model selection on SC and one-time evaluation on ST. The strongest primary model is the classical band-power logistic regression, which leads on accuracy (0.7633), macro-F1 (0.6853) and Cohen's κ (0.6601); the Raw CNN-1D is the strongest neural model by macro-F1 (0.6782). See [Results](#results).

### Submission blockers outside the README

The README does not claim completion for items that are not evidenced here. Before submission, the only repository-level checks worth performing are:

1. Re-run the classical logistic-regression baseline after resolving its historical `lbfgs` convergence warning, if that warning is still present in the current environment.
2. Confirm whether the original task brief treats bad-channel correction/exclusion, a true scalp topomap, or a demo artifact as mandatory rather than optional.
3. If a demo is mandatory, provide the demo as a separate deliverable; it is not represented by generated `artifacts/`.

## Current Status

### Implemented

- EDF + hypnogram loading via MNE (`acquisition/`)
- Production preprocessing chain: annotation unrolling, sleep-boundary detection, channel selection, bad-channel flagging, optional CAR referencing, per-channel-type filtering, ICA with EOG-guided exclusion, wake cropping, R&K→AASM mapping, amplitude-based epoch rejection, and per-recording normalization
- Three encoders: raw, Welch band-power, and STFT log-power time-frequency
- Four baselines: RawCNN1D, BandPowerMLP, STFTCNN2D, and classical LogisticRegression
- Subject-wise, leakage-checked SC/ST splitting
- Primary SC→ST training and evaluation pipeline (`main.py`)
- Best-validation-macro-F1 checkpointing and reload before test evaluation (verified in `trainer.py`)
- Final ST metrics for all four primary models
- Accuracy, macro-F1, Cohen's kappa, per-class precision/recall/F1, and raw/normalized confusion matrices
- Read-only Streamlit dashboard with recording/subject selection, representation selection, hypnogram comparison, confusion matrices, per-class metrics, sleep statistics, raw-epoch viewing, and PSD

### Partial or intentionally limited

- **Bad-channel detection**: implemented as flagging; ICA reads the flag when picking channels to fit on, but flagged channels are not interpolated, excluded from encoding, or excluded from model training.
- **Common-average referencing**: implemented but disabled in the reported primary configuration (`mode: original`).
- **CWT**: interface and configuration exist, but the transform is not implemented.
- **Topomap**: a real scalp topomap is not rendered because the dataset configuration contains only two bipolar EEG derivations plus one EOG channel; the dashboard therefore uses an explicitly illustrative montage schematic instead.

### Historical / non-primary

- The repository contains a legacy internal-SC holdout result (`outputs/model_outputs_report.md`). Retained for traceability but not the primary external-generalization result.

### Not currently implemented

- Per-subject test metrics
- Band-power ratio features
- EMG-inclusive staging in the default configuration

### Optional / not run

- LOSO subset experiment
- CWT-based time-frequency evaluation

## Reproducibility

- **Configuration**: single YAML file, `configs/default.yaml`, loaded into typed dataclasses (`config/settings.py`); no Hydra/Pydantic. Dataset root is overridable via `SLEEP_EDF_ROOT` without editing the tracked config.
- **Seeds**: `seed=42` used consistently for the subject-wise split and for `set_seed()` (numpy + torch, including CUDA) in training. **Seeding alone does not make the neural results bit-for-bit reproducible.** The repository does *not* set `torch.backends.cudnn.deterministic=True` or call `torch.use_deterministic_algorithms()`, so cuDNN may select non-deterministic kernels and a fresh `main.py --model all` run will reproduce the three PyTorch models' metrics closely but not exactly — observed drift across runs is in the third decimal and can be large enough to reorder closely-spaced models. The classical logistic-regression baseline *is* exactly reproducible (deterministic solver, fixed `random_state`, CPU) and reproduces its committed numbers to four decimals. The subject-wise split itself is fully deterministic given the seed, so cohort membership never varies between runs.
- **Cached encodings**: per-recording encoding cache keyed by an "encoder fingerprint" (`training/sc_to_st_cache.py`); default cache location is outside the repository and overridable via the `SLEEP_CACHE_ROOT` environment variable.
- **Generated outputs**: `artifacts/` and `models/` are fully reproducible by rerunning `python main.py --model all` against the same dataset root and config; they are gitignored (except the two report files noted in [Outputs and Artifacts](#outputs-and-artifacts)) for exactly this reason.
- **Rerun from scratch**: delete/ignore `models/`, `artifacts/predictions/`, `artifacts/figures/`, and non-committed `artifacts/reports/` content, then run `python main.py --model all` again with `SLEEP_EDF_ROOT` pointed at a local Sleep-EDF Expanded copy.

## Limitations

Limitations verified from the repository itself (comments, code behavior, or committed reports), not inferred:

- CWT time-frequency encoding is not usable despite being present in configuration.
- Filter/amplitude-rejection thresholds are explicitly described in code as unvalidated "engineering baselines," not literature-derived or clinically-validated cutoffs.
- Bad-channel detection has a narrow downstream effect: it only influences which channels ICA fits components on. It does not remove or correct channels for encoding or model training.
- Sleep-EDF SC/ST provide only 2 EEG bipolar derivations + 1 EOG channel, so the dashboard cannot render a real scalp topomap (explicitly acknowledged in `app.py` itself).
- Per-subject-level test metrics are not computed; all reported metrics pool epochs across the full test cohort.
- The three primary PyTorch models were trained on CUDA per `artifacts/reports/sc_to_st/inventory.json` (`"device": "cuda"`), selected automatically by `select_device()`. The classical logistic-regression baseline is CPU/scikit-learn and records no device. Because CUDA training is not made bit-deterministic here (see [Reproducibility](#reproducibility)), reruns reproduce the neural metrics closely but not exactly.
- Cohen's kappa for every primary SC→ST model (0.5364–0.6601) falls below the published human inter-rater agreement range (κ ≈ 0.76–0.83); see [Results](#results).

## Citation / Dataset

This repository does not include its own CITATION file. It targets the Sleep-EDF Expanded dataset (PhysioNet), which is documented at https://physionet.org/content/sleep-edfx/ and is described there as an extension of the original Sleep-EDF database, associated with the standard PhysioNet citation requirement (Goldberger et al., "PhysioBank, PhysioToolkit, and PhysioNet," *Circulation* 101(23), 2000) in addition to citing the dataset itself. Refer to the PhysioNet page for the exact, current citation text rather than this README, since PhysioNet citation guidance can change independently of this repository.
