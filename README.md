# Sleep Staging Pipeline

Production-oriented, modular clinical EEG sleep staging pipeline for the
**Sleep-EDF Expanded** (PhysioNet) dataset.

## Pipeline stages

1. **Data Acquisition** ← complete
2. **Preprocessing** ← complete
3. **Representations** ← complete (Raw, BandPower, STFT encoders)
4. **Models** ← complete (baseline CNN-1D, MLP, CNN-2D)
5. **Evaluation** ← complete
6. **Dashboard** ← complete (Streamlit)

## Requirements

- Python 3.12+
- MNE (EDF loading / preprocessing)
- PyTorch (models & training)

## Installation

```bash
cd sleep-staging-pipeline
pip install -e ".[dev]"
```

Dependencies are pinned in `pyproject.toml` to the exact versions the reported
results were produced with. For GPU training, install the matching CUDA build of
`torch==2.11.0` from the PyTorch index; the CPU wheel of the same version
produces the same results, only slower.

## Reproduce / demo

Point the pipeline at your Sleep-EDF Expanded copy (no need to edit tracked
config — the environment variable wins over `configs/default.yaml`):

```bash
set SLEEP_EDF_ROOT=D:/SleepEDFX
```

Then, in order:

```bash
pytest
```

```bash
python main.py --model all
```

```bash
streamlit run app.py
```

- `pytest` — full suite (178 tests). Tests needing the real corpus skip
  automatically when it is absent, so the suite is green on a bare checkout.
- `main.py` — the primary SC→ST experiment end to end: encode all 197
  recordings (153 SC + 44 ST), train all four models, then write checkpoints to
  `models/` and metrics / predictions / hypnograms to `artifacts/`. Per-recording
  encodings are cached, so only the first run pays the full preprocessing cost.
  Add `--model raw|bandpower|time_frequency|classical` to run a single model.
- `app.py` — Streamlit dashboard over the artifacts produced above.

Encoding cache location defaults to `D:/SleepStagingCache/sc_to_st` and is
overridable with `SLEEP_CACHE_ROOT`. It is deliberately kept outside the repo:
at full scale it is several GB per representation.

## The experiment: SC → ST

`SC` and `ST` are the two **Sleep-EDF Expanded** cohorts:

| Cohort | Full name | Recordings | Population | Role here |
|---|---|---|---|---|
| **SC** | Sleep **C**assette | 153 | Healthy subjects, studied at home | Train + validation |
| **ST** | Sleep **T**elemetry | 44 | Mild difficulty falling asleep, studied in hospital | External test |

So **SC→ST** = *train on Sleep Cassette, test on Sleep Telemetry*. This is a
deliberately harder protocol than a random split: the test cohort differs in
both population and recording setting, so the score reflects genuine external
generalization rather than memorized within-cohort quirks.

Splitting is **subject-wise** (never recording-wise) and asserted leak-free:
55 SC subjects train, 12 SC subjects validate (model selection + early
stopping), and all 22 ST subjects are held out as a clean test set that is
evaluated exactly once.

## Where outputs go

Two roots, with a deliberate rule:

| Directory | Contents | Committed? |
|---|---|---|
| `artifacts/` | **Generated.** Written by `python main.py` — metrics, prediction CSVs, hypnograms. Safe to delete; rerun to rebuild. | No (gitignored) |
| `models/` | **Generated.** Trained checkpoints + metadata per model. | No (gitignored) |
| `outputs/` | **Authored.** Committed analysis: the representation-comparison report, amplitude-rejection QC, whole-night exports. | Yes |
| `$SLEEP_CACHE_ROOT` | **Generated.** Per-recording encoding cache (several GB). Lives outside the repo by design. | No |

Rule of thumb: `outputs/` is written by a human (or reviewed once and kept),
`artifacts/` + `models/` are disposable build products of `main.py`.

## Configuration

Paths and pipeline options live in `configs/default.yaml`. Relative paths are
resolved from the project root. Configuration is plain YAML → dataclasses
(no Hydra / Pydantic).

```yaml
acquisition:
  data_root: D:/SleepEDFX    # or set SLEEP_EDF_ROOT
  preload: false

preprocessing:
  epoch_duration_sec: 30
  wake_crop:
    enabled: true
    minutes: 30
  stage_map:
    unmapped_policy: ignore
  channels:
    - Fpz-Cz
    - Pz-Oz
  filter:
    enabled: true
    l_freq: 0.5
    h_freq: 30.0
    notch_freqs: [50.0]
  normalize:
    enabled: true
    method: zscore   # or: robust
```

## Quick start

```python
from pathlib import Path

from sleep_staging.acquisition import load_recording
from sleep_staging.common import configure_logging
from sleep_staging.config import load_settings
from sleep_staging.preprocessing import preprocess_recording

configure_logging()
settings = load_settings(Path("configs/default.yaml"))

recording = load_recording(
    "SC4001E0-PSG.edf",
    settings=settings.acquisition,
)
preprocessed = preprocess_recording(recording, settings.preprocessing)

print(preprocessed.summary())
print(preprocessed.epoch_labels.labels[:10])
print(preprocessed.boundaries)
print(preprocessed.channel_names)
```

Transforms can also be composed manually:

```python
from sleep_staging.preprocessing import (
    AnnotationUnroller,
    ChannelSelector,
    PreprocessPipeline,
    RecordingNormalizer,
    SignalFilter,
    SleepBoundaryDetector,
    StageMapper,
    WakeCropper,
)

pipeline = PreprocessPipeline([
    AnnotationUnroller(),
    SleepBoundaryDetector(),
    ChannelSelector(names=["Fpz-Cz", "Pz-Oz"]),
    SignalFilter(l_freq=0.5, h_freq=30.0),
    WakeCropper(minutes=30),
    StageMapper(),
    RecordingNormalizer(method="zscore"),
])
preprocessed = pipeline.run(recording, preload=True)
```

## Representations (Phase 3)

| Piece | Role |
|-------|------|
| `EpochTensorBatch` | Encoder input `(N, C, T)` from preprocessing |
| `BaseEncoder` | Abstract `encode` / `describe` interface |
| `RawSignalEncoder` | → `(N, C, 3000)` |
| `BandPowerEncoder` | → `(N, C, 10)` via Welch (5 bands × log-abs + relative) |
| `TimeFrequencyEncoder` | → `(N, C, F, T)` via STFT (CWT backend placeholder) |
| `EncodedDataset` | Per-recording output + `RepresentationMetadata` |

```python
from sleep_staging.representations import build_encoder
from sleep_staging.config import load_settings

settings = load_settings("configs/default.yaml")
encoder = build_encoder(settings.encodings)
```

## Preprocessing order

1. **AnnotationUnroller** — variable-length hypnogram bouts → fixed 30 s labels on the global epoch grid
2. **SleepBoundaryDetector** — sleep onset / offset (no cropping)
3. **ChannelSelector** — config-driven channel picks (before expensive filtering)
4. **SignalFilter** — band-pass / notch on the continuous recording (edge transients settle in wake tails)
5. **WakeCropper** — optional crop to sleep ± buffer, snapped to the 30 s grid
6. **StageMapper** — R&K → AASM; Movement / `?` → `IGNORE` (not deleted)
7. **RecordingNormalizer** — per-recording `zscore` or `robust` on the cropped window

## MNE boundaries

| Layer | Interact with MNE? | Notes |
|-------|--------------------|-------|
| `sleep_staging.acquisition.loader` | **Yes** | EDF I/O, attach annotations |
| `sleep_staging.acquisition.metadata` | **Yes** (read-only) | Pulls fields from `Raw` |
| `SleepRecording.raw` / `.annotations` | **Yes** | Downstream preprocessing uses these |
| `preprocessing` transforms | **Yes** (via `raw`) | Crop / pick / filter / normalize |
| `representations` | **No** (after epoch handoff) | NumPy only; MNE stopped at preprocessing |
| `RecordingMetadata`, `utils`, `config`, `common` | **No** | Paths, IDs, YAML, logging |

Annotations have a **single authoritative store**: `recording.raw.annotations`
(also exposed as `recording.annotations`). After wake cropping, epoch labels on
`PreprocessedRecording` are the canonical fixed-epoch label sequence.

## Project layout

```text
sleep-staging-pipeline/
├── configs/default.yaml
├── main.py                          # CLI entry point
├── app.py                           # Streamlit dashboard
├── src/sleep_staging/
│   ├── acquisition/                 # Phase 1 — EDF + hypnogram loading
│   ├── preprocessing/               # Phase 2 — composable transforms
│   ├── representations/             # Phase 3 — encoders (raw, bandpower, STFT)
│   ├── evaluation/                  # Evaluation & output utilities
│   ├── training/                    # Phase 4 — PyTorch training
│   ├── models/                      # Baseline model architectures
│   ├── dashboard/                   # Dashboard package marker
│   ├── config/                      # Central typed settings
│   └── common/                      # Shared logging helpers
├── scripts/
│   ├── evaluation/                  # Whole-night artifact export
│   ├── utilities/                   # Dataset stats, class distribution, QC
│   └── run_classical_baseline.py    # Standalone classical baseline runner
├── tests/                           # Full suite (pytest)
├── models/                          # Trained checkpoints (main.py output)
│   ├── raw_cnn/  bandpower_mlp/  stft_cnn/  classical/
├── artifacts/                       # main.py outputs
│   ├── reports/sc_to_st/            # inventory.json + per-model metrics
│   ├── predictions/sc_to_st/        # Per-epoch prediction CSVs
│   └── figures/sc_to_st/            # Hypnogram plots
├── outputs/                         # Committed analysis reports
│   ├── phase4_final_report.md       # Representation comparison writeup
│   ├── qc_amplitude/                # Amplitude-rejection threshold QC
│   └── phase4_outputs/              # Per-recording whole-night exports
└── pyproject.toml
```

## Tests

```bash
pytest
```

## License

MIT
