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

## Configuration

Paths and pipeline options live in `configs/default.yaml`. Relative paths are
resolved from the project root. Configuration is plain YAML → dataclasses
(no Hydra / Pydantic).

```yaml
acquisition:
  data_root: D:/SleepEDFX
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
│   ├── experiments/                 # Full experiment runners
│   ├── evaluation/                  # Checkpoint evaluation & export
│   ├── utilities/                   # Dataset stats, class distribution, etc.
│   └── smoke/                       # Smoke tests for pipeline stages
├── tests/
├── outputs/
│   ├── checkpoints/                 # Trained model checkpoints
│   ├── predictions/                 # Exported predictions
│   ├── reports/                     # Generated reports
│   ├── figures/                     # Generated figures
│   └── phase4_outputs/              # Phase 4 exported artifacts
└── pyproject.toml
```

## Tests

```bash
pytest
```

## License

MIT
