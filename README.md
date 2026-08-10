# Sleep Staging Pipeline

Production-oriented, modular clinical EEG sleep staging pipeline for the
**Sleep-EDF Expanded** (PhysioNet) dataset.

## Pipeline stages

1. **Data Acquisition** ← complete
2. **Preprocessing** ← complete
3. **Encodings / Representations** ← architecture (algorithms next)
4. Models *(not yet implemented)*
5. Validation *(not yet implemented)*
6. Metrics *(not yet implemented)*
7. Dashboard *(not yet implemented)*

## Requirements

- Python 3.12+
- MNE (EDF loading / preprocessing)
- PyTorch will be introduced in the Models phase

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

## Preprocessing order

1. **AnnotationUnroller** — variable-length hypnogram bouts → fixed 30 s labels on the global epoch grid
2. **SleepBoundaryDetector** — sleep onset / offset (no cropping)
3. **ChannelSelector** — config-driven channel picks (before expensive filtering)
4. **SignalFilter** — band-pass / notch on the continuous recording (edge transients settle in wake tails)
5. **WakeCropper** — optional crop to sleep ± buffer, snapped to the 30 s grid
6. **StageMapper** — R&K → AASM; Movement / `?` → `IGNORE` (not deleted)
7. **RecordingNormalizer** — per-recording `zscore` or `robust` on the cropped window

## Encodings architecture (Phase 3)

| Piece | Role |
|-------|------|
| `EpochTensorBatch` | Encoder input `(N, C, T)` from preprocessing |
| `BaseEncoder` | Abstract `encode` / `describe` interface |
| `RawSignalEncoder` | → `(N, C, 3000)` |
| `BandPowerEncoder` | → `(N, C, 10)` via Welch (5 bands × log-abs + relative) |
| `TimeFrequencyEncoder` | → `(N, C, F, T)` via STFT (CWT backend placeholder) |
| `EncodedDataset` | Per-recording output + `RepresentationMetadata` |

```python
from sleep_staging.encodings import build_encoder
from sleep_staging.config import load_settings

settings = load_settings("configs/default.yaml")
encoder = build_encoder(settings.encodings)  # skeleton; encode() raises until DSP lands
```

## MNE boundaries

| Layer | Interact with MNE? | Notes |
|-------|--------------------|-------|
| `sleep_staging.acquisition.loader` | **Yes** | EDF I/O, attach annotations |
| `sleep_staging.acquisition.metadata` | **Yes** (read-only) | Pulls fields from `Raw` |
| `SleepRecording.raw` / `.annotations` | **Yes** | Downstream preprocessing uses these |
| `preprocessing` transforms | **Yes** (via `raw`) | Crop / pick / filter / normalize |
| `encodings` | **No** (after epoch handoff) | NumPy only; MNE stopped at preprocessing |
| `RecordingMetadata`, `utils`, `config`, `common` | **No** | Paths, IDs, YAML, logging |

Annotations have a **single authoritative store**: `recording.raw.annotations`
(also exposed as `recording.annotations`). After wake cropping, epoch labels on
`PreprocessedRecording` are the canonical fixed-epoch label sequence.

## Project layout

```text
sleep-staging-pipeline/
├── analysis/                 # Offline dataset statistics (not runtime)
├── configs/default.yaml
├── src/sleep_staging/
│   ├── acquisition/          # Phase 1 — EDF + hypnogram loading
│   ├── preprocessing/        # Phase 2 — composable transforms
│   ├── encodings/            # Phase 3 — representations (arch)
│   ├── config/               # Central typed settings
│   └── common/               # Shared logging helpers
├── tests/
└── pyproject.toml
```

## Tests

```bash
pytest
```

## License

MIT
