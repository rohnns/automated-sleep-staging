"""Data acquisition for Sleep-EDF Expanded PSG recordings.

Public API
----------
- :class:`SleepEDFLoader` — primary loader class
- :func:`load_recording` / :func:`load_recordings` / :func:`discover_recordings`
- :class:`SleepRecording` — complete recording container
- :class:`RecordingMetadata` — MNE-agnostic summary metadata

MNE boundary
------------
**Interact with MNE directly**

- ``loader.py`` — EDF I/O and annotation attachment
- ``metadata.py`` — reading fields from ``Raw`` / ``Annotations``
- ``SleepRecording.raw`` / ``SleepRecording.annotations`` — consumed by later
  MNE-based preprocessing and epoching

**Remain library-agnostic**

- ``utils.py`` — Sleep-EDF path / filename conventions
- ``RecordingMetadata``, ``ChannelInfo``, ``AnnotationRecord``
- ``config`` / ``common`` packages
"""

from sleep_staging.acquisition.dataclasses import (
    AnnotationRecord,
    ChannelInfo,
    RecordingMetadata,
    SleepRecording,
    annotations_to_records,
)
from sleep_staging.acquisition.exceptions import (
    AcquisitionError,
    InvalidEDFFileError,
    MetadataExtractionError,
    MissingHypnogramFileError,
    MissingPSGFileError,
    RecordingValidationError,
)
from sleep_staging.acquisition.loader import (
    SleepEDFLoader,
    discover_recordings,
    load_recording,
    load_recordings,
)
from sleep_staging.acquisition.utils import SleepEDFFileIds, parse_psg_filename, resolve_hypnogram_path

__all__ = [
    "AcquisitionError",
    "AnnotationRecord",
    "ChannelInfo",
    "InvalidEDFFileError",
    "MetadataExtractionError",
    "MissingHypnogramFileError",
    "MissingPSGFileError",
    "RecordingMetadata",
    "RecordingValidationError",
    "SleepEDFFileIds",
    "SleepEDFLoader",
    "SleepRecording",
    "annotations_to_records",
    "discover_recordings",
    "load_recording",
    "load_recordings",
    "parse_psg_filename",
    "resolve_hypnogram_path",
]
