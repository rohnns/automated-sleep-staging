"""Public API for Phase 3 encodings / representations.

Raw, band-power (Welch), and STFT time-frequency encoders are implemented.
CWT remains a future backend option.
"""

from sleep_staging.encodings.backends import CWTBackend, STFTBackend, TimeFrequencyBackend
from sleep_staging.encodings.base import BaseEncoder, build_encoded_dataset
from sleep_staging.encodings.encoders import (
    DEFAULT_BANDS,
    BandPowerEncoder,
    RawSignalEncoder,
    TimeFrequencyEncoder,
)
from sleep_staging.encodings.epoching import preprocessed_to_epoch_batch
from sleep_staging.encodings.exceptions import (
    EncoderNotImplementedError,
    EncodingError,
    EpochExtractionError,
    ShapeValidationError,
)

# EpochExtractionError is part of the public handoff contract.
from sleep_staging.encodings.factory import build_encoder, build_time_frequency_backend
from sleep_staging.encodings.types import (
    EncodedDataset,
    EncodedDatasetCollection,
    EpochTensorBatch,
    LabelVocabulary,
    RepresentationMetadata,
    RepresentationType,
)

__all__ = [
    "DEFAULT_BANDS",
    "BandPowerEncoder",
    "BaseEncoder",
    "CWTBackend",
    "EncodedDataset",
    "EncodedDatasetCollection",
    "EncoderNotImplementedError",
    "EncodingError",
    "EpochExtractionError",
    "EpochTensorBatch",
    "LabelVocabulary",
    "RawSignalEncoder",
    "RepresentationMetadata",
    "RepresentationType",
    "STFTBackend",
    "ShapeValidationError",
    "TimeFrequencyBackend",
    "TimeFrequencyEncoder",
    "build_encoded_dataset",
    "build_encoder",
    "build_time_frequency_backend",
    "preprocessed_to_epoch_batch",
]
