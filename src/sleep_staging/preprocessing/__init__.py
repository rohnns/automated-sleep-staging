"""Public API for Sleep-EDF preprocessing transforms and pipeline.

Transforms are independent and composable. Default order::

    AnnotationUnroller → SleepBoundaryDetector → ChannelSelector →
    BadChannelDetector → (optional Reference) → SignalFilter → ICA →
    WakeCropper → StageMapper → AmplitudeEpochRejector → RecordingNormalizer

This package does **not** create ML encodings, tensors, or train models.
"""

from sleep_staging.preprocessing.amplitude_reject import AmplitudeEpochRejector
from sleep_staging.preprocessing.annotation_unroller import (
    AnnotationUnroller,
    unroll_annotations,
)
from sleep_staging.preprocessing.channel_selector import (
    DEFAULT_CHANNEL_NAMES,
    ChannelSelector,
    resolve_channel_picks,
)
from sleep_staging.preprocessing.epoch_grid import (
    align_crop_window,
    ceil_to_epoch_grid,
    floor_to_epoch_grid,
    grid_epoch_indices_in_bout,
)
from sleep_staging.preprocessing.exceptions import (
    MissingChannelsError,
    NoSleepBoundaryError,
    PreprocessingError,
    TransformError,
)
from sleep_staging.preprocessing.filtering import SignalFilter
from sleep_staging.preprocessing.ica import ICATransform
from sleep_staging.preprocessing.normalization import RecordingNormalizer
from sleep_staging.preprocessing.reference import ReferenceTransform
from sleep_staging.preprocessing.pipeline import (
    PreprocessPipeline,
    build_default_pipeline,
    preprocess_recording,
)
from sleep_staging.preprocessing.sleep_boundaries import (
    DEFAULT_SLEEP_LABELS,
    SleepBoundaryDetector,
    detect_boundaries_from_annotations,
    detect_boundaries_from_epochs,
)
from sleep_staging.preprocessing.stage_mapper import (
    AASM_STAGES,
    DEFAULT_RK_TO_AASM,
    IGNORE_LABEL,
    StageMapper,
    map_epoch_labels,
)
from sleep_staging.preprocessing.types import (
    EpochLabels,
    PreprocessedRecording,
    SleepBoundaries,
    Transform,
)
from sleep_staging.preprocessing.wake_crop import WakeCropper
from sleep_staging.preprocessing.bad_channels import BadChannelDetector

__all__ = [
    "AASM_STAGES",
    "AmplitudeEpochRejector",
    "AnnotationUnroller",
    "ChannelSelector",
    "DEFAULT_CHANNEL_NAMES",
    "DEFAULT_RK_TO_AASM",
    "DEFAULT_SLEEP_LABELS",
    "IGNORE_LABEL",
    "EpochLabels",
    "ICATransform",
    "MissingChannelsError",
    "NoSleepBoundaryError",
    "PreprocessPipeline",
    "PreprocessedRecording",
    "PreprocessingError",
    "RecordingNormalizer",
    "ReferenceTransform",
    "SignalFilter",
    "SleepBoundaries",
    "SleepBoundaryDetector",
    "StageMapper",
    "Transform",
    "TransformError",
    "WakeCropper",
    "BadChannelDetector",
    "align_crop_window",
    "build_default_pipeline",
    "ceil_to_epoch_grid",
    "detect_boundaries_from_annotations",
    "detect_boundaries_from_epochs",
    "floor_to_epoch_grid",
    "grid_epoch_indices_in_bout",
    "map_epoch_labels",
    "preprocess_recording",
    "resolve_channel_picks",
    "unroll_annotations",
]
