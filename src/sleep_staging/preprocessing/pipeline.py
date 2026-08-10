"""Composable preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from sleep_staging.acquisition.dataclasses import SleepRecording
from sleep_staging.common.logging_utils import get_logger
from sleep_staging.config.settings import PreprocessingSettings
from sleep_staging.preprocessing.annotation_unroller import AnnotationUnroller
from sleep_staging.preprocessing.channel_selector import ChannelSelector
from sleep_staging.preprocessing.filtering import SignalFilter
from sleep_staging.preprocessing.normalization import RecordingNormalizer
from sleep_staging.preprocessing.sleep_boundaries import SleepBoundaryDetector
from sleep_staging.preprocessing.stage_mapper import StageMapper
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform
from sleep_staging.preprocessing.wake_crop import WakeCropper

logger = get_logger(__name__)


class PreprocessPipeline:
    """Run an ordered sequence of independent transforms."""

    def __init__(self, transforms: Sequence[Transform]) -> None:
        self.transforms = tuple(transforms)

    def run(
        self,
        recording: SleepRecording,
        *,
        copy: bool = True,
        preload: bool = False,
    ) -> PreprocessedRecording:
        """Apply all transforms to a sleep recording."""
        state = PreprocessedRecording.from_sleep_recording(
            recording,
            copy=copy,
            preload=preload,
        )
        for transform in self.transforms:
            logger.debug("Applying transform: %s", transform.name)
            state = transform(state)
        logger.info("Preprocessing complete: %s", state.summary())
        return state

    def __call__(
        self,
        recording: SleepRecording,
        *,
        copy: bool = True,
        preload: bool = False,
    ) -> PreprocessedRecording:
        return self.run(recording, copy=copy, preload=preload)


def build_default_pipeline(settings: PreprocessingSettings) -> PreprocessPipeline:
    """Build the standard Sleep-EDF preprocessing chain from settings.

    Scientifically motivated order:
    1. Unroll annotations onto the global 30 s grid
    2. Detect sleep onset/offset (no cropping)
    3. Select channels (before expensive filtering)
    4. Filter on the continuous recording (edge transients in wake tails)
    5. Optional wake crop with epoch-grid-aligned bounds
    6. Map R&K → AASM (IGNORE for Movement / '?')
    7. Per-recording normalize on the cropped window
    """
    if settings.stage_map.unmapped_policy == "drop":
        raise ValueError(
            "Default pipeline refuses stage_map.unmapped_policy='drop' because "
            "it desynchronizes contiguous signal/label indexing. Use "
            "unmapped_policy='ignore', or build a custom PreprocessPipeline "
            "with StageMapper(allow_length_change=True)."
        )

    transforms: list[Transform] = [
        AnnotationUnroller(
            epoch_duration_sec=settings.epoch_duration_sec,
            min_remainder_sec=settings.min_remainder_sec,
            align_to_grid=True,
        ),
        SleepBoundaryDetector(),
        ChannelSelector(
            names=settings.channels.names,
            types=settings.channels.types,
            require_all_names=settings.channels.require_all_names,
        ),
    ]

    if settings.filter.enabled:
        transforms.append(
            SignalFilter(
                l_freq=settings.filter.l_freq,
                h_freq=settings.filter.h_freq,
                notch_freqs=settings.filter.notch_freqs,
            )
        )

    if settings.wake_crop.enabled:
        transforms.append(
            WakeCropper(
                buffer_sec=settings.wake_crop.buffer_sec,
                require_sleep=settings.wake_crop.require_sleep,
                align_to_epoch_grid=True,
                epoch_duration_sec=settings.epoch_duration_sec,
            )
        )

    transforms.append(
        StageMapper(
            unmapped_policy=settings.stage_map.unmapped_policy,
            ignore_label=settings.stage_map.ignore_label,
        )
    )

    if settings.normalize.enabled:
        transforms.append(
            RecordingNormalizer(
                method=settings.normalize.method,
                eps=settings.normalize.eps,
            )
        )

    return PreprocessPipeline(transforms)


def preprocess_recording(
    recording: SleepRecording,
    settings: PreprocessingSettings,
    *,
    copy: bool = True,
) -> PreprocessedRecording:
    """Convenience: build the default pipeline and run it."""
    pipeline = build_default_pipeline(settings)
    # Filter/normalize need preloaded data; wake crop also loads as needed.
    needs_preload = settings.filter.enabled or settings.normalize.enabled
    return pipeline.run(recording, copy=copy, preload=needs_preload)
