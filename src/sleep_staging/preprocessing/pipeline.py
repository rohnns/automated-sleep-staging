"""Composable preprocessing pipeline."""

from __future__ import annotations

from collections.abc import Sequence

from sleep_staging.acquisition.dataclasses import SleepRecording
from sleep_staging.common.logging_utils import get_logger
from sleep_staging.config.settings import PreprocessingSettings
from sleep_staging.preprocessing.amplitude_reject import AmplitudeEpochRejector
from sleep_staging.preprocessing.annotation_unroller import AnnotationUnroller
from sleep_staging.preprocessing.bad_channels import BadChannelDetector
from sleep_staging.preprocessing.channel_selector import ChannelSelector
from sleep_staging.preprocessing.filtering import SignalFilter
from sleep_staging.preprocessing.ica import ICATransform
from sleep_staging.preprocessing.normalization import RecordingNormalizer
from sleep_staging.preprocessing.reference import ReferenceTransform
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
    4. Optional bad-channel marking / reference
    5. Filter on the continuous recording (edge transients in wake tails)
    6. Optional ICA on EEG (EOG-guided exclusion when available)
    7. Optional wake crop with epoch-grid-aligned bounds
    8. Map R&K → AASM (IGNORE for Movement / '?')
    9. Optional epoch amplitude rejection (mark IGNORE; do not drop epochs)
    10. Per-recording normalize on the cropped window
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

    # Bad channel detection (non-destructive marking) — runs after channel selection
    if settings.bad_channel.enabled:
        transforms.append(
            BadChannelDetector(
                flat_std_threshold=settings.bad_channel.flat_std_threshold,
                nan_frac_threshold=settings.bad_channel.nan_frac_threshold,
                saturation_frac_threshold=settings.bad_channel.saturation_frac_threshold,
                eeg_high_std_threshold=settings.bad_channel.eeg_high_std_threshold,
                eeg_peak_to_peak_threshold=settings.bad_channel.eeg_peak_to_peak_threshold,
                eog_high_std_threshold=settings.bad_channel.eog_high_std_threshold,
                eog_peak_to_peak_threshold=settings.bad_channel.eog_peak_to_peak_threshold,
                emg_high_std_threshold=settings.bad_channel.emg_high_std_threshold,
                emg_peak_to_peak_threshold=settings.bad_channel.emg_peak_to_peak_threshold,
                mark_mne_bads=settings.bad_channel.mark_mne_bads,
            )
        )

    # Reference: default "original" leaves bipolar SC derivations unchanged (no-op by omission).
    # Opt-in CAR is inserted after bad-channel marking and before filtering.
    if settings.reference.mode is not None and settings.reference.mode != "original":
        transforms.append(ReferenceTransform(mode=settings.reference.mode))

    if settings.filter.enabled:
        transforms.append(
            SignalFilter(
                l_freq=settings.filter.l_freq,
                h_freq=settings.filter.h_freq,
                eeg_l_freq=settings.filter.eeg_l_freq,
                eeg_h_freq=settings.filter.eeg_h_freq,
                eog_l_freq=settings.filter.eog_l_freq,
                eog_h_freq=settings.filter.eog_h_freq,
                emg_l_freq=settings.filter.emg_l_freq,
                emg_h_freq=settings.filter.emg_h_freq,
                notch_freqs=settings.filter.notch_freqs,
            )
        )

    # ICA after filtering, before wake crop (edge transients remain in wake tails).
    if settings.ica.enabled:
        transforms.append(
            ICATransform(
                n_components=settings.ica.n_components,
                random_state=settings.ica.random_state,
                method=settings.ica.method,
                max_iter=settings.ica.max_iter,
                detect_eog=settings.ica.detect_eog,
                eog_threshold=settings.ica.eog_threshold,
                eog_measure=settings.ica.eog_measure,
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

    if settings.amplitude_reject.enabled:
        transforms.append(
            AmplitudeEpochRejector(
                eeg_peak_to_peak=settings.amplitude_reject.eeg_peak_to_peak,
                eog_peak_to_peak=settings.amplitude_reject.eog_peak_to_peak,
                emg_peak_to_peak=settings.amplitude_reject.emg_peak_to_peak,
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
    # Filter/ICA/normalize need preloaded data; wake crop also loads as needed.
    needs_preload = (
        settings.filter.enabled
        or settings.ica.enabled
        or settings.amplitude_reject.enabled
        or settings.normalize.enabled
    )
    return pipeline.run(recording, copy=copy, preload=needs_preload)
