"""Tests for sleep boundaries, wake cropping, channels, filter, normalize."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from sleep_staging.config import PreprocessingSettings, WakeCropSettings
from sleep_staging.preprocessing import (
    AnnotationUnroller,
    ChannelSelector,
    MissingChannelsError,
    NoSleepBoundaryError,
    RecordingNormalizer,
    SignalFilter,
    SleepBoundaryDetector,
    WakeCropper,
    align_crop_window,
    build_default_pipeline,
    detect_boundaries_from_epochs,
)
from sleep_staging.preprocessing.types import EpochLabels, PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def test_sleep_boundary_detector() -> None:
    labels = EpochLabels(
        onsets_sec=(0.0, 30.0, 60.0, 90.0, 120.0),
        duration_sec=30.0,
        labels=(
            "Sleep stage W",
            "Sleep stage 2",
            "Sleep stage 2",
            "Sleep stage R",
            "Sleep stage W",
        ),
    )
    bounds = detect_boundaries_from_epochs(labels)
    assert bounds.has_scored_sleep
    assert bounds.onset_sec == 30.0
    assert bounds.offset_sec == 120.0


def test_wake_cropper_keeps_buffer() -> None:
    # 0-120 W, 120-240 sleep, 240-600 W  → with buffer 60 keep [60, 300]
    annot = mne.Annotations(
        onset=[0.0, 120.0, 240.0],
        duration=[120.0, 120.0, 360.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage W"],
    )
    recording = make_sleep_recording(duration_sec=600.0, annotations=annot)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state = AnnotationUnroller()(state)
    state = SleepBoundaryDetector()(state)
    state = WakeCropper(buffer_sec=60.0)(state)

    assert state.duration_sec == pytest.approx(240.0, abs=1.0 / state.sampling_frequency)
    assert state.boundaries is not None
    assert state.boundaries.onset_sec == pytest.approx(60.0)
    assert state.epoch_labels is not None
    assert state.epoch_labels.onsets_sec[0] == pytest.approx(0.0)
    assert all(onset % 30.0 == pytest.approx(0.0) for onset in state.epoch_labels.onsets_sec)


def test_wake_cropper_aligns_off_grid_buffer() -> None:
    # Sleep 120–240; buffer 45 → requested [75, 285] → aligned [60, 300]
    annot = mne.Annotations(
        onset=[0.0, 120.0, 240.0],
        duration=[120.0, 120.0, 360.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage W"],
    )
    recording = make_sleep_recording(duration_sec=600.0, annotations=annot)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state = AnnotationUnroller()(state)
    state = SleepBoundaryDetector()(state)
    state = WakeCropper(buffer_sec=45.0, align_to_epoch_grid=True)(state)

    assert state.extras["wake_crop"]["tmin_requested_sec"] == pytest.approx(75.0)
    assert state.extras["wake_crop"]["tmin_sec"] == pytest.approx(60.0)
    assert state.extras["wake_crop"]["tmax_sec"] == pytest.approx(300.0)
    assert state.epoch_labels is not None
    assert state.epoch_labels.onsets_sec[0] == pytest.approx(0.0)


def test_align_crop_window_helper() -> None:
    tmin, tmax = align_crop_window(
        75.0,
        285.0,
        epoch_duration_sec=30.0,
        recording_duration_sec=600.0,
    )
    assert tmin == 60.0
    assert tmax == 300.0


def test_wake_cropper_minutes_alias() -> None:
    cropper = WakeCropper(minutes=30)
    assert cropper.buffer_sec == 1800.0


def test_wake_cropper_requires_sleep() -> None:
    annot = mne.Annotations(onset=[0.0], duration=[300.0], description=["Sleep stage W"])
    recording = make_sleep_recording(duration_sec=300.0, annotations=annot)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state = AnnotationUnroller()(state)
    state = SleepBoundaryDetector()(state)
    with pytest.raises(NoSleepBoundaryError):
        WakeCropper(buffer_sec=30.0, require_sleep=True)(state)


def test_channel_selector() -> None:
    recording = make_sleep_recording(
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal", "submental", "Event marker"],
        ch_types=["eeg", "eeg", "eog", "emg", "stim"],
        annotations=mne.Annotations(onset=[0.0], duration=[30.0], description=["Sleep stage 2"]),
    )
    state = PreprocessedRecording.from_sleep_recording(recording)
    state = ChannelSelector(names=["Fpz-Cz", "horizontal"])(state)
    assert state.channel_names == ("Fpz-Cz", "horizontal")


def test_channel_selector_missing() -> None:
    recording = make_sleep_recording(
        ch_names=["Fpz-Cz", "Pz-Oz"],
        ch_types=["eeg", "eeg"],
        annotations=mne.Annotations(onset=[0.0], duration=[30.0], description=["Sleep stage 2"]),
    )
    state = PreprocessedRecording.from_sleep_recording(recording)
    with pytest.raises(MissingChannelsError):
        ChannelSelector(names=["Fpz-Cz", "missing"])(state)


def test_filter_and_normalize_zscore() -> None:
    annot = mne.Annotations(
        onset=[0.0, 60.0],
        duration=[60.0, 120.0],
        description=["Sleep stage W", "Sleep stage 2"],
    )
    recording = make_sleep_recording(duration_sec=300.0, annotations=annot)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    # 50 Hz notch is skipped automatically at 100 Hz (Nyquist).
    state = SignalFilter(l_freq=0.5, h_freq=30.0, notch_freqs=(50.0,))(state)
    assert state.extras["filter"]["applied_notch_freqs"] == []
    state = RecordingNormalizer(method="zscore")(state)
    data = state.raw.get_data()
    assert np.allclose(data.mean(axis=1), 0.0, atol=1e-6)
    assert np.allclose(data.std(axis=1), 1.0, atol=1e-5)


def test_robust_normalizer() -> None:
    recording = make_sleep_recording(
        duration_sec=60.0,
        annotations=mne.Annotations(onset=[0.0], duration=[60.0], description=["Sleep stage 2"]),
    )
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    # Inject a large outlier on channel 0 via public API path after normalize setup.
    data = state.raw.get_data()
    data[0, 0] = 1e-2
    state.raw = mne.io.RawArray(data, state.raw.info, verbose="ERROR")
    state = RecordingNormalizer(method="robust")(state)
    data = state.raw.get_data()
    medians = np.median(data, axis=1)
    assert np.allclose(medians, 0.0, atol=1e-6)
    q75 = np.percentile(data, 75, axis=1)
    q25 = np.percentile(data, 25, axis=1)
    assert np.allclose(q75 - q25, 1.0, atol=1e-5)


def test_default_pipeline_end_to_end() -> None:
    annot = mne.Annotations(
        onset=[0.0, 180.0, 360.0],
        duration=[180.0, 180.0, 240.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage W"],
    )
    recording = make_sleep_recording(
        duration_sec=600.0,
        annotations=annot,
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal", "submental"],
        ch_types=["eeg", "eeg", "eog", "emg"],
    )
    settings = PreprocessingSettings(
        wake_crop=WakeCropSettings(enabled=True, buffer_sec=60.0, require_sleep=True)
    )
    result = build_default_pipeline(settings).run(recording, preload=True)
    assert result.epoch_labels is not None
    assert set(result.epoch_labels.labels) <= {"W", "N1", "N2", "N3", "REM", "IGNORE"}
    assert result.channel_names == ("Fpz-Cz", "Pz-Oz", "horizontal", "submental")
    assert result.boundaries is not None
    assert result.boundaries.has_scored_sleep
    # Filter must precede wake crop so edge transients land in discarded wake.
    assert result.applied_transforms.index("signal_filter") < result.applied_transforms.index(
        "wake_cropper"
    )
    assert result.applied_transforms.index("channel_selector") < result.applied_transforms.index(
        "signal_filter"
    )
    assert result.epoch_labels.onsets_sec[0] == pytest.approx(0.0)


def test_default_pipeline_rejects_drop_policy() -> None:
    from sleep_staging.config import StageMapSettings

    settings = PreprocessingSettings(
        stage_map=StageMapSettings(unmapped_policy="drop"),
    )
    with pytest.raises(ValueError, match="refuses"):
        build_default_pipeline(settings)
