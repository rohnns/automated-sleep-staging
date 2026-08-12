"""Unit tests for the non-destructive BadChannelDetector."""

from __future__ import annotations

import numpy as np

from sleep_staging.preprocessing.bad_channels import BadChannelDetector
from sleep_staging.preprocessing.types import PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def test_bad_channel_marks_flat_and_nan() -> None:
    # Short recording to keep test small: 10 s @ 100 Hz -> 1000 samples
    recording = make_sleep_recording(duration_sec=10.0)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)

    # Mutate data: channel 0 flat (zeros); channel 1 many NaNs; channel 2 saturated
    data = state.raw.get_data().copy()
    data[0, :] = 0.0
    data[1, :500] = np.nan  # 50% NaN fraction -> above default 1%
    # Saturation: set channel 2 to a constant large value for most samples
    data[2, :] = 1e-3
    state.raw = state.raw.__class__(data, state.raw.info, verbose="ERROR")

    detector = BadChannelDetector(
        flat_std_threshold=1e-9,
        nan_frac_threshold=0.01,
        saturation_frac_threshold=0.9,
        mark_mne_bads=True,
    )
    before_data = state.raw.get_data().copy()
    state = detector(state)

    report = state.extras.get("bad_channels", {})
    assert "Fpz-Cz" in report["flat"]
    # second channel (Pz-Oz) should be flagged for NaNs
    assert "Pz-Oz" in report["nan"]
    # third channel is saturated by construction
    assert "horizontal" in report["saturation"] or "horizontal" in report["extreme_amplitude"]
    # Raw data should be unchanged in shape and values
    after_data = state.raw.get_data()
    assert after_data.shape == before_data.shape
    assert np.allclose(after_data, before_data, equal_nan=True)


def test_bad_channel_sets_mne_bads_field() -> None:
    recording = make_sleep_recording(duration_sec=5.0)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    data = state.raw.get_data()
    # Make channel 0 flat
    data[0, :] = 0.0
    state.raw = state.raw.__class__(data, state.raw.info, verbose="ERROR")

    detector = BadChannelDetector(flat_std_threshold=1e-9, mark_mne_bads=True)
    state = detector(state)
    report = state.extras.get("bad_channels", {})
    assert "Fpz-Cz" in report["flat"]
    assert "Fpz-Cz" in state.raw.info.get("bads", [])
