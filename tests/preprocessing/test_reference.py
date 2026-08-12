"""Unit tests for ReferenceTransform: original (no-op) and common-average (EEG-only CAR)."""

from __future__ import annotations

import numpy as np

import mne
from sleep_staging.preprocessing.reference import ReferenceTransform
from sleep_staging.preprocessing.types import PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def test_reference_original_noop():
    recording = make_sleep_recording(ch_names=["Fpz-Cz", "Pz-Oz", "horizontal"], ch_types=["eeg", "eeg", "eog"], duration_sec=1.0)
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    before = state.raw.get_data().copy()
    state = ReferenceTransform(mode="original")(state)
    after = state.raw.get_data()
    assert after.shape == before.shape
    assert np.allclose(after, before)
    assert state.extras.get("reference", {}).get("method") == "original"


def test_reference_common_average_eeg_only():
    # Create 3 channels: eeg1, eeg2, eog
    sfreq = 100.0
    t = np.arange(0, 1.0, 1.0 / sfreq)
    eeg1 = np.sin(2 * np.pi * 1.0 * t)  # channel 0
    eeg2 = 2.0 * np.sin(2 * np.pi * 1.0 * t)  # channel 1 (double amplitude)
    eog = 100.0 * np.ones_like(t)  # channel 2 (should not be included in CAR)

    data = np.vstack([eeg1, eeg2, eog])
    info = mne.create_info(ch_names=["Fpz-Cz", "Pz-Oz", "horizontal"], sfreq=sfreq, ch_types=["eeg", "eeg", "eog"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    # Build PreprocessedRecording manually
    from sleep_staging.acquisition.dataclasses import RecordingMetadata, ChannelInfo
    from pathlib import Path

    channels = (
        ChannelInfo(name="Fpz-Cz", ch_type="eeg", unit="V", sampling_frequency=sfreq),
        ChannelInfo(name="Pz-Oz", ch_type="eeg", unit="V", sampling_frequency=sfreq),
        ChannelInfo(name="horizontal", ch_type="eog", unit="V", sampling_frequency=sfreq),
    )
    metadata = RecordingMetadata(
        subject_id="00",
        recording_id="1",
        study="SC",
        sampling_frequency=sfreq,
        duration_sec=1.0,
        n_channels=3,
        channel_names=("Fpz-Cz", "Pz-Oz", "horizontal"),
        channel_types=("eeg", "eeg", "eog"),
        channels=channels,
        units={"Fpz-Cz": "V", "Pz-Oz": "V", "horizontal": "V"},
        reference="test",
        montage=None,
        meas_date=None,
        psg_path=Path("SC0001-PSG.edf"),
        hypnogram_path=Path("SC0001-Hyp.edf"),
        n_annotations=0,
    )
    state = PreprocessedRecording(raw=raw, metadata=metadata)

    # Apply CAR
    state = ReferenceTransform(mode="common_average")(state)
    after = state.raw.get_data()

    # CAR should subtract the mean of eeg1 and eeg2 from those channels
    # Mean at each sample = (eeg1 + eeg2)/2 -> expected eeg1' = eeg1 - (eeg1+eeg2)/2 = (eeg1 - eeg2)/2
    expected_eeg1 = (eeg1 - eeg2) / 2.0
    expected_eeg2 = (eeg2 - (eeg1 + eeg2) / 2.0)
    assert np.allclose(after[0], expected_eeg1)
    assert np.allclose(after[1], expected_eeg2)

    # EOG channel must be unchanged
    assert np.allclose(after[2], eog)

    # Channel ordering preserved
    assert state.raw.ch_names == ["Fpz-Cz", "Pz-Oz", "horizontal"]
    assert state.extras.get("reference", {}).get("method") == "common_average"
    assert set(state.extras.get("reference", {}).get("applied_channels", [])) == {"Fpz-Cz", "Pz-Oz"}
