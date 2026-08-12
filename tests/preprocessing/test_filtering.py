"""Unit tests for channel-type-aware SignalFilter."""
from __future__ import annotations

import numpy as np
import mne

from sleep_staging.preprocessing.filtering import SignalFilter
from sleep_staging.preprocessing.types import PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def test_filter_preserves_shape_and_names():
    # Make a 3-channel recording: eeg, eog, emg
    sfreq = 100.0
    t = np.arange(0, 1.0, 1.0 / sfreq)
    eeg = np.sin(2 * np.pi * 1.0 * t)
    eog = 100.0 * np.ones_like(t)
    emg = np.random.randn(len(t)) * 0.001

    data = np.vstack([eeg, eog, emg])
    info = mne.create_info(ch_names=["Fpz-Cz", "horizontal", "submental"], sfreq=sfreq, ch_types=["eeg", "eog", "emg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    # Build PreprocessedRecording
    from sleep_staging.acquisition.dataclasses import RecordingMetadata, ChannelInfo
    from pathlib import Path

    channels = (
        ChannelInfo(name="Fpz-Cz", ch_type="eeg", unit="V", sampling_frequency=sfreq),
        ChannelInfo(name="horizontal", ch_type="eog", unit="V", sampling_frequency=sfreq),
        ChannelInfo(name="submental", ch_type="emg", unit="V", sampling_frequency=sfreq),
    )
    metadata = RecordingMetadata(
        subject_id="00",
        recording_id="1",
        study="SC",
        sampling_frequency=sfreq,
        duration_sec=1.0,
        n_channels=3,
        channel_names=("Fpz-Cz", "horizontal", "submental"),
        channel_types=("eeg", "eog", "emg"),
        channels=channels,
        units={"Fpz-Cz": "V", "horizontal": "V", "submental": "V"},
        reference="test",
        montage=None,
        meas_date=None,
        psg_path=Path("SC0001-PSG.edf"),
        hypnogram_path=Path("SC0001-Hyp.edf"),
        n_annotations=0,
    )
    state = PreprocessedRecording(raw=raw, metadata=metadata)

    # Apply SignalFilter with per-type defaults
    filt = SignalFilter(
        l_freq=0.5,
        h_freq=30.0,
        eog_l_freq=0.5,
        eog_h_freq=15.0,
        emg_l_freq=10.0,
        emg_h_freq=30.0,
        notch_freqs=(),
    )
    state2 = filt(state)

    # Shape and names preserved
    assert state2.raw.get_data().shape == data.shape
    assert state2.raw.ch_names == ["Fpz-Cz", "horizontal", "submental"]
    # Extras recorded
    assert "filter" in state2.extras
    per_ch = state2.extras["filter"]["per_channel"]
    assert per_ch["Fpz-Cz"]["mne_type"] == "eeg"
    assert per_ch["Fpz-Cz"]["l_freq"] == 0.5
    assert per_ch["Fpz-Cz"]["h_freq"] == 30.0
    assert per_ch["horizontal"]["mne_type"] == "eog"
    assert per_ch["horizontal"]["h_freq"] == 15.0
    assert per_ch["submental"]["mne_type"] == "emg"
    assert per_ch["submental"]["l_freq"] == 10.0


def test_unknown_channel_type_skipped():
    # Unknown channel type (e.g., temperature) should not get band-pass by default
    sfreq = 100.0
    t = np.arange(0, 1.0, 1.0 / sfreq)
    temp = 36.5 * np.ones_like(t)
    data = np.vstack([temp])
    info = mne.create_info(ch_names=["Temperature"], sfreq=sfreq, ch_types=["temperature"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    from sleep_staging.acquisition.dataclasses import RecordingMetadata, ChannelInfo
    from pathlib import Path

    channels = (
        ChannelInfo(name="Temperature", ch_type="temperature", unit="degC", sampling_frequency=sfreq),
    )
    metadata = RecordingMetadata(
        subject_id="00",
        recording_id="1",
        study="SC",
        sampling_frequency=sfreq,
        duration_sec=1.0,
        n_channels=1,
        channel_names=("Temperature",),
        channel_types=("temperature",),
        channels=channels,
        units={"Temperature": "degC"},
        reference="test",
        montage=None,
        meas_date=None,
        psg_path=Path("SC0001-PSG.edf"),
        hypnogram_path=Path("SC0001-Hyp.edf"),
        n_annotations=0,
    )
    state = PreprocessedRecording(raw=raw, metadata=metadata)

    filt = SignalFilter(l_freq=0.5, h_freq=30.0, notch_freqs=())
    state2 = filt(state)
    per_ch = state2.extras["filter"]["per_channel"]
    assert per_ch["Temperature"]["l_freq"] is None
    assert per_ch["Temperature"]["h_freq"] is None


def test_signal_filter_default_instantiation_uses_type_cutoffs():
    sfreq = 100.0
    t = np.arange(0, 1.0, 1.0 / sfreq)
    eeg = np.sin(2 * np.pi * 1.0 * t)
    eog = 100.0 * np.ones_like(t)
    data = np.vstack([eeg, eog])
    info = mne.create_info(ch_names=["Fpz-Cz", "horizontal"], sfreq=sfreq, ch_types=["eeg", "eog"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")

    from sleep_staging.acquisition.dataclasses import RecordingMetadata, ChannelInfo
    from pathlib import Path

    channels = (
        ChannelInfo(name="Fpz-Cz", ch_type="eeg", unit="V", sampling_frequency=sfreq),
        ChannelInfo(name="horizontal", ch_type="eog", unit="V", sampling_frequency=sfreq),
    )
    metadata = RecordingMetadata(
        subject_id="00",
        recording_id="1",
        study="SC",
        sampling_frequency=sfreq,
        duration_sec=1.0,
        n_channels=2,
        channel_names=("Fpz-Cz", "horizontal"),
        channel_types=("eeg", "eog"),
        channels=channels,
        units={"Fpz-Cz": "V", "horizontal": "V"},
        reference="test",
        montage=None,
        meas_date=None,
        psg_path=Path("SC0001-PSG.edf"),
        hypnogram_path=Path("SC0001-Hyp.edf"),
        n_annotations=0,
    )
    state = PreprocessedRecording(raw=raw, metadata=metadata)

    # Instantiate SignalFilter with NO arguments
    filt = SignalFilter()
    state2 = filt(state)
    per_ch = state2.extras["filter"]["per_channel"]

    assert per_ch["Fpz-Cz"]["h_freq"] == 30.0
    assert per_ch["horizontal"]["h_freq"] == 15.0

