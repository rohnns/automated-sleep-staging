"""Tests ensuring ChannelSelector defaults are staging-relevant and configurable."""

from __future__ import annotations

import mne

from sleep_staging.config import PreprocessingSettings
from sleep_staging.preprocessing import ChannelSelector, resolve_channel_picks
from sleep_staging.preprocessing.types import PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def test_default_staging_selection_is_eeg_eog_only():
    # Recording with expected channel names and types
    recording = make_sleep_recording(
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal", "submental", "oro-nasal", "rectal", "Event marker"],
        ch_types=["eeg", "eeg", "eog", "emg", "resp", "temperature", "stim"],
    )
    state = PreprocessedRecording.from_sleep_recording(recording)

    selector = ChannelSelector()
    state = selector(state)
    assert state.channel_names == ("Fpz-Cz", "Pz-Oz", "horizontal")


def test_auxiliary_excluded_by_default():
    recording = make_sleep_recording(
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal", "submental", "oro-nasal", "rectal", "Event marker"],
        ch_types=["eeg", "eeg", "eog", "emg", "resp", "temperature", "stim"],
    )
    state = PreprocessedRecording.from_sleep_recording(recording)
    selector = ChannelSelector()
    state = selector(state)
    assert "submental" not in state.channel_names
    assert "oro-nasal" not in state.channel_names
    assert "rectal" not in state.channel_names
    assert "Event marker" not in state.channel_names


def test_emg_supported_when_explicitly_enabled():
    recording = make_sleep_recording(
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal", "submental", "oro-nasal", "rectal", "Event marker"],
        ch_types=["eeg", "eeg", "eog", "emg", "resp", "temperature", "stim"],
    )
    state = PreprocessedRecording.from_sleep_recording(recording)
    selector = ChannelSelector(include_emg=True)
    state = selector(state)
    assert state.channel_names == ("Fpz-Cz", "Pz-Oz", "horizontal", "submental")


def test_explicit_selection_can_include_auxiliary():
    recording = make_sleep_recording(
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal", "submental", "oro-nasal", "rectal", "Event marker"],
        ch_types=["eeg", "eeg", "eog", "emg", "resp", "temperature", "stim"],
    )
    state = PreprocessedRecording.from_sleep_recording(recording)
    # Explicit names list retains ordering and can pick auxiliary channels
    selector = ChannelSelector(names=["Fpz-Cz", "oro-nasal", "rectal"])
    state = selector(state)
    assert state.channel_names == ("Fpz-Cz", "oro-nasal", "rectal")

    # Or selection by type
    state = PreprocessedRecording.from_sleep_recording(recording)
    selector = ChannelSelector(names=None, types=["eeg", "eog", "emg"], require_all_names=False)
    state = selector(state)
    assert set(state.channel_names) >= {"Fpz-Cz", "Pz-Oz", "horizontal", "submental"}
