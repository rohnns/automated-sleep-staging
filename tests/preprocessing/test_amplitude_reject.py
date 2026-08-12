"""Unit and pipeline tests for epoch amplitude rejection."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mne
import numpy as np

from sleep_staging.config import (
    AmplitudeRejectSettings,
    FilterSettings,
    ICASettings,
    PreprocessingSettings,
    WakeCropSettings,
    load_settings,
)
from sleep_staging.preprocessing import AmplitudeEpochRejector, build_default_pipeline
from sleep_staging.preprocessing.types import EpochLabels, PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def _state_with_epochs(
    *,
    duration_sec: float = 120.0,
    sfreq: float = 100.0,
    spike_epoch: int | None = 1,
    spike_amp: float = 1.0e-3,
    eog_spike: bool = False,
) -> PreprocessedRecording:
    n = int(duration_sec * sfreq)
    rng = np.random.default_rng(0)
    eeg1 = rng.normal(0, 1e-6, n)
    eeg2 = rng.normal(0, 1e-6, n)
    eog = rng.normal(0, 1e-6, n)
    if spike_epoch is not None:
        start = int(spike_epoch * 30 * sfreq)
        mid = start + int(15 * sfreq)
        if eog_spike:
            eog[mid] = spike_amp
            eog[mid + 1] = -spike_amp
        else:
            eeg1[mid] = spike_amp
            eeg1[mid + 1] = 0.0
    info = mne.create_info(
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg", "eog"],
    )
    raw = mne.io.RawArray(np.vstack([eeg1, eeg2, eog]), info, verbose="ERROR")
    recording = make_sleep_recording(
        duration_sec=duration_sec,
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal"],
        ch_types=["eeg", "eeg", "eog"],
        annotations=mne.Annotations(
            onset=[0.0, 30.0, 60.0, 90.0],
            duration=[30.0, 30.0, 30.0, 30.0],
            description=["Sleep stage W", "Sleep stage 2", "Sleep stage 1", "Sleep stage W"],
        ),
    )
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state.raw = raw
    state.epoch_labels = EpochLabels(
        onsets_sec=(0.0, 30.0, 60.0, 90.0),
        duration_sec=30.0,
        labels=("W", "N2", "N1", "W"),
    )
    return state


def test_amplitude_reject_marks_high_ptp_epoch_as_ignore() -> None:
    state = _state_with_epochs(spike_epoch=1, spike_amp=1.0e-3)
    onsets_before = state.epoch_labels.onsets_sec
    n_before = state.epoch_labels.n_epochs
    data_before = state.raw.get_data().copy()

    out = AmplitudeEpochRejector(
        eeg_peak_to_peak=5.0e-4,
        eog_peak_to_peak=1.0e-3,
    )(state)

    assert out.epoch_labels.n_epochs == n_before
    assert out.epoch_labels.onsets_sec == onsets_before
    assert out.epoch_labels.labels[1] == "IGNORE"
    assert out.epoch_labels.labels[0] == "W"
    assert out.epoch_labels.labels[2] == "N1"
    assert np.allclose(out.raw.get_data(), data_before)

    report = out.extras["amplitude_reject"]
    assert report["rule"] == "peak_to_peak"
    assert report["n_rejected"] == 1
    assert report["rejected_epoch_indices"] == [1]
    assert any("peak_to_peak" in r for r in report["reasons"]["1"])
    assert report["thresholds"]["eeg_peak_to_peak"] == 5.0e-4


def test_amplitude_reject_type_specific_thresholds() -> None:
    # EEG spike below EEG threshold but above a tiny EOG threshold should not
    # reject via EOG; EOG spike above EOG threshold should reject.
    state = _state_with_epochs(spike_epoch=2, spike_amp=8.0e-4, eog_spike=True)
    out = AmplitudeEpochRejector(
        eeg_peak_to_peak=5.0e-4,
        eog_peak_to_peak=5.0e-4,
        emg_peak_to_peak=None,
    )(state)
    assert out.epoch_labels.labels[2] == "IGNORE"
    reasons = out.extras["amplitude_reject"]["reasons"]["2"]
    assert any(r.startswith("eog:horizontal:peak_to_peak") for r in reasons)


def test_amplitude_reject_nonfinite_marks_ignore() -> None:
    state = _state_with_epochs(spike_epoch=None)
    data = state.raw.get_data()
    data[0, int(30 * 100) + 10] = np.nan
    state.raw = mne.io.RawArray(data, state.raw.info.copy(), verbose="ERROR")
    out = AmplitudeEpochRejector(eeg_peak_to_peak=5.0e-4)(state)
    assert out.epoch_labels.labels[1] == "IGNORE"
    assert any("nonfinite" in r for r in out.extras["amplitude_reject"]["reasons"]["1"])


def test_default_pipeline_amplitude_reject_position_and_config() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = load_settings(project_root / "configs" / "default.yaml")
    assert settings.preprocessing.amplitude_reject.enabled is True
    assert settings.preprocessing.amplitude_reject.eeg_peak_to_peak == 5.0e-4
    assert settings.preprocessing.amplitude_reject.eog_peak_to_peak == 1.0e-3

    names = [t.name for t in build_default_pipeline(settings.preprocessing).transforms]
    assert "amplitude_epoch_rejector" in names
    assert names.index("stage_mapper") < names.index("amplitude_epoch_rejector")
    assert names.index("amplitude_epoch_rejector") < names.index("recording_normalizer")


def test_default_pipeline_amplitude_reject_disabled() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base = load_settings(project_root / "configs" / "default.yaml").preprocessing
    settings = replace(
        base,
        amplitude_reject=replace(base.amplitude_reject, enabled=False),
    )
    assert "amplitude_epoch_rejector" not in [
        t.name for t in build_default_pipeline(settings).transforms
    ]


def test_default_pipeline_amplitude_reject_integration() -> None:
    """Synthetic recording with one spiked epoch is marked IGNORE via default path."""
    settings = PreprocessingSettings(
        wake_crop=WakeCropSettings(enabled=False),
        ica=ICASettings(enabled=False),
        filter=FilterSettings(enabled=False),
        amplitude_reject=AmplitudeRejectSettings(
            enabled=True,
            eeg_peak_to_peak=5.0e-4,
            eog_peak_to_peak=1.0e-3,
            emg_peak_to_peak=1.0e-3,
        ),
    )
    sfreq = 100.0
    duration_sec = 120.0
    n = int(duration_sec * sfreq)
    rng = np.random.default_rng(1)
    eeg1 = rng.normal(0, 1e-6, n)
    eeg2 = rng.normal(0, 1e-6, n)
    eog = rng.normal(0, 1e-6, n)
    mid = int(45 * sfreq)
    eeg1[mid] = 2.0e-3
    info = mne.create_info(
        ["Fpz-Cz", "Pz-Oz", "horizontal"], sfreq, ["eeg", "eeg", "eog"]
    )
    raw = mne.io.RawArray(np.vstack([eeg1, eeg2, eog]), info, verbose="ERROR")
    annot = mne.Annotations(
        onset=[0.0, 30.0, 60.0, 90.0],
        duration=[30.0, 30.0, 30.0, 30.0],
        description=[
            "Sleep stage W",
            "Sleep stage 2",
            "Sleep stage 1",
            "Sleep stage W",
        ],
    )
    recording = make_sleep_recording(
        duration_sec=duration_sec,
        annotations=annot,
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal"],
        ch_types=["eeg", "eeg", "eog"],
    )
    # Replace synthetic noise with spiked raw while keeping metadata/annotations.
    recording.raw = raw
    recording.raw.set_annotations(annot, emit_warning=False)

    result = build_default_pipeline(settings).run(recording, preload=True)
    assert result.epoch_labels is not None
    assert result.epoch_labels.n_epochs == 4
    assert result.epoch_labels.onsets_sec == (0.0, 30.0, 60.0, 90.0)
    assert result.epoch_labels.labels[1] == "IGNORE"
    assert result.extras["amplitude_reject"]["n_rejected"] >= 1
    assert "amplitude_epoch_rejector" in result.applied_transforms
