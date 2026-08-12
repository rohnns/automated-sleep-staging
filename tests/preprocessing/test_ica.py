"""Unit tests for MNE ICATransform."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import mne
import numpy as np

from sleep_staging.config import ICASettings, PreprocessingSettings, WakeCropSettings, load_settings
from sleep_staging.preprocessing import ICATransform, build_default_pipeline
from sleep_staging.preprocessing.types import EpochLabels, PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def _blinky_raw(
    *,
    duration_sec: float = 120.0,
    sfreq: float = 100.0,
    include_eog: bool = True,
    n_eeg: int = 2,
    seed: int = 1,
) -> mne.io.RawArray:
    n = int(duration_sec * sfreq)
    rng = np.random.default_rng(seed)
    blinks = np.zeros(n)
    for b in range(5, int(duration_sec) - 5, 8):
        i = int(b * sfreq)
        blinks[i : i + 30] = np.hanning(30) * 2e-4

    channels: list[np.ndarray] = []
    names: list[str] = []
    types: list[str] = []
    eeg_names = ["Fpz-Cz", "Pz-Oz", "EEG3"]
    for idx in range(n_eeg):
        sig = rng.normal(0, 1e-5, n) + blinks * (1.0 - 0.1 * idx)
        channels.append(sig)
        names.append(eeg_names[idx])
        types.append("eeg")
    if include_eog:
        channels.append(blinks * 1.2 + rng.normal(0, 5e-7, n))
        names.append("horizontal")
        types.append("eog")

    info = mne.create_info(ch_names=names, sfreq=sfreq, ch_types=types)
    return mne.io.RawArray(np.vstack(channels), info, verbose="ERROR")


def _state_from_raw(raw: mne.io.RawArray) -> PreprocessedRecording:
    recording = make_sleep_recording(
        duration_sec=float(raw.n_times) / float(raw.info["sfreq"]),
        ch_names=list(raw.ch_names),
        ch_types=list(raw.get_channel_types()),
        annotations=mne.Annotations(
            onset=[0.0, 30.0, 60.0],
            duration=[30.0, 30.0, 60.0],
            description=["Sleep stage W", "Sleep stage 2", "Sleep stage W"],
        ),
    )
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state.raw = raw.copy().load_data()
    state.epoch_labels = EpochLabels(
        onsets_sec=(0.0, 30.0, 60.0, 90.0),
        duration_sec=30.0,
        labels=("W", "N2", "W", "W"),
    )
    return state


def test_ica_enabled_runs_and_records_extras() -> None:
    state = _state_from_raw(_blinky_raw(include_eog=True, n_eeg=2))
    names_before = list(state.raw.ch_names)
    labels_before = state.epoch_labels.labels if state.epoch_labels else None
    n_times_before = state.raw.n_times
    sfreq_before = state.sampling_frequency

    out = ICATransform(random_state=42, eog_measure="correlation", eog_threshold=0.8)(state)
    ica = out.extras["ica"]

    assert ica["ran"] is True
    assert ica["skipped_reason"] is None
    assert ica["eeg_channels"] == ["Fpz-Cz", "Pz-Oz"]
    assert ica["eog_channels"] == ["horizontal"]
    assert ica["n_components_fitted"] == 2
    assert ica["random_state"] == 42
    assert ica["method"] == "fastica"
    assert ica["eog_detection"]["performed"] is True
    assert ica["eog_detection"]["ch_name"] == "horizontal"
    assert isinstance(ica["excluded_components"], list)
    assert isinstance(ica["eog_detection"]["scores"], list)
    assert list(out.raw.ch_names) == names_before
    assert out.raw.n_times == n_times_before
    assert out.sampling_frequency == sfreq_before
    assert out.epoch_labels is not None
    assert out.epoch_labels.labels == labels_before


def test_ica_reproducible_with_fixed_random_state() -> None:
    raw = _blinky_raw(include_eog=True, n_eeg=2, seed=3)
    a = ICATransform(random_state=7, detect_eog=True)(_state_from_raw(raw.copy()))
    b = ICATransform(random_state=7, detect_eog=True)(_state_from_raw(raw.copy()))
    assert np.allclose(a.raw.get_data(), b.raw.get_data())
    assert a.extras["ica"]["excluded_components"] == b.extras["ica"]["excluded_components"]
    assert a.extras["ica"]["eog_detection"]["scores"] == b.extras["ica"]["eog_detection"]["scores"]


def test_ica_without_eog_still_fits() -> None:
    state = _state_from_raw(_blinky_raw(include_eog=False, n_eeg=2))
    out = ICATransform(random_state=42, detect_eog=True)(state)
    ica = out.extras["ica"]
    assert ica["ran"] is True
    assert ica["eog_channels"] == []
    assert ica["eog_detection"]["performed"] is False
    assert ica["eog_detection"]["skipped_reason"] == "no EOG channel available"
    assert ica["excluded_components"] == []
    assert list(out.raw.ch_names) == ["Fpz-Cz", "Pz-Oz"]


def test_ica_with_eog_can_exclude_components() -> None:
    state = _state_from_raw(_blinky_raw(include_eog=True, n_eeg=2))
    before = state.raw.get_data().copy()
    out = ICATransform(
        random_state=42,
        detect_eog=True,
        eog_measure="correlation",
        eog_threshold=0.5,
    )(state)
    after = out.raw.get_data()
    ica = out.extras["ica"]
    assert ica["eog_detection"]["performed"] is True
    # EOG channel must remain untouched.
    assert np.allclose(before[2], after[2])
    if ica["excluded_components"]:
        assert not np.allclose(before[:2], after[:2])


def test_ica_skips_with_too_few_eeg_channels() -> None:
    state = _state_from_raw(_blinky_raw(include_eog=True, n_eeg=1))
    before = state.raw.get_data().copy()
    out = ICATransform(random_state=42)(state)
    ica = out.extras["ica"]
    assert ica["ran"] is False
    assert "fewer than 2" in ica["skipped_reason"]
    assert np.allclose(out.raw.get_data(), before)


def test_default_pipeline_ica_enabled_from_config() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = load_settings(project_root / "configs" / "default.yaml")
    assert settings.preprocessing.ica.enabled is True
    assert settings.preprocessing.ica.random_state == 42
    assert settings.preprocessing.ica.eog_measure == "correlation"

    pipeline = build_default_pipeline(settings.preprocessing)
    names = [t.name for t in pipeline.transforms]
    assert "ica" in names
    assert names.index("signal_filter") < names.index("ica")
    assert names.index("ica") < names.index("wake_cropper")


def test_default_pipeline_ica_disabled_path() -> None:
    project_root = Path(__file__).resolve().parents[2]
    base = load_settings(project_root / "configs" / "default.yaml").preprocessing
    settings = replace(base, ica=replace(base.ica, enabled=False))
    pipeline = build_default_pipeline(settings)
    assert "ica" not in [t.name for t in pipeline.transforms]

    annot = mne.Annotations(
        onset=[0.0, 180.0, 360.0],
        duration=[180.0, 180.0, 240.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage W"],
    )
    recording = make_sleep_recording(
        duration_sec=600.0,
        annotations=annot,
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal"],
        ch_types=["eeg", "eeg", "eog"],
    )
    result = pipeline.run(recording, preload=True)
    assert "ica" not in result.applied_transforms
    assert "ica" not in result.extras


def test_default_pipeline_ica_preserves_labels_and_channels() -> None:
    settings = PreprocessingSettings(
        wake_crop=WakeCropSettings(enabled=False, buffer_sec=60.0),
        ica=ICASettings(enabled=True, random_state=42),
    )
    annot = mne.Annotations(
        onset=[0.0, 180.0, 360.0],
        duration=[180.0, 180.0, 240.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage W"],
    )
    recording = make_sleep_recording(
        duration_sec=600.0,
        annotations=annot,
        ch_names=["Fpz-Cz", "Pz-Oz", "horizontal"],
        ch_types=["eeg", "eeg", "eog"],
    )
    result = build_default_pipeline(settings).run(recording, preload=True)
    assert result.channel_names == ("Fpz-Cz", "Pz-Oz", "horizontal")
    assert result.epoch_labels is not None
    assert result.epoch_labels.n_epochs > 0
    assert result.extras["ica"]["ran"] is True
    assert "ica" in result.applied_transforms
