"""Unit tests for SleepEDFLoader with mocked MNE I/O."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import mne
import numpy as np
import pytest

from sleep_staging.acquisition import SleepEDFLoader, SleepRecording, load_recording
from sleep_staging.acquisition.exceptions import (
    InvalidEDFFileError,
    MissingPSGFileError,
    RecordingValidationError,
)
from sleep_staging.config import AcquisitionSettings


def _make_raw(n_channels: int = 2, n_times: int = 10_000, sfreq: float = 100.0) -> mne.io.RawArray:
    data = np.random.default_rng(0).normal(size=(n_channels, n_times))
    info = mne.create_info(
        ch_names=["EEG Fpz-Cz", "EEG Pz-Oz"][:n_channels],
        sfreq=sfreq,
        ch_types=["eeg"] * n_channels,
    )
    return mne.io.RawArray(data, info, verbose="ERROR")


def _touch_pair(directory: Path, stem: str = "SC4001E0") -> tuple[Path, Path]:
    psg = directory / f"{stem}-PSG.edf"
    hyp = directory / f"{stem}-Hypnogram.edf"
    psg.write_bytes(b"dummy-psg")
    hyp.write_bytes(b"dummy-hyp")
    return psg, hyp


def test_load_recording_success(tmp_path: Path) -> None:
    psg, hyp = _touch_pair(tmp_path)
    raw = _make_raw()
    annot = mne.Annotations(
        onset=[0.0, 30.0],
        duration=[30.0, 30.0],
        description=["Sleep stage W", "Sleep stage 2"],
    )
    settings = AcquisitionSettings(data_root=tmp_path, preload=False)

    with (
        patch("sleep_staging.acquisition.loader.mne.io.read_raw_edf", return_value=raw) as read_psg,
        patch("sleep_staging.acquisition.loader.mne.read_annotations", return_value=annot) as read_hyp,
    ):
        recording = SleepEDFLoader(settings).load_recording(psg.name)

    assert isinstance(recording, SleepRecording)
    assert recording.subject_id == "00"
    assert recording.recording_id == "1"
    assert recording.metadata.n_annotations == 2
    assert recording.metadata.psg_path == psg.resolve()
    assert recording.metadata.hypnogram_path == hyp.resolve()
    # Single authoritative store: property delegates to raw.annotations
    assert recording.annotations is recording.raw.annotations
    assert len(recording.annotations) == 2
    assert len(recording.annotation_records) == 2
    assert recording.annotation_records[0].description == "Sleep stage W"
    read_psg.assert_called_once()
    read_hyp.assert_called_once()


def test_load_recording_missing_psg(tmp_path: Path) -> None:
    settings = AcquisitionSettings(data_root=tmp_path)
    with pytest.raises(MissingPSGFileError):
        SleepEDFLoader(settings).load_recording("SC4001E0-PSG.edf")


def test_load_recording_invalid_edf(tmp_path: Path) -> None:
    psg, _ = _touch_pair(tmp_path)
    settings = AcquisitionSettings(data_root=tmp_path)

    with patch(
        "sleep_staging.acquisition.loader.mne.io.read_raw_edf",
        side_effect=RuntimeError("corrupt"),
    ):
        with pytest.raises(InvalidEDFFileError, match="Failed to read PSG"):
            SleepEDFLoader(settings).load_recording(psg)


def test_discover_and_iter(tmp_path: Path) -> None:
    _touch_pair(tmp_path, "SC4001E0")
    _touch_pair(tmp_path, "SC4011E0")
    settings = AcquisitionSettings(data_root=tmp_path)
    loader = SleepEDFLoader(settings)

    discovered = loader.discover()
    assert len(discovered) == 2

    raw = _make_raw()
    annot = mne.Annotations(onset=[0.0], duration=[30.0], description=["Sleep stage W"])
    with (
        patch("sleep_staging.acquisition.loader.mne.io.read_raw_edf", return_value=raw),
        patch("sleep_staging.acquisition.loader.mne.read_annotations", return_value=annot),
    ):
        recordings = loader.load_all()

    assert len(recordings) == 2
    subjects = {rec.subject_id for rec in recordings}
    assert subjects == {"00", "01"}


def test_convenience_load_recording(tmp_path: Path) -> None:
    psg, _ = _touch_pair(tmp_path)
    raw = _make_raw()
    annot = mne.Annotations(onset=[0.0], duration=[30.0], description=["Sleep stage REM"])
    settings = AcquisitionSettings(data_root=tmp_path)

    with (
        patch("sleep_staging.acquisition.loader.mne.io.read_raw_edf", return_value=raw),
        patch("sleep_staging.acquisition.loader.mne.read_annotations", return_value=annot),
    ):
        recording = load_recording(psg, settings=settings)

    assert "SleepRecording" in repr(recording)
    assert recording.metadata.study == "SC"


def test_load_rejects_empty_hypnogram(tmp_path: Path) -> None:
    psg, _ = _touch_pair(tmp_path)
    raw = _make_raw()
    empty = mne.Annotations(onset=[], duration=[], description=[])
    settings = AcquisitionSettings(data_root=tmp_path)

    with (
        patch("sleep_staging.acquisition.loader.mne.io.read_raw_edf", return_value=raw),
        patch("sleep_staging.acquisition.loader.mne.read_annotations", return_value=empty),
    ):
        with pytest.raises(RecordingValidationError):
            SleepEDFLoader(settings).load_recording(psg)
