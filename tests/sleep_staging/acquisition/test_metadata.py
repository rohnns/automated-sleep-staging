"""Unit tests for metadata extraction helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import mne
import numpy as np
import pytest

from sleep_staging.acquisition.dataclasses import RecordingMetadata, annotations_to_records
from sleep_staging.acquisition.exceptions import RecordingValidationError
from sleep_staging.acquisition.metadata import extract_metadata, validate_recording
from sleep_staging.acquisition.utils import parse_psg_filename


def _make_raw(n_channels: int = 2, n_times: int = 100, sfreq: float = 100.0) -> mne.io.RawArray:
    data = np.zeros((n_channels, n_times), dtype=np.float64)
    ch_names = ["EEG Fpz-Cz", "EEG Pz-Oz"][:n_channels]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg"] * n_channels)
    return mne.io.RawArray(data, info, verbose="ERROR")


def test_annotations_to_records() -> None:
    annot = mne.Annotations(
        onset=[0.0, 30.0],
        duration=[30.0, 30.0],
        description=["Sleep stage W", "Sleep stage 1"],
    )
    records = annotations_to_records(annot)
    assert len(records) == 2
    assert records[0].description == "Sleep stage W"
    assert records[1].onset == 30.0


def test_extract_metadata() -> None:
    raw = _make_raw()
    annot = mne.Annotations(onset=[0.0], duration=[30.0], description=["Sleep stage W"])
    raw.set_annotations(annot, emit_warning=False)
    psg = Path("/data/SC4001E0-PSG.edf")
    hyp = Path("/data/SC4001E0-Hypnogram.edf")
    meta = extract_metadata(
        raw,
        psg_path=psg,
        hypnogram_path=hyp,
        file_ids=parse_psg_filename(psg),
    )
    assert meta.subject_id == "00"
    assert meta.recording_id == "1"
    assert meta.study == "SC"
    assert meta.sampling_frequency == 100.0
    assert meta.n_channels == 2
    assert meta.n_annotations == 1
    assert "annotations" not in RecordingMetadata.__dataclass_fields__
    assert meta.reference is not None
    assert "Fpz-Cz" in meta.reference or "Bipolar" in meta.reference


def test_validate_recording_rejects_empty_annotations() -> None:
    raw = _make_raw()
    raw.set_annotations(mne.Annotations(onset=[], duration=[], description=[]), emit_warning=False)
    with pytest.raises(RecordingValidationError):
        validate_recording(
            raw,
            psg_path=Path("SC4001E0-PSG.edf"),
            hypnogram_path=Path("SC4001E0-Hypnogram.edf"),
        )


def test_validate_recording_rejects_no_channels() -> None:
    raw = MagicMock()
    raw.ch_names = []
    raw.n_times = 10
    raw.info = {"sfreq": 100.0}
    raw.annotations = mne.Annotations(onset=[0.0], duration=[30.0], description=["Sleep stage W"])
    with pytest.raises(RecordingValidationError):
        validate_recording(
            raw,
            psg_path=Path("SC4001E0-PSG.edf"),
            hypnogram_path=Path("SC4001E0-Hypnogram.edf"),
        )
