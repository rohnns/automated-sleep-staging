"""Unit tests for RawSignalEncoder and epoch slicing."""

from __future__ import annotations

import mne
import numpy as np
import pytest

from sleep_staging.representations import (
    EpochExtractionError,
    LabelVocabulary,
    RawSignalEncoder,
    preprocessed_to_epoch_batch,
)
from sleep_staging.preprocessing import (
    AnnotationUnroller,
    StageMapper,
)
from sleep_staging.preprocessing.types import EpochLabels, PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def _make_preprocessed_with_known_signals() -> tuple[PreprocessedRecording, np.ndarray]:
    """Build a short recording with deterministic per-epoch channel markers."""
    sfreq = 100.0
    duration_sec = 150.0  # 5 × 30 s epochs
    n_times = int(duration_sec * sfreq)
    ch_names = ["Fpz-Cz", "Pz-Oz"]
    data = np.zeros((len(ch_names), n_times), dtype=np.float64)

    # Epoch i samples get value (i + 1) on ch0 and -(i + 1) on ch1.
    for epoch_idx in range(5):
        start = epoch_idx * 3000
        stop = start + 3000
        data[0, start:stop] = float(epoch_idx + 1)
        data[1, start:stop] = -float(epoch_idx + 1)

    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=["eeg", "eeg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    annot = mne.Annotations(
        onset=[0.0, 30.0, 60.0, 90.0, 120.0],
        duration=[30.0] * 5,
        description=[
            "Sleep stage W",
            "Sleep stage 1",
            "Sleep stage 2",
            "Movement time",
            "Sleep stage R",
        ],
    )
    raw.set_annotations(annot, emit_warning=False)

    recording = make_sleep_recording(
        duration_sec=duration_sec,
        ch_names=ch_names,
        ch_types=["eeg", "eeg"],
        annotations=annot,
        subject_id="42",
        recording_id="7",
    )
    # Replace synthetic noise with deterministic markers while keeping metadata.
    recording.raw = raw

    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state = AnnotationUnroller()(state)
    state = StageMapper()(state)
    assert state.epoch_labels is not None
    assert state.epoch_labels.labels == ("W", "N1", "N2", "IGNORE", "REM")
    return state, data


def test_preprocessed_to_epoch_batch_shape_and_alignment() -> None:
    preprocessed, continuous = _make_preprocessed_with_known_signals()
    batch = preprocessed_to_epoch_batch(preprocessed)

    assert batch.n_epochs == 5
    assert batch.signals.shape == (5, 2, 3000)
    assert batch.labels.shape == (5,)
    assert batch.onsets_sec.tolist() == [0.0, 30.0, 60.0, 90.0, 120.0]
    assert batch.channel_names == ("Fpz-Cz", "Pz-Oz")
    assert batch.sfreq == 100.0
    assert batch.subject_id == "42"
    assert batch.recording_id == "7"
    assert batch.ignore_index == -100

    for epoch_idx in range(5):
        start = epoch_idx * 3000
        stop = start + 3000
        np.testing.assert_allclose(batch.signals[epoch_idx], continuous[:, start:stop])


def test_label_mapping_includes_ignore_minus_100() -> None:
    preprocessed, _ = _make_preprocessed_with_known_signals()
    batch = preprocessed_to_epoch_batch(preprocessed)
    assert batch.labels.tolist() == [0, 1, 2, -100, 4]

    vocab = LabelVocabulary(ignore_index=-100)
    assert vocab.encode(["W", "N1", "N2", "N3", "REM", "IGNORE"]).tolist() == [
        0,
        1,
        2,
        3,
        4,
        -100,
    ]


def test_raw_signal_encoder_from_preprocessed_recording() -> None:
    preprocessed, continuous = _make_preprocessed_with_known_signals()
    encoder = RawSignalEncoder(dtype="float32")
    encoded = encoder.encode_recording(preprocessed)

    assert encoded.n_epochs == 5
    assert encoded.features.shape == (5, 2, 3000)
    assert encoded.features.dtype == np.float32
    assert encoded.labels.tolist() == [0, 1, 2, -100, 4]
    assert encoded.ignore_index == -100
    assert encoded.subject_id == "42"
    assert encoded.recording_id == "7"
    assert encoded.channel_names == ("Fpz-Cz", "Pz-Oz")
    assert encoded.sfreq == 100.0
    assert encoded.metadata.representation == "raw"
    assert encoded.metadata.feature_shape == (2, 3000)
    assert encoded.metadata.channel_names == ("Fpz-Cz", "Pz-Oz")
    assert encoded.metadata.extras["ignore_index"] == -100
    assert encoded.onsets_sec is not None
    assert encoded.onsets_sec.tolist() == [0.0, 30.0, 60.0, 90.0, 120.0]

    # Signal-label alignment: epoch i features match continuous slice and label i.
    for epoch_idx, expected_label in enumerate([0, 1, 2, -100, 4]):
        start = epoch_idx * 3000
        stop = start + 3000
        np.testing.assert_allclose(
            encoded.features[epoch_idx],
            continuous[:, start:stop].astype(np.float32),
        )
        assert int(encoded.labels[epoch_idx]) == expected_label


def test_raw_signal_encoder_call_accepts_preprocessed() -> None:
    preprocessed, _ = _make_preprocessed_with_known_signals()
    encoded = RawSignalEncoder()(preprocessed)
    assert encoded.features.shape == (5, 2, 3000)


def test_raw_signal_encoder_single_channel_shape() -> None:
    sfreq = 100.0
    duration_sec = 90.0
    n_times = int(duration_sec * sfreq)
    data = np.arange(n_times, dtype=np.float64).reshape(1, n_times)
    info = mne.create_info(ch_names=["Fpz-Cz"], sfreq=sfreq, ch_types=["eeg"])
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    annot = mne.Annotations(
        onset=[0.0, 30.0, 60.0],
        duration=[30.0, 30.0, 30.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage R"],
    )
    raw.set_annotations(annot, emit_warning=False)

    recording = make_sleep_recording(
        duration_sec=duration_sec,
        ch_names=["Fpz-Cz"],
        ch_types=["eeg"],
        annotations=annot,
        subject_id="01",
        recording_id="1",
    )
    recording.raw = raw

    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state = AnnotationUnroller()(state)
    state = StageMapper()(state)

    encoded = RawSignalEncoder().encode_recording(state)
    assert encoded.features.shape == (3, 1, 3000)
    assert encoded.channel_names == ("Fpz-Cz",)
    assert encoded.n_epochs == 3


def test_raw_signal_encoder_features_are_independent_copy() -> None:
    preprocessed, _ = _make_preprocessed_with_known_signals()
    batch = preprocessed_to_epoch_batch(preprocessed)
    encoded = RawSignalEncoder().encode(batch)
    encoded.features[0, 0, 0] = 12345.0
    assert batch.signals[0, 0, 0] != 12345.0


def test_epoch_count_matches_epoch_labels() -> None:
    preprocessed, _ = _make_preprocessed_with_known_signals()
    assert preprocessed.epoch_labels is not None
    encoded = RawSignalEncoder().encode_recording(preprocessed)
    assert encoded.n_epochs == preprocessed.epoch_labels.n_epochs


def test_missing_epoch_labels_raises() -> None:
    recording = make_sleep_recording(
        duration_sec=60.0,
        ch_names=["Fpz-Cz"],
        ch_types=["eeg"],
        annotations=mne.Annotations(
            onset=[0.0], duration=[60.0], description=["Sleep stage 2"]
        ),
    )
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    with pytest.raises(EpochExtractionError, match="epoch_labels"):
        preprocessed_to_epoch_batch(state)


def test_overflowing_epoch_raises() -> None:
    recording = make_sleep_recording(
        duration_sec=60.0,
        ch_names=["Fpz-Cz"],
        ch_types=["eeg"],
        annotations=mne.Annotations(
            onset=[0.0], duration=[60.0], description=["Sleep stage 2"]
        ),
        subject_id="09",
        recording_id="1",
    )
    state = PreprocessedRecording.from_sleep_recording(recording, preload=True)
    state.epoch_labels = EpochLabels(
        onsets_sec=(0.0, 45.0),  # second epoch needs samples through 75 s
        duration_sec=30.0,
        labels=("N2", "N2"),
    )
    with pytest.raises(EpochExtractionError, match="overflows"):
        preprocessed_to_epoch_batch(state)
