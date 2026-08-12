"""Architecture tests for Phase 3 encodings."""

from __future__ import annotations

import numpy as np
import pytest

from sleep_staging.config import EncodingSettings
from sleep_staging.representations import (
    DEFAULT_BANDS,
    BandPowerEncoder,
    EncoderNotImplementedError,
    EncodedDataset,
    EncodedDatasetCollection,
    EpochTensorBatch,
    LabelVocabulary,
    RawSignalEncoder,
    RepresentationMetadata,
    STFTBackend,
    TimeFrequencyEncoder,
    build_encoder,
)


def _fake_batch(*, n_epochs: int = 4, n_channels: int = 2, n_times: int = 3000) -> EpochTensorBatch:
    rng = np.random.default_rng(0)
    return EpochTensorBatch(
        signals=rng.normal(size=(n_epochs, n_channels, n_times)),
        labels=np.asarray([0, 1, 2, -100][:n_epochs], dtype=np.int64),
        onsets_sec=np.arange(n_epochs, dtype=np.float64) * 30.0,
        channel_names=tuple(f"ch{i}" for i in range(n_channels)),
        sfreq=100.0,
        epoch_duration_sec=30.0,
        subject_id="00",
        recording_id="1",
        ignore_index=-100,
    )


def test_label_vocabulary_roundtrip() -> None:
    vocab = LabelVocabulary()
    encoded = vocab.encode(["W", "N1", "IGNORE", "REM"])
    assert encoded.tolist() == [0, 1, -100, 4]
    assert vocab.decode(encoded) == ("W", "N1", "IGNORE", "REM")


def test_epoch_tensor_batch_shape_guard() -> None:
    with pytest.raises(ValueError, match="shape"):
        EpochTensorBatch(
            signals=np.zeros((2, 2, 10)),
            labels=np.zeros((3,), dtype=np.int64),
            onsets_sec=np.zeros((2,)),
            channel_names=("a", "b"),
            sfreq=100.0,
            epoch_duration_sec=30.0,
            subject_id="00",
            recording_id="1",
        )


def test_encoded_dataset_validates_feature_shape() -> None:
    meta = RepresentationMetadata(
        representation="raw",
        channel_names=("Fpz-Cz", "Pz-Oz"),
        sfreq=100.0,
        epoch_duration_sec=30.0,
        feature_shape=(2, 3000),
    )
    features = np.zeros((3, 2, 3000))
    labels = np.zeros((3,), dtype=np.int64)
    ds = EncodedDataset(
        features=features,
        labels=labels,
        metadata=meta,
        subject_id="00",
        recording_id="1",
    )
    assert ds.n_epochs == 3
    assert ds.representation == "raw"

    with pytest.raises(ValueError, match="feature_shape"):
        EncodedDataset(
            features=np.zeros((3, 2, 100)),
            labels=labels,
            metadata=meta,
            subject_id="00",
            recording_id="1",
        )


def test_raw_and_bandpower_describe_shapes() -> None:
    batch = _fake_batch()
    raw = RawSignalEncoder()
    raw_meta = raw.describe(
        n_channels=batch.n_channels,
        n_times=batch.n_times,
        sfreq=batch.sfreq,
        epoch_duration_sec=batch.epoch_duration_sec,
        channel_names=batch.channel_names,
    )
    assert raw_meta.representation == "raw"
    assert raw_meta.feature_shape == (2, 3000)

    bp = BandPowerEncoder()
    assert bp.n_bands == 5
    assert bp.n_features == 10
    assert len(DEFAULT_BANDS) == 5
    bp_meta = bp.describe(
        n_channels=batch.n_channels,
        n_times=batch.n_times,
        sfreq=batch.sfreq,
        epoch_duration_sec=batch.epoch_duration_sec,
        channel_names=batch.channel_names,
    )
    assert bp_meta.feature_shape == (2, 10)


def test_time_frequency_stft_describe_is_ncft() -> None:
    batch = _fake_batch()
    encoder = TimeFrequencyEncoder(backend=STFTBackend())
    meta = encoder.describe(
        n_channels=batch.n_channels,
        n_times=batch.n_times,
        sfreq=batch.sfreq,
        epoch_duration_sec=batch.epoch_duration_sec,
        channel_names=batch.channel_names,
    )
    assert meta.representation == "time_frequency"
    assert meta.algorithm == "stft"
    assert len(meta.feature_shape) == 3
    assert meta.feature_shape[0] == batch.n_channels
    assert meta.feature_shape[1:] == STFTBackend().output_hw(sfreq=100.0, n_times=3000)
    assert meta.freqs_hz is not None and len(meta.freqs_hz) == meta.feature_shape[1]
    assert meta.times_sec is not None and len(meta.times_sec) == meta.feature_shape[2]


def test_stft_encoder_encode_batch() -> None:
    batch = _fake_batch()
    encoded = TimeFrequencyEncoder(backend=STFTBackend()).encode(batch)
    n_freqs, n_frames = STFTBackend().output_hw(sfreq=100.0, n_times=3000)
    assert encoded.features.shape == (4, 2, n_freqs, n_frames)
    assert encoded.labels.tolist() == [0, 1, 2, -100]
    assert np.isfinite(encoded.features).all()


def test_cwt_still_unimplemented() -> None:
    from sleep_staging.representations import CWTBackend

    batch = _fake_batch()
    with pytest.raises(EncoderNotImplementedError):
        TimeFrequencyEncoder(backend=CWTBackend()).encode(batch)


def test_bandpower_encoder_encode_batch() -> None:
    batch = _fake_batch()
    encoded = BandPowerEncoder().encode(batch)
    assert encoded.features.shape == (4, 2, 10)
    assert encoded.labels.tolist() == [0, 1, 2, -100]
    assert np.isfinite(encoded.features).all()


def test_raw_encoder_encode_batch() -> None:
    batch = _fake_batch()
    encoded = RawSignalEncoder().encode(batch)
    assert encoded.features.shape == (4, 2, 3000)
    assert encoded.labels.tolist() == [0, 1, 2, -100]


def test_factory_builds_each_representation() -> None:
    assert isinstance(build_encoder(EncodingSettings(representation="raw")), RawSignalEncoder)
    assert isinstance(
        build_encoder(EncodingSettings(representation="bandpower")), BandPowerEncoder
    )
    tf = build_encoder(EncodingSettings(representation="time_frequency"))
    assert isinstance(tf, TimeFrequencyEncoder)
    assert tf.backend.name == "stft"


def test_encoded_dataset_collection_groups_by_subject() -> None:
    meta = RepresentationMetadata(
        representation="bandpower",
        channel_names=("Fpz-Cz",),
        sfreq=100.0,
        epoch_duration_sec=30.0,
        feature_shape=(1, 10),
        band_names=("delta", "theta", "alpha", "sigma", "beta"),
    )

    def make_item(subject: str, recording: str) -> EncodedDataset:
        return EncodedDataset(
            features=np.zeros((2, 1, 10)),
            labels=np.zeros((2,), dtype=np.int64),
            metadata=meta,
            subject_id=subject,
            recording_id=recording,
        )

    collection = EncodedDatasetCollection(
        items=(make_item("00", "1"), make_item("00", "2"), make_item("01", "1"))
    )
    grouped = collection.by_subject()
    assert len(grouped["00"]) == 2
    assert len(grouped["01"]) == 1
