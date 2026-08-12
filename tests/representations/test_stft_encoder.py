"""Unit tests for STFT TimeFrequencyEncoder."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sleep_staging.config import EncodingSettings, load_settings
from sleep_staging.representations import (
    CWTBackend,
    EncoderNotImplementedError,
    EncodingError,
    EpochTensorBatch,
    STFTBackend,
    TimeFrequencyEncoder,
    build_encoder,
)


def _sinusoid_batch(
    *,
    freqs_hz: list[float],
    n_epochs: int = 2,
    sfreq: float = 100.0,
    duration_sec: float = 30.0,
    amplitude: float = 1.0,
    labels: list[int] | None = None,
) -> EpochTensorBatch:
    n_times = int(round(duration_sec * sfreq))
    t = np.arange(n_times, dtype=np.float64) / sfreq
    n_channels = len(freqs_hz)
    signals = np.zeros((n_epochs, n_channels, n_times), dtype=np.float64)
    for ch, f0 in enumerate(freqs_hz):
        tone = amplitude * np.sin(2.0 * np.pi * f0 * t)
        for epoch in range(n_epochs):
            signals[epoch, ch] = tone

    if labels is None:
        label_arr = np.arange(n_epochs, dtype=np.int64) % 5
    else:
        label_arr = np.asarray(labels, dtype=np.int64)

    return EpochTensorBatch(
        signals=signals,
        labels=label_arr,
        onsets_sec=np.arange(n_epochs, dtype=np.float64) * duration_sec,
        channel_names=tuple(f"ch{i}" for i in range(n_channels)),
        sfreq=sfreq,
        epoch_duration_sec=duration_sec,
        subject_id="00",
        recording_id="1",
        ignore_index=-100,
    )


def _default_encoder() -> TimeFrequencyEncoder:
    return TimeFrequencyEncoder(backend=STFTBackend())


def test_stft_output_rank_and_shape() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0, 2.0], n_epochs=3)
    encoded = _default_encoder().encode(batch)
    backend = STFTBackend()
    n_freqs, n_frames = backend.output_hw(sfreq=100.0, n_times=3000)

    assert encoded.features.ndim == 4
    assert encoded.features.shape == (3, 2, n_freqs, n_frames)
    assert n_freqs == 75  # bins in [0.5, 30] at Δf = 100/256
    assert n_frames == 28  # 1 + (3000 - 256) // 100
    assert encoded.metadata.feature_shape == (2, n_freqs, n_frames)
    assert encoded.metadata.algorithm == "stft"


def test_stft_frequency_bin_metadata() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    encoded = _default_encoder().encode(batch)
    freqs = encoded.metadata.freqs_hz
    assert freqs is not None
    assert freqs[0] >= 0.5
    assert freqs[-1] <= 30.0
    assert len(freqs) == encoded.features.shape[2]
    # Uniform Δf = fs / n_fft
    df = np.diff(freqs)
    np.testing.assert_allclose(df, df[0], rtol=0, atol=1e-9)
    assert df[0] == pytest.approx(100.0 / 256.0)


def test_stft_time_frame_metadata() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    encoded = _default_encoder().encode(batch)
    times = encoded.metadata.times_sec
    assert times is not None
    assert len(times) == encoded.features.shape[3]
    # First frame center at win_length/2 / fs = 1.28 s; hop = 1.0 s
    assert times[0] == pytest.approx(256 / 2 / 100.0)
    if len(times) > 1:
        assert times[1] - times[0] == pytest.approx(1.0)


def test_alpha_sinusoid_peaks_at_expected_bin() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1, amplitude=5.0)
    encoded = _default_encoder().encode(batch)
    freqs = np.asarray(encoded.metadata.freqs_hz)
    # Mean log-power over time frames → frequency profile
    profile = encoded.features[0, 0].mean(axis=-1)
    peak_idx = int(np.argmax(profile))
    assert freqs[peak_idx] == pytest.approx(10.0, abs=100.0 / 256.0)


def test_multichannel_independent_frequency_peaks() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0, 2.0], n_epochs=1, amplitude=4.0)
    encoded = _default_encoder().encode(batch)
    freqs = np.asarray(encoded.metadata.freqs_hz)
    peak0 = freqs[int(np.argmax(encoded.features[0, 0].mean(axis=-1)))]
    peak1 = freqs[int(np.argmax(encoded.features[0, 1].mean(axis=-1)))]
    assert peak0 == pytest.approx(10.0, abs=100.0 / 256.0)
    assert peak1 == pytest.approx(2.0, abs=100.0 / 256.0)


def test_flat_signal_is_finite() -> None:
    batch = EpochTensorBatch(
        signals=np.zeros((2, 2, 3000), dtype=np.float64),
        labels=np.asarray([0, -100], dtype=np.int64),
        onsets_sec=np.asarray([0.0, 30.0]),
        channel_names=("Fpz-Cz", "Pz-Oz"),
        sfreq=100.0,
        epoch_duration_sec=30.0,
        subject_id="00",
        recording_id="1",
        ignore_index=-100,
    )
    encoded = _default_encoder().encode(batch)
    assert np.isfinite(encoded.features).all()


def test_ignore_labels_preserved() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=3, labels=[0, -100, 4])
    encoded = _default_encoder().encode(batch)
    assert encoded.labels.tolist() == [0, -100, 4]
    assert encoded.ignore_index == -100
    assert encoded.subject_id == "00"
    assert encoded.recording_id == "1"
    assert encoded.n_epochs == 3


def test_unexpected_sfreq_rejected() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], sfreq=200.0, duration_sec=30.0)
    with pytest.raises(EncodingError, match="expected sfreq"):
        _default_encoder().encode(batch)


def test_no_nan_inf_on_valid_input() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0, 15.0], n_epochs=2, amplitude=3.0)
    encoded = _default_encoder().encode(batch)
    assert np.isfinite(encoded.features).all()


def test_features_are_independent_copy() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    encoded = _default_encoder().encode(batch)
    original = float(encoded.features[0, 0, 0, 0])
    encoded.features[0, 0, 0, 0] = 12345.0
    again = _default_encoder().encode(batch)
    assert float(again.features[0, 0, 0, 0]) == pytest.approx(original)


def test_factory_builds_stft_encoder_from_config() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = load_settings(project_root / "configs" / "default.yaml")
    enc_settings = EncodingSettings(
        representation="time_frequency",
        time_frequency=settings.encodings.time_frequency,
    )
    encoder = build_encoder(enc_settings)
    assert isinstance(encoder, TimeFrequencyEncoder)
    assert isinstance(encoder.backend, STFTBackend)
    assert encoder.backend.n_fft == 256
    assert encoder.backend.hop_length == 100
    assert encoder.backend.log_scale is True

    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    encoded = encoder.encode(batch)
    assert encoded.features.ndim == 4
    assert encoded.representation == "time_frequency"


def test_cwt_backend_still_unimplemented() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    with pytest.raises(EncoderNotImplementedError):
        TimeFrequencyEncoder(backend=CWTBackend()).encode(batch)


def test_stft_deterministic() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    a = _default_encoder().encode(batch).features
    b = _default_encoder().encode(batch).features
    np.testing.assert_array_equal(a, b)
