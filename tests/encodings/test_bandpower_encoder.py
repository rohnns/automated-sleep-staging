"""Unit tests for BandPowerEncoder (Welch band-power features)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sleep_staging.config import EncodingSettings, load_settings
from sleep_staging.encodings import (
    BandPowerEncoder,
    EncodingError,
    EpochTensorBatch,
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
    """Build epochs where channel ``i`` is a pure tone at ``freqs_hz[i]``."""
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


def test_output_shape_is_n_c_10() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0, 2.0], n_epochs=3)
    encoded = BandPowerEncoder().encode(batch)
    assert encoded.features.shape == (3, 2, 10)
    assert encoded.n_epochs == 3
    assert encoded.metadata.feature_shape == (2, 10)
    assert encoded.metadata.band_names == ("delta", "theta", "alpha", "sigma", "beta")
    assert len(encoded.metadata.extras["feature_names"]) == 10


def test_n_features_matches_bands_times_two() -> None:
    encoder = BandPowerEncoder()
    assert encoder.n_bands == 5
    assert encoder.n_features == 10
    assert encoder.feature_names == (
        "log_delta",
        "log_theta",
        "log_alpha",
        "log_sigma",
        "log_beta",
        "rel_delta",
        "rel_theta",
        "rel_alpha",
        "rel_sigma",
        "rel_beta",
    )


def test_alpha_sinusoid_dominates_alpha_band() -> None:
    # 10 Hz sits clearly inside alpha (8–12 Hz).
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1, amplitude=5.0)
    encoded = BandPowerEncoder().encode(batch)
    abs_powers_proxy = encoded.features[0, 0, :5]  # log-abs
    rel = encoded.features[0, 0, 5:]
    alpha_idx = 2
    assert int(np.argmax(abs_powers_proxy)) == alpha_idx
    assert int(np.argmax(rel)) == alpha_idx
    assert float(rel[alpha_idx]) > 0.5


def test_delta_sinusoid_dominates_delta_band() -> None:
    batch = _sinusoid_batch(freqs_hz=[2.0], n_epochs=1, amplitude=5.0)
    encoded = BandPowerEncoder().encode(batch)
    rel = encoded.features[0, 0, 5:]
    assert int(np.argmax(rel)) == 0  # delta
    assert float(rel[0]) > 0.5


def test_relative_powers_sum_to_one() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0, 2.0], n_epochs=2, amplitude=3.0)
    encoded = BandPowerEncoder().encode(batch)
    rel = encoded.features[:, :, 5:]
    sums = rel.sum(axis=-1)
    np.testing.assert_allclose(sums, 1.0, rtol=1e-5, atol=1e-5)


def test_log_absolute_is_finite() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=2)
    encoded = BandPowerEncoder().encode(batch)
    log_abs = encoded.features[:, :, :5]
    assert np.isfinite(log_abs).all()


def test_flat_signal_has_no_nan_inf() -> None:
    n_epochs, n_channels, n_times = 2, 2, 3000
    batch = EpochTensorBatch(
        signals=np.zeros((n_epochs, n_channels, n_times), dtype=np.float64),
        labels=np.asarray([0, -100], dtype=np.int64),
        onsets_sec=np.asarray([0.0, 30.0]),
        channel_names=("Fpz-Cz", "Pz-Oz"),
        sfreq=100.0,
        epoch_duration_sec=30.0,
        subject_id="00",
        recording_id="1",
        ignore_index=-100,
    )
    encoded = BandPowerEncoder().encode(batch)
    assert np.isfinite(encoded.features).all()
    # Near-zero total power → relative powers defined as zeros.
    rel = encoded.features[:, :, 5:]
    np.testing.assert_allclose(rel, 0.0, atol=1e-7)


def test_ignore_labels_preserved() -> None:
    batch = _sinusoid_batch(
        freqs_hz=[10.0],
        n_epochs=3,
        labels=[0, -100, 4],
    )
    encoded = BandPowerEncoder().encode(batch)
    assert encoded.labels.tolist() == [0, -100, 4]
    assert encoded.ignore_index == -100
    assert encoded.n_epochs == 3


def test_multichannel_independent_peaks() -> None:
    # ch0 = 10 Hz (alpha), ch1 = 2 Hz (delta)
    batch = _sinusoid_batch(freqs_hz=[10.0, 2.0], n_epochs=1, amplitude=4.0)
    encoded = BandPowerEncoder().encode(batch)
    rel0 = encoded.features[0, 0, 5:]
    rel1 = encoded.features[0, 1, 5:]
    assert int(np.argmax(rel0)) == 2  # alpha
    assert int(np.argmax(rel1)) == 0  # delta


def test_unexpected_sfreq_is_rejected() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], sfreq=200.0, duration_sec=30.0)
    # 200 Hz → T=6000; encoder still rejects mismatched expected_sfreq.
    with pytest.raises(EncodingError, match="expected sfreq"):
        BandPowerEncoder(expected_sfreq=100.0).encode(batch)


def test_features_are_independent_copy() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    encoded = BandPowerEncoder().encode(batch)
    original = float(encoded.features[0, 0, 0])
    encoded.features[0, 0, 0] = 12345.0
    again = BandPowerEncoder().encode(batch)
    assert float(again.features[0, 0, 0]) == pytest.approx(original)
    assert float(again.features[0, 0, 0]) != 12345.0


def test_subject_recording_metadata_preserved() -> None:
    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    encoded = BandPowerEncoder().encode(batch)
    assert encoded.subject_id == "00"
    assert encoded.recording_id == "1"
    assert encoded.channel_names == ("ch0",)
    assert encoded.onsets_sec is not None
    assert encoded.onsets_sec.tolist() == [0.0]


def test_factory_builds_configured_bandpower() -> None:
    project_root = Path(__file__).resolve().parents[2]
    settings = load_settings(project_root / "configs" / "default.yaml")
    settings_bp = EncodingSettings(
        representation="bandpower",
        bandpower=settings.encodings.bandpower,
    )
    encoder = build_encoder(settings_bp)
    assert isinstance(encoder, BandPowerEncoder)
    assert encoder.n_bands == 5
    assert encoder.n_features == 10
    assert encoder.welch.nperseg == 400
    assert encoder.welch.window == "hamming"

    batch = _sinusoid_batch(freqs_hz=[10.0], n_epochs=1)
    encoded = encoder.encode(batch)
    assert encoded.features.shape == (1, 1, 10)


def test_include_ratios_rejected_until_implemented() -> None:
    with pytest.raises(EncodingError, match="ratios"):
        BandPowerEncoder(include_ratios=True)


def test_closed_band_masks_share_boundary_bins_without_area_gaps() -> None:
    """Regression: adjacent bands must use closed [lo, hi] masks.

    Half-open ``[lo, hi)`` omits the trapezoid segment that ends on the shared
    edge (e.g. [3.75, 4.0] Hz), creating an integration gap. Closed masks share
    the boundary sample; consecutive trapezoids meet at that abscissa and do
    not overlap in area.
    """
    from sleep_staging.encodings.encoders import DEFAULT_BANDS, _frequency_mask

    # Welch geometry for Sleep-EDF defaults: df = 100/400 = 0.25 Hz.
    freqs = np.arange(0.0, 50.0 + 1e-12, 0.25, dtype=np.float64)
    psd = np.ones_like(freqs)  # flat density → easy analytic trapz checks

    shared_edges = (4.0, 8.0, 12.0, 16.0)
    for edge in shared_edges:
        assert edge in freqs
        left = next(band for band in DEFAULT_BANDS if band[2] == edge)
        right = next(band for band in DEFAULT_BANDS if band[1] == edge)
        assert bool(_frequency_mask(freqs, left[1], left[2])[freqs == edge][0])
        assert bool(_frequency_mask(freqs, right[1], right[2])[freqs == edge][0])

    band_powers = []
    for _name, lo, hi in DEFAULT_BANDS:
        mask = _frequency_mask(freqs, lo, hi)
        band_powers.append(float(np.trapezoid(psd[mask], freqs[mask])))
    closed_sum = sum(band_powers)

    full_mask = (freqs >= 0.5) & (freqs <= 30.0)
    full_integral = float(np.trapezoid(psd[full_mask], freqs[full_mask]))
    assert closed_sum == pytest.approx(full_integral, rel=0.0, abs=1e-12)

    # Half-open masks (old behavior) miss the segment abutting each shared edge.
    half_open_sum = 0.0
    for idx, (_name, lo, hi) in enumerate(DEFAULT_BANDS):
        if idx == len(DEFAULT_BANDS) - 1:
            mask = (freqs >= lo) & (freqs <= hi)
        else:
            mask = (freqs >= lo) & (freqs < hi)
        half_open_sum += float(np.trapezoid(psd[mask], freqs[mask]))
    assert half_open_sum < full_integral
    # Four interior edges × one 0.25 Hz segment of height 1 → gap of 1.0 total.
    assert full_integral - half_open_sum == pytest.approx(1.0, abs=1e-12)
