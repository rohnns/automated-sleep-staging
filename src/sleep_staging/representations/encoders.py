"""Concrete encoder implementations."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.config.settings import WelchPSDSettings
from sleep_staging.representations.base import BaseEncoder, build_encoded_dataset
from sleep_staging.representations.backends import TimeFrequencyBackend
from sleep_staging.representations.exceptions import EncoderNotImplementedError, EncodingError
from sleep_staging.representations.types import (
    EncodedDataset,
    EpochTensorBatch,
    LabelVocabulary,
    RepresentationMetadata,
)
from sleep_staging.preprocessing.types import PreprocessedRecording

logger = get_logger(__name__)

# Default Sleep-EDF band-power set (5 bands). Sigma is kept separate from beta
# because sleep spindles (12–16 Hz) are clinically distinct from broadband beta.
# Gamma is omitted: preprocessing band-passes at ~0.5–30 Hz.
DEFAULT_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)

DEFAULT_RAW_VOCABULARY = LabelVocabulary(ignore_index=-100)

_SFREQ_TOL = 1e-6


class RawSignalEncoder(BaseEncoder):
    """Pass-through multichannel waveforms.

    Accepts either an :class:`EpochTensorBatch` via :meth:`encode` or a
    :class:`~sleep_staging.preprocessing.types.PreprocessedRecording` via
    :meth:`encode_recording` / ``__call__``.

    Output shape: ``(N, C, T)`` with ``T = 3000`` for 30 s @ 100 Hz.
    Labels: W=0, N1=1, N2=2, N3=3, REM=4, IGNORE=-100.
    """

    name = "raw_signal"
    representation = "raw"

    def __init__(
        self,
        *,
        dtype: str = "float32",
        vocabulary: LabelVocabulary | None = None,
    ) -> None:
        self.dtype = np.dtype(dtype)
        self.vocabulary = vocabulary or DEFAULT_RAW_VOCABULARY

    def encode(self, batch: EpochTensorBatch) -> EncodedDataset:
        expected_times = int(round(batch.epoch_duration_sec * batch.sfreq))
        if batch.n_times != expected_times:
            raise ValueError(
                f"RawSignalEncoder expected T={expected_times} "
                f"(duration={batch.epoch_duration_sec}s × sfreq={batch.sfreq}), "
                f"got T={batch.n_times}"
            )

        features = np.array(batch.signals, dtype=self.dtype, copy=True)
        metadata = self.describe(
            n_channels=batch.n_channels,
            n_times=batch.n_times,
            sfreq=batch.sfreq,
            epoch_duration_sec=batch.epoch_duration_sec,
            channel_names=batch.channel_names,
        )
        encoded = build_encoded_dataset(features=features, batch=batch, metadata=metadata)
        if batch.ignore_index != self.vocabulary.ignore_index:
            logger.warning(
                "Batch ignore_index=%s differs from encoder vocabulary ignore_index=%s",
                batch.ignore_index,
                self.vocabulary.ignore_index,
            )
        logger.info(
            "RawSignalEncoder produced %s for subject=%s recording=%s",
            features.shape,
            encoded.subject_id,
            encoded.recording_id,
        )
        return encoded

    def encode_recording(
        self,
        preprocessed: PreprocessedRecording,
        *,
        vocabulary: LabelVocabulary | None = None,
    ) -> EncodedDataset:
        """Slice ``preprocessed`` with this encoder's vocabulary, then encode."""
        return super().encode_recording(
            preprocessed,
            vocabulary=vocabulary or self.vocabulary,
        )

    def describe(
        self,
        *,
        n_channels: int,
        n_times: int,
        sfreq: float,
        epoch_duration_sec: float,
        channel_names: tuple[str, ...],
    ) -> RepresentationMetadata:
        return RepresentationMetadata(
            representation="raw",
            channel_names=channel_names,
            sfreq=sfreq,
            epoch_duration_sec=epoch_duration_sec,
            feature_shape=(n_channels, n_times),
            algorithm=None,
            extras={
                "dtype": str(self.dtype),
                "layout": "NCT",
                "ignore_index": self.vocabulary.ignore_index,
                "label_stages": self.vocabulary.stages,
            },
        )


class BandPowerEncoder(BaseEncoder):
    """Per-channel spectral band-power features via Welch PSD.

    Default output shape: ``(N, C, 10)`` — five bands
    (delta, theta, alpha, sigma, beta) × (log-absolute + relative).

    Feature layout along the last axis
    ----------------------------------
    ``[0 : B)``   log(band_power + eps) for each band
    ``[B : 2B)``  relative powers (band / sum of selected bands)

    Epochs are processed independently (no cross-epoch PSD pooling).
    IGNORE labels are preserved; epochs are never dropped.
    """

    name = "bandpower"
    representation = "bandpower"

    def __init__(
        self,
        *,
        bands: tuple[tuple[str, float, float], ...] = DEFAULT_BANDS,
        method: str = "welch",
        include_log_absolute: bool = True,
        include_relative: bool = True,
        include_ratios: bool = False,
        eps: float = 1e-10,
        expected_sfreq: float = 100.0,
        welch: WelchPSDSettings | None = None,
        dtype: str = "float32",
        ratios: tuple[tuple[str, str, str], ...] = (),
    ) -> None:
        if len(bands) == 0:
            raise ValueError("bands must be non-empty")
        for name, lo, hi in bands:
            if not (0.0 <= lo < hi):
                raise ValueError(f"Invalid band {name!r}: require 0 <= fmin < fmax")
        if method != "welch":
            raise ValueError(
                f"BandPowerEncoder only supports method='welch' (got {method!r})"
            )
        if not include_log_absolute and not include_relative:
            raise ValueError("At least one of include_log_absolute / include_relative")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if expected_sfreq <= 0:
            raise ValueError("expected_sfreq must be positive")
        if include_ratios:
            raise EncodingError(
                "Spectral ratios are not implemented yet; set include_ratios=False"
            )

        self.bands = bands
        self.method = method
        self.include_log_absolute = include_log_absolute
        self.include_relative = include_relative
        self.include_ratios = include_ratios
        self.eps = float(eps)
        self.expected_sfreq = float(expected_sfreq)
        self.welch = welch if welch is not None else WelchPSDSettings()
        self.dtype = np.dtype(dtype)
        self.ratios = ratios
        self._validate_welch_settings()

    def _validate_welch_settings(self) -> None:
        w = self.welch
        if w.nperseg <= 0 or w.nfft <= 0:
            raise ValueError("welch.nperseg and welch.nfft must be positive")
        if w.noverlap < 0 or w.noverlap >= w.nperseg:
            raise ValueError("welch.noverlap must satisfy 0 <= noverlap < nperseg")
        if w.nfft < w.nperseg:
            raise ValueError("welch.nfft must be >= nperseg")
        if w.average not in {"mean", "median"}:
            raise ValueError("welch.average must be 'mean' or 'median'")
        if w.scaling not in {"density", "spectrum"}:
            raise ValueError("welch.scaling must be 'density' or 'spectrum'")

    @property
    def band_names(self) -> tuple[str, ...]:
        return tuple(name for name, _lo, _hi in self.bands)

    @property
    def n_bands(self) -> int:
        return len(self.bands)

    @property
    def n_features(self) -> int:
        n = 0
        if self.include_log_absolute:
            n += self.n_bands
        if self.include_relative:
            n += self.n_bands
        return n

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.include_log_absolute:
            names.extend(f"log_{name}" for name in self.band_names)
        if self.include_relative:
            names.extend(f"rel_{name}" for name in self.band_names)
        return tuple(names)

    def encode(self, batch: EpochTensorBatch) -> EncodedDataset:
        self._validate_batch(batch)

        signals = np.asarray(batch.signals, dtype=np.float64)
        if not np.isfinite(signals).all():
            logger.warning(
                "BandPowerEncoder: non-finite samples replaced with 0 "
                "(subject=%s recording=%s)",
                batch.subject_id,
                batch.recording_id,
            )
            signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        band_powers = self._band_powers(signals, sfreq=float(batch.sfreq))
        features = self._assemble_features(band_powers)
        features = np.array(features, dtype=self.dtype, copy=True)

        if not np.isfinite(features).all():
            raise EncodingError("BandPowerEncoder produced non-finite features")

        metadata = self.describe(
            n_channels=batch.n_channels,
            n_times=batch.n_times,
            sfreq=batch.sfreq,
            epoch_duration_sec=batch.epoch_duration_sec,
            channel_names=batch.channel_names,
        )
        encoded = build_encoded_dataset(features=features, batch=batch, metadata=metadata)
        logger.info(
            "BandPowerEncoder produced %s for subject=%s recording=%s",
            features.shape,
            encoded.subject_id,
            encoded.recording_id,
        )
        return encoded

    def describe(
        self,
        *,
        n_channels: int,
        n_times: int,
        sfreq: float,
        epoch_duration_sec: float,
        channel_names: tuple[str, ...],
    ) -> RepresentationMetadata:
        _ = n_times
        return RepresentationMetadata(
            representation="bandpower",
            channel_names=channel_names,
            sfreq=sfreq,
            epoch_duration_sec=epoch_duration_sec,
            feature_shape=(n_channels, self.n_features),
            algorithm=self.method,
            band_names=self.band_names,
            extras={
                "layout": "NCF",
                "feature_names": self.feature_names,
                "include_log_absolute": self.include_log_absolute,
                "include_relative": self.include_relative,
                "include_ratios": self.include_ratios,
                "eps": self.eps,
                "expected_sfreq": self.expected_sfreq,
                "bands_hz": tuple((name, lo, hi) for name, lo, hi in self.bands),
                "welch": {
                    "nperseg": self.welch.nperseg,
                    "noverlap": self.welch.noverlap,
                    "nfft": self.welch.nfft,
                    "window": self.welch.window,
                    "average": self.welch.average,
                    "detrend": self.welch.detrend,
                    "scaling": self.welch.scaling,
                },
            },
        )

    def _validate_batch(self, batch: EpochTensorBatch) -> None:
        if abs(float(batch.sfreq) - self.expected_sfreq) > _SFREQ_TOL:
            raise EncodingError(
                f"BandPowerEncoder expected sfreq={self.expected_sfreq} Hz "
                f"(Welch/band config), got {batch.sfreq}. "
                "Reconfigure expected_sfreq / Welch parameters explicitly; "
                "do not silently reuse 100 Hz defaults at another rate."
            )

        expected_times = int(round(batch.epoch_duration_sec * batch.sfreq))
        if batch.n_times != expected_times:
            raise EncodingError(
                f"BandPowerEncoder expected T={expected_times} "
                f"(duration={batch.epoch_duration_sec}s × sfreq={batch.sfreq}), "
                f"got T={batch.n_times}"
            )

        if batch.n_times < self.welch.nperseg:
            raise EncodingError(
                f"Epoch length T={batch.n_times} < welch.nperseg={self.welch.nperseg}"
            )

        nyquist = float(batch.sfreq) / 2.0
        for name, lo, hi in self.bands:
            if hi > nyquist + _SFREQ_TOL:
                raise EncodingError(
                    f"Band {name!r} upper edge {hi} Hz exceeds Nyquist {nyquist} Hz"
                )

    def _band_powers(
        self,
        signals: NDArray[np.floating],
        *,
        sfreq: float,
    ) -> NDArray[np.floating]:
        """Return absolute band powers with shape ``(N, C, B)``.

        Each epoch×channel is transformed independently via Welch PSD.

        Bands are closed intervals ``[lo, hi]`` so adjacent bands share their
        boundary frequency sample (e.g. 4 Hz belongs to both delta and theta).
        Sharing a single abscissa point does not double-count trapezoidal area:
        delta's last segment is ``[f_{k-1}, 4]`` and theta's first is ``[4, f_{k+1}]``.
        Half-open masks would omit the segment abutting the shared edge.
        """
        w = self.welch
        freqs, psd = signal.welch(
            signals,
            fs=sfreq,
            window=w.window,
            nperseg=w.nperseg,
            noverlap=w.noverlap,
            nfft=w.nfft,
            detrend=w.detrend,
            return_onesided=True,
            scaling=w.scaling,
            average=w.average,
            axis=-1,
        )
        # Numerical floor: density estimates should be non-negative.
        psd = np.maximum(np.nan_to_num(psd, nan=0.0, posinf=0.0, neginf=0.0), 0.0)

        n_epochs, n_channels, _ = signals.shape
        powers = np.zeros((n_epochs, n_channels, self.n_bands), dtype=np.float64)
        for band_idx, (_name, lo, hi) in enumerate(self.bands):
            mask = _frequency_mask(freqs, lo, hi)
            if not np.any(mask):
                raise EncodingError(
                    f"No Welch frequency bins fall in band [{lo}, {hi}]; "
                    f"check nfft/sfreq (df={sfreq / w.nfft:.4f} Hz)"
                )
            # Integrate PSD over frequency → band power (V^2 if density scaling).
            powers[:, :, band_idx] = np.trapezoid(psd[:, :, mask], freqs[mask], axis=-1)

        return np.maximum(powers, 0.0)

    def _assemble_features(self, band_powers: NDArray[np.floating]) -> NDArray[np.floating]:
        """Build ``(N, C, F)`` from absolute band powers."""
        parts: list[NDArray[np.floating]] = []
        if self.include_log_absolute:
            parts.append(np.log(band_powers + self.eps))
        if self.include_relative:
            total = band_powers.sum(axis=-1, keepdims=True)
            relative = np.divide(
                band_powers,
                total,
                out=np.zeros_like(band_powers),
                where=total > self.eps,
            )
            parts.append(relative)
        return np.concatenate(parts, axis=-1)


def _frequency_mask(
    freqs: NDArray[np.floating],
    lo: float,
    hi: float,
) -> NDArray[np.bool_]:
    """Closed interval mask ``[lo, hi]`` over Welch frequency bins.

    Adjacent bands intentionally share the boundary sample. Trapezoidal
    integration only accumulates area between consecutive selected bins, so a
    shared endpoint contributes zero measure and does not double-count power.
    """
    return (freqs >= lo) & (freqs <= hi)


class TimeFrequencyEncoder(BaseEncoder):
    """Time-frequency image encoder with a swappable backend (STFT / CWT).

    Output shape: ``(N, C, F, T)`` regardless of backend. Downstream code keys
    off ``representation="time_frequency"`` and ``metadata.algorithm``.

    Epochs are transformed independently. IGNORE labels are preserved; epochs
    are never dropped. Channel selection is inherited from preprocessing
    (same contract as :class:`RawSignalEncoder` / :class:`BandPowerEncoder`).
    """

    name = "time_frequency"
    representation = "time_frequency"

    def __init__(self, *, backend: TimeFrequencyBackend) -> None:
        self.backend = backend

    def encode(self, batch: EpochTensorBatch) -> EncodedDataset:
        expected_times = int(round(batch.epoch_duration_sec * batch.sfreq))
        if batch.n_times != expected_times:
            raise EncodingError(
                f"TimeFrequencyEncoder expected T={expected_times} "
                f"(duration={batch.epoch_duration_sec}s × sfreq={batch.sfreq}), "
                f"got T={batch.n_times}"
            )

        features = self.backend.transform(batch.signals, sfreq=float(batch.sfreq))
        features = np.array(features, dtype=np.asarray(features).dtype, copy=True)

        metadata = self.describe(
            n_channels=batch.n_channels,
            n_times=batch.n_times,
            sfreq=batch.sfreq,
            epoch_duration_sec=batch.epoch_duration_sec,
            channel_names=batch.channel_names,
        )
        if tuple(features.shape[1:]) != metadata.feature_shape:
            raise EncodingError(
                f"STFT features tail {tuple(features.shape[1:])} != "
                f"metadata.feature_shape {metadata.feature_shape}"
            )
        if features.shape[0] != batch.n_epochs:
            raise EncodingError("STFT features N axis does not match epoch count")

        encoded = build_encoded_dataset(features=features, batch=batch, metadata=metadata)
        logger.info(
            "TimeFrequencyEncoder(%s) produced %s for subject=%s recording=%s",
            self.backend.name,
            features.shape,
            encoded.subject_id,
            encoded.recording_id,
        )
        return encoded

    def describe(
        self,
        *,
        n_channels: int,
        n_times: int,
        sfreq: float,
        epoch_duration_sec: float,
        channel_names: tuple[str, ...],
    ) -> RepresentationMetadata:
        n_freqs, n_frames = self.backend.output_hw(sfreq=sfreq, n_times=n_times)
        return RepresentationMetadata(
            representation="time_frequency",
            channel_names=channel_names,
            sfreq=sfreq,
            epoch_duration_sec=epoch_duration_sec,
            feature_shape=(n_channels, n_freqs, n_frames),
            algorithm=self.backend.name,
            freqs_hz=self.backend.frequency_axis(sfreq=sfreq, n_times=n_times),
            times_sec=self.backend.time_axis(sfreq=sfreq, n_times=n_times),
            extras={"layout": "NCFT", **self.backend.describe_params()},
        )
