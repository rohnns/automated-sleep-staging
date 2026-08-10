"""Interchangeable time-frequency backends (STFT now, CWT later).

``TimeFrequencyEncoder`` depends only on this interface. Swapping STFT ↔ CWT
requires no changes to models that consume ``EncodedDataset`` with
``representation=\"time_frequency\"`` and shape ``(N, C, F, T)``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import signal

from sleep_staging.encodings.exceptions import EncoderNotImplementedError, EncodingError

_SFREQ_TOL = 1e-6


class TimeFrequencyBackend(ABC):
    """Algorithm plugin that maps ``(N, C, T)`` → ``(N, C, F, T_frames)``."""

    name: str

    @abstractmethod
    def transform(
        self,
        signals: NDArray[np.floating],
        *,
        sfreq: float,
    ) -> NDArray[np.floating]:
        """Transform a batch of epochs → ``(N, C, F, T_frames)``."""

    @abstractmethod
    def frequency_axis(self, *, sfreq: float, n_times: int) -> tuple[float, ...]:
        """Center frequencies (Hz) for axis ``F``."""

    @abstractmethod
    def time_axis(self, *, sfreq: float, n_times: int) -> tuple[float, ...]:
        """Frame centers (seconds from epoch start) for axis ``T_frames``."""

    @abstractmethod
    def output_hw(self, *, sfreq: float, n_times: int) -> tuple[int, int]:
        """Return ``(F, T_frames)`` from config geometry (no DSP)."""

    def describe_params(self) -> dict[str, Any]:
        return {"algorithm": self.name}


class STFTBackend(TimeFrequencyBackend):
    """Short-time Fourier transform backend.

    Baseline Sleep-EDF defaults (100 Hz, 30 s epochs)::

        n_fft = win_length = 256   → Δf ≈ 0.391 Hz, 2.56 s window
        hop_length = 100          → 1.0 s frame stride (~61% overlap)
        fmin, fmax = 0.5, 30.0    → matches preprocessing band-pass
        power = 2, log_scale      → log-power spectrogram

    Frames are computed without boundary padding so the hop grid matches
    ``1 + (T - win_length) // hop_length`` for ``T >= win_length``.
    """

    name = "stft"

    def __init__(
        self,
        *,
        n_fft: int = 256,
        hop_length: int = 100,
        win_length: int | None = 256,
        window: str = "hann",
        fmin: float = 0.5,
        fmax: float = 30.0,
        power: int = 2,
        log_scale: bool = True,
        eps: float = 1e-10,
        expected_sfreq: float = 100.0,
        dtype: str = "float32",
    ) -> None:
        if n_fft <= 0:
            raise ValueError("n_fft must be positive")
        if hop_length <= 0:
            raise ValueError("hop_length must be positive")
        if fmin < 0 or fmax <= fmin:
            raise ValueError("require 0 <= fmin < fmax")
        if power < 1:
            raise ValueError("power must be >= 1")
        if eps <= 0:
            raise ValueError("eps must be positive")
        if expected_sfreq <= 0:
            raise ValueError("expected_sfreq must be positive")

        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.win_length = int(win_length) if win_length is not None else int(n_fft)
        self.window = window
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.power = int(power)
        self.log_scale = bool(log_scale)
        self.eps = float(eps)
        self.expected_sfreq = float(expected_sfreq)
        self.dtype = np.dtype(dtype)

        if self.win_length <= 0:
            raise ValueError("win_length must be positive")
        if self.win_length > self.n_fft:
            raise ValueError("win_length must be <= n_fft")
        if self.hop_length > self.win_length:
            raise ValueError("hop_length must be <= win_length")

    def transform(
        self,
        signals: NDArray[np.floating],
        *,
        sfreq: float,
    ) -> NDArray[np.floating]:
        if signals.ndim != 3:
            raise EncodingError("STFTBackend.transform expects signals shaped (N, C, T)")

        n_epochs, n_channels, n_times = signals.shape
        self._validate_geometry(sfreq=float(sfreq), n_times=n_times)

        data = np.asarray(signals, dtype=np.float64)
        if not np.isfinite(data).all():
            data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

        noverlap = self.win_length - self.hop_length
        flat = data.reshape(n_epochs * n_channels, n_times)
        freqs, _times, zxx = signal.stft(
            flat,
            fs=float(sfreq),
            window=self.window,
            nperseg=self.win_length,
            noverlap=noverlap,
            nfft=self.n_fft,
            detrend=False,
            return_onesided=True,
            boundary=None,
            padded=False,
            axis=-1,
        )

        # Magnitude / power / log-power spectrogram.
        spec = np.abs(zxx)
        if self.power != 1:
            spec = np.power(spec, self.power)
        if self.log_scale:
            spec = np.log(spec + self.eps)

        freq_mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        if not np.any(freq_mask):
            raise EncodingError(
                f"No STFT bins fall in [{self.fmin}, {self.fmax}] Hz "
                f"(df={float(sfreq) / self.n_fft:.4f} Hz)"
            )
        spec = spec[:, freq_mask, :]

        expected_f, expected_t = self.output_hw(sfreq=float(sfreq), n_times=n_times)
        if spec.shape[-2] != expected_f or spec.shape[-1] != expected_t:
            raise EncodingError(
                f"STFT geometry mismatch: got F,T={spec.shape[-2:]} "
                f"expected ({expected_f}, {expected_t})"
            )

        features = spec.reshape(n_epochs, n_channels, expected_f, expected_t)
        features = np.array(features, dtype=self.dtype, copy=True)
        if not np.isfinite(features).all():
            raise EncodingError("STFTBackend produced non-finite features")
        return features

    def frequency_axis(self, *, sfreq: float, n_times: int) -> tuple[float, ...]:
        _ = n_times
        freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / sfreq)
        mask = (freqs >= self.fmin) & (freqs <= self.fmax)
        return tuple(float(value) for value in freqs[mask])

    def time_axis(self, *, sfreq: float, n_times: int) -> tuple[float, ...]:
        _n_freqs, n_frames = self.output_hw(sfreq=sfreq, n_times=n_times)
        # Frame centers for unpadded segments of length win_length.
        centers = (
            np.arange(n_frames, dtype=np.float64) * self.hop_length + self.win_length / 2.0
        ) / sfreq
        return tuple(float(value) for value in centers)

    def output_hw(self, *, sfreq: float, n_times: int) -> tuple[int, int]:
        n_freqs = len(self.frequency_axis(sfreq=sfreq, n_times=n_times))
        if n_times < self.win_length:
            n_frames = 1 if n_times > 0 else 0
        else:
            n_frames = 1 + (n_times - self.win_length) // self.hop_length
        return n_freqs, int(n_frames)

    def describe_params(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "window": self.window,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "power": self.power,
            "log_scale": self.log_scale,
            "eps": self.eps,
            "expected_sfreq": self.expected_sfreq,
            "dtype": str(self.dtype),
            "boundary": None,
            "padded": False,
        }

    def _validate_geometry(self, *, sfreq: float, n_times: int) -> None:
        if abs(sfreq - self.expected_sfreq) > _SFREQ_TOL:
            raise EncodingError(
                f"STFTBackend expected sfreq={self.expected_sfreq} Hz, got {sfreq}. "
                "Reconfigure expected_sfreq / STFT parameters explicitly."
            )
        if n_times < self.win_length:
            raise EncodingError(
                f"Epoch length T={n_times} < win_length={self.win_length}"
            )
        nyquist = sfreq / 2.0
        if self.fmax > nyquist + _SFREQ_TOL:
            raise EncodingError(
                f"STFT fmax={self.fmax} Hz exceeds Nyquist {nyquist} Hz"
            )


class CWTBackend(TimeFrequencyBackend):
    """Continuous wavelet transform backend (transform not implemented).

    Shape geometry uses ``n_freqs`` scales spanning ``[fmin, fmax]`` and keeps
    full temporal resolution (``T_frames == n_times``) as the default contract.
    """

    name = "cwt"

    def __init__(
        self,
        *,
        wavelet: str = "morlet",
        fmin: float = 0.5,
        fmax: float = 30.0,
        n_freqs: int = 30,
        output: str = "magnitude",
        log_scale: bool = True,
    ) -> None:
        if n_freqs <= 0:
            raise ValueError("n_freqs must be positive")
        if fmin < 0 or fmax <= fmin:
            raise ValueError("require 0 <= fmin < fmax")
        self.wavelet = wavelet
        self.fmin = fmin
        self.fmax = fmax
        self.n_freqs = n_freqs
        self.output = output
        self.log_scale = log_scale

    def transform(
        self,
        signals: NDArray[np.floating],
        *,
        sfreq: float,
    ) -> NDArray[np.floating]:
        raise EncoderNotImplementedError("CWTBackend.transform is not implemented yet")

    def frequency_axis(self, *, sfreq: float, n_times: int) -> tuple[float, ...]:
        _ = (sfreq, n_times)
        freqs = np.geomspace(self.fmin, self.fmax, num=self.n_freqs)
        return tuple(float(value) for value in freqs)

    def time_axis(self, *, sfreq: float, n_times: int) -> tuple[float, ...]:
        times = np.arange(n_times, dtype=np.float64) / sfreq
        return tuple(float(value) for value in times)

    def output_hw(self, *, sfreq: float, n_times: int) -> tuple[int, int]:
        _ = sfreq
        return self.n_freqs, int(n_times)

    def describe_params(self) -> dict[str, Any]:
        return {
            "algorithm": self.name,
            "wavelet": self.wavelet,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "n_freqs": self.n_freqs,
            "output": self.output,
            "log_scale": self.log_scale,
        }
