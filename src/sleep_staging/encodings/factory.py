"""Factory: build encoders from :class:`EncodingSettings`."""

from __future__ import annotations

from sleep_staging.config.settings import (
    BandPowerEncodingSettings,
    EncodingSettings,
    STFTEncodingSettings,
    CWTEncodingSettings,
)
from sleep_staging.encodings.backends import CWTBackend, STFTBackend, TimeFrequencyBackend
from sleep_staging.encodings.base import BaseEncoder
from sleep_staging.encodings.encoders import (
    DEFAULT_BANDS,
    BandPowerEncoder,
    RawSignalEncoder,
    TimeFrequencyEncoder,
)
from sleep_staging.encodings.exceptions import EncodingError


def build_time_frequency_backend(settings: EncodingSettings) -> TimeFrequencyBackend:
    """Construct the TF algorithm plugin selected by config."""
    method = settings.time_frequency.method
    if method == "stft":
        stft = settings.time_frequency.stft
        return STFTBackend(
            n_fft=stft.n_fft,
            hop_length=stft.hop_length,
            win_length=stft.win_length,
            window=stft.window,
            fmin=stft.fmin,
            fmax=stft.fmax,
            power=stft.power,
            log_scale=stft.log_scale,
            eps=stft.eps,
            expected_sfreq=stft.expected_sfreq,
            dtype=stft.dtype,
        )
    if method == "cwt":
        cwt = settings.time_frequency.cwt
        return CWTBackend(
            wavelet=cwt.wavelet,
            fmin=cwt.fmin,
            fmax=cwt.fmax,
            n_freqs=cwt.n_freqs,
            output=cwt.output,
            log_scale=cwt.log_scale,
        )
    raise EncodingError(f"Unknown time_frequency.method: {method!r}")


def build_encoder(settings: EncodingSettings) -> BaseEncoder:
    """Dispatch ``settings.representation`` to the matching encoder."""
    representation = settings.representation
    if representation == "raw":
        from sleep_staging.encodings.types import LabelVocabulary

        vocabulary = LabelVocabulary(
            ignore_label=settings.ignore_label,
            ignore_index=settings.ignore_index,
        )
        return RawSignalEncoder(dtype=settings.raw.dtype, vocabulary=vocabulary)
    if representation == "bandpower":
        bp = settings.bandpower
        return BandPowerEncoder(
            bands=_bands_from_settings(bp),
            method=bp.method,
            include_log_absolute=bp.include_log_absolute,
            include_relative=bp.include_relative,
            include_ratios=bp.include_ratios,
            eps=bp.eps,
            expected_sfreq=bp.expected_sfreq,
            welch=bp.welch,
            ratios=bp.ratios,
        )
    if representation == "time_frequency":
        return TimeFrequencyEncoder(backend=build_time_frequency_backend(settings))
    raise EncodingError(f"Unknown encodings.representation: {representation!r}")


def _bands_from_settings(
    settings: BandPowerEncodingSettings,
) -> tuple[tuple[str, float, float], ...]:
    if not settings.bands:
        return DEFAULT_BANDS
    return tuple((name, float(lo), float(hi)) for name, lo, hi in settings.bands)


# Re-export settings types used by factory callers / type checkers.
__all__ = [
    "CWTEncodingSettings",
    "STFTEncodingSettings",
    "build_encoder",
    "build_time_frequency_backend",
]
