"""Configurable signal filtering via MNE.

Defaults follow common Sleep-EDF staging practice on 100 Hz recordings:
band-pass about 0.5–30 Hz and an optional line-noise notch. Notch frequencies
at or above Nyquist are skipped automatically (e.g. 50 Hz on 100 Hz data).
"""

from __future__ import annotations

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)


class SignalFilter(Transform):
    """Apply band-pass and optional notch filtering to ``state.raw``.

    Parameters
    ----------
    l_freq, h_freq:
        Band-pass edges in Hz. Either may be ``None`` to skip that side.
    notch_freqs:
        Frequencies for notch filtering (e.g. ``(50.0,)``). Values at or
        above the Nyquist frequency are skipped with a warning.
    picks:
        Channel selection for filtering. ``None`` filters all channels.
    pad:
        Edge padding method forwarded to MNE.
    verbose:
        MNE verbosity.
    """

    name = "signal_filter"

    def __init__(
        self,
        *,
        l_freq: float | None = 0.5,
        h_freq: float | None = 30.0,
        notch_freqs: list[float] | tuple[float, ...] | None = (50.0,),
        picks: str | list[str] | None = None,
        pad: str = "reflect_limited",
        verbose: str | bool | None = "ERROR",
    ) -> None:
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = tuple(notch_freqs) if notch_freqs is not None else ()
        self.picks = picks
        self.pad = pad
        self.verbose = verbose

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        if not state.raw.ch_names:
            raise TransformError("Cannot filter a recording with no channels")
        if not state.raw.preload:
            state.raw.load_data()

        nyquist = state.sampling_frequency / 2.0
        applied_notch = _usable_notch_freqs(self.notch_freqs, nyquist=nyquist)
        skipped = [freq for freq in self.notch_freqs if freq not in applied_notch]
        if skipped:
            logger.warning(
                "Skipping notch freq(s) >= Nyquist (%.1f Hz): %s",
                nyquist,
                skipped,
            )

        if applied_notch:
            state.raw.notch_filter(
                freqs=list(applied_notch),
                picks=self.picks,
                verbose=self.verbose,
            )

        if self.l_freq is not None or self.h_freq is not None:
            h_freq = self.h_freq
            if h_freq is not None and h_freq >= nyquist:
                # Keep a small margin below Nyquist for FIR design.
                h_freq = max(0.0, nyquist - 0.5)
                logger.warning(
                    "Clamping h_freq from %s to %.1f Hz (Nyquist=%.1f)",
                    self.h_freq,
                    h_freq,
                    nyquist,
                )
            state.raw.filter(
                l_freq=self.l_freq,
                h_freq=h_freq,
                picks=self.picks,
                pad=self.pad,
                verbose=self.verbose,
            )

        state.extras["filter"] = {
            "l_freq": self.l_freq,
            "h_freq": self.h_freq,
            "notch_freqs": list(self.notch_freqs),
            "applied_notch_freqs": list(applied_notch),
        }
        logger.info(
            "Filtered signals (bandpass=%s–%s Hz, notch=%s)",
            self.l_freq,
            self.h_freq,
            list(applied_notch),
        )
        return state


def _usable_notch_freqs(
    freqs: tuple[float, ...],
    *,
    nyquist: float,
) -> tuple[float, ...]:
    """Keep notch frequencies strictly below Nyquist."""
    return tuple(freq for freq in freqs if 0.0 < float(freq) < nyquist)
