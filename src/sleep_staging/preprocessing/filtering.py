"""Configurable, channel-type-aware signal filtering via MNE.

Applies notch filtering (global) then per-channel-type band-pass filters
(EEG / EOG / EMG) using MNE. Cutoffs are configuration-driven engineering
baselines for this project, not universal Sleep-EDF or AASM requirements.
Published sleep-staging pipelines use different EOG passbands (e.g. ~0.5–10 Hz
or up to ~30 Hz); the current project baseline is 0.5–15 Hz and is overridable.

Design principles:
- Filtering runs on the continuous recording before epoch slicing.
- Per-channel-type band-passes (EEG/EOG/EMG); unknown/auxiliary types are not
  band-passed (avoids accidental amplitude changes on non-staging channels).
- Notch is separate and applied globally (or to ``picks`` when provided).
- Sample counts, channel order, and annotations are preserved.
- Applied settings are recorded in ``state.extras['filter']['per_channel']``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import mne

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)


class SignalFilter(Transform):
    """Apply channel-type-aware band-pass and optional notch filtering.

    Parameters
    ----------
    l_freq, h_freq:
        Global fallback band-pass edges in Hz (used when per-type values are
        ``None``).
    eeg_l_freq, eeg_h_freq, eog_l_freq, eog_h_freq, emg_l_freq, emg_h_freq:
        Optional per-channel-type band-pass edges. If ``None``, the global
        ``l_freq``/``h_freq`` are used. Constructor defaults mirror the project
        engineering baseline (EOG high-cut 15 Hz is configurable, not mandated).
    notch_freqs:
        Frequencies for notch filtering applied before band-pass. Values at or
        above the Nyquist frequency are skipped with a warning.
    picks:
        Channel selection for operations (passed through to MNE).
    pad, verbose:
        Forwarded to MNE filtering functions.
    """

    name = "signal_filter"

    def __init__(
        self,
        *,
        l_freq: Optional[float] = 0.5,
        h_freq: Optional[float] = 30.0,
        eeg_l_freq: Optional[float] = 0.5,
        eeg_h_freq: Optional[float] = 30.0,
        eog_l_freq: Optional[float] = 0.5,
        eog_h_freq: Optional[float] = 15.0,
        emg_l_freq: Optional[float] = 10.0,
        emg_h_freq: Optional[float] = 30.0,
        notch_freqs: Tuple[float, ...] | List[float] | None = (50.0,),
        picks: str | list[str] | None = None,
        pad: str = "reflect_limited",
        verbose: str | bool | None = "ERROR",
    ) -> None:
        # Global fallback
        self.l_freq = l_freq
        self.h_freq = h_freq
        # Per-type overrides (None → use global)
        self.eeg_l_freq = eeg_l_freq
        self.eeg_h_freq = eeg_h_freq
        self.eog_l_freq = eog_l_freq
        self.eog_h_freq = eog_h_freq
        self.emg_l_freq = emg_l_freq
        self.emg_h_freq = emg_h_freq

        self.notch_freqs = tuple(notch_freqs) if notch_freqs is not None else ()
        self.picks = picks
        self.pad = pad
        self.verbose = verbose

    def _type_band(self, ch_type: str) -> Tuple[Optional[float], Optional[float]]:
        """Return (l_freq, h_freq) for a given MNE channel type."""
        if ch_type == "eeg":
            return (
                self.eeg_l_freq if self.eeg_l_freq is not None else self.l_freq,
                self.eeg_h_freq if self.eeg_h_freq is not None else self.h_freq,
            )
        if ch_type == "eog":
            return (
                self.eog_l_freq if self.eog_l_freq is not None else self.l_freq,
                self.eog_h_freq if self.eog_h_freq is not None else self.h_freq,
            )
        if ch_type == "emg":
            return (
                self.emg_l_freq if self.emg_l_freq is not None else self.l_freq,
                self.emg_h_freq if self.emg_h_freq is not None else self.h_freq,
            )
        # Unknown/aux channel types: do not apply band-pass by default
        return (None, None)

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        if not state.raw.ch_names:
            raise TransformError("Cannot filter a recording with no channels")
        if not state.raw.preload:
            state.raw.load_data()

        sfreq = state.sampling_frequency
        nyquist = sfreq / 2.0

        # Determine usable notch freqs
        applied_notch = _usable_notch_freqs(self.notch_freqs, nyquist=nyquist)
        skipped_notch = [f for f in self.notch_freqs if f not in applied_notch]
        if skipped_notch:
            logger.warning(
                "Skipping notch freq(s) >= Nyquist (%.1f Hz): %s", nyquist, skipped_notch
            )

        # Apply notch globally (or to picks if provided). Keep as a single step
        # so we don't repeatedly call notch_filter per-channel.
        if applied_notch:
            state.raw.notch_filter(freqs=list(applied_notch), picks=self.picks, verbose=self.verbose)

        # Build per-channel configuration map and apply band-pass per channel-type
        per_channel: Dict[str, Dict[str, Optional[float] | list[float]]] = {}

        # Map of mne channel types to channel names
        ch_names = state.raw.info["ch_names"]
        # Query MNE for channel picks by type
        picks_eeg = mne.pick_types(state.raw.info, eeg=True, eog=False, emg=False, exclude=[])  # indices
        picks_eog = mne.pick_types(state.raw.info, eeg=False, eog=True, emg=False, exclude=[])
        picks_emg = mne.pick_types(state.raw.info, eeg=False, eog=False, emg=True, exclude=[])

        # Helper to apply filter to a list of indices
        def _apply_to_indices(indices, l_freq, h_freq, type_name: str):
            if len(indices) == 0:
                return
            names = [ch_names[i] for i in indices]
            # Adjust h_freq to be below Nyquist if necessary
            if h_freq is not None and h_freq >= nyquist:
                h_adj = max(0.0, nyquist - 0.5)
                logger.warning(
                    "Clamping %s h_freq from %s to %.1f Hz (Nyquist=%.1f)", type_name, h_freq, h_adj, nyquist
                )
                h_use = h_adj
            else:
                h_use = h_freq
            # Call MNE filtering on the selected channel names
            # If both l_freq and h_use are None, skip
            if l_freq is None and h_use is None:
                return
            state.raw.filter(l_freq=l_freq, h_freq=h_use, picks=names, pad=self.pad, verbose=self.verbose)

        # EEG
        l_eeg, h_eeg = self._type_band("eeg")
        _apply_to_indices(picks_eeg, l_eeg, h_eeg, "EEG")

        # EOG
        l_eog, h_eog = self._type_band("eog")
        _apply_to_indices(picks_eog, l_eog, h_eog, "EOG")

        # EMG
        l_emg, h_emg = self._type_band("emg")
        _apply_to_indices(picks_emg, l_emg, h_emg, "EMG")

        # Fill per_channel map for all channels: record applied l/h and notch list
        # Get channel types aligned with ch_names
        ch_types = state.raw.get_channel_types()
        for ch_idx, ch in enumerate(ch_names):
            ch_type = ch_types[ch_idx]
            l_applied, h_applied = self._type_band(ch_type)
            # If the type is unknown, we did not apply band-pass (None)
            per_channel[ch] = {
                "mne_type": ch_type,
                "l_freq": l_applied,
                "h_freq": h_applied,
                "notch_freqs": list(applied_notch),
            }

        state.extras["filter"] = {
            "global": {"l_freq": self.l_freq, "h_freq": self.h_freq, "notch_freqs": list(self.notch_freqs)},
            "per_channel": per_channel,
            "applied_notch_freqs": list(applied_notch),
        }

        logger.info("Applied channel-type-aware filtering; recorded per-channel config in extras['filter']")
        return state


def _usable_notch_freqs(freqs: Tuple[float, ...], *, nyquist: float) -> Tuple[float, ...]:
    """Keep notch frequencies strictly below Nyquist."""
    return tuple(freq for freq in freqs if 0.0 < float(freq) < nyquist)
