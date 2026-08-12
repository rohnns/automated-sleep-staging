"""Detect bad channels by simple statistical heuristics (non-destructive).

Marks channels with near-flat signals, excessive non-finite samples, saturation,
pathological variance, or extreme peak-to-peak amplitude. Does NOT remove or
reorder channels; findings are recorded in ``state.extras['bad_channels']`` and
optionally appended to ``raw.info['bads']`` for compatibility with MNE tooling.

This is intentionally conservative and configurable; downstream policies decide
how to handle channels marked bad.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)


class BadChannelDetector(Transform):
    """Non-destructive bad-channel detection with channel-type-aware thresholds.

    Parameters
    ----------
    flat_std_threshold:
        Channels with standard deviation below this (Volts) are considered flat.
        (applied to all channel types)
    nan_frac_threshold:
        Fraction of non-finite samples required to mark a channel as bad.
        (applied to all channel types)
    saturation_frac_threshold:
        Fraction of samples equal to the channel min or max value considered
        evidence of digital saturation. (applied to all channel types)
    eeg_high_std_threshold, eeg_peak_to_peak_threshold:
        Amplitude/variance thresholds for EEG channels.
    eog_high_std_threshold, eog_peak_to_peak_threshold:
        Amplitude/variance thresholds for EOG channels (more permissive).
    emg_high_std_threshold, emg_peak_to_peak_threshold:
        Amplitude/variance thresholds for EMG channels.
    mark_mne_bads:
        If true, extend raw.info['bads'] with detected channels (non-destructive).
    """

    name = "bad_channel_detector"

    def __init__(
        self,
        *,
        flat_std_threshold: float = 1e-8,
        nan_frac_threshold: float = 0.01,
        saturation_frac_threshold: float = 0.99,
        # EEG defaults (preserve previous global defaults)
        eeg_high_std_threshold: float = 1e-2,
        eeg_peak_to_peak_threshold: float = 1e-3,
        # EOG defaults (more permissive for eye movements)
        eog_high_std_threshold: float = 2e-4,
        eog_peak_to_peak_threshold: float = 3e-3,
        # EMG defaults
        emg_high_std_threshold: float = 5e-4,
        emg_peak_to_peak_threshold: float = 5e-3,
        mark_mne_bads: bool = True,
    ) -> None:
        self.flat_std_threshold = float(flat_std_threshold)
        self.nan_frac_threshold = float(nan_frac_threshold)
        self.saturation_frac_threshold = float(saturation_frac_threshold)

        self.eeg_high_std_threshold = float(eeg_high_std_threshold)
        self.eeg_peak_to_peak_threshold = float(eeg_peak_to_peak_threshold)

        self.eog_high_std_threshold = float(eog_high_std_threshold)
        self.eog_peak_to_peak_threshold = float(eog_peak_to_peak_threshold)

        self.emg_high_std_threshold = float(emg_high_std_threshold)
        self.emg_peak_to_peak_threshold = float(emg_peak_to_peak_threshold)

        self.mark_mne_bads = bool(mark_mne_bads)

    def _get_thresholds_for_type(self, ch_type: str) -> tuple[float | None, float | None]:
        """Return (high_std_threshold, peak_to_peak_threshold) for channel type.

        For channel types other than eeg/eog/emg, return (None, None) so amplitude
        checks are skipped (those channels are handled by ChannelSelector).
        """
        t = ch_type.lower() if ch_type is not None else ""
        if t == "eeg":
            return self.eeg_high_std_threshold, self.eeg_peak_to_peak_threshold
        if t == "eog":
            return self.eog_high_std_threshold, self.eog_peak_to_peak_threshold
        if t == "emg":
            return self.emg_high_std_threshold, self.emg_peak_to_peak_threshold
        return None, None

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        raw = state.raw
        if not raw.preload:
            raw.load_data()

        data = raw.get_data()
        n_ch, n_times = data.shape

        flat: list[str] = []
        nan: list[str] = []
        saturation: list[str] = []
        high_variance: list[str] = []
        extreme_amplitude: list[str] = []

        # Obtain channel types aligned with raw.ch_names
        try:
            ch_types = raw.get_channel_types()
        except Exception:
            # Fallback: unknown types
            ch_types = ["unknown"] * len(raw.ch_names)

        for ch_idx, (ch_name, ch_type) in enumerate(zip(raw.ch_names, ch_types, strict=True)):
            vals = data[ch_idx, :]
            finite = np.isfinite(vals)
            n_finite = int(finite.sum())
            nan_frac = 1.0 - (n_finite / float(n_times))

            # Use nan-aware statistics to avoid crashing on NaN-full channels
            std = float(np.nanstd(vals))
            p2p = float(np.nanmax(vals) - np.nanmin(vals))

            # Saturation heuristic: large fraction of samples equal to min or max
            if n_finite > 0:
                maxv = float(np.nanmax(vals))
                minv = float(np.nanmin(vals))
                # Use exact equality; saturation typically produces identical samples
                max_count = int((vals == maxv).sum())
                min_count = int((vals == minv).sum())
                sat_frac = max(max_count, min_count) / float(n_times)
            else:
                sat_frac = 1.0

            # Flat / NaN / Saturation checks apply to all channels
            if std <= self.flat_std_threshold:
                flat.append(ch_name)
            if nan_frac >= self.nan_frac_threshold:
                nan.append(ch_name)
            if sat_frac >= self.saturation_frac_threshold:
                saturation.append(ch_name)

            # Channel-type-aware amplitude/variance checks
            high_std_thresh, p2p_thresh = self._get_thresholds_for_type(ch_type)
            if high_std_thresh is not None and std >= high_std_thresh:
                high_variance.append(ch_name)
            if p2p_thresh is not None and p2p >= p2p_thresh:
                extreme_amplitude.append(ch_name)

            logger.debug(
                "BadChannel check: %s type=%s std=%.6g p2p=%.6g nan_frac=%.3g sat_frac=%.3g",
                ch_name,
                ch_type,
                std,
                p2p,
                nan_frac,
                sat_frac,
            )

        all_flagged = sorted(set(flat + nan + saturation + high_variance + extreme_amplitude))

        report = {
            "flat": flat,
            "nan": nan,
            "saturation": saturation,
            "high_variance": high_variance,
            "extreme_amplitude": extreme_amplitude,
            "all": all_flagged,
            "params": {
                "flat_std_threshold": self.flat_std_threshold,
                "nan_frac_threshold": self.nan_frac_threshold,
                "saturation_frac_threshold": self.saturation_frac_threshold,
                "eeg_high_std_threshold": self.eeg_high_std_threshold,
                "eeg_peak_to_peak_threshold": self.eeg_peak_to_peak_threshold,
                "eog_high_std_threshold": self.eog_high_std_threshold,
                "eog_peak_to_peak_threshold": self.eog_peak_to_peak_threshold,
                "emg_high_std_threshold": self.emg_high_std_threshold,
                "emg_peak_to_peak_threshold": self.emg_peak_to_peak_threshold,
                "mark_mne_bads": self.mark_mne_bads,
            },
        }

        state.extras["bad_channels"] = report

        if self.mark_mne_bads and all_flagged:
            # Extend MNE bad list non-destructively
            existing = list(raw.info.get("bads", []) or [])
            for ch in all_flagged:
                if ch not in existing:
                    existing.append(ch)
            raw.info["bads"] = existing

        logger.info("Bad channel detection: %d channels flagged", len(all_flagged))
        return state
