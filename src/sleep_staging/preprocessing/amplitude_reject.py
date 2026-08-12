"""Epoch-level amplitude artifact rejection.

Marks epochs with extreme peak-to-peak amplitude as ``IGNORE`` without deleting
epochs or shifting the 30 s grid. Thresholds are configurable engineering
baselines in Volts (MNE units) and are **not** claimed physiological standards;
they must be validated on real Sleep-EDF SC data before being treated as final.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.stage_mapper import IGNORE_LABEL
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)


class AmplitudeEpochRejector(Transform):
    """Reject epochs whose peak-to-peak amplitude exceeds type-specific limits.

    For each epoch window ``[onset, onset + duration)`` (same indexing as
    encodings), compute per-channel peak-to-peak ``max - min``. If any checked
    channel exceeds its type threshold, the epoch label is set to
    ``ignore_label``. Non-finite samples also trigger rejection.

    Parameters
    ----------
    eeg_peak_to_peak, eog_peak_to_peak, emg_peak_to_peak:
        Peak-to-peak thresholds in Volts. ``None`` disables checks for that type.
    ignore_label:
        Label written for rejected epochs (default ``IGNORE`` → encodings ``-100``).
    """

    name = "amplitude_epoch_rejector"

    def __init__(
        self,
        *,
        eeg_peak_to_peak: float | None = 5.0e-4,
        eog_peak_to_peak: float | None = 1.0e-3,
        emg_peak_to_peak: float | None = 1.0e-3,
        ignore_label: str = IGNORE_LABEL,
    ) -> None:
        for name, value in (
            ("eeg_peak_to_peak", eeg_peak_to_peak),
            ("eog_peak_to_peak", eog_peak_to_peak),
            ("emg_peak_to_peak", emg_peak_to_peak),
        ):
            if value is not None and float(value) <= 0:
                raise ValueError(f"{name} must be positive when set")
        self.eeg_peak_to_peak = None if eeg_peak_to_peak is None else float(eeg_peak_to_peak)
        self.eog_peak_to_peak = None if eog_peak_to_peak is None else float(eog_peak_to_peak)
        self.emg_peak_to_peak = None if emg_peak_to_peak is None else float(emg_peak_to_peak)
        self.ignore_label = str(ignore_label)

    def _threshold_for_type(self, ch_type: str) -> float | None:
        t = (ch_type or "").lower()
        if t == "eeg":
            return self.eeg_peak_to_peak
        if t == "eog":
            return self.eog_peak_to_peak
        if t == "emg":
            return self.emg_peak_to_peak
        return None

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        if state.epoch_labels is None:
            raise TransformError(
                "AmplitudeEpochRejector requires epoch_labels (AnnotationUnroller first)"
            )
        if not state.raw.preload:
            state.raw.load_data()

        epoch_labels = state.epoch_labels
        sfreq = float(state.sampling_frequency)
        n_times = int(round(float(epoch_labels.duration_sec) * sfreq))
        if n_times <= 0:
            raise TransformError("Invalid epoch sample count for amplitude rejection")

        data = state.raw.get_data()
        n_channels, n_samples = data.shape
        ch_names = list(state.raw.ch_names)
        ch_types = [t.lower() for t in state.raw.get_channel_types()]

        new_labels: list[str] = []
        rejected_indices: list[int] = []
        reasons: dict[str, list[str]] = {}
        counts_by_reason: dict[str, int] = {}
        n_already_ignore = 0

        for idx, (onset_sec, label) in enumerate(
            zip(epoch_labels.onsets_sec, epoch_labels.labels, strict=True)
        ):
            if label == self.ignore_label:
                n_already_ignore += 1

            start = int(round(float(onset_sec) * sfreq))
            stop = start + n_times
            epoch_reasons: list[str] = []

            if start < 0 or stop > n_samples:
                reason = "epoch_out_of_bounds"
                epoch_reasons.append(reason)
                counts_by_reason[reason] = counts_by_reason.get(reason, 0) + 1
            else:
                window = data[:, start:stop]
                for ch_i in range(n_channels):
                    thr = self._threshold_for_type(ch_types[ch_i])
                    if thr is None:
                        continue
                    channel = window[ch_i]
                    ch_name = ch_names[ch_i]
                    ch_type = ch_types[ch_i]
                    if not np.isfinite(channel).all():
                        reason = f"{ch_type}:{ch_name}:nonfinite"
                        epoch_reasons.append(reason)
                        counts_by_reason[reason] = counts_by_reason.get(reason, 0) + 1
                        continue
                    ptp = float(np.ptp(channel))
                    if ptp > thr:
                        reason = f"{ch_type}:{ch_name}:peak_to_peak"
                        epoch_reasons.append(reason)
                        counts_by_reason[reason] = counts_by_reason.get(reason, 0) + 1

            if epoch_reasons:
                rejected_indices.append(idx)
                reasons[str(idx)] = epoch_reasons
                new_labels.append(self.ignore_label)
            else:
                new_labels.append(label)

        state.epoch_labels = epoch_labels.relabeled(tuple(new_labels))
        extras: dict[str, Any] = {
            "rule": "peak_to_peak",
            "thresholds": {
                "eeg_peak_to_peak": self.eeg_peak_to_peak,
                "eog_peak_to_peak": self.eog_peak_to_peak,
                "emg_peak_to_peak": self.emg_peak_to_peak,
            },
            "ignore_label": self.ignore_label,
            "n_epochs": epoch_labels.n_epochs,
            "n_rejected": len(rejected_indices),
            "n_already_ignore": n_already_ignore,
            "rejected_epoch_indices": rejected_indices,
            "reasons": reasons,
            "counts_by_reason": counts_by_reason,
        }
        state.extras["amplitude_reject"] = extras
        logger.info(
            "Amplitude epoch rejection: %d/%d epoch(s) marked %s (rule=peak_to_peak)",
            len(rejected_indices),
            epoch_labels.n_epochs,
            self.ignore_label,
        )
        return state
