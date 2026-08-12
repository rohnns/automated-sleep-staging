"""Reference transform: support 'original' (no-op) and opt-in 'common_average' limited to EEG channels.

Behavior:
- 'original': no change; records method='original' in state.extras['reference']
- 'common_average': compute CAR across EEG channels only and subtract the instantaneous mean
  from each EEG channel. EOG/EMG/aux channels are not included.

The transform is non-destructive for ordering and number of channels; it modifies the
Raw data buffer but preserves channel names and annotations. Records affected channels in
state.extras['reference'] for reproducibility.
"""
from __future__ import annotations

from typing import List

import numpy as np

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)


class ReferenceTransform(Transform):
    """Apply optional reference transformations to the recording.

    Parameters
    ----------
    mode:
        'original' (no-op) or 'common_average' (CAR on EEG channels only).
    """

    name = "reference_transform"

    def __init__(self, *, mode: str = "original") -> None:
        if mode not in {"original", "common_average"}:
            raise ValueError("mode must be 'original' or 'common_average'")
        self.mode = mode

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        raw = state.raw
        # Record default extras even if no-op
        state.extras.setdefault("reference", {})

        if self.mode == "original":
            state.extras["reference"] = {"method": "original", "applied_channels": []}
            logger.info("Reference: original (no-op)")
            return state

        # common_average
        if not raw.preload:
            raw.load_data()

        # Determine EEG channel indices
        try:
            ch_types = raw.get_channel_types()
        except Exception as exc:
            raise TransformError(f"Failed to get channel types for referencing: {exc}")

        eeg_indices: List[int] = [i for i, t in enumerate(ch_types) if (t or "").lower() == "eeg"]
        eeg_names: List[str] = [raw.ch_names[i] for i in eeg_indices]

        if len(eeg_indices) < 2:
            # CAR not meaningful with <2 EEG channels; skip but record
            state.extras["reference"] = {
                "method": "common_average",
                "applied_channels": [],
                "skipped_reason": "not enough EEG channels",
            }
            logger.warning("CAR skipped: fewer than 2 EEG channels available")
            return state

        data = raw.get_data()
        eeg_data = data[eeg_indices, :]

        # Compute instantaneous mean across EEG channels
        mean_across = np.mean(eeg_data, axis=0, keepdims=True)
        # Subtract mean from EEG channels (in-place on a copy then replace raw)
        new_eeg = eeg_data - mean_across

        # Construct a replacement data array preserving other channels
        replaced = data.copy()
        replaced[eeg_indices, :] = new_eeg

        # Replace raw._data by creating a new RawArray to avoid mutating internals
        import mne

        replacement = mne.io.RawArray(replaced.astype(np.float64, copy=False), raw.info.copy(), verbose="ERROR")
        # Preserve annotations
        if raw.annotations is not None and len(raw.annotations) > 0:
            replacement.set_annotations(raw.annotations.copy(), emit_warning=False)

        state.raw = replacement
        state.extras["reference"] = {"method": "common_average", "applied_channels": eeg_names}
        logger.info("Applied common-average reference to EEG channels: %s", eeg_names)
        return state
