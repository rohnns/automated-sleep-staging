"""Epoch-grid alignment helpers for sample-exact signal/label sync."""

from __future__ import annotations

import math


def floor_to_epoch_grid(time_sec: float, epoch_duration_sec: float) -> float:
    """Largest grid boundary ``<= time_sec``."""
    if epoch_duration_sec <= 0:
        raise ValueError("epoch_duration_sec must be positive")
    return math.floor(time_sec / epoch_duration_sec + 1e-12) * epoch_duration_sec


def ceil_to_epoch_grid(time_sec: float, epoch_duration_sec: float) -> float:
    """Smallest grid boundary ``>= time_sec``."""
    if epoch_duration_sec <= 0:
        raise ValueError("epoch_duration_sec must be positive")
    return math.ceil(time_sec / epoch_duration_sec - 1e-12) * epoch_duration_sec


def align_crop_window(
    tmin_sec: float,
    tmax_sec: float,
    *,
    epoch_duration_sec: float,
    recording_duration_sec: float,
) -> tuple[float, float]:
    """Snap a crop window outward onto the recording epoch grid.

    ``tmin`` is floored and ``tmax`` is ceiled so contiguous 30 s slices from
    sample 0 of the cropped recording stay phase-aligned with epoch labels.
    """
    tmin_aligned = floor_to_epoch_grid(tmin_sec, epoch_duration_sec)
    tmax_aligned = ceil_to_epoch_grid(tmax_sec, epoch_duration_sec)
    tmin_aligned = max(0.0, tmin_aligned)
    tmax_aligned = min(float(recording_duration_sec), tmax_aligned)
    return tmin_aligned, tmax_aligned


def grid_epoch_indices_in_bout(
    bout_onset_sec: float,
    bout_end_sec: float,
    *,
    epoch_duration_sec: float,
) -> range:
    """Return grid indices ``k`` whose full epochs lie inside a bout.

    Epoch ``k`` covers ``[k * d, (k + 1) * d)``. Using ``ceil`` for the first
    index (instead of ``round(bout_onset)``) avoids back-dating a bout onto a
    previous segment when annotations leave a short gap.
    """
    if bout_end_sec <= bout_onset_sec:
        return range(0, 0)
    first = math.ceil(bout_onset_sec / epoch_duration_sec - 1e-12)
    last_exclusive = math.floor(bout_end_sec / epoch_duration_sec + 1e-12)
    if last_exclusive < first:
        return range(0, 0)
    return range(int(first), int(last_exclusive))
