"""Optional wake cropping around the scored sleep period.

Dataset evidence: wake before/after sleep is typically several hours on
Sleep-EDF Expanded. Following common practice (e.g. MNE sleep tutorial), keep
a configurable buffer (default 30 minutes) before first sleep and after last
sleep, then drop the remaining long Wake tails.

Crop windows are snapped to the epoch grid by default so contiguous 30 s
signal slices from sample 0 stay phase-aligned with ``EpochLabels``.
"""

from __future__ import annotations

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.epoch_grid import align_crop_window
from sleep_staging.preprocessing.exceptions import NoSleepBoundaryError, TransformError
from sleep_staging.preprocessing.types import PreprocessedRecording, Transform

logger = get_logger(__name__)


class WakeCropper(Transform):
    """Crop ``raw`` (and epoch labels) to sleep period ± buffer.

    Parameters
    ----------
    buffer_sec:
        Seconds of context kept before sleep onset and after sleep offset.
        Default ``1800`` (30 min), aligned with analysis design notes.
    minutes:
        Optional convenience alias; if set, overrides ``buffer_sec`` as
        ``minutes * 60``.
    require_sleep:
        If true, raise when sleep boundaries are missing. If false, skip crop.
    align_to_epoch_grid:
        If true (default), floor ``tmin`` and ceil ``tmax`` onto the epoch
        grid before cropping. Prevents phase mismatch when encodings slice
        contiguous ``epoch_duration_sec`` windows from sample 0.
    epoch_duration_sec:
        Grid spacing used when ``align_to_epoch_grid`` is true. If ``None``,
        uses ``state.epoch_labels.duration_sec`` when available, else ``30``.
    """

    name = "wake_cropper"

    def __init__(
        self,
        *,
        buffer_sec: float = 1800.0,
        minutes: float | None = None,
        require_sleep: bool = True,
        align_to_epoch_grid: bool = True,
        epoch_duration_sec: float | None = None,
    ) -> None:
        resolved = float(minutes) * 60.0 if minutes is not None else float(buffer_sec)
        if resolved < 0:
            raise ValueError("wake crop buffer must be >= 0")
        if epoch_duration_sec is not None and epoch_duration_sec <= 0:
            raise ValueError("epoch_duration_sec must be positive")
        self.buffer_sec = resolved
        self.require_sleep = require_sleep
        self.align_to_epoch_grid = align_to_epoch_grid
        self.epoch_duration_sec = (
            None if epoch_duration_sec is None else float(epoch_duration_sec)
        )

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        if state.boundaries is None:
            raise TransformError(
                "WakeCropper requires SleepBoundaryDetector to run first"
            )
        if not state.boundaries.has_scored_sleep:
            if self.require_sleep:
                raise NoSleepBoundaryError(
                    "Cannot crop wake: no scored sleep stages were found"
                )
            logger.warning("WakeCropper skipped: no scored sleep in recording")
            return state

        assert state.boundaries.onset_sec is not None
        assert state.boundaries.offset_sec is not None

        tmin = max(0.0, state.boundaries.onset_sec - self.buffer_sec)
        tmax = min(state.duration_sec, state.boundaries.offset_sec + self.buffer_sec)

        epoch_duration = self._resolve_epoch_duration(state)
        tmin_requested, tmax_requested = tmin, tmax
        if self.align_to_epoch_grid:
            tmin, tmax = align_crop_window(
                tmin,
                tmax,
                epoch_duration_sec=epoch_duration,
                recording_duration_sec=state.duration_sec,
            )

        if tmax <= tmin:
            raise TransformError(
                f"Invalid crop window after buffering: tmin={tmin}, tmax={tmax}"
            )

        if not state.raw.preload:
            state.raw.load_data()

        # Crop signals; MNE also trims annotations to the retained window.
        max_raw_time = float(state.raw.times[-1])
        tmax_crop = min(tmax, max_raw_time)
        state.raw.crop(tmin=tmin, tmax=tmax_crop, include_tmax=True)

        if state.epoch_labels is not None:
            state.epoch_labels = state.epoch_labels.filtered(
                tmin_sec=tmin,
                tmax_sec=tmax,
                shift_to_zero=True,
            )

        # Boundaries on the new timeline (origin shifted by tmin).
        state.boundaries = type(state.boundaries)(
            onset_sec=max(0.0, state.boundaries.onset_sec - tmin),
            offset_sec=max(0.0, state.boundaries.offset_sec - tmin),
            has_scored_sleep=True,
        )
        state.extras["wake_crop"] = {
            "tmin_sec": tmin,
            "tmax_sec": tmax,
            "tmin_requested_sec": tmin_requested,
            "tmax_requested_sec": tmax_requested,
            "buffer_sec": self.buffer_sec,
            "align_to_epoch_grid": self.align_to_epoch_grid,
            "epoch_duration_sec": epoch_duration,
        }
        logger.info(
            "Wake crop kept [%.1f, %.1f] s (buffer=%.1f s, grid_align=%s); "
            "duration now %.1f s; epochs=%s",
            tmin,
            tmax,
            self.buffer_sec,
            self.align_to_epoch_grid,
            state.duration_sec,
            None if state.epoch_labels is None else state.epoch_labels.n_epochs,
        )
        return state

    def _resolve_epoch_duration(self, state: PreprocessedRecording) -> float:
        if self.epoch_duration_sec is not None:
            return self.epoch_duration_sec
        if state.epoch_labels is not None:
            return float(state.epoch_labels.duration_sec)
        return 30.0
