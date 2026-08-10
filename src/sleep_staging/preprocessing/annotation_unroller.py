"""Expand variable-length hypnogram bouts into fixed 30-second epoch labels.

Dataset evidence (``analysis/dataset_statistics.py``): Sleep-EDF Expanded
hypnogram annotations are contiguous multi-epoch stage bouts with many
durations that are multiples of 30 s (and occasional remainders / zeros).
Clinical staging and the planned CNN pipeline use fixed 30 s epochs.

By default, epoch onsets are emitted on the recording-level epoch grid
``0, 30, 60, ...`` so labels stay phase-aligned with contiguous sample
slicing after wake cropping.
"""

from __future__ import annotations

import mne

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.epoch_grid import grid_epoch_indices_in_bout
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.types import EpochLabels, PreprocessedRecording, Transform

logger = get_logger(__name__)


class AnnotationUnroller(Transform):
    """Convert variable-length annotations into fixed-duration epoch labels.

    Parameters
    ----------
    epoch_duration_sec:
        Target epoch length. Defaults to 30 s (Sleep-EDF / AASM standard).
    min_remainder_sec:
        Only used when ``align_to_grid=False``. Minimum leftover bout duration
        to keep as a final partial epoch. Defaults to ``epoch_duration_sec``.
    align_to_grid:
        If true (default), emit only full epochs whose onsets lie on the
        global recording grid ``k * epoch_duration_sec``. Partial leading /
        trailing fragments inside a bout are dropped rather than creating
        off-grid onsets.
    source:
        ``"raw"`` reads ``state.raw.annotations`` (authoritative).
    """

    name = "annotation_unroller"

    def __init__(
        self,
        *,
        epoch_duration_sec: float = 30.0,
        min_remainder_sec: float | None = None,
        align_to_grid: bool = True,
        source: str = "raw",
    ) -> None:
        if epoch_duration_sec <= 0:
            raise ValueError("epoch_duration_sec must be positive")
        self.epoch_duration_sec = float(epoch_duration_sec)
        self.min_remainder_sec = (
            self.epoch_duration_sec if min_remainder_sec is None else float(min_remainder_sec)
        )
        self.align_to_grid = align_to_grid
        if source != "raw":
            raise ValueError("Only source='raw' is supported")
        self.source = source

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        annotations = state.raw.annotations
        if annotations is None or len(annotations) == 0:
            raise TransformError("Cannot unroll annotations: raw.annotations is empty")

        epoch_labels = unroll_annotations(
            annotations,
            epoch_duration_sec=self.epoch_duration_sec,
            min_remainder_sec=self.min_remainder_sec,
            recording_duration_sec=state.duration_sec,
            align_to_grid=self.align_to_grid,
        )
        state.epoch_labels = epoch_labels
        logger.info(
            "Unrolled %d annotation bout(s) into %d x %.0f s epoch(s) (align_to_grid=%s)",
            len(annotations),
            epoch_labels.n_epochs,
            self.epoch_duration_sec,
            self.align_to_grid,
        )
        return state


def unroll_annotations(
    annotations: mne.Annotations,
    *,
    epoch_duration_sec: float = 30.0,
    min_remainder_sec: float | None = None,
    recording_duration_sec: float | None = None,
    align_to_grid: bool = True,
) -> EpochLabels:
    """Pure helper: expand annotation bouts into fixed-length epochs.

    When ``align_to_grid`` is true, each bout contributes every full grid
    epoch ``[k d, (k+1) d)`` that is fully contained in the bout. This uses
    ``ceil`` for the first index — not ``round(bout_onset)`` — so a bout that
    starts mid-grid (e.g. after a short gap) never back-dates into a prior
    segment.
    """
    min_remainder = (
        epoch_duration_sec if min_remainder_sec is None else float(min_remainder_sec)
    )
    onsets: list[float] = []
    labels: list[str] = []

    for bout_onset, bout_duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        strict=True,
    ):
        bout_onset_f = float(bout_onset)
        bout_duration_f = float(bout_duration)
        label = str(description)

        if bout_duration_f <= 0:
            continue

        if recording_duration_sec is not None:
            bout_duration_f = min(
                bout_duration_f,
                max(0.0, float(recording_duration_sec) - bout_onset_f),
            )
            if bout_duration_f <= 0:
                continue

        bout_end = bout_onset_f + bout_duration_f

        if align_to_grid:
            for index in grid_epoch_indices_in_bout(
                bout_onset_f,
                bout_end,
                epoch_duration_sec=epoch_duration_sec,
            ):
                onsets.append(index * epoch_duration_sec)
                labels.append(label)
            continue

        n_full = int(bout_duration_f // epoch_duration_sec)
        for index in range(n_full):
            onsets.append(bout_onset_f + index * epoch_duration_sec)
            labels.append(label)

        remainder = bout_duration_f - n_full * epoch_duration_sec
        if remainder >= min_remainder - 1e-9:
            onsets.append(bout_onset_f + n_full * epoch_duration_sec)
            labels.append(label)

    return EpochLabels(
        onsets_sec=tuple(onsets),
        duration_sec=float(epoch_duration_sec),
        labels=tuple(labels),
    )
