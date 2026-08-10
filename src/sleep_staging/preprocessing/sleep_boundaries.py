"""Detect sleep onset/offset from hypnogram labels.

Dataset evidence: Sleep-EDF Expanded recordings typically contain multi-hour
Wake before the first sleep stage and after the last sleep stage. Downstream
wake cropping needs reliable sleep boundaries.
"""

from __future__ import annotations

import mne

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.preprocessing.exceptions import TransformError
from sleep_staging.preprocessing.types import (
    EpochLabels,
    PreprocessedRecording,
    SleepBoundaries,
    Transform,
)

logger = get_logger(__name__)

DEFAULT_SLEEP_LABELS = frozenset(
    {
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
        "N1",
        "N2",
        "N3",
        "REM",
    }
)


class SleepBoundaryDetector(Transform):
    """Compute sleep onset/offset from epoch labels or raw annotations."""

    name = "sleep_boundary_detector"

    def __init__(
        self,
        *,
        sleep_labels: frozenset[str] | set[str] | None = None,
        prefer_epoch_labels: bool = True,
    ) -> None:
        self.sleep_labels = frozenset(sleep_labels or DEFAULT_SLEEP_LABELS)
        self.prefer_epoch_labels = prefer_epoch_labels

    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        if self.prefer_epoch_labels and state.epoch_labels is not None:
            boundaries = detect_boundaries_from_epochs(
                state.epoch_labels,
                sleep_labels=self.sleep_labels,
            )
        else:
            boundaries = detect_boundaries_from_annotations(
                state.raw.annotations,
                sleep_labels=self.sleep_labels,
                recording_duration_sec=state.duration_sec,
            )
        state.boundaries = boundaries
        logger.info(
            "Sleep boundaries: onset=%s offset=%s has_sleep=%s",
            None if boundaries.onset_sec is None else f"{boundaries.onset_sec:.1f}s",
            None if boundaries.offset_sec is None else f"{boundaries.offset_sec:.1f}s",
            boundaries.has_scored_sleep,
        )
        return state


def detect_boundaries_from_epochs(
    epoch_labels: EpochLabels,
    *,
    sleep_labels: frozenset[str] = DEFAULT_SLEEP_LABELS,
) -> SleepBoundaries:
    """Detect sleep onset/offset from fixed epoch labels."""
    sleep_epochs = [
        (onset, onset + epoch_labels.duration_sec)
        for onset, label in zip(epoch_labels.onsets_sec, epoch_labels.labels, strict=True)
        if label in sleep_labels
    ]
    if not sleep_epochs:
        return SleepBoundaries(onset_sec=None, offset_sec=None, has_scored_sleep=False)
    return SleepBoundaries(
        onset_sec=min(start for start, _ in sleep_epochs),
        offset_sec=max(end for _, end in sleep_epochs),
        has_scored_sleep=True,
    )


def detect_boundaries_from_annotations(
    annotations: mne.Annotations,
    *,
    sleep_labels: frozenset[str] = DEFAULT_SLEEP_LABELS,
    recording_duration_sec: float | None = None,
) -> SleepBoundaries:
    """Detect sleep onset/offset from variable-length annotations."""
    if annotations is None or len(annotations) == 0:
        raise TransformError("Cannot detect sleep boundaries: annotations are empty")

    sleep_spans: list[tuple[float, float]] = []
    for onset, duration, description in zip(
        annotations.onset,
        annotations.duration,
        annotations.description,
        strict=True,
    ):
        if str(description) not in sleep_labels:
            continue
        start = float(onset)
        end = float(onset) + float(duration)
        if recording_duration_sec is not None:
            end = min(end, float(recording_duration_sec))
        if end > start:
            sleep_spans.append((start, end))

    if not sleep_spans:
        return SleepBoundaries(onset_sec=None, offset_sec=None, has_scored_sleep=False)
    return SleepBoundaries(
        onset_sec=min(start for start, _ in sleep_spans),
        offset_sec=max(end for _, end in sleep_spans),
        has_scored_sleep=True,
    )
