"""Unit tests for offline dataset analysis helpers (no full dataset required)."""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "utilities"))
from dataset_statistics import (
    AnnotationSegment,
    compute_sleep_boundaries,
)


def test_compute_sleep_boundaries_basic() -> None:
    segments = [
        AnnotationSegment(0.0, 3600.0, "Sleep stage W"),
        AnnotationSegment(3600.0, 30.0, "Sleep stage 1"),
        AnnotationSegment(3630.0, 30.0, "Sleep stage 2"),
        AnnotationSegment(3660.0, 30.0, "Sleep stage W"),
        AnnotationSegment(3690.0, 30.0, "Sleep stage R"),
        AnnotationSegment(3720.0, 1800.0, "Sleep stage W"),
    ]
    stats = compute_sleep_boundaries(segments, recording_duration_sec=6000.0)
    assert stats.has_scored_sleep is True
    assert stats.sleep_onset_sec == 3600.0
    assert stats.sleep_offset_sec == 3720.0
    assert abs(stats.wake_before_sleep_sec - 3600.0) < 1e-6
    assert abs(stats.wake_during_sleep_sec - 30.0) < 1e-6
    assert abs(stats.wake_after_sleep_sec - 1800.0) < 1e-6
    assert abs(stats.sleep_period_duration_sec - 120.0) < 1e-6


def test_compute_sleep_boundaries_no_sleep() -> None:
    segments = [
        AnnotationSegment(0.0, 100.0, "Sleep stage W"),
        AnnotationSegment(100.0, 50.0, "Sleep stage ?"),
    ]
    stats = compute_sleep_boundaries(segments, recording_duration_sec=150.0)
    assert stats.has_scored_sleep is False
    assert stats.sleep_onset_sec is None
    assert abs(stats.wake_before_sleep_sec - 100.0) < 1e-6
