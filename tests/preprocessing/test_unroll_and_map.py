"""Tests for annotation unrolling and stage mapping."""

from __future__ import annotations

import mne
import pytest

from sleep_staging.preprocessing import (
    IGNORE_LABEL,
    AnnotationUnroller,
    StageMapper,
    map_epoch_labels,
    unroll_annotations,
)
from sleep_staging.preprocessing.types import EpochLabels, PreprocessedRecording
from tests.preprocessing.conftest import make_sleep_recording


def test_unroll_annotations_expands_bouts() -> None:
    annot = mne.Annotations(
        onset=[0.0, 90.0, 180.0],
        duration=[90.0, 45.0, 0.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage ?"],
    )
    epochs = unroll_annotations(annot, epoch_duration_sec=30.0, min_remainder_sec=30.0)
    assert epochs.n_epochs == 4  # 3 from W + 1 from stage 2 (45→30, drop 15)
    assert epochs.labels == (
        "Sleep stage W",
        "Sleep stage W",
        "Sleep stage W",
        "Sleep stage 2",
    )
    assert epochs.onsets_sec == (0.0, 30.0, 60.0, 90.0)


def test_unroll_snaps_to_global_grid_without_backdating() -> None:
    # Gap 300–310 s: bout-relative unrolling would start at 310, 340, ...
    # Grid mode starts at 330 (ceil), never claims the 300–330 gap as stage 2.
    annot = mne.Annotations(
        onset=[0.0, 310.0],
        duration=[300.0, 90.0],
        description=["Sleep stage W", "Sleep stage 2"],
    )
    epochs = unroll_annotations(annot, epoch_duration_sec=30.0, align_to_grid=True)
    stage2_onsets = [
        onset
        for onset, label in zip(epochs.onsets_sec, epochs.labels, strict=True)
        if label == "Sleep stage 2"
    ]
    assert stage2_onsets == [330.0, 360.0]
    assert epochs.onsets_sec[-3:] == (270.0, 330.0, 360.0)


def test_unroll_bout_relative_legacy_mode() -> None:
    annot = mne.Annotations(
        onset=[310.0],
        duration=[90.0],
        description=["Sleep stage 2"],
    )
    epochs = unroll_annotations(annot, epoch_duration_sec=30.0, align_to_grid=False)
    assert epochs.onsets_sec == (310.0, 340.0, 370.0)


def test_annotation_unroller_transform() -> None:
    annot = mne.Annotations(
        onset=[0.0, 60.0],
        duration=[60.0, 60.0],
        description=["Sleep stage W", "Sleep stage 1"],
    )
    recording = make_sleep_recording(duration_sec=300.0, annotations=annot)
    state = PreprocessedRecording.from_sleep_recording(recording)
    state = AnnotationUnroller()(state)
    assert state.epoch_labels is not None
    assert state.epoch_labels.n_epochs == 4
    assert "annotation_unroller" in state.applied_transforms


def test_stage_mapper_rk_to_aasm() -> None:
    annot = mne.Annotations(
        onset=[0.0, 30.0, 60.0, 90.0, 120.0, 150.0],
        duration=[30.0] * 6,
        description=[
            "Sleep stage W",
            "Sleep stage 1",
            "Sleep stage 2",
            "Sleep stage 3",
            "Sleep stage 4",
            "Sleep stage R",
        ],
    )
    recording = make_sleep_recording(duration_sec=300.0, annotations=annot)
    state = PreprocessedRecording.from_sleep_recording(recording)
    state = AnnotationUnroller()(state)
    state = StageMapper()(state)
    assert state.epoch_labels is not None
    assert state.epoch_labels.labels == ("W", "N1", "N2", "N3", "N3", "REM")


def test_stage_mapper_marks_ignore_without_dropping() -> None:
    annot = mne.Annotations(
        onset=[0.0, 30.0, 60.0],
        duration=[30.0, 30.0, 30.0],
        description=["Sleep stage 2", "Movement time", "Sleep stage ?"],
    )
    recording = make_sleep_recording(duration_sec=120.0, annotations=annot)
    state = PreprocessedRecording.from_sleep_recording(recording)
    state = AnnotationUnroller()(state)
    assert state.epoch_labels is not None
    before = state.epoch_labels.n_epochs
    state = StageMapper(unmapped_policy="ignore")(state)
    assert state.epoch_labels.n_epochs == before
    assert state.epoch_labels.labels == ("N2", IGNORE_LABEL, IGNORE_LABEL)
    assert state.epoch_labels.onsets_sec == (0.0, 30.0, 60.0)


def test_stage_mapper_rejects_drop_without_opt_in() -> None:
    with pytest.raises(ValueError, match="allow_length_change"):
        StageMapper(unmapped_policy="drop")


def test_map_epoch_labels_drop_helper_still_available() -> None:
    labels = EpochLabels(
        onsets_sec=(0.0, 30.0),
        duration_sec=30.0,
        labels=("Sleep stage 2", "Movement time"),
    )
    mapped = map_epoch_labels(labels, unmapped_policy="drop")
    assert mapped.labels == ("N2",)


def test_stage_mapper_error_policy() -> None:
    labels = EpochLabels(
        onsets_sec=(0.0,),
        duration_sec=30.0,
        labels=("not a real stage",),
    )
    with pytest.raises(Exception):
        map_epoch_labels(labels, unmapped_policy="error")
