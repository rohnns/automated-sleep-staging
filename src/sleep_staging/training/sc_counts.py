"""Lightweight SC subject discovery and IGNORE-aware epoch counting."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import mne

from sleep_staging.acquisition.utils import discover_psg_files, parse_psg_filename, resolve_hypnogram_path
from sleep_staging.preprocessing.annotation_unroller import unroll_annotations
from sleep_staging.preprocessing.epoch_grid import align_crop_window
from sleep_staging.preprocessing.sleep_boundaries import detect_boundaries_from_epochs
from sleep_staging.preprocessing.stage_mapper import map_epoch_labels
from sleep_staging.training.dataset import SplitEpochStats
from sleep_staging.training.split import SubjectSplit, sleep_edf_subject_key, subject_wise_split


@dataclass(frozen=True, slots=True)
class RecordingEpochCount:
    subject_key: str
    recording_id: str
    psg_name: str
    n_epochs: int
    n_ignore: int
    n_supervised: int


def discover_sc_subject_keys(data_root: Path | str) -> tuple[str, ...]:
    """Unique canonical SC subject keys under ``data_root``."""
    root = Path(data_root)
    keys: set[str] = set()
    for psg in discover_psg_files(root):
        ids = parse_psg_filename(psg)
        if ids.study != "SC":
            continue
        keys.add(sleep_edf_subject_key(ids))
    return tuple(sorted(keys))


def _count_recording_epochs(
    psg_path: Path,
    *,
    wake_buffer_sec: float = 1800.0,
    epoch_duration_sec: float = 30.0,
) -> RecordingEpochCount:
    """Count supervised epochs using hypnogram-only logic (no EEG load)."""
    ids = parse_psg_filename(psg_path)
    hyp = resolve_hypnogram_path(psg_path)
    annotations = mne.read_annotations(str(hyp))

    # Approximate recording duration from last annotation end.
    if len(annotations) == 0:
        raise RuntimeError(f"Empty hypnogram: {hyp}")
    duration_sec = float(max(onset + dur for onset, dur in zip(annotations.onset, annotations.duration)))

    epoch_labels = unroll_annotations(
        annotations,
        epoch_duration_sec=epoch_duration_sec,
        recording_duration_sec=duration_sec,
        align_to_grid=True,
    )
    boundaries = detect_boundaries_from_epochs(epoch_labels)
    if boundaries.has_scored_sleep and boundaries.onset_sec is not None and boundaries.offset_sec is not None:
        tmin = max(0.0, boundaries.onset_sec - wake_buffer_sec)
        tmax = min(duration_sec, boundaries.offset_sec + wake_buffer_sec)
        tmin, tmax = align_crop_window(
            tmin,
            tmax,
            epoch_duration_sec=epoch_duration_sec,
            recording_duration_sec=duration_sec,
        )
        epoch_labels = epoch_labels.filtered(tmin_sec=tmin, tmax_sec=tmax, shift_to_zero=True)

    mapped = map_epoch_labels(epoch_labels, unmapped_policy="ignore")
    n_epochs = mapped.n_epochs
    n_ignore = sum(1 for label in mapped.labels if label == "IGNORE")
    return RecordingEpochCount(
        subject_key=sleep_edf_subject_key(ids),
        recording_id=ids.recording_id,
        psg_name=psg_path.name,
        n_epochs=n_epochs,
        n_ignore=n_ignore,
        n_supervised=n_epochs - n_ignore,
    )


def count_sc_epochs_by_subject(
    data_root: Path | str,
    *,
    wake_buffer_sec: float = 1800.0,
) -> dict[str, SplitEpochStats]:
    """Aggregate hypnogram-derived epoch counts per SC subject key."""
    root = Path(data_root)
    per_subject_epochs: dict[str, int] = defaultdict(int)
    per_subject_ignore: dict[str, int] = defaultdict(int)

    for psg in discover_psg_files(root):
        ids = parse_psg_filename(psg)
        if ids.study != "SC":
            continue
        counts = _count_recording_epochs(psg, wake_buffer_sec=wake_buffer_sec)
        per_subject_epochs[counts.subject_key] += counts.n_epochs
        per_subject_ignore[counts.subject_key] += counts.n_ignore

    out: dict[str, SplitEpochStats] = {}
    for subject in sorted(per_subject_epochs):
        n_epochs = per_subject_epochs[subject]
        n_ignore = per_subject_ignore[subject]
        out[subject] = SplitEpochStats(
            n_subjects=1,
            n_epochs=n_epochs,
            n_epochs_ignore=n_ignore,
            n_epochs_supervised=n_epochs - n_ignore,
            subjects=(subject,),
        )
    return out


def summarize_sc_subject_split(
    data_root: Path | str,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    wake_buffer_sec: float = 1800.0,
) -> tuple[SubjectSplit, dict[str, SplitEpochStats]]:
    """Build the Phase 4a SC subject split and IGNORE-aware epoch totals."""
    by_subject = count_sc_epochs_by_subject(data_root, wake_buffer_sec=wake_buffer_sec)
    split = subject_wise_split(tuple(by_subject.keys()), ratios=ratios, seed=seed)

    def _agg(subjects: tuple[str, ...]) -> SplitEpochStats:
        n_epochs = sum(by_subject[s].n_epochs for s in subjects)
        n_ignore = sum(by_subject[s].n_epochs_ignore for s in subjects)
        return SplitEpochStats(
            n_subjects=len(subjects),
            n_epochs=n_epochs,
            n_epochs_ignore=n_ignore,
            n_epochs_supervised=n_epochs - n_ignore,
            subjects=subjects,
        )

    stats = {
        "train": _agg(split.train),
        "val": _agg(split.val),
        "test": _agg(split.test),
    }
    return split, stats
