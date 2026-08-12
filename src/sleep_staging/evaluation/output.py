"""Whole-night prediction, hypnogram, and sleep-statistics utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes

STAGE_NAMES: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
STAGE_TO_INDEX = {stage: idx for idx, stage in enumerate(STAGE_NAMES)}
STAGE_COLORS = {
    "W": "#bdbdbd",
    "N1": "#fdae61",
    "N2": "#4daf4a",
    "N3": "#377eb8",
    "REM": "#984ea3",
}


@dataclass(frozen=True, slots=True)
class EpochPredictionRecord:
    subject_id: str
    recording_id: str
    epoch_index: int
    onset_sec: float
    duration_sec: float
    target: int
    prediction: int


@dataclass(frozen=True, slots=True)
class SleepStatistics:
    total_sleep_time_sec: float
    sleep_efficiency: float
    sleep_onset_latency_sec: float | None
    rem_latency_sec: float | None
    time_in_stage_sec: dict[str, float]
    pct_in_stage: dict[str, float]


def stage_index_to_name(index: int) -> str:
    if index < 0 or index >= len(STAGE_NAMES):
        return "IGNORE"
    return STAGE_NAMES[index]


def decode_stage_sequence(indices: Iterable[int]) -> list[str]:
    return [stage_index_to_name(int(idx)) for idx in indices]


def compute_sleep_statistics(labels: Iterable[int], *, epoch_duration_sec: float = 30.0) -> SleepStatistics:
    seq = [int(x) for x in labels if int(x) >= 0]
    total_recorded_sec = len(seq) * epoch_duration_sec
    time_in_stage = {stage: 0.0 for stage in STAGE_NAMES}
    for idx in seq:
        if 0 <= idx < len(STAGE_NAMES):
            time_in_stage[STAGE_NAMES[idx]] += epoch_duration_sec
    total_sleep_time_sec = sum(time_in_stage[stage] for stage in STAGE_NAMES if stage != "W")
    sleep_efficiency = (total_sleep_time_sec / total_recorded_sec) if total_recorded_sec > 0 else 0.0

    sleep_onset_latency_sec = None
    rem_latency_sec = None
    for i, idx in enumerate(seq):
        stage = STAGE_NAMES[idx] if 0 <= idx < len(STAGE_NAMES) else None
        if sleep_onset_latency_sec is None and stage is not None and stage != "W":
            sleep_onset_latency_sec = i * epoch_duration_sec
        if rem_latency_sec is None and stage == "REM":
            rem_latency_sec = i * epoch_duration_sec
        if sleep_onset_latency_sec is not None and rem_latency_sec is not None:
            break

    pct = {
        stage: (time_in_stage[stage] / total_recorded_sec * 100.0) if total_recorded_sec > 0 else 0.0
        for stage in STAGE_NAMES
    }
    return SleepStatistics(
        total_sleep_time_sec=total_sleep_time_sec,
        sleep_efficiency=sleep_efficiency,
        sleep_onset_latency_sec=sleep_onset_latency_sec,
        rem_latency_sec=rem_latency_sec,
        time_in_stage_sec=time_in_stage,
        pct_in_stage=pct,
    )


def build_epoch_predictions(
    *,
    subject_id: str,
    recording_id: str,
    onsets_sec: Iterable[float],
    duration_sec: float,
    y_true: Iterable[int],
    y_pred: Iterable[int],
) -> tuple[EpochPredictionRecord, ...]:
    return tuple(
        EpochPredictionRecord(
            subject_id=subject_id,
            recording_id=recording_id,
            epoch_index=i,
            onset_sec=float(onset),
            duration_sec=duration_sec,
            target=int(t),
            prediction=int(p),
        )
        for i, (onset, t, p) in enumerate(zip(onsets_sec, y_true, y_pred, strict=True))
    )


def save_predictions_csv(records: Iterable[EpochPredictionRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        "subject_id,recording_id,epoch_index,onset_sec,duration_sec,target,prediction,target_stage,prediction_stage"
    ]
    for r in records:
        rows.append(
            f"{r.subject_id},{r.recording_id},{r.epoch_index},{r.onset_sec:.1f},{r.duration_sec:.1f},"
            f"{r.target},{r.prediction},{stage_index_to_name(r.target)},{stage_index_to_name(r.prediction)}"
        )
    path.write_text("\n".join(rows), encoding="utf-8")


def plot_hypnogram(
    *,
    onsets_sec: Iterable[float],
    y_true: Iterable[int],
    y_pred: Iterable[int],
    out_path: Path,
    title: str,
    epoch_duration_sec: float = 30.0,
) -> None:
    onsets = np.asarray(list(onsets_sec), dtype=float)
    true = np.asarray(list(y_true), dtype=int)
    pred = np.asarray(list(y_pred), dtype=int)
    fig, ax = plt.subplots(figsize=(16, 5), constrained_layout=True)
    _plot_stage_track(ax, onsets, true, label="Expert", y_offset=0.0)
    _plot_stage_track(ax, onsets, pred, label="Predicted", y_offset=0.15)
    ax.set_yticks([0.0, 0.15])
    ax.set_yticklabels(["Expert", "Predicted"])
    ax.set_xlabel("Time (s)")
    ax.set_title(title)
    ax.legend(loc="upper right")
    ax.set_ylim(-0.15, 0.35)
    ax.grid(True, axis="x", alpha=0.2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_stage_track(ax: Axes, onsets: np.ndarray, labels: np.ndarray, *, label: str, y_offset: float) -> None:
    for onset, stage_idx in zip(onsets, labels, strict=True):
        stage = stage_index_to_name(int(stage_idx))
        color = STAGE_COLORS.get(stage, "#999999")
        ax.broken_barh([(float(onset), 30.0)], (y_offset, 0.12), facecolors=color)
    ax.text(0.0, y_offset + 0.05, label, va="center", ha="left", fontsize=9)
