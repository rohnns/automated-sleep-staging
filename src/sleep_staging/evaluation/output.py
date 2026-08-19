"""Whole-night prediction, hypnogram, and sleep-statistics utilities."""

from __future__ import annotations

import json
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
    # A single broken_barh() call with one (onset, width) tuple + one color
    # per epoch renders as one vectorized collection. The previous version
    # called broken_barh() once per epoch (one artist each) -- fine for a
    # handful of epochs, but for a full ST test set (~42,550 epochs) that's
    # 42,550 separate matplotlib artists per track, which made legend
    # scanning, layout, and rendering take hours instead of seconds. Same
    # per-epoch stage->color mapping, just batched into one draw call.
    xranges = [(float(onset), 30.0) for onset in onsets]
    colors = [STAGE_COLORS.get(stage_index_to_name(int(stage_idx)), "#999999") for stage_idx in labels]
    if xranges:
        ax.broken_barh(xranges, (y_offset, 0.12), facecolors=colors)
    ax.text(0.0, y_offset + 0.05, label, va="center", ha="left", fontsize=9)


def save_training_history(
    history: Iterable[dict[str, float]],
    out_path: Path,
    *,
    best_epoch: int | None = None,
) -> Path:
    """Persist the per-epoch training history as JSON.

    ``train_baseline`` already tracks ``train_loss``, ``val_loss``,
    ``val_macro_f1`` and an explicit ``loss_gap_train_minus_val`` overfitting
    signal per epoch, but historically kept them only in memory. Writing them
    out makes the training dynamics auditable after the fact, and is the
    machine-readable companion to :func:`plot_loss_curves`.
    """
    rows = [dict(row) for row in history]
    payload: dict[str, object] = {"best_epoch": best_epoch, "epochs": rows}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return out_path


def plot_loss_curves(
    history: Iterable[dict[str, float]],
    out_path: Path,
    *,
    title: str,
    best_epoch: int | None = None,
) -> Path:
    """Plot train vs validation loss per epoch, with the overfitting gap.

    Top panel: train and validation loss on shared axes -- the two curves
    separating is the visual signature of overfitting. Bottom panel: the
    signed ``train_loss - val_loss`` gap the trainer already records, so the
    same information is readable as a single trend line.

    ``best_epoch`` (the epoch whose validation macro-F1 was selected as the
    checkpoint) is marked on both panels, since that is the model actually
    evaluated on the test cohort -- not the final epoch.
    """
    rows = [dict(row) for row in history]
    if not rows:
        raise ValueError("history is empty; nothing to plot")

    epochs = [float(r["epoch"]) for r in rows]
    train = [float(r["train_loss"]) for r in rows]
    val = [float(r["val_loss"]) for r in rows]
    gap = [float(r.get("loss_gap_train_minus_val", t - v)) for r, t, v in zip(rows, train, val)]

    fig, (ax_loss, ax_gap) = plt.subplots(
        2, 1, figsize=(9, 6), sharex=True, height_ratios=[3, 1], constrained_layout=True
    )

    ax_loss.plot(epochs, train, marker="o", ms=3, lw=1.6, color="#377eb8", label="Train loss")
    ax_loss.plot(epochs, val, marker="o", ms=3, lw=1.6, color="#e41a1c", label="Validation loss")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title(title)
    ax_loss.grid(True, alpha=0.25)

    ax_gap.axhline(0.0, color="#666666", lw=1.0)
    ax_gap.plot(epochs, gap, marker="o", ms=3, lw=1.4, color="#4daf4a")
    ax_gap.set_ylabel("train − val")
    ax_gap.set_xlabel("Epoch")
    ax_gap.grid(True, alpha=0.25)

    if best_epoch is not None and best_epoch >= 0:
        for ax in (ax_loss, ax_gap):
            ax.axvline(
                float(best_epoch),
                color="#984ea3",
                ls="--",
                lw=1.3,
                label="Best val macro-F1 (checkpoint)" if ax is ax_loss else None,
            )

    ax_loss.legend(loc="upper right", fontsize=9)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
