"""Classification metrics with IGNORE excluded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


STAGE_NAMES: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")


@dataclass(frozen=True, slots=True)
class ClassificationMetrics:
    """Aggregate + per-class metrics on supervised epochs only."""

    accuracy: float
    macro_f1: float
    cohen_kappa: float
    per_class_precision: tuple[float, ...]
    per_class_recall: tuple[float, ...]
    per_class_f1: tuple[float, ...]
    confusion_matrix: NDArray[np.int64]
    normalized_confusion_matrix: NDArray[np.float64]
    n_samples: int
    class_names: tuple[str, ...] = STAGE_NAMES

    def as_dict(self) -> dict[str, object]:
        return {
            "accuracy": self.accuracy,
            "macro_f1": self.macro_f1,
            "cohen_kappa": self.cohen_kappa,
            "per_class_precision": dict(zip(self.class_names, self.per_class_precision, strict=True)),
            "per_class_recall": dict(zip(self.class_names, self.per_class_recall, strict=True)),
            "per_class_f1": dict(zip(self.class_names, self.per_class_f1, strict=True)),
            "confusion_matrix": self.confusion_matrix.tolist(),
            "normalized_confusion_matrix": self.normalized_confusion_matrix.tolist(),
            "n_samples": self.n_samples,
        }


def _filter_ignore(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    *,
    ignore_index: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    mask = y_true != ignore_index
    return y_true[mask], y_pred[mask]


def confusion_matrix(
    y_true: NDArray[np.int64],
    y_pred: NDArray[np.int64],
    *,
    n_classes: int = 5,
) -> NDArray[np.int64]:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist(), strict=True):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def normalize_confusion_matrix(cm: NDArray[np.integer]) -> NDArray[np.float64]:
    """Row-normalize a multiclass confusion matrix by true class support."""
    cm_float = np.asarray(cm, dtype=np.float64)
    row_sums = cm_float.sum(axis=1, keepdims=True)
    return np.divide(cm_float, row_sums, out=np.zeros_like(cm_float), where=row_sums > 0)


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def cohen_kappa(cm: NDArray[np.integer]) -> float:
    cm = np.asarray(cm, dtype=np.int64)
    n = float(cm.sum())
    if n <= 0:
        return 0.0
    po = float(np.trace(cm) / n)
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    pe = float(np.sum(row * col) / (n * n))
    if pe >= 1.0:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def compute_classification_metrics(
    y_true: Sequence[int] | NDArray[np.integer],
    y_pred: Sequence[int] | NDArray[np.integer],
    *,
    ignore_index: int = -100,
    class_names: tuple[str, ...] = STAGE_NAMES,
) -> ClassificationMetrics:
    """Compute metrics after dropping IGNORE targets."""
    yt = np.asarray(y_true, dtype=np.int64).reshape(-1)
    yp = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if yt.shape != yp.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    yt, yp = _filter_ignore(yt, yp, ignore_index=ignore_index)
    n_classes = len(class_names)
    cm = confusion_matrix(yt, yp, n_classes=n_classes)
    n = int(cm.sum())
    accuracy = _safe_div(float(np.trace(cm)), float(n))

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for k in range(n_classes):
        tp = float(cm[k, k])
        fp = float(cm[:, k].sum() - tp)
        fn = float(cm[k, :].sum() - tp)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2.0 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return ClassificationMetrics(
        accuracy=accuracy,
        macro_f1=float(np.mean(f1s)) if f1s else 0.0,
        cohen_kappa=cohen_kappa(cm),
        per_class_precision=tuple(precisions),
        per_class_recall=tuple(recalls),
        per_class_f1=tuple(f1s),
        confusion_matrix=cm,
        normalized_confusion_matrix=normalize_confusion_matrix(cm),
        n_samples=n,
        class_names=class_names,
    )
