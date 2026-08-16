"""Single classical baseline: BandPower → LogisticRegression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LogisticRegression

from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.metrics import ClassificationMetrics, compute_classification_metrics


@dataclass(frozen=True, slots=True)
class EpochPrediction:
    """Per-epoch prediction record preserving whole-night reconstruction keys."""

    subject_id: str
    recording_id: str
    epoch_index: int
    onset_sec: float | None
    target: int
    prediction: int


@dataclass(frozen=True, slots=True)
class ClassicalBaselineResult:
    """Result of the fixed classical baseline."""

    validation_metrics: ClassificationMetrics
    test_metrics: ClassificationMetrics | None
    predictions: tuple[EpochPrediction, ...]
    model: Any


def _features_and_labels(
    dataset: EpochDataset,
    *,
    ignore_index: int,
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    xs: list[np.ndarray] = []
    ys: list[int] = []
    for example in dataset.examples:
        if int(example.label) == ignore_index:
            continue
        xs.append(np.asarray(example.features, dtype=np.float64).reshape(-1))
        ys.append(int(example.label))
    if not xs:
        raise ValueError("Classical baseline requires at least one supervised example")
    return np.stack(xs, axis=0), np.asarray(ys, dtype=np.int64)


def _predict_with_metadata(
    model: LogisticRegression,
    dataset: EpochDataset,
    *,
    ignore_index: int,
) -> tuple[NDArray[np.int64], NDArray[np.int64], tuple[EpochPrediction, ...]]:
    xs: list[np.ndarray] = []
    kept = []
    for example in dataset.examples:
        if int(example.label) == ignore_index:
            continue
        xs.append(np.asarray(example.features, dtype=np.float64).reshape(-1))
        kept.append(example)
    if not xs:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64), ()
    y_pred = np.asarray(model.predict(np.stack(xs, axis=0)), dtype=np.int64)
    y_true = np.asarray([int(ex.label) for ex in kept], dtype=np.int64)
    predictions = tuple(
        EpochPrediction(
            subject_id=ex.subject_id,
            recording_id=ex.recording_id,
            epoch_index=ex.epoch_index,
            onset_sec=ex.onset_sec,
            target=int(ex.label),
            prediction=int(pred),
        )
        for ex, pred in zip(kept, y_pred.tolist(), strict=True)
    )
    return y_true, y_pred, predictions


def train_bandpower_logistic_regression(
    *,
    train_dataset: EpochDataset,
    val_dataset: EpochDataset,
    test_dataset: EpochDataset | None = None,
    ignore_index: int = -100,
    max_iter: int = 1000,
    evaluate_test: bool = False,
    random_state: int = 42,
) -> ClassicalBaselineResult:
    """Train the required single classical baseline on train subjects only."""
    x_train, y_train = _features_and_labels(train_dataset, ignore_index=ignore_index)
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=max_iter,
        random_state=random_state,
    )
    model.fit(x_train, y_train)

    y_val, pred_val, val_predictions = _predict_with_metadata(
        model, val_dataset, ignore_index=ignore_index
    )
    val_metrics = compute_classification_metrics(y_val, pred_val, ignore_index=ignore_index)

    test_metrics = None
    test_predictions: tuple[EpochPrediction, ...] = ()
    if evaluate_test:
        if test_dataset is None:
            raise ValueError("evaluate_test=True requires test_dataset")
        y_test, pred_test, test_predictions = _predict_with_metadata(
            model, test_dataset, ignore_index=ignore_index
        )
        test_metrics = compute_classification_metrics(y_test, pred_test, ignore_index=ignore_index)

    return ClassicalBaselineResult(
        validation_metrics=val_metrics,
        test_metrics=test_metrics,
        predictions=(*val_predictions, *test_predictions),
        model=model,
    )
