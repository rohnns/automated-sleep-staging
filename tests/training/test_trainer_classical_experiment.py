from __future__ import annotations

from pathlib import Path

import torch

from tests.phase4_helpers import make_collection
from sleep_staging.models import BandPowerMLP
from sleep_staging.training.classical import train_bandpower_logistic_regression
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.experiment import build_shared_subject_split, run_controlled_pytorch_experiments
from sleep_staging.training.trainer import TrainRecipe, compute_class_weights_from_dataset, select_device, train_baseline


def test_class_weights_are_train_only() -> None:
    collection = make_collection("bandpower", subjects=("S1",), n_epochs=6)
    ds = EpochDataset(collection)
    weights = compute_class_weights_from_dataset(ds)
    assert weights.shape == (5,)
    assert torch.isfinite(weights).all()
    assert weights[-1] > 0 or weights[-1] == 0


def test_trainer_checkpoint_and_early_stopping(tmp_path: Path) -> None:
    collection = make_collection("bandpower", subjects=("S1", "S2", "S3"), n_epochs=8)
    train_ds = EpochDataset(collection, subject_ids=("S1",))
    val_ds = EpochDataset(collection, subject_ids=("S2",))
    test_ds = EpochDataset(collection, subject_ids=("S3",))
    recipe = TrainRecipe(
        seed=1,
        batch_size=4,
        max_epochs=3,
        learning_rate=1e-2,
        early_stopping_patience=1,
        checkpoint_dir=tmp_path,
    )
    result = train_baseline(
        BandPowerMLP(in_channels=1, n_features=10),
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        recipe=recipe,
        device=torch.device("cpu"),
        evaluate_test=True,
    )
    assert result.checkpoint_path is not None
    assert result.checkpoint_path.exists()
    assert result.class_weights is not None
    assert result.test_metrics is not None
    assert len(result.history) <= recipe.max_epochs


def test_select_device_returns_supported_kind() -> None:
    assert select_device().type in {"cuda", "mps", "cpu"}


def test_classical_bandpower_baseline_preserves_prediction_metadata() -> None:
    collection = make_collection("bandpower", subjects=("S1", "S2", "S3"), n_epochs=8)
    result = train_bandpower_logistic_regression(
        train_dataset=EpochDataset(collection, subject_ids=("S1",)),
        val_dataset=EpochDataset(collection, subject_ids=("S2",)),
        test_dataset=EpochDataset(collection, subject_ids=("S3",)),
        evaluate_test=True,
        max_iter=200,
    )
    assert result.validation_metrics.n_samples > 0
    assert result.test_metrics is not None
    assert result.predictions
    pred = result.predictions[0]
    assert pred.subject_id in {"S2", "S3"}
    assert pred.recording_id.endswith("R1")
    assert pred.onset_sec is not None


def test_controlled_runner_reuses_same_subject_split(tmp_path: Path) -> None:
    subjects = ("S1", "S2", "S3")
    collections = {
        "raw": make_collection("raw", subjects=subjects, n_epochs=4),
        "bandpower": make_collection("bandpower", subjects=subjects, n_epochs=4),
        "time_frequency": make_collection("time_frequency", subjects=subjects, n_epochs=4),
    }
    split = build_shared_subject_split(collections["raw"], ratios=(1, 1, 1), seed=2)
    recipe = TrainRecipe(
        seed=2,
        batch_size=2,
        max_epochs=1,
        early_stopping_patience=None,
        checkpoint_dir=None,
    )
    result = run_controlled_pytorch_experiments(
        collections,
        split=split,
        recipe=recipe,
        checkpoint_root=tmp_path,
        device=torch.device("cpu"),
    )
    assert set(result.results) == {"raw", "bandpower", "time_frequency"}
    assert all(run.split is split for run in result.results.values())
    assert (tmp_path / "raw" / "best_model.pt").exists()
