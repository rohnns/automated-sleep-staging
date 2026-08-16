"""Minimal controlled experiment runner.

This module intentionally consumes already-encoded collections. Full Sleep-EDF
preprocess/encode orchestration is left for a later script so tests and smoke
runs can verify the train/val/test/model infrastructure without touching all
197 recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from sleep_staging.representations.types import EncodedDatasetCollection
from sleep_staging.models.factory import build_baseline_model
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.split import SubjectSplit, assert_no_subject_leakage, subject_wise_split
from sleep_staging.training.trainer import TrainRecipe, TrainResult, train_baseline

REPRESENTATION_ORDER: tuple[str, ...] = ("raw", "bandpower", "time_frequency")


@dataclass(frozen=True, slots=True)
class RepresentationRunResult:
    representation: str
    split: SubjectSplit
    train_result: TrainResult


@dataclass(frozen=True, slots=True)
class ControlledExperimentResult:
    """Results for the controlled Raw/BandPower/STFT PyTorch bake-off."""

    split: SubjectSplit
    results: dict[str, RepresentationRunResult]


def build_shared_subject_split(
    collection: EncodedDatasetCollection,
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SubjectSplit:
    """Build one split reusable across Raw, BandPower, and STFT collections."""
    split = subject_wise_split(
        tuple(item.subject_id for item in collection.items), ratios=ratios, seed=seed
    )
    assert_no_subject_leakage(split)
    return split


def _assert_collection_subjects_match(
    collections: Mapping[str, EncodedDatasetCollection],
) -> None:
    expected: tuple[str, ...] | None = None
    for name, collection in collections.items():
        subjects = tuple(sorted({item.subject_id for item in collection.items}))
        if expected is None:
            expected = subjects
        elif subjects != expected:
            raise ValueError(
                f"Collection {name!r} has subjects {subjects}, expected {expected}. "
                "All representations must use the same subject set."
            )


def run_controlled_pytorch_experiments(
    collections: Mapping[str, EncodedDatasetCollection],
    *,
    split: SubjectSplit | None = None,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    recipe: TrainRecipe | None = None,
    checkpoint_root: Path | None = None,
    evaluate_test: bool = False,
    device: torch.device | None = None,
) -> ControlledExperimentResult:
    """Run RawCNN1D, BandPowerMLP, and STFTCNN2D with a shared split/recipe."""
    missing = set(REPRESENTATION_ORDER) - set(collections)
    if missing:
        raise ValueError(f"Missing encoded collections for representations: {sorted(missing)}")
    _assert_collection_subjects_match(collections)

    if split is None:
        split = build_shared_subject_split(collections["raw"], ratios=ratios, seed=seed)
    assert_no_subject_leakage(split)

    base_recipe = recipe or TrainRecipe(seed=seed)
    results: dict[str, RepresentationRunResult] = {}
    for representation in REPRESENTATION_ORDER:
        collection = collections[representation]
        if collection.items and collection.items[0].representation != representation:
            raise ValueError(
                f"Collection for {representation!r} reports "
                f"{collection.items[0].representation!r}"
            )
        train_ds = EpochDataset(collection, subject_ids=split.train, drop_ignore=False)
        val_ds = EpochDataset(collection, subject_ids=split.val, drop_ignore=False)
        test_ds = EpochDataset(collection, subject_ids=split.test, drop_ignore=False)
        in_channels = collection.items[0].metadata.feature_shape[0]
        n_band_features = (
            collection.items[0].metadata.feature_shape[-1]
            if representation == "bandpower"
            else 10
        )
        model = build_baseline_model(
            representation,
            in_channels=in_channels,
            n_band_features=n_band_features,
        )
        run_recipe = base_recipe
        if checkpoint_root is not None:
            run_recipe = TrainRecipe(
                seed=base_recipe.seed,
                batch_size=base_recipe.batch_size,
                max_epochs=base_recipe.max_epochs,
                learning_rate=base_recipe.learning_rate,
                weight_decay=base_recipe.weight_decay,
                num_workers=base_recipe.num_workers,
                ignore_index=base_recipe.ignore_index,
                class_weighting=base_recipe.class_weighting,
                early_stopping_patience=base_recipe.early_stopping_patience,
                checkpoint_dir=checkpoint_root / representation,
                checkpoint_name=base_recipe.checkpoint_name,
            )
        train_result = train_baseline(
            model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            recipe=run_recipe,
            device=device,
            evaluate_test=evaluate_test,
        )
        results[representation] = RepresentationRunResult(
            representation=representation,
            split=split,
            train_result=train_result,
        )
    return ControlledExperimentResult(split=split, results=results)
