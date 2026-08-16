"""Fixed-recipe trainer for the representation baselines."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.training.dataset import EpochDataset, collate_epoch_batch
from sleep_staging.training.metrics import ClassificationMetrics, compute_classification_metrics
from sleep_staging.training.sampler import LocalityAwareSampler

logger = get_logger(__name__)

IGNORE_INDEX = -100


@dataclass(frozen=True, slots=True)
class TrainRecipe:
    """One training recipe, shared unchanged by every representation."""

    seed: int = 42
    batch_size: int = 32
    max_epochs: int = 20
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    num_workers: int = 0
    ignore_index: int = IGNORE_INDEX
    class_weighting: str = "balanced"
    early_stopping_patience: int | None = 5
    checkpoint_dir: Path | None = None
    checkpoint_name: str = "best_model.pt"
    block_size: int = 4


@dataclass(frozen=True, slots=True)
class EpochResult:
    loss: float
    metrics: ClassificationMetrics


@dataclass(frozen=True, slots=True)
class TrainResult:
    history: tuple[dict[str, float], ...]
    best_epoch: int
    best_val_macro_f1: float
    best_state_dict: dict[str, torch.Tensor]
    test_metrics: ClassificationMetrics | None
    checkpoint_path: Path | None = None
    device: str = "cpu"
    class_weights: tuple[float, ...] | None = None
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Select CUDA, then MPS, then CPU for PyTorch training/inference."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_class_weights_from_dataset(
    dataset: EpochDataset,
    *,
    n_classes: int = 5,
    ignore_index: int = IGNORE_INDEX,
) -> torch.Tensor:
    """Balanced class weights computed from TRAIN examples only."""
    counts = np.zeros(n_classes, dtype=np.float64)
    for example in dataset.examples:
        if 0 <= int(example.label) < n_classes and int(example.label) != ignore_index:
            counts[int(example.label)] += 1.0
    supervised = float(counts.sum())
    weights = np.ones(n_classes, dtype=np.float32)
    nonzero = counts > 0
    if supervised > 0 and np.any(nonzero):
        weights[nonzero] = supervised / (float(n_classes) * counts[nonzero])
        weights[~nonzero] = 0.0
    return torch.as_tensor(weights, dtype=torch.float32)


def autocast_for_device(device: torch.device):
    """bf16 autocast on CUDA (Ada/RTX-40-series supports bf16 tensor cores
    natively, so no GradScaler is needed -- bf16 has fp32's exponent range,
    just less mantissa precision). No-op elsewhere (CPU/MPS), so behavior on
    those devices -- including existing CPU-run tests -- is unchanged.
    """
    enabled = device.type == "cuda"
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=enabled)


def _log_cuda_memory(device: torch.device, *, tag: str) -> None:
    if device.type != "cuda":
        return
    logger.info(
        "%s: cuda memory allocated=%.1fMB reserved=%.1fMB max_allocated=%.1fMB",
        tag,
        torch.cuda.memory_allocated(device) / 1e6,
        torch.cuda.memory_reserved(device) / 1e6,
        torch.cuda.max_memory_allocated(device) / 1e6,
    )


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        "features": batch["features"].to(device),
        "label": batch["label"].to(device),
        "target": batch.get("target", batch["label"]).to(device),
        "subject_id": batch["subject_id"],
        "recording_id": batch["recording_id"],
        "epoch_index": batch["epoch_index"],
        "onset_sec": batch.get("onset_sec"),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    ignore_index: int = IGNORE_INDEX,
    criterion: nn.Module | None = None,
) -> EpochResult:
    model.eval()
    if criterion is None:
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    total_loss = 0.0
    total_weight = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        with autocast_for_device(device):
            logits = model(batch["features"])
            if logits.ndim != 2 or logits.shape[1] != 5:
                raise RuntimeError(f"Expected logits (B, 5), got {tuple(logits.shape)}")
            labels = batch["label"]
            supervised = labels != ignore_index
            n_sup = int(supervised.sum().item())
            if n_sup > 0:
                loss = criterion(logits, labels)

        if n_sup > 0:
            total_loss += float(loss.item()) * n_sup
            total_weight += n_sup

        preds = torch.argmax(logits, dim=1)
        y_true.extend(int(v) for v in labels.detach().cpu().tolist())
        y_pred.extend(int(v) for v in preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, ignore_index=ignore_index)
    avg_loss = total_loss / max(total_weight, 1)
    return EpochResult(loss=avg_loss, metrics=metrics)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    criterion: nn.Module,
    ignore_index: int = IGNORE_INDEX,
) -> EpochResult:
    model.train()
    total_loss = 0.0
    total_weight = 0
    y_true: list[int] = []
    y_pred: list[int] = []

    for batch in loader:
        batch = _move_batch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_for_device(device):
            logits = model(batch["features"])
            if logits.ndim != 2 or logits.shape[1] != 5:
                raise RuntimeError(f"Expected logits (B, 5), got {tuple(logits.shape)}")
            labels = batch["label"]
            supervised = labels != ignore_index
            n_sup = int(supervised.sum().item())
            if n_sup == 0:
                continue
            loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.item()) * n_sup
        total_weight += n_sup
        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(int(v) for v in labels.detach().cpu().tolist())
        y_pred.extend(int(v) for v in preds.detach().cpu().tolist())

    metrics = compute_classification_metrics(y_true, y_pred, ignore_index=ignore_index)
    return EpochResult(loss=total_loss / max(total_weight, 1), metrics=metrics)


def make_loader(
    dataset: EpochDataset,
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int = 0,
    block_size: int = 4,
) -> DataLoader:
    """Build a DataLoader.

    When ``shuffle=True`` this uses ``LocalityAwareSampler`` instead of
    PyTorch's default ``RandomSampler`` -- see ``training/sampler.py`` for
    why: a fully global shuffle over a memory-mapped dataset causes OS
    page-cache thrashing once the dataset no longer fits comfortably in free
    RAM. ``shuffle=False`` (validation/test) is unaffected -- it already
    iterates in the dataset's natural, already-recording-contiguous order,
    which was never the thrashing source.
    """
    if shuffle:
        sampler = LocalityAwareSampler(dataset, block_size=block_size, seed=seed)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_epoch_batch,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_epoch_batch,
    )


def _save_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    epoch: int,
    val_macro_f1: float,
    recipe: TrainRecipe,
    class_weights: torch.Tensor | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "epoch": epoch,
            "val_macro_f1": val_macro_f1,
            "recipe": recipe,
            "class_weights": None if class_weights is None else class_weights.detach().cpu(),
        },
        path,
    )


def train_baseline(
    model: nn.Module,
    *,
    train_dataset: EpochDataset,
    val_dataset: EpochDataset,
    test_dataset: EpochDataset | None = None,
    recipe: TrainRecipe | None = None,
    device: torch.device | None = None,
    evaluate_test: bool = False,
) -> TrainResult:
    """Train with a fixed recipe; select checkpoint by validation macro-F1.

    Class weights are computed from ``train_dataset`` only. The test set is
    evaluated only when ``evaluate_test=True`` and never used for checkpoint
    selection, early stopping, or weighting.
    """
    recipe = recipe or TrainRecipe()
    device = device or select_device()
    set_seed(recipe.seed)
    model = model.to(device)

    logger.info(
        "train_baseline: dataset lengths train=%d val=%d test=%s",
        len(train_dataset),
        len(val_dataset),
        "n/a" if test_dataset is None else len(test_dataset),
    )
    if len(train_dataset) == 0:
        raise ValueError(
            "train_dataset has 0 examples -- cannot build a DataLoader "
            "(torch.utils.data.RandomSampler requires num_samples > 0). "
            "This is almost always a subject-id/split mismatch upstream "
            "(the split builder's subject keys don't match "
            "EncodedDataset.subject_id), not a trainer bug."
        )
    if len(val_dataset) == 0:
        raise ValueError(
            "val_dataset has 0 examples -- cannot build a DataLoader. "
            "Check the val split's subject keys against EncodedDataset.subject_id."
        )

    train_loader = make_loader(
        train_dataset,
        batch_size=recipe.batch_size,
        shuffle=True,
        seed=recipe.seed,
        num_workers=recipe.num_workers,
        block_size=recipe.block_size,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=recipe.batch_size,
        shuffle=False,
        seed=recipe.seed,
        num_workers=recipe.num_workers,
    )

    class_weights = None
    if recipe.class_weighting == "balanced":
        class_weights = compute_class_weights_from_dataset(
            train_dataset, ignore_index=recipe.ignore_index
        ).to(device)
    elif recipe.class_weighting not in {"none", "off", ""}:
        raise ValueError("class_weighting must be 'balanced' or 'none'")

    criterion = nn.CrossEntropyLoss(weight=class_weights, ignore_index=recipe.ignore_index)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=recipe.learning_rate,
        weight_decay=recipe.weight_decay,
    )

    history: list[dict[str, float]] = []
    best_epoch = -1
    best_val_macro_f1 = float("-inf")
    best_state: dict[str, torch.Tensor] | None = None
    checkpoint_path = None
    epochs_without_improvement = 0
    stopped_early = False

    if recipe.checkpoint_dir is not None:
        checkpoint_path = Path(recipe.checkpoint_dir) / recipe.checkpoint_name

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(recipe.max_epochs):
        try:
            train_result = train_one_epoch(
                model,
                train_loader,
                optimizer=optimizer,
                device=device,
                criterion=criterion,
                ignore_index=recipe.ignore_index,
            )
            val_result = evaluate(
                model,
                val_loader,
                device=device,
                ignore_index=recipe.ignore_index,
                criterion=criterion,
            )
        except RuntimeError as exc:
            # torch's OOM exception type has changed across versions
            # (torch.cuda.OutOfMemoryError vs. torch.AcceleratorError, both
            # RuntimeError subclasses) -- match on message instead of a
            # specific class so this stays correct across torch versions.
            if device.type != "cuda" or "out of memory" not in str(exc).lower():
                raise
            # Diagnose (don't swallow): dump the allocator's own breakdown of
            # what actually holds memory at the moment of failure, then
            # re-raise so the run still fails loudly.
            logger.error(
                "CUDA OOM at epoch=%d. Allocator summary:\n%s",
                epoch,
                torch.cuda.memory_summary(device),
            )
            raise

        _log_cuda_memory(device, tag=f"epoch={epoch} end")
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        overfit_gap = train_result.loss - val_result.loss
        row = {
            "epoch": float(epoch),
            "train_loss": train_result.loss,
            "train_macro_f1": train_result.metrics.macro_f1,
            "val_loss": val_result.loss,
            "val_macro_f1": val_result.metrics.macro_f1,
            "loss_gap_train_minus_val": overfit_gap,
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.4f val_loss=%.4f val_macro_f1=%.4f",
            epoch,
            train_result.loss,
            val_result.loss,
            val_result.metrics.macro_f1,
        )
        if val_result.metrics.macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_result.metrics.macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if checkpoint_path is not None:
                _save_checkpoint(
                    path=checkpoint_path,
                    model=model,
                    epoch=epoch,
                    val_macro_f1=best_val_macro_f1,
                    recipe=recipe,
                    class_weights=class_weights,
                )
        else:
            epochs_without_improvement += 1
            if (
                recipe.early_stopping_patience is not None
                and recipe.early_stopping_patience >= 0
                and epochs_without_improvement >= recipe.early_stopping_patience
            ):
                stopped_early = True
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = max(len(history) - 1, 0)
        best_val_macro_f1 = history[-1]["val_macro_f1"] if history else 0.0

    model.load_state_dict(best_state)

    test_metrics = None
    if evaluate_test:
        if test_dataset is None:
            raise ValueError("evaluate_test=True requires test_dataset")
        test_loader = make_loader(
            test_dataset,
            batch_size=recipe.batch_size,
            shuffle=False,
            seed=recipe.seed,
            num_workers=recipe.num_workers,
        )
        test_metrics = evaluate(
            model,
            test_loader,
            device=device,
            ignore_index=recipe.ignore_index,
            criterion=criterion,
        ).metrics

    class_weight_tuple = None
    if class_weights is not None:
        class_weight_tuple = tuple(float(x) for x in class_weights.detach().cpu().tolist())

    return TrainResult(
        history=tuple(history),
        best_epoch=best_epoch,
        best_val_macro_f1=best_val_macro_f1,
        best_state_dict=best_state,
        test_metrics=test_metrics,
        checkpoint_path=checkpoint_path,
        device=str(device),
        class_weights=class_weight_tuple,
        stopped_early=stopped_early,
    )
