"""Torch dataset over encoded epochs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator, Sequence

import numpy as np
import torch
from numpy.typing import NDArray
from torch.utils.data import Dataset

from sleep_staging.representations.types import EncodedDataset, EncodedDatasetCollection
from sleep_staging.training.split import SubjectSplit


@dataclass(frozen=True, slots=True)
class EpochExample:
    """One training/eval epoch with identity metadata preserved."""

    features: NDArray[np.floating]
    label: int
    subject_id: str
    recording_id: str
    epoch_index: int
    onset_sec: float | None = None


@dataclass(frozen=True, slots=True)
class SplitEpochStats:
    """Subject / epoch counts after optional IGNORE filtering."""

    n_subjects: int
    n_epochs: int
    n_epochs_ignore: int
    n_epochs_supervised: int
    subjects: tuple[str, ...]


def iter_epoch_examples(
    collection: EncodedDatasetCollection | Sequence[EncodedDataset],
    *,
    drop_ignore: bool = False,
) -> Iterator[EpochExample]:
    """Yield epoch examples while preserving subject/recording/epoch identity."""
    items = collection.items if isinstance(collection, EncodedDatasetCollection) else collection
    for encoded in items:
        labels = np.asarray(encoded.labels, dtype=np.int64)
        onsets = encoded.onsets_sec
        for epoch_index in range(encoded.n_epochs):
            label = int(labels[epoch_index])
            if drop_ignore and label == encoded.ignore_index:
                continue
            onset_sec = None if onsets is None else float(onsets[epoch_index])
            yield EpochExample(
                features=np.asarray(encoded.features[epoch_index]),
                label=label,
                subject_id=str(encoded.subject_id),
                recording_id=str(encoded.recording_id),
                epoch_index=int(epoch_index),
                onset_sec=onset_sec,
            )


def filter_collection_by_subjects(
    collection: EncodedDatasetCollection | Sequence[EncodedDataset],
    subject_ids: Sequence[str],
) -> tuple[EncodedDataset, ...]:
    wanted = set(subject_ids)
    items = collection.items if isinstance(collection, EncodedDatasetCollection) else collection
    return tuple(item for item in items if item.subject_id in wanted)


def summarize_partition(
    collection: EncodedDatasetCollection | Sequence[EncodedDataset],
    subject_ids: Sequence[str],
    *,
    ignore_index: int = -100,
) -> SplitEpochStats:
    """Count subjects/epochs for a subject partition (IGNORE tallied separately)."""
    subset = filter_collection_by_subjects(collection, subject_ids)
    n_epochs = 0
    n_ignore = 0
    subjects: set[str] = set()
    for encoded in subset:
        subjects.add(encoded.subject_id)
        labels = np.asarray(encoded.labels, dtype=np.int64)
        n_epochs += int(labels.size)
        n_ignore += int(np.sum(labels == ignore_index))
    return SplitEpochStats(
        n_subjects=len(subjects),
        n_epochs=n_epochs,
        n_epochs_ignore=n_ignore,
        n_epochs_supervised=n_epochs - n_ignore,
        subjects=tuple(sorted(subjects)),
    )


def summarize_split(
    collection: EncodedDatasetCollection | Sequence[EncodedDataset],
    split: SubjectSplit,
    *,
    ignore_index: int = -100,
) -> dict[str, SplitEpochStats]:
    return {
        "train": summarize_partition(collection, split.train, ignore_index=ignore_index),
        "val": summarize_partition(collection, split.val, ignore_index=ignore_index),
        "test": summarize_partition(collection, split.test, ignore_index=ignore_index),
    }


class EpochDataset(Dataset):
    """Flat epoch dataset built from encoded recordings.

    By default IGNORE epochs are kept in the index so signal/label alignment
    stays inspectable; the training loss uses ``ignore_index=-100`` and metrics
    exclude IGNORE. Set ``drop_ignore=True`` only when a loader should never see
    IGNORE rows.
    """

    def __init__(
        self,
        collection: EncodedDatasetCollection | Sequence[EncodedDataset],
        *,
        subject_ids: Sequence[str] | None = None,
        drop_ignore: bool = False,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if subject_ids is not None:
            items = filter_collection_by_subjects(collection, subject_ids)
        else:
            items = (
                collection.items
                if isinstance(collection, EncodedDatasetCollection)
                else tuple(collection)
            )
        self.examples: tuple[EpochExample, ...] = tuple(
            iter_epoch_examples(items, drop_ignore=drop_ignore)
        )
        self.dtype = dtype
        self.drop_ignore = drop_ignore

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        # np.array(..., copy=True) materializes just this one epoch's slice
        # (tens of KB) out of a possibly memory-mapped, read-only backing
        # array (see sc_to_st_cache._load_features_memmapped), producing a
        # normal writable array for torch instead of aliasing mmap pages.
        return {
            "features": torch.as_tensor(np.array(example.features), dtype=self.dtype),
            "label": torch.tensor(example.label, dtype=torch.long),
            "target": torch.tensor(example.label, dtype=torch.long),
            "subject_id": example.subject_id,
            "recording_id": example.recording_id,
            "epoch_index": example.epoch_index,
            "onset_sec": example.onset_sec,
        }


def collate_epoch_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-free epoch dicts into a training batch."""
    return {
        "features": torch.stack([item["features"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
        "target": torch.stack([item["target"] for item in batch], dim=0),
        "subject_id": [item["subject_id"] for item in batch],
        "recording_id": [item["recording_id"] for item in batch],
        "epoch_index": torch.tensor(
            [item["epoch_index"] for item in batch], dtype=torch.long
        ),
        "onset_sec": [item["onset_sec"] for item in batch],
    }
