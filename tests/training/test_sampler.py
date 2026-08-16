"""Tests for LocalityAwareSampler -- the memmap-thrashing fix.

All fixtures here are small, fully synthetic, in-memory EncodedDataset
collections. None of these tests touch the real ~8 GB Sleep-EDF cache.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch.utils.data import RandomSampler

from sleep_staging.representations.types import (
    EncodedDataset,
    EncodedDatasetCollection,
    RepresentationMetadata,
)
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.sampler import LocalityAwareSampler, group_indices_by_recording
from sleep_staging.training.trainer import make_loader


def _make_synthetic_collection(
    *,
    n_recordings: int = 6,
    epochs_per_recording: int = 20,
    n_channels: int = 3,
    n_times: int = 10,
) -> EncodedDatasetCollection:
    """Small in-memory EncodedDatasetCollection -- no real dataset needed."""
    metadata = RepresentationMetadata(
        representation="raw",
        channel_names=tuple(f"ch{i}" for i in range(n_channels)),
        sfreq=100.0,
        epoch_duration_sec=30.0,
        feature_shape=(n_channels, n_times),
    )
    items = []
    for rec in range(n_recordings):
        subject_id = f"{rec:02d}"
        features = np.full(
            (epochs_per_recording, n_channels, n_times), fill_value=float(rec), dtype=np.float32
        )
        labels = np.arange(epochs_per_recording, dtype=np.int64) % 5
        items.append(
            EncodedDataset(
                features=features,
                labels=labels,
                metadata=metadata,
                subject_id=subject_id,
                recording_id="1",
            )
        )
    return EncodedDatasetCollection(items=tuple(items))


def _torch_gen(seed: int) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Grouping correctness
# ---------------------------------------------------------------------------
def test_group_indices_by_recording_partitions_dataset_exactly() -> None:
    collection = _make_synthetic_collection(n_recordings=5, epochs_per_recording=7)
    dataset = EpochDataset(collection, drop_ignore=False)
    groups = group_indices_by_recording(dataset)

    assert len(groups) == 5
    all_indices = np.concatenate(groups)
    assert sorted(all_indices.tolist()) == list(range(len(dataset)))
    for g in groups:
        assert len(g) == 7
        assert (g[-1] - g[0]) == 6  # contiguous run


# ---------------------------------------------------------------------------
# Full coverage per epoch (no sample dropped, none duplicated)
# ---------------------------------------------------------------------------
def test_sampler_covers_every_index_exactly_once_per_epoch() -> None:
    collection = _make_synthetic_collection(n_recordings=9, epochs_per_recording=11)
    dataset = EpochDataset(collection, drop_ignore=False)
    sampler = LocalityAwareSampler(dataset, block_size=3, seed=123)

    for _ in range(3):  # simulate several training epochs
        seen = list(sampler)
        assert sorted(seen) == list(range(len(dataset)))
        assert len(seen) == len(dataset)


def test_sampler_reshuffles_every_epoch() -> None:
    """Real resampling each epoch, not a fixed/precomputed order."""
    collection = _make_synthetic_collection(n_recordings=10, epochs_per_recording=8)
    dataset = EpochDataset(collection, drop_ignore=False)
    sampler = LocalityAwareSampler(dataset, block_size=2, seed=7)

    orders = [list(sampler) for _ in range(3)]
    assert orders[0] != orders[1]
    assert orders[1] != orders[2]


def test_sampler_deterministic_given_same_seed() -> None:
    collection = _make_synthetic_collection(n_recordings=6, epochs_per_recording=5)
    dataset = EpochDataset(collection, drop_ignore=False)

    sampler_a = LocalityAwareSampler(dataset, block_size=2, seed=99)
    sampler_b = LocalityAwareSampler(dataset, block_size=2, seed=99)
    for _ in range(2):
        assert list(sampler_a) == list(sampler_b)


def test_invalid_block_size_rejected() -> None:
    collection = _make_synthetic_collection(n_recordings=2, epochs_per_recording=3)
    dataset = EpochDataset(collection, drop_ignore=False)
    with pytest.raises(ValueError, match="block_size"):
        LocalityAwareSampler(dataset, block_size=0)


# ---------------------------------------------------------------------------
# The core fix: locality vs. global scatter
# ---------------------------------------------------------------------------
def test_sampler_is_locality_aware_vs_global_random_sampler() -> None:
    """block_size=1: every batch is drawn from exactly one recording.

    Contrast with plain RandomSampler over the same dataset, which scatters
    across essentially all recordings per batch -- this is the actual
    mechanism behind the observed memmap thrashing (every batch could touch
    up to batch_size distinct on-disk recording files).
    """
    n_recordings = 20
    epochs_per_recording = 32
    collection = _make_synthetic_collection(
        n_recordings=n_recordings, epochs_per_recording=epochs_per_recording
    )
    dataset = EpochDataset(collection, drop_ignore=False)
    recording_of = [dataset.examples[i].subject_id for i in range(len(dataset))]
    batch_size = 8

    sampler = LocalityAwareSampler(dataset, block_size=1, seed=0)
    indices = list(sampler)
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    distinct_per_batch = [len({recording_of[idx] for idx in batch}) for batch in batches]
    assert max(distinct_per_batch) == 1

    rng_indices = list(RandomSampler(dataset, generator=_torch_gen(0)))
    rng_batches = [rng_indices[i : i + batch_size] for i in range(0, len(rng_indices), batch_size)]
    rng_distinct = [len({recording_of[idx] for idx in batch}) for batch in rng_batches]
    assert sum(rng_distinct) / len(rng_distinct) > 1.5  # clearly scattered vs. ==1


def test_sampler_block_size_bounds_working_set() -> None:
    """Across any run of consecutive indices matching one block's length, at
    most block_size distinct recordings are touched -- the actual "hot
    working set" bound. Window is aligned to the true block length (all
    synthetic recordings have equal epoch counts here, so
    epochs_per_recording * block_size is exactly one block) rather than an
    arbitrary size, since an unaligned window can straddle a block boundary
    and spuriously touch one extra recording.
    """
    n_recordings = 12
    epochs_per_recording = 10
    collection = _make_synthetic_collection(
        n_recordings=n_recordings, epochs_per_recording=epochs_per_recording
    )
    dataset = EpochDataset(collection, drop_ignore=False)
    recording_of = [dataset.examples[i].subject_id for i in range(len(dataset))]

    for block_size in (1, 2, 4):
        sampler = LocalityAwareSampler(dataset, block_size=block_size, seed=1)
        indices = list(sampler)
        window = epochs_per_recording * block_size
        max_distinct = 0
        for i in range(0, len(indices), window):
            chunk = indices[i : i + window]
            max_distinct = max(max_distinct, len({recording_of[idx] for idx in chunk}))
        assert max_distinct <= block_size


# ---------------------------------------------------------------------------
# DataLoader wiring
# ---------------------------------------------------------------------------
def test_make_loader_uses_locality_sampler_when_shuffle_true() -> None:
    collection = _make_synthetic_collection(n_recordings=4, epochs_per_recording=6)
    dataset = EpochDataset(collection, drop_ignore=False)

    loader = make_loader(dataset, batch_size=4, shuffle=True, seed=0, block_size=2)
    assert isinstance(loader.sampler, LocalityAwareSampler)

    seen_labels = []
    for batch in loader:
        seen_labels.extend(batch["label"].tolist())
    assert len(seen_labels) == len(dataset)


def test_make_loader_val_test_unaffected_by_sampler_change() -> None:
    """shuffle=False (validation/test) never used LocalityAwareSampler and
    still doesn't -- only the training loader's sampling strategy changed.
    """
    collection = _make_synthetic_collection(n_recordings=3, epochs_per_recording=5)
    dataset = EpochDataset(collection, drop_ignore=False)
    loader = make_loader(dataset, batch_size=4, shuffle=False, seed=0)
    assert not isinstance(loader.sampler, LocalityAwareSampler)

    seen_labels = []
    for batch in loader:
        seen_labels.extend(batch["label"].tolist())
    assert len(seen_labels) == len(dataset)


# ---------------------------------------------------------------------------
# Split integrity: sampler cannot introduce train/val/test leakage
# ---------------------------------------------------------------------------
def test_train_val_test_split_remains_disjoint_with_new_sampler() -> None:
    """The sampler only reorders indices within one EpochDataset. Subject-
    level train/val/test separation is enforced entirely upstream by
    EpochDataset(subject_ids=...) filtering; this proves the sampler cannot
    leak across that boundary regardless of block_size.
    """
    collection = _make_synthetic_collection(n_recordings=10, epochs_per_recording=6)
    train_subjects = tuple(f"{i:02d}" for i in range(6))
    val_subjects = tuple(f"{i:02d}" for i in range(6, 8))
    test_subjects = tuple(f"{i:02d}" for i in range(8, 10))

    train_ds = EpochDataset(collection, subject_ids=train_subjects, drop_ignore=False)
    val_ds = EpochDataset(collection, subject_ids=val_subjects, drop_ignore=False)
    test_ds = EpochDataset(collection, subject_ids=test_subjects, drop_ignore=False)

    train_seen = {ex.subject_id for ex in train_ds.examples}
    val_seen = {ex.subject_id for ex in val_ds.examples}
    test_seen = {ex.subject_id for ex in test_ds.examples}
    assert not (train_seen & val_seen)
    assert not (train_seen & test_seen)
    assert not (val_seen & test_seen)

    sampler = LocalityAwareSampler(train_ds, block_size=2, seed=0)
    sampled_subjects = {train_ds.examples[i].subject_id for i in sampler}
    assert sampled_subjects == train_seen
    assert not (sampled_subjects & val_seen)
    assert not (sampled_subjects & test_seen)
