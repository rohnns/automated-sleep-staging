"""Subject-wise splitting by subject ID (no recording-level leakage)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from sleep_staging.acquisition.utils import SleepEDFFileIds


def sleep_edf_subject_key(ids: SleepEDFFileIds) -> str:
    """Canonical Sleep-EDF subject grouping key, e.g. ``SC400``.

    Uses study + series + subject so night/recording IDs never enter the key.
    """
    return f"{ids.study}{ids.series}{ids.subject_id}"


@dataclass(frozen=True, slots=True)
class SubjectSplit:
    """Disjoint subject ID partitions for train / validation / test."""

    train: tuple[str, ...]
    val: tuple[str, ...]
    test: tuple[str, ...]
    seed: int
    ratios: tuple[float, float, float]

    def __post_init__(self) -> None:
        sets = (set(self.train), set(self.val), set(self.test))
        if len(sets[0]) != len(self.train) or len(sets[1]) != len(self.val) or len(
            sets[2]
        ) != len(self.test):
            raise ValueError("SubjectSplit partitions must not contain duplicate IDs")
        if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
            raise ValueError("SubjectSplit partitions must be disjoint")

    @property
    def all_subjects(self) -> tuple[str, ...]:
        return (*self.train, *self.val, *self.test)

    def partition_for(self, subject_id: str) -> str:
        if subject_id in self.train:
            return "train"
        if subject_id in self.val:
            return "val"
        if subject_id in self.test:
            return "test"
        raise KeyError(f"Subject {subject_id!r} is not in this split")

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}


def subject_wise_split(
    subject_ids: Sequence[str],
    *,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> SubjectSplit:
    """Split unique subject IDs into train/val/test with a fixed seed.

    Grouping is by subject ID only. Recording / night identifiers must not be
    passed here. The same ``subject_ids`` + ``seed`` + ``ratios`` always yield
    the same partitions for every representation.
    """
    if len(ratios) != 3:
        raise ValueError("ratios must be (train, val, test)")
    if any(r < 0 for r in ratios) or sum(ratios) <= 0:
        raise ValueError("ratios must be non-negative and sum to a positive value")

    unique = sorted(set(subject_ids))
    if not unique:
        raise ValueError("subject_ids must be non-empty")

    rng = np.random.default_rng(seed)
    order = np.array(unique, dtype=object)
    rng.shuffle(order)

    n = len(order)
    train_r, val_r, _test_r = ratios
    total = float(sum(ratios))
    n_train = int(round(n * train_r / total))
    n_val = int(round(n * val_r / total))
    # Ensure every non-empty input gets at least an empty-safe partition and
    # leftover subjects go to test so counts sum to n.
    if n >= 3:
        n_train = min(max(n_train, 1), n - 2)
        n_val = min(max(n_val, 1), n - n_train - 1)
    elif n == 2:
        n_train, n_val = 1, 1
    else:
        n_train, n_val = 1, 0
    n_test = n - n_train - n_val

    train = tuple(str(x) for x in order[:n_train])
    val = tuple(str(x) for x in order[n_train : n_train + n_val])
    test = tuple(str(x) for x in order[n_train + n_val :])
    assert len(train) + len(val) + len(test) == n
    return SubjectSplit(
        train=train,
        val=val,
        test=test,
        seed=seed,
        ratios=(train_r / total, val_r / total, _test_r / total),
    )


def assert_no_subject_leakage(split: SubjectSplit) -> None:
    """Raise if any subject appears in more than one partition."""
    counts: dict[str, int] = {}
    for part in (split.train, split.val, split.test):
        for subject in part:
            counts[subject] = counts.get(subject, 0) + 1
    leaked = {sid: n for sid, n in counts.items() if n > 1}
    if leaked:
        raise AssertionError(f"Subject leakage detected: {leaked}")


def split_membership(split: SubjectSplit) -> Mapping[str, str]:
    """Map subject_id → {'train'|'val'|'test'}."""
    mapping: dict[str, str] = {}
    for name, subjects in split.as_dict().items():
        for subject in subjects:
            mapping[subject] = name
    return mapping
