"""Locality-aware epoch sampler for memory-mapped training datasets.

``EpochDataset`` (see ``training/dataset.py``) is typically backed by
per-recording memory-mapped ``.npy`` sidecars (see
``sc_to_st_cache._load_features_memmapped``): each Sleep-EDF recording's
features live in its own on-disk array, and touching a recording's memmap
for the first time in a while requires the OS to page it in from disk.

PyTorch's default ``DataLoader(shuffle=True)`` uses ``RandomSampler``, which
draws indices uniformly at random across the *entire* flat dataset. For a
55-recording SC training split that means every single batch can touch up
to ``batch_size`` different recordings' memmaps, in a completely different
random combination each batch, for the whole epoch. The "hot" working set
needed to avoid page faults is then effectively *all* recordings at once --
observed in practice as CPU pegged near 100%, GPU utilization near 0%, and
epoch time growing from ~30s to a multi-minute stall once free RAM ran low
enough that the OS could no longer keep that whole working set cached.

``LocalityAwareSampler`` fixes this by keeping full stochastic per-epoch
coverage (every example is visited exactly once per epoch; both recording
order and within-block window order are reshuffled every epoch -- true
resampling, not a fixed/precomputed order) while bounding how many distinct
recordings are "hot" at any moment to a small, configurable ``block_size``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import numpy as np
from torch.utils.data import Sampler

if TYPE_CHECKING:
    from sleep_staging.training.dataset import EpochDataset


def group_indices_by_recording(dataset: "EpochDataset") -> tuple[np.ndarray, ...]:
    """Partition ``range(len(dataset))`` into contiguous per-recording index runs.

    ``EpochDataset.examples`` is built by iterating source recordings in
    collection order and, for each, its epochs in increasing ``epoch_index``
    order (see ``iter_epoch_examples``) -- so examples from the same
    recording are always contiguous in ``dataset.examples``. This groups by
    exactly that contiguity, keyed on ``(subject_id, recording_id)`` so
    distinct recordings are never merged even though a bare ``recording_id``
    (e.g. a Sleep-EDF night digit like ``"1"``) repeats across subjects.
    """
    examples = dataset.examples
    if not examples:
        return ()

    groups: list[np.ndarray] = []
    current_key = (examples[0].subject_id, examples[0].recording_id)
    start = 0
    for i in range(1, len(examples)):
        key = (examples[i].subject_id, examples[i].recording_id)
        if key != current_key:
            groups.append(np.arange(start, i, dtype=np.int64))
            current_key = key
            start = i
    groups.append(np.arange(start, len(examples), dtype=np.int64))
    return tuple(groups)


class LocalityAwareSampler(Sampler[int]):
    """Epoch-shuffled sampler bounding the "hot" memmap working set to a block of recordings.

    Each time the sampler is iterated (once per training epoch -- PyTorch's
    ``DataLoader`` calls ``iter()`` on its sampler fresh every time
    ``for batch in loader:`` starts, so no explicit ``set_epoch()`` call is
    needed):

    1. The order of *recordings* (contiguous index groups) is reshuffled.
    2. Recordings are chunked into blocks of ``block_size`` consecutive
       (in that shuffled order) recordings.
    3. Within each block, the pooled window indices are shuffled.
    4. Indices are yielded block by block, in that shuffled intra-block order.

    Every index in ``range(len(dataset))`` is yielded exactly once per
    epoch -- full coverage, no duplication, no leakage -- only the *order*
    changes relative to a plain global ``RandomSampler``. ``block_size=1``
    is maximally local (one recording "hot" at a time); larger values trade
    some locality for batch composition closer to a fully global shuffle.
    """

    def __init__(
        self,
        dataset: "EpochDataset",
        *,
        block_size: int = 4,
        seed: int = 42,
    ) -> None:
        if block_size < 1:
            raise ValueError("block_size must be >= 1")
        self.dataset = dataset
        self.block_size = int(block_size)
        self.seed = int(seed)
        self._groups = group_indices_by_recording(dataset)
        self._call_count = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self._call_count)
        self._call_count += 1

        if not self._groups:
            return

        order = np.arange(len(self._groups))
        rng.shuffle(order)

        for start in range(0, len(order), self.block_size):
            block_group_indices = order[start : start + self.block_size]
            block = np.concatenate([self._groups[g] for g in block_group_indices])
            rng.shuffle(block)
            yield from block.tolist()
