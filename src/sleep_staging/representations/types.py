"""Core data containers for the representations stage.

Encoders never receive MNE ``Raw`` objects. Preprocessing owns MNE;
this module consumes NumPy epoch windows plus integer labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Literal, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray

RepresentationType = Literal["raw", "bandpower", "time_frequency"]


@dataclass(frozen=True, slots=True)
class LabelVocabulary:
    """Stable AASM string ↔ int mapping shared by encodings and models."""

    stages: tuple[str, ...] = ("W", "N1", "N2", "N3", "REM")
    ignore_label: str = "IGNORE"
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("LabelVocabulary.stages must be unique")
        if self.ignore_label in self.stages:
            raise ValueError("ignore_label must not appear in stages")

    @property
    def n_classes(self) -> int:
        return len(self.stages)

    def to_index(self, label: str) -> int:
        if label == self.ignore_label:
            return self.ignore_index
        try:
            return self.stages.index(label)
        except ValueError as exc:
            raise KeyError(f"Unknown stage label: {label!r}") from exc

    def encode(self, labels: Sequence[str]) -> NDArray[np.int64]:
        return np.asarray([self.to_index(label) for label in labels], dtype=np.int64)

    def decode(self, indices: NDArray[np.int64] | Sequence[int]) -> tuple[str, ...]:
        decoded: list[str] = []
        for index in indices:
            value = int(index)
            if value == self.ignore_index:
                decoded.append(self.ignore_label)
            else:
                decoded.append(self.stages[value])
        return tuple(decoded)


@dataclass(frozen=True, slots=True)
class EpochTensorBatch:
    """Encoder input: one recording's fixed-length multichannel epochs.

    Shape contract
    --------------
    signals : ``(N, C, T)`` with ``T = round(epoch_duration_sec * sfreq)``
              (3000 samples for 30 s @ 100 Hz).
    labels  : ``(N,)`` int64; AASM classes or ``ignore_index``.
    """

    signals: NDArray[np.floating]
    labels: NDArray[np.int64]
    onsets_sec: NDArray[np.floating]
    channel_names: tuple[str, ...]
    sfreq: float
    epoch_duration_sec: float
    subject_id: str
    recording_id: str
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.signals.ndim != 3:
            raise ValueError("signals must have shape (N, C, T)")
        n_epochs, n_channels, _n_times = self.signals.shape
        if self.labels.shape != (n_epochs,):
            raise ValueError("labels must have shape (N,)")
        if self.onsets_sec.shape != (n_epochs,):
            raise ValueError("onsets_sec must have shape (N,)")
        if len(self.channel_names) != n_channels:
            raise ValueError("channel_names length must match signals axis 1")
        if self.sfreq <= 0:
            raise ValueError("sfreq must be positive")
        if self.epoch_duration_sec <= 0:
            raise ValueError("epoch_duration_sec must be positive")

    @property
    def n_epochs(self) -> int:
        return int(self.signals.shape[0])

    @property
    def n_channels(self) -> int:
        return int(self.signals.shape[1])

    @property
    def n_times(self) -> int:
        return int(self.signals.shape[2])


@dataclass(frozen=True, slots=True)
class RepresentationMetadata:
    """Metadata describing how an encoded tensor was produced.

    Downstream models should read layout from here rather than hardcoding
    shapes. Frequency axes are optional and only set for TF / band-power.
    """

    representation: RepresentationType
    channel_names: tuple[str, ...]
    sfreq: float
    epoch_duration_sec: float
    feature_shape: tuple[int, ...]
    """Shape of one epoch's features (no leading N)."""

    algorithm: str | None = None
    """Backend name for TF: ``"stft"`` / ``"cwt"``; unused for raw."""

    freqs_hz: tuple[float, ...] | None = None
    """Frequency-bin centers (Hz) for TF images, length F."""

    times_sec: tuple[float, ...] | None = None
    """Frame centers (s) within the epoch for TF images, length T_frames."""

    band_names: tuple[str, ...] | None = None
    """Band labels for band-power features, length B (default 5)."""

    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EncodedDataset:
    """Container for one recording's encoded representation.

    Naming
    ------
    Called *Dataset* (not Batch) because it is the durable unit passed to
    subject-wise splitting, caching, and later ``torch.utils.data`` adapters.
    One instance == one Sleep-EDF recording after encoding.

    Tensor shapes (leading axis ``N`` = epochs)
    -------------------------------------------
    - raw            : ``(N, C, 3000)``
    - bandpower      : ``(N, C, 10)``  (5 bands × log-abs + relative)
    - time_frequency : ``(N, C, F, T)``
    """

    features: NDArray[np.floating]
    labels: NDArray[np.int64]
    metadata: RepresentationMetadata
    subject_id: str
    recording_id: str
    onsets_sec: NDArray[np.floating] | None = None
    ignore_index: int = -100

    def __post_init__(self) -> None:
        if self.features.shape[0] != self.labels.shape[0]:
            raise ValueError("features and labels must share the same N (epoch) axis")
        expected_tail = self.metadata.feature_shape
        actual_tail = tuple(self.features.shape[1:])
        if actual_tail != expected_tail:
            raise ValueError(
                f"features shape tail {actual_tail} != metadata.feature_shape {expected_tail}"
            )
        if self.onsets_sec is not None and self.onsets_sec.shape != (self.n_epochs,):
            raise ValueError("onsets_sec must have shape (N,)")

    @property
    def n_epochs(self) -> int:
        return int(self.features.shape[0])

    @property
    def representation(self) -> RepresentationType:
        return self.metadata.representation

    @property
    def channel_names(self) -> tuple[str, ...]:
        return self.metadata.channel_names

    @property
    def sfreq(self) -> float:
        return self.metadata.sfreq


@dataclass(frozen=True, slots=True)
class EncodedDatasetCollection:
    """Ordered collection of per-recording :class:`EncodedDataset` objects.

    Used when encoding many recordings before subject-wise split. Keeps
    recordings separate so subject IDs are never mixed inside one tensor.
    """

    items: tuple[EncodedDataset, ...]

    def __post_init__(self) -> None:
        if not self.items:
            return
        reps = {item.representation for item in self.items}
        if len(reps) != 1:
            raise ValueError(
                f"All EncodedDataset items must share representation; got {reps}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterator[EncodedDataset]:
        return iter(self.items)

    def __getitem__(self, index: int) -> EncodedDataset:
        return self.items[index]

    @property
    def representation(self) -> RepresentationType | None:
        if not self.items:
            return None
        return self.items[0].representation

    def by_subject(self) -> Mapping[str, tuple[EncodedDataset, ...]]:
        grouped: dict[str, list[EncodedDataset]] = {}
        for item in self.items:
            grouped.setdefault(item.subject_id, []).append(item)
        return {key: tuple(value) for key, value in grouped.items()}
