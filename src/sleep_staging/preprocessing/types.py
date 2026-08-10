"""Shared types and the mutable preprocessing state object."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import mne
import numpy as np
from numpy.typing import NDArray

from sleep_staging.acquisition.dataclasses import RecordingMetadata, SleepRecording


@dataclass(frozen=True, slots=True)
class EpochLabels:
    """Fixed-length epoch labels derived from variable-length hypnogram bouts.

    Attributes
    ----------
    onsets_sec:
        Epoch start times in seconds relative to the current ``raw`` time origin.
    duration_sec:
        Constant epoch length (30 s for Sleep-EDF staging).
    labels:
        Stage label per epoch (R&K or AASM depending on pipeline stage).
    """

    onsets_sec: tuple[float, ...]
    duration_sec: float
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.onsets_sec) != len(self.labels):
            raise ValueError("onsets_sec and labels must have the same length")
        if self.duration_sec <= 0:
            raise ValueError("duration_sec must be positive")

    @property
    def n_epochs(self) -> int:
        return len(self.labels)

    def with_shifted_onsets(self, delta_sec: float) -> EpochLabels:
        """Return a copy with all onsets shifted by ``delta_sec``."""
        return EpochLabels(
            onsets_sec=tuple(onset + delta_sec for onset in self.onsets_sec),
            duration_sec=self.duration_sec,
            labels=self.labels,
        )

    def filtered(
        self,
        *,
        tmin_sec: float,
        tmax_sec: float,
        shift_to_zero: bool = True,
    ) -> EpochLabels:
        """Keep epochs fully contained in ``[tmin_sec, tmax_sec]``."""
        kept_onsets: list[float] = []
        kept_labels: list[str] = []
        for onset, label in zip(self.onsets_sec, self.labels, strict=True):
            end = onset + self.duration_sec
            if onset >= tmin_sec and end <= tmax_sec + 1e-9:
                new_onset = onset - tmin_sec if shift_to_zero else onset
                kept_onsets.append(new_onset)
                kept_labels.append(label)
        return EpochLabels(
            onsets_sec=tuple(kept_onsets),
            duration_sec=self.duration_sec,
            labels=tuple(kept_labels),
        )

    def relabeled(self, labels: tuple[str, ...]) -> EpochLabels:
        if len(labels) != self.n_epochs:
            raise ValueError("relabeled() requires the same number of labels")
        return EpochLabels(
            onsets_sec=self.onsets_sec,
            duration_sec=self.duration_sec,
            labels=labels,
        )

    def mask(self, keep: NDArray[np.bool_] | list[bool]) -> EpochLabels:
        keep_arr = np.asarray(keep, dtype=bool)
        if keep_arr.shape != (self.n_epochs,):
            raise ValueError("mask length must match n_epochs")
        return EpochLabels(
            onsets_sec=tuple(
                onset for onset, flag in zip(self.onsets_sec, keep_arr, strict=True) if flag
            ),
            duration_sec=self.duration_sec,
            labels=tuple(
                label for label, flag in zip(self.labels, keep_arr, strict=True) if flag
            ),
        )


@dataclass(frozen=True, slots=True)
class SleepBoundaries:
    """Sleep period boundaries on the current recording timeline."""

    onset_sec: float | None
    offset_sec: float | None
    has_scored_sleep: bool

    @property
    def sleep_period_duration_sec(self) -> float | None:
        if self.onset_sec is None or self.offset_sec is None:
            return None
        return max(0.0, self.offset_sec - self.onset_sec)


@dataclass(slots=True)
class PreprocessedRecording:
    """Working state passed through composable preprocessing transforms.

    Starts from a :class:`~sleep_staging.acquisition.SleepRecording` and is
    enriched by each transform. ``raw`` is a writable working copy and may be
    cropped, filtered, and normalized in place by later stages.
    """

    raw: mne.io.BaseRaw
    metadata: RecordingMetadata
    epoch_labels: EpochLabels | None = None
    boundaries: SleepBoundaries | None = None
    applied_transforms: tuple[str, ...] = ()
    extras: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_sleep_recording(
        cls,
        recording: SleepRecording,
        *,
        copy: bool = True,
        preload: bool = False,
    ) -> PreprocessedRecording:
        """Create preprocessing state from an acquired recording."""
        raw = recording.raw.copy() if copy else recording.raw
        if preload and not raw.preload:
            raw.load_data()
        return cls(raw=raw, metadata=recording.metadata)

    @property
    def sampling_frequency(self) -> float:
        return float(self.raw.info["sfreq"])

    @property
    def duration_sec(self) -> float:
        return float(self.raw.n_times) / self.sampling_frequency

    @property
    def channel_names(self) -> tuple[str, ...]:
        return tuple(self.raw.ch_names)

    def mark_applied(self, name: str) -> None:
        self.applied_transforms = (*self.applied_transforms, name)

    def summary(self) -> str:
        n_epochs = self.epoch_labels.n_epochs if self.epoch_labels is not None else 0
        return (
            f"PreprocessedRecording(subject={self.metadata.subject_id}, "
            f"recording={self.metadata.recording_id}, "
            f"sfreq={self.sampling_frequency:.2f} Hz, "
            f"duration={self.duration_sec:.1f} s, "
            f"channels={len(self.channel_names)}, "
            f"epochs={n_epochs}, "
            f"transforms={list(self.applied_transforms)})"
        )

    def __repr__(self) -> str:
        return self.summary()


class Transform(ABC):
    """Single-responsibility preprocessing transform."""

    name: str

    @abstractmethod
    def apply(self, state: PreprocessedRecording) -> PreprocessedRecording:
        """Apply this transform and return the updated state."""

    def __call__(self, state: PreprocessedRecording) -> PreprocessedRecording:
        result = self.apply(state)
        result.mark_applied(self.name)
        return result
