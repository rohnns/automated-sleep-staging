"""Typed data containers for Sleep-EDF acquisition results.

MNE boundary
------------
``SleepRecording.raw`` is an MNE ``Raw`` object. Annotations live on that object
(``raw.annotations``) and are the single authoritative runtime representation.
``RecordingMetadata`` is intentionally free of MNE types so callers can inspect
paths, IDs, and channel summaries without depending on MNE object state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import mne
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class ChannelInfo:
    """Metadata describing a single recorded channel.

    Attributes
    ----------
    name:
        Channel label as exposed by MNE after loading.
    ch_type:
        MNE channel type (e.g. ``eeg``, ``eog``, ``emg``, ``stim``).
    unit:
        Physical unit string when available (e.g. ``V``, ``°C``).
    sampling_frequency:
        Per-channel sampling rate in Hz when available.
    """

    name: str
    ch_type: str
    unit: str | None = None
    sampling_frequency: float | None = None


@dataclass(frozen=True, slots=True)
class AnnotationRecord:
    """Plain (library-agnostic) view of one hypnogram annotation.

    Prefer ``SleepRecording.annotations`` (MNE) for processing. Use this type
    only when a simple serializable record is needed.
    """

    onset: float
    duration: float
    description: str


def annotations_to_records(annotations: mne.Annotations) -> tuple[AnnotationRecord, ...]:
    """Convert MNE annotations into immutable plain records.

    This is a derived view helper, not a second stored copy of the hypnogram.
    """
    return tuple(
        AnnotationRecord(
            onset=float(onset),
            duration=float(duration),
            description=str(description),
        )
        for onset, duration, description in zip(
            annotations.onset,
            annotations.duration,
            annotations.description,
            strict=True,
        )
    )


@dataclass(frozen=True, slots=True)
class RecordingMetadata:
    """Structured metadata for one Sleep-EDF PSG recording.

    This container is MNE-agnostic: it stores paths, identifiers, and summary
    fields only. Annotation *content* is not duplicated here; see
    ``SleepRecording.annotations``.

    ``n_annotations`` is a load-time count taken from the authoritative
    ``raw.annotations`` after attachment (useful for logging/summaries).
    """

    subject_id: str
    recording_id: str
    study: str
    sampling_frequency: float
    duration_sec: float
    n_channels: int
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    channels: tuple[ChannelInfo, ...]
    units: dict[str, str | None]
    reference: str | None
    montage: str | None
    meas_date: datetime | None
    psg_path: Path
    hypnogram_path: Path
    n_annotations: int
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class SleepRecording:
    """Complete Sleep-EDF recording: MNE signals plus summary metadata.

    Downstream stages should accept this type rather than calling MNE I/O
    themselves. Signal processing that needs MNE APIs should use ``raw``
    (and ``annotations``) directly.

    Attributes
    ----------
    raw:
        MNE ``Raw`` containing PSG signals. Hypnogram annotations are attached
        here and are the authoritative annotation store
        (``raw.annotations`` / :attr:`annotations`).
    metadata:
        Immutable, MNE-agnostic recording summary.
    """

    raw: mne.io.BaseRaw
    metadata: RecordingMetadata

    @property
    def annotations(self) -> mne.Annotations:
        """Authoritative hypnogram annotations (delegates to ``raw.annotations``)."""
        return self.raw.annotations

    @property
    def annotation_records(self) -> tuple[AnnotationRecord, ...]:
        """Derived plain-record view of :attr:`annotations` (not a stored copy)."""
        return annotations_to_records(self.raw.annotations)

    @property
    def subject_id(self) -> str:
        """Subject identifier for convenience."""
        return self.metadata.subject_id

    @property
    def recording_id(self) -> str:
        """Recording / night identifier for convenience."""
        return self.metadata.recording_id

    @property
    def sampling_frequency(self) -> float:
        """Primary sampling frequency in Hz."""
        return self.metadata.sampling_frequency

    @property
    def duration_sec(self) -> float:
        """Recording duration in seconds."""
        return self.metadata.duration_sec

    @property
    def channel_names(self) -> tuple[str, ...]:
        """Ordered channel names."""
        return self.metadata.channel_names

    def get_data(
        self,
        *,
        picks: str | list[str] | None = None,
        start: float | None = None,
        stop: float | None = None,
    ) -> NDArray[np.floating[Any]]:
        """Return signal data as a NumPy array.

        Parameters
        ----------
        picks:
            Channel selection passed to ``Raw.get_data``.
        start, stop:
            Optional time window in seconds.

        Returns
        -------
        ndarray
            Array of shape ``(n_channels, n_times)``.
        """
        kwargs: dict[str, Any] = {}
        if picks is not None:
            kwargs["picks"] = picks
        if start is not None:
            kwargs["tmin"] = start
        if stop is not None:
            kwargs["tmax"] = stop
        return self.raw.get_data(**kwargs)

    def summary(self) -> str:
        """Return a compact human-readable summary of the recording."""
        meta = self.metadata
        return (
            f"SleepRecording(study={meta.study}, subject={meta.subject_id}, "
            f"recording={meta.recording_id}, sfreq={meta.sampling_frequency:.2f} Hz, "
            f"duration={meta.duration_sec:.1f} s, channels={meta.n_channels}, "
            f"annotations={meta.n_annotations})"
        )

    def __repr__(self) -> str:
        return self.summary()
