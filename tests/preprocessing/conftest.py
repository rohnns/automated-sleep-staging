"""Helpers to build synthetic SleepRecording objects for preprocessing tests."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from sleep_staging.acquisition.dataclasses import (
    ChannelInfo,
    RecordingMetadata,
    SleepRecording,
)


def make_sleep_recording(
    *,
    n_channels: int = 4,
    sfreq: float = 100.0,
    duration_sec: float = 600.0,
    ch_names: list[str] | None = None,
    ch_types: list[str] | None = None,
    annotations: mne.Annotations | None = None,
    subject_id: str = "00",
    recording_id: str = "1",
) -> SleepRecording:
    """Create a minimal in-memory SleepRecording for unit tests."""
    n_times = int(duration_sec * sfreq)
    names = ch_names or ["Fpz-Cz", "Pz-Oz", "horizontal", "submental"][:n_channels]
    types = ch_types or ["eeg", "eeg", "eog", "emg"][: len(names)]
    rng = np.random.default_rng(0)
    data = rng.normal(scale=1e-5, size=(len(names), n_times))
    info = mne.create_info(ch_names=list(names), sfreq=sfreq, ch_types=list(types))
    raw = mne.io.RawArray(data, info, verbose="ERROR")
    if annotations is not None:
        raw.set_annotations(annotations, emit_warning=False)

    channels = tuple(
        ChannelInfo(name=n, ch_type=t, unit="V", sampling_frequency=sfreq)
        for n, t in zip(names, types, strict=True)
    )
    metadata = RecordingMetadata(
        subject_id=subject_id,
        recording_id=recording_id,
        study="SC",
        sampling_frequency=sfreq,
        duration_sec=duration_sec,
        n_channels=len(names),
        channel_names=tuple(names),
        channel_types=tuple(types),
        channels=channels,
        units={n: "V" for n in names},
        reference="test",
        montage=None,
        meas_date=None,
        psg_path=Path("SC4001E0-PSG.edf"),
        hypnogram_path=Path("SC4001EC-Hypnogram.edf"),
        n_annotations=0 if annotations is None else len(annotations),
    )
    return SleepRecording(raw=raw, metadata=metadata)
