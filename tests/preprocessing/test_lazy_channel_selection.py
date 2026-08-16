"""Regression tests: preprocessing must select channels before loading data.

Context
-------
``preprocess_recording()`` used to force ``preload=True`` before running any
transform (see git history of ``pipeline.py``). For a real Sleep-EDF EDF
file, ``raw.load_data()`` on a still-multi-channel ``Raw`` (EEG/EOG at
100 Hz alongside EMG/Resp/Temp/Marker at ~1 Hz) forces MNE to resample every
channel onto one common grid for the *entire* recording -- the actual cause
of the ``std::bad_alloc`` seen on long (8-24h) ST recordings.

The fix: never preload before ``ChannelSelector`` runs. ``Raw.pick()`` is a
metadata-only operation MNE supports on unloaded (lazy) ``Raw`` objects, so
narrowing to the 2-3 needed, already-homogeneous-rate channels can happen
before a single sample is read from disk.

``mne.io.RawArray`` (used by ``tests/preprocessing/conftest.py``) is always
preloaded from construction, so it cannot exercise "lazy" behavior. These
tests use a minimal duck-typed stand-in for ``mne.io.BaseRaw`` that starts
unloaded and records call order, to verify the intended ordering without
needing a real multi-hour EDF file.
"""

from __future__ import annotations

from pathlib import Path

import mne
import pytest

from sleep_staging.acquisition.dataclasses import RecordingMetadata, SleepRecording
from sleep_staging.config import load_settings
from sleep_staging.preprocessing import (
    BadChannelDetector,
    PreprocessPipeline,
    build_default_pipeline,
    preprocess_recording,
)
from sleep_staging.preprocessing import pipeline as pipeline_module
from tests.preprocessing.conftest import make_sleep_recording


def _settings():
    return load_settings(Path("configs/default.yaml"))


# ---------------------------------------------------------------------------
# 1. preprocess_recording() must never force an eager full-recording preload.
# ---------------------------------------------------------------------------
def test_preprocess_recording_does_not_eager_preload(monkeypatch) -> None:
    settings = _settings().preprocessing

    # Sanity: the default config enables stages that used to trigger the
    # (buggy) eager preload, so this test isn't vacuously true.
    assert settings.filter.enabled
    assert settings.bad_channel.enabled

    captured: dict[str, object] = {}
    original_run = pipeline_module.PreprocessPipeline.run

    def spy_run(self, recording, *, copy=True, preload=False):
        captured["preload"] = preload
        return original_run(self, recording, copy=copy, preload=preload)

    monkeypatch.setattr(pipeline_module.PreprocessPipeline, "run", spy_run)

    annotations = mne.Annotations(
        onset=[0.0, 30.0, 60.0],
        duration=[30.0, 30.0, 30.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage 2"],
    )
    recording = make_sleep_recording(duration_sec=90.0, annotations=annotations)

    preprocess_recording(recording, settings, copy=True)

    assert captured["preload"] is False


# ---------------------------------------------------------------------------
# 2. Channel selection must happen before any data is materialized, and the
#    channels that are never used (heterogeneous low-rate EMG/marker/etc.)
#    must never reach load_data().
# ---------------------------------------------------------------------------
class _LazyRawStub:
    """Minimal mne.io.BaseRaw stand-in that starts unloaded (preload=False).

    Implements only what AnnotationUnroller / SleepBoundaryDetector /
    ChannelSelector / BadChannelDetector touch, and logs call order so tests
    can assert channels are picked *before* data is ever loaded.
    """

    def __init__(self, *, ch_names, ch_types, sfreq, n_times, annotations, call_log):
        self.ch_names = list(ch_names)
        self._ch_types = list(ch_types)
        self.info = {"sfreq": sfreq, "bads": []}
        self.preload = False
        self.n_times = n_times
        self.annotations = annotations
        self._call_log = call_log

    def copy(self) -> "_LazyRawStub":
        clone = _LazyRawStub(
            ch_names=self.ch_names,
            ch_types=self._ch_types,
            sfreq=self.info["sfreq"],
            n_times=self.n_times,
            annotations=self.annotations,
            call_log=self._call_log,
        )
        clone.preload = self.preload
        return clone

    def get_channel_types(self):
        return list(self._ch_types)

    def set_annotations(self, annotations, emit_warning=False):
        self.annotations = annotations

    def pick(self, picks):
        self._call_log.append(("pick", tuple(picks)))
        idx = [self.ch_names.index(p) for p in picks]
        self.ch_names = [self.ch_names[i] for i in idx]
        self._ch_types = [self._ch_types[i] for i in idx]
        return self

    def load_data(self):
        self._call_log.append(("load_data", tuple(self.ch_names)))
        self.preload = True
        return self

    def get_data(self):
        import numpy as np

        return np.zeros((len(self.ch_names), self.n_times))


def _fake_recording_with_lazy_raw(call_log: list) -> SleepRecording:
    # Mirrors a real ST EDF file: EEG/EOG at 100 Hz plus lower-rate auxiliary
    # channels (EMG/marker) that the pipeline never actually needs.
    ch_names = ["Fpz-Cz", "Pz-Oz", "horizontal", "submental", "marker"]
    ch_types = ["eeg", "eeg", "eog", "emg", "misc"]
    sfreq = 100.0
    n_times = int(100.0 * 90.0)

    annotations = mne.Annotations(
        onset=[0.0, 30.0, 60.0],
        duration=[30.0, 30.0, 30.0],
        description=["Sleep stage W", "Sleep stage 2", "Sleep stage 2"],
    )
    raw = _LazyRawStub(
        ch_names=ch_names,
        ch_types=ch_types,
        sfreq=sfreq,
        n_times=n_times,
        annotations=annotations,
        call_log=call_log,
    )

    metadata = RecordingMetadata(
        subject_id="7191",
        recording_id="J0",
        study="ST",
        sampling_frequency=sfreq,
        duration_sec=90.0,
        n_channels=len(ch_names),
        channel_names=tuple(ch_names),
        channel_types=tuple(ch_types),
        channels=(),
        units={},
        reference=None,
        montage=None,
        meas_date=None,
        psg_path=Path("ST7191J0-PSG.edf"),
        hypnogram_path=Path("ST7191JP-Hypnogram.edf"),
        n_annotations=len(annotations),
    )
    return SleepRecording(raw=raw, metadata=metadata)


def test_channel_selection_happens_before_first_data_load() -> None:
    call_log: list = []
    recording = _fake_recording_with_lazy_raw(call_log)
    settings = _settings().preprocessing

    # The fake Raw above only implements the surface AnnotationUnroller /
    # SleepBoundaryDetector / ChannelSelector / BadChannelDetector touch, so
    # truncate the real, config-driven pipeline to that prefix (everything
    # up to and including the first data-touching transform) rather than
    # re-implementing a full mne.io.BaseRaw fake.
    full_pipeline = build_default_pipeline(settings)
    cutoff = next(
        i
        for i, transform in enumerate(full_pipeline.transforms)
        if isinstance(transform, BadChannelDetector)
    )
    prefix_pipeline = PreprocessPipeline(full_pipeline.transforms[: cutoff + 1])

    result = prefix_pipeline.run(recording, copy=True, preload=False)

    pick_indices = [i for i, (op, _) in enumerate(call_log) if op == "pick"]
    load_indices = [i for i, (op, _) in enumerate(call_log) if op == "load_data"]

    assert pick_indices, "ChannelSelector never ran"
    assert load_indices, "no transform ever loaded data"
    assert pick_indices[0] < load_indices[0], (
        "data was loaded before channel selection ran -- this forces MNE to "
        "materialize/resample every channel (including unused low-rate "
        "EMG/marker channels) across the whole recording"
    )

    # The channels present at the moment of the first load must already be
    # narrowed to the target set -- the unwanted heterogeneous-rate channels
    # were never loaded at all.
    first_load_channels = call_log[load_indices[0]][1]
    assert set(first_load_channels) == {"Fpz-Cz", "Pz-Oz", "horizontal"}
    assert "submental" not in first_load_channels
    assert "marker" not in first_load_channels

    assert result.channel_names == ("Fpz-Cz", "Pz-Oz", "horizontal")
    assert result.sampling_frequency == pytest.approx(100.0)
