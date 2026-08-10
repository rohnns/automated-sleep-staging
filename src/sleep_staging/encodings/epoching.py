"""Handoff from preprocessing → encodings.

Slices ``PreprocessedRecording.raw`` into fixed-length epoch windows using
``epoch_labels.onsets_sec``, producing an :class:`EpochTensorBatch`.
"""

from __future__ import annotations

import numpy as np

from sleep_staging.common.logging_utils import get_logger
from sleep_staging.encodings.exceptions import EpochExtractionError
from sleep_staging.encodings.types import EpochTensorBatch, LabelVocabulary
from sleep_staging.preprocessing.types import PreprocessedRecording

logger = get_logger(__name__)

DEFAULT_VOCABULARY = LabelVocabulary(ignore_index=-100)


def preprocessed_to_epoch_batch(
    preprocessed: PreprocessedRecording,
    *,
    vocabulary: LabelVocabulary | None = None,
) -> EpochTensorBatch:
    """Slice ``preprocessed.raw`` into ``(N, C, T)`` windows at epoch onsets.

    Required preprocessing state
    ----------------------------
    - ``epoch_labels`` present (AnnotationUnroller + StageMapper)
    - labels are AASM / IGNORE strings
    - wake crop grid-aligned so contiguous indexing matches onsets
    """
    if preprocessed.epoch_labels is None:
        raise EpochExtractionError(
            "PreprocessedRecording.epoch_labels is required before encoding"
        )

    epoch_labels = preprocessed.epoch_labels
    if epoch_labels.n_epochs == 0:
        raise EpochExtractionError("epoch_labels contains no epochs")

    vocab = vocabulary or DEFAULT_VOCABULARY
    sfreq = float(preprocessed.sampling_frequency)
    duration_sec = float(epoch_labels.duration_sec)
    n_times = int(round(duration_sec * sfreq))
    if n_times <= 0:
        raise EpochExtractionError(
            f"Invalid epoch sample count: duration={duration_sec}s sfreq={sfreq}"
        )

    if not preprocessed.raw.preload:
        preprocessed.raw.load_data()

    continuous = preprocessed.raw.get_data()
    n_channels, n_samples = continuous.shape
    channel_names = tuple(preprocessed.raw.ch_names)
    if len(channel_names) != n_channels:
        raise EpochExtractionError("Channel name count does not match signal data")

    epochs: list[np.ndarray] = []
    kept_onsets: list[float] = []
    kept_labels: list[str] = []

    for onset_sec, label in zip(epoch_labels.onsets_sec, epoch_labels.labels, strict=True):
        start = int(round(float(onset_sec) * sfreq))
        stop = start + n_times
        if start < 0:
            raise EpochExtractionError(f"Negative epoch start sample for onset={onset_sec}")
        if stop > n_samples:
            raise EpochExtractionError(
                f"Epoch at onset={onset_sec}s overflows recording "
                f"(need samples [{start}, {stop}), have {n_samples})"
            )
        epochs.append(continuous[:, start:stop])
        kept_onsets.append(float(onset_sec))
        kept_labels.append(str(label))

    signals = np.stack(epochs, axis=0)
    labels = vocab.encode(kept_labels)

    batch = EpochTensorBatch(
        signals=signals,
        labels=labels,
        onsets_sec=np.asarray(kept_onsets, dtype=np.float64),
        channel_names=channel_names,
        sfreq=sfreq,
        epoch_duration_sec=duration_sec,
        subject_id=preprocessed.metadata.subject_id,
        recording_id=preprocessed.metadata.recording_id,
        ignore_index=vocab.ignore_index,
    )
    logger.info(
        "Extracted %d epoch(s) → shape %s (sfreq=%.1f Hz, subject=%s, recording=%s)",
        batch.n_epochs,
        batch.signals.shape,
        sfreq,
        batch.subject_id,
        batch.recording_id,
    )
    return batch
