"""Abstract encoder interface and shared helpers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from sleep_staging.encodings.epoching import preprocessed_to_epoch_batch
from sleep_staging.encodings.types import (
    EncodedDataset,
    EpochTensorBatch,
    LabelVocabulary,
    RepresentationMetadata,
    RepresentationType,
)
from sleep_staging.preprocessing.types import PreprocessedRecording


class BaseEncoder(ABC):
    """Transform epoch tensors into an :class:`EncodedDataset`.

    Design rules
    ------------
    - Stateless across recordings: all knobs come from construction / config.
    - Preserve epoch axis alignment: ``features.shape[0] == labels.shape[0]``.
    - Do not drop IGNORE epochs; models mask via ``ignore_index``.
    - Do not import training / model code.
    - Return NumPy arrays; Torch conversion belongs to the Models phase.
    """

    name: str
    representation: RepresentationType

    @abstractmethod
    def encode(self, batch: EpochTensorBatch) -> EncodedDataset:
        """Encode all epochs in ``batch``."""

    @abstractmethod
    def describe(
        self,
        *,
        n_channels: int,
        n_times: int,
        sfreq: float,
        epoch_duration_sec: float,
        channel_names: tuple[str, ...],
    ) -> RepresentationMetadata:
        """Declare output layout without running the transform."""

    def encode_recording(
        self,
        preprocessed: PreprocessedRecording,
        *,
        vocabulary: LabelVocabulary | None = None,
    ) -> EncodedDataset:
        """Slice a :class:`PreprocessedRecording` then encode it."""
        batch = preprocessed_to_epoch_batch(preprocessed, vocabulary=vocabulary)
        return self.encode(batch)

    def __call__(
        self,
        source: EpochTensorBatch | PreprocessedRecording,
        *,
        vocabulary: LabelVocabulary | None = None,
    ) -> EncodedDataset:
        if isinstance(source, PreprocessedRecording):
            return self.encode_recording(source, vocabulary=vocabulary)
        return self.encode(source)


def build_encoded_dataset(
    *,
    features: NDArray[np.floating],
    batch: EpochTensorBatch,
    metadata: RepresentationMetadata,
) -> EncodedDataset:
    """Shared constructor used by encoder implementations."""
    return EncodedDataset(
        features=features,
        labels=batch.labels.copy(),
        metadata=metadata,
        subject_id=batch.subject_id,
        recording_id=batch.recording_id,
        onsets_sec=batch.onsets_sec.copy(),
        ignore_index=batch.ignore_index,
    )
