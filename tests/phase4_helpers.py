"""Shared synthetic encoded collections for Phase 4 tests."""

from __future__ import annotations

import numpy as np

from sleep_staging.representations.types import EncodedDataset, EncodedDatasetCollection, RepresentationMetadata


def make_collection(
    representation: str,
    *,
    subjects: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6"),
    n_epochs: int = 6,
) -> EncodedDatasetCollection:
    items = []
    rng = np.random.default_rng(123)
    if representation == "raw":
        tail = (1, 3000)
    elif representation == "bandpower":
        tail = (1, 10)
    elif representation == "time_frequency":
        tail = (1, 75, 28)
    else:
        raise ValueError(representation)

    for idx, subject in enumerate(subjects):
        labels = np.asarray([(idx + j) % 5 for j in range(n_epochs)], dtype=np.int64)
        labels[-1] = -100
        features = rng.normal(size=(n_epochs, *tail)).astype(np.float32)
        # Inject an easy signal for tiny smoke training/classical tests.
        for e, label in enumerate(labels):
            if label != -100:
                features[e] += float(label) * 0.5
        metadata = RepresentationMetadata(
            representation=representation,  # type: ignore[arg-type]
            channel_names=("Fpz-Cz",),
            sfreq=100.0,
            epoch_duration_sec=30.0,
            feature_shape=tail,
            algorithm="stft" if representation == "time_frequency" else None,
        )
        items.append(
            EncodedDataset(
                features=features,
                labels=labels,
                metadata=metadata,
                subject_id=subject,
                recording_id=f"{subject}R1",
                onsets_sec=np.arange(n_epochs, dtype=np.float64) * 30.0,
                ignore_index=-100,
            )
        )
    return EncodedDatasetCollection(tuple(items))
