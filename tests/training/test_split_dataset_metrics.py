from __future__ import annotations

import numpy as np
import pytest

from tests.phase4_helpers import make_collection
from sleep_staging.training.dataset import EpochDataset, collate_epoch_batch
from sleep_staging.training.metrics import compute_classification_metrics, normalize_confusion_matrix
from sleep_staging.training.split import SubjectSplit, assert_no_subject_leakage, subject_wise_split


def test_subject_split_is_deterministic_and_leak_free() -> None:
    subjects = ["S1", "S2", "S1", "S3", "S4", "S5"]
    split1 = subject_wise_split(subjects, seed=7)
    split2 = subject_wise_split(subjects, seed=7)
    assert split1 == split2
    assert_no_subject_leakage(split1)
    assert set(split1.all_subjects) == set(subjects)


def test_subject_split_rejects_leakage() -> None:
    with pytest.raises(ValueError, match="disjoint"):
        SubjectSplit(train=("S1",), val=("S1",), test=("S2",), seed=1, ratios=(0.7, 0.15, 0.15))


def test_epoch_dataset_preserves_reconstruction_metadata() -> None:
    collection = make_collection("bandpower", subjects=("S1",), n_epochs=3)
    dataset = EpochDataset(collection, drop_ignore=False)
    item = dataset[0]
    assert item["subject_id"] == "S1"
    assert item["recording_id"] == "S1R1"
    assert item["epoch_index"] == 0
    assert item["onset_sec"] == 0.0
    assert int(item["target"]) == int(item["label"])
    batch = collate_epoch_batch([dataset[0], dataset[1]])
    assert batch["subject_id"] == ["S1", "S1"]
    assert batch["onset_sec"] == [0.0, 30.0]


def test_metrics_exclude_ignore_and_normalize_rows() -> None:
    y_true = [0, 0, 1, 1, 2, -100]
    y_pred = [0, 1, 1, 2, 2, 4]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics.n_samples == 5
    assert metrics.confusion_matrix[0, 0] == 1
    assert metrics.confusion_matrix[0, 1] == 1
    assert metrics.confusion_matrix.sum() == 5
    assert np.allclose(metrics.normalized_confusion_matrix[0], [0.5, 0.5, 0, 0, 0])
    assert "normalized_confusion_matrix" in metrics.as_dict()


def test_normalized_confusion_matrix_handles_empty_rows() -> None:
    cm = np.asarray([[1, 1], [0, 0]], dtype=np.int64)
    norm = normalize_confusion_matrix(cm)
    assert np.allclose(norm[0], [0.5, 0.5])
    assert np.allclose(norm[1], [0.0, 0.0])
