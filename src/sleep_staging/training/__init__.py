"""Public API for training utilities."""

from sleep_staging.training.dataset import (
    EpochDataset,
    EpochExample,
    SplitEpochStats,
    collate_epoch_batch,
    filter_collection_by_subjects,
    iter_epoch_examples,
    summarize_partition,
    summarize_split,
)
from sleep_staging.training.classical import (
    ClassicalBaselineResult,
    EpochPrediction,
    train_bandpower_logistic_regression,
)
from sleep_staging.training.experiment import (
    ControlledExperimentResult,
    RepresentationRunResult,
    build_shared_subject_split,
    run_controlled_pytorch_experiments,
)
from sleep_staging.training.metrics import (
    STAGE_NAMES,
    ClassificationMetrics,
    compute_classification_metrics,
    normalize_confusion_matrix,
)
from sleep_staging.training.split import (
    SubjectSplit,
    assert_no_subject_leakage,
    sleep_edf_subject_key,
    split_membership,
    subject_wise_split,
)
from sleep_staging.training.trainer import (
    IGNORE_INDEX,
    TrainRecipe,
    TrainResult,
    compute_class_weights_from_dataset,
    evaluate,
    select_device,
    set_seed,
    train_baseline,
)

__all__ = [
    "ClassificationMetrics",
    "ClassicalBaselineResult",
    "ControlledExperimentResult",
    "EpochDataset",
    "EpochExample",
    "EpochPrediction",
    "IGNORE_INDEX",
    "RepresentationRunResult",
    "STAGE_NAMES",
    "SplitEpochStats",
    "SubjectSplit",
    "TrainRecipe",
    "TrainResult",
    "assert_no_subject_leakage",
    "collate_epoch_batch",
    "compute_class_weights_from_dataset",
    "compute_classification_metrics",
    "normalize_confusion_matrix",
    "evaluate",
    "filter_collection_by_subjects",
    "iter_epoch_examples",
    "set_seed",
    "select_device",
    "sleep_edf_subject_key",
    "split_membership",
    "subject_wise_split",
    "summarize_partition",
    "summarize_split",
    "build_shared_subject_split",
    "run_controlled_pytorch_experiments",
    "train_bandpower_logistic_regression",
    "train_baseline",
]
