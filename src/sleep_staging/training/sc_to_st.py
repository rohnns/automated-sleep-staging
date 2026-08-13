"""Primary SC→ST external-generalization experiment orchestration.

This module intentionally keeps the primary experiment separate from legacy
SC-only artifacts. It discovers SC and ST recordings independently, builds a
SC-only subject-wise train/validation split, and evaluates on the full ST
cohort as the external test set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import dataclasses
import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from sleep_staging import __version__ as PACKAGE_VERSION
from sleep_staging.acquisition.loader import discover_recordings, load_recording
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.config import load_settings
from sleep_staging.evaluation.output import EpochPredictionRecord, build_epoch_predictions, compute_sleep_statistics, plot_hypnogram, save_predictions_csv
from sleep_staging.models.factory import build_baseline_model
from sleep_staging.preprocessing.pipeline import preprocess_recording
from sleep_staging.representations.factory import build_encoder
from sleep_staging.representations.types import EncodedDataset, EncodedDatasetCollection, LabelVocabulary
from sleep_staging.training.classical import train_bandpower_logistic_regression
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.experiment import build_shared_subject_split
from sleep_staging.training.metrics import compute_classification_metrics
from sleep_staging.training.split import assert_no_subject_leakage, sleep_edf_subject_key, subject_wise_split
from sleep_staging.training.trainer import TrainRecipe, select_device, train_baseline
from sleep_staging.training.trainer import collate_epoch_batch, make_loader


PRIMARY_EXPERIMENT_NAME = "sc_to_st_external_generalization"
PRIMARY_ARTIFACT_ROOT = Path("artifacts")
PRIMARY_MODEL_ROOT = Path("models")
DEFAULT_DATASET_ROOT = Path("D:/SleepEDFX")

REPRESENTATION_TO_ARTIFACT = {
    "raw": "raw",
    "bandpower": "bandpower",
    "time_frequency": "time_frequency",
}
REPRESENTATION_TO_MODEL_DIR = {
    "raw": "raw_cnn",
    "bandpower": "bandpower_mlp",
    "time_frequency": "stft_cnn",
}


@dataclass(frozen=True, slots=True)
class CohortArtifacts:
    sc_train_subjects: tuple[str, ...]
    sc_val_subjects: tuple[str, ...]
    st_test_subjects: tuple[str, ...]
    sc_recordings: tuple[Path, ...]
    st_recordings: tuple[Path, ...]


@dataclass(frozen=True, slots=True)
class ExperimentArtifactSummary:
    experiment_name: str
    model_name: str
    representation: str
    model_dir: Path
    predictions_path: Path
    metrics_path: Path
    metadata_path: Path


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_sc_st_recordings(dataset_root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    files = discover_recordings(dataset_root)
    sc = tuple(p for p in files if parse_psg_filename(p).study == "SC")
    st = tuple(p for p in files if parse_psg_filename(p).study == "ST")
    return sc, st


def _subject_key(path: Path) -> str:
    return sleep_edf_subject_key(parse_psg_filename(path))


def build_sc_split(
    sc_recordings: Iterable[Path],
    *,
    seed: int = 42,
    train_subjects: int = 55,
    val_subjects: int = 12,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    subjects = sorted({_subject_key(path) for path in sc_recordings})
    split = subject_wise_split(subjects, ratios=(0.70, 0.15, 0.15), seed=seed)
    assert_no_subject_leakage(split)
    if len(split.train) != train_subjects or len(split.val) != val_subjects:
        raise AssertionError(
            f"Expected SC split {train_subjects}/{val_subjects}, got {len(split.train)}/{len(split.val)}"
        )
    return split.train, split.val


def _load_settings_with_root(config_path: Path, dataset_root: Path | None) -> object:
    settings = load_settings(config_path)
    if dataset_root is not None:
        settings = dataclasses.replace(settings, acquisition=dataclasses.replace(settings.acquisition, data_root=dataset_root))
    return settings


def _preprocess_and_encode_recordings(
    recording_paths: Iterable[Path],
    *,
    settings,
    representation: str,
) -> EncodedDatasetCollection:
    enc_settings = dataclasses.replace(settings.encodings, representation=representation)
    encoder = build_encoder(enc_settings)
    vocab = LabelVocabulary(ignore_label=settings.encodings.ignore_label, ignore_index=settings.encodings.ignore_index)
    datasets: list[EncodedDataset] = []
    for path in recording_paths:
        recording = load_recording(path, settings=settings.acquisition, preload=settings.acquisition.preload)
        preprocessed = preprocess_recording(recording, settings.preprocessing, copy=False)
        datasets.append(encoder(preprocessed, vocabulary=vocab))
    return EncodedDatasetCollection(items=tuple(datasets))


def build_primary_collections(
    *,
    config_path: Path,
    dataset_root: Path | None = None,
    max_sc_recordings: int | None = None,
    max_st_recordings: int | None = None,
) -> tuple[object, tuple[Path, ...], tuple[Path, ...], dict[str, EncodedDatasetCollection]]:
    settings = _load_settings_with_root(config_path, dataset_root)
    sc_recordings, st_recordings = discover_sc_st_recordings(settings.acquisition.data_root)
    if max_sc_recordings is not None:
        sc_recordings = sc_recordings[:max_sc_recordings]
    if max_st_recordings is not None:
        st_recordings = st_recordings[:max_st_recordings]
    if not sc_recordings:
        raise ValueError("No SC recordings discovered")
    if not st_recordings:
        raise ValueError("No ST recordings discovered")
    collections = {
        rep: _preprocess_and_encode_recordings(
            sc_recordings if rep != "time_frequency" else sc_recordings,
            settings=settings,
            representation=rep,
        )
        for rep in ("raw", "bandpower", "time_frequency")
    }
    # Validate identical subject set across representations.
    raw_subjects = tuple(sorted({item.subject_id for item in collections["raw"].items}))
    for rep in ("bandpower", "time_frequency"):
        subj = tuple(sorted({item.subject_id for item in collections[rep].items}))
        if subj != raw_subjects:
            raise ValueError(f"Representation {rep} has mismatched subjects")
    return settings, sc_recordings, st_recordings, collections


def _artifact_paths(model_dir: Path, representation: str) -> tuple[Path, Path, Path]:
    pred_dir = _ensure_dir(PRIMARY_ARTIFACT_ROOT / "predictions" / "sc_to_st" / representation)
    rep_dir = _ensure_dir(PRIMARY_ARTIFACT_ROOT / "reports" / "sc_to_st" / representation)
    fig_dir = _ensure_dir(PRIMARY_ARTIFACT_ROOT / "figures" / "sc_to_st" / representation)
    return pred_dir, rep_dir, fig_dir


def _predict_epoch_dataset(
    model: torch.nn.Module,
    dataset: EpochDataset,
    *,
    recipe: TrainRecipe,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[EpochPredictionRecord, ...]]:
    loader = make_loader(dataset, batch_size=recipe.batch_size, shuffle=False, seed=recipe.seed, num_workers=recipe.num_workers)
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    onsets: list[float] = []
    records: list[EpochPredictionRecord] = []
    flat_examples = list(dataset.examples)
    offset = 0
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            logits = model(batch["features"])
            preds = torch.argmax(logits, dim=1).detach().cpu().tolist()
            labels = batch["label"].detach().cpu().tolist()
            batch_size = len(preds)
            y_true.extend(int(v) for v in labels)
            y_pred.extend(int(v) for v in preds)
            onset_list = batch.get("onset_sec", [])
            for onset in onset_list:
                onsets.append(float(onset) if onset is not None else 0.0)
            for local_idx, pred in enumerate(preds):
                ex = flat_examples[offset + local_idx]
                records.append(
                    EpochPredictionRecord(
                        subject_id=ex.subject_id,
                        recording_id=ex.recording_id,
                        epoch_index=ex.epoch_index,
                        onset_sec=0.0 if ex.onset_sec is None else float(ex.onset_sec),
                        duration_sec=30.0,
                        target=int(ex.label),
                        prediction=int(pred),
                    )
                )
            offset += batch_size
    return np.asarray(onsets, dtype=float), np.asarray(y_true, dtype=np.int64), np.asarray(y_pred, dtype=np.int64), tuple(records)


def _save_model_metadata(
    *,
    model_dir: Path,
    experiment_name: str,
    representation: str,
    settings,
    sc_train_subjects: tuple[str, ...],
    sc_val_subjects: tuple[str, ...],
    st_test_subjects: tuple[str, ...],
) -> Path:
    _ensure_dir(model_dir)
    payload = {
        "experiment_name": experiment_name,
        "representation": representation,
        "dataset_identity": {
            "sc_train_subjects": list(sc_train_subjects),
            "sc_val_subjects": list(sc_val_subjects),
            "st_test_subjects": list(st_test_subjects),
        },
        "preprocessing": asdict(settings.preprocessing),
        "encoding": asdict(dataclasses.replace(settings.encodings, representation=representation)),
        "training": asdict(settings.experiment.train),
        "package_version": PACKAGE_VERSION,
    }
    meta_path = model_dir / "metadata.json"
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return meta_path


def _save_metrics_bundle(
    *,
    rep_name: str,
    result,
    pred_dir: Path,
    rep_dir: Path,
    fig_dir: Path,
    dataset_name: str,
) -> dict[str, object]:
    train_result = result.train_result
    metrics = train_result.test_metrics
    assert metrics is not None
    pred_path = pred_dir / f"{rep_name}_predictions.csv"
    metrics_path = rep_dir / f"{rep_name}_metrics.json"
    fig_path = fig_dir / f"{rep_name}_hypnogram.png"
    preds = train_result.test_metrics  # placeholder: predictions are written below if available
    payload = {
        "experiment_name": PRIMARY_EXPERIMENT_NAME,
        "dataset_name": dataset_name,
        "representation": rep_name,
        "metrics": metrics.as_dict(),
        "split": result.split.as_dict(),
    }
    metrics_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "metrics_path": metrics_path,
        "predictions_path": pred_path,
        "figure_path": fig_path,
    }


def run_primary_experiment(
    *,
    config_path: Path,
    dataset_root: Path | None = None,
    model: str = "all",
    max_sc_recordings: int | None = None,
    max_st_recordings: int | None = None,
    smoke: bool = False,
) -> dict[str, object]:
    settings, sc_recordings, st_recordings, collections = build_primary_collections(
        config_path=config_path,
        dataset_root=dataset_root,
        max_sc_recordings=max_sc_recordings,
        max_st_recordings=max_st_recordings,
    )
    if smoke:
        unique_sc = sorted({_subject_key(path) for path in sc_recordings})
        split = subject_wise_split(unique_sc, ratios=(0.7, 0.15, 0.15), seed=42)
        assert_no_subject_leakage(split)
        sc_train_subjects, sc_val_subjects = split.train, split.val
    else:
        sc_train_subjects, sc_val_subjects = build_sc_split(sc_recordings, seed=42, train_subjects=55, val_subjects=12)
    st_subjects = tuple(sorted({_subject_key(path) for path in st_recordings}))
    if len(st_subjects) != 22 and not smoke:
        raise AssertionError(f"Expected 22 ST subjects, got {len(st_subjects)}")

    sc_train = sc_train_subjects
    sc_val = sc_val_subjects
    train_ds_by_rep: dict[str, EpochDataset] = {}
    val_ds_by_rep: dict[str, EpochDataset] = {}
    test_ds_by_rep: dict[str, EpochDataset] = {}
    for rep, col in collections.items():
        train_ds_by_rep[rep] = EpochDataset(col, subject_ids=sc_train, drop_ignore=False)
        val_ds_by_rep[rep] = EpochDataset(col, subject_ids=sc_val, drop_ignore=False)
        test_ds_by_rep[rep] = EpochDataset(
            _preprocess_and_encode_recordings(st_recordings, settings=settings, representation=rep),
            drop_ignore=False,
        )

    model_names = [model] if model != "all" else ["raw", "bandpower", "time_frequency", "classical"]
    results: dict[str, object] = {
        "experiment_name": PRIMARY_EXPERIMENT_NAME,
        "sc_train_subjects": list(sc_train_subjects),
        "sc_val_subjects": list(sc_val_subjects),
        "st_test_subjects": list(st_subjects),
        "dataset_root": str(settings.acquisition.data_root),
        "models": {},
    }
    device = select_device()
    for rep in model_names:
        if rep == "classical":
            bp_col = collections["bandpower"]
            train_ds = EpochDataset(bp_col, subject_ids=sc_train, drop_ignore=False)
            val_ds = EpochDataset(bp_col, subject_ids=sc_val, drop_ignore=False)
            st_bp = _preprocess_and_encode_recordings(st_recordings, settings=settings, representation="bandpower")
            test_ds = EpochDataset(st_bp, drop_ignore=False)
            res = train_bandpower_logistic_regression(
                train_dataset=train_ds,
                val_dataset=val_ds,
                test_dataset=test_ds,
                ignore_index=settings.experiment.train.ignore_index,
                max_iter=settings.experiment.classical_baseline.max_iter,
                evaluate_test=True,
                random_state=settings.experiment.train.seed,
            )
            model_dir = _ensure_dir(PRIMARY_MODEL_ROOT / "classical")
            with (model_dir / "model.pkl").open("wb") as fh:
                pickle.dump(res.model, fh)
            meta_path = _save_model_metadata(
                model_dir=model_dir,
                experiment_name=PRIMARY_EXPERIMENT_NAME,
                representation="bandpower",
                settings=settings,
                sc_train_subjects=sc_train,
                sc_val_subjects=sc_val,
                st_test_subjects=st_subjects,
            )
            pred_dir, rep_dir, fig_dir = _artifact_paths(model_dir, "classical")
            pred_records = res.predictions
            pred_csv = pred_dir / "classical_predictions.csv"
            save_predictions_csv(
                build_epoch_predictions(
                    subject_id=pred_records[0].subject_id if pred_records else "",
                    recording_id=pred_records[0].recording_id if pred_records else "",
                    onsets_sec=[p.onset_sec or 0.0 for p in pred_records],
                    duration_sec=30.0,
                    y_true=[p.target for p in pred_records],
                    y_pred=[p.prediction for p in pred_records],
                ),
                pred_csv,
            )
            metrics_payload = {
                "experiment_name": PRIMARY_EXPERIMENT_NAME,
                "model": "classical_logistic_regression",
                "representation": "bandpower",
                "validation_metrics": res.validation_metrics.as_dict(),
                "test_metrics": None if res.test_metrics is None else res.test_metrics.as_dict(),
                "split": {"train": list(sc_train), "val": list(sc_val), "test": [], "st_test": list(st_subjects)},
                "model_metadata": str(meta_path),
                "predictions": str(pred_csv),
            }
            metrics_path = rep_dir / "classical_metrics.json"
            metrics_path.write_text(json.dumps(metrics_payload, indent=2, sort_keys=True), encoding="utf-8")
            results["models"][rep] = {
                "model_dir": str(model_dir),
                "metrics_path": str(metrics_path),
                "predictions_path": str(pred_csv),
                "metadata_path": str(meta_path),
                "test_metrics": None if res.test_metrics is None else res.test_metrics.as_dict(),
            }
            continue

        col = collections[rep]
        model_impl = build_baseline_model(rep, in_channels=col.items[0].metadata.feature_shape[0], n_band_features=col.items[0].metadata.feature_shape[-1] if rep == "bandpower" else 10)
        recipe = TrainRecipe(
            seed=settings.experiment.train.seed,
            batch_size=settings.experiment.train.batch_size,
            max_epochs=settings.experiment.train.max_epochs,
            learning_rate=settings.experiment.train.learning_rate,
            weight_decay=settings.experiment.train.weight_decay,
            num_workers=settings.experiment.train.num_workers,
            ignore_index=settings.experiment.train.ignore_index,
            class_weighting=settings.experiment.train.class_weighting,
            early_stopping_patience=settings.experiment.train.early_stopping_patience,
            checkpoint_dir=_ensure_dir(PRIMARY_MODEL_ROOT / REPRESENTATION_TO_MODEL_DIR[rep]),
        )
        train_ds = EpochDataset(col, subject_ids=sc_train, drop_ignore=False)
        val_ds = EpochDataset(col, subject_ids=sc_val, drop_ignore=False)
        test_ds = EpochDataset(
            _preprocess_and_encode_recordings(st_recordings, settings=settings, representation=rep),
            drop_ignore=False,
        )
        train_result = train_baseline(
            model_impl,
            train_dataset=train_ds,
            val_dataset=val_ds,
            test_dataset=test_ds,
            recipe=recipe,
            device=device,
            evaluate_test=True,
        )
        model_dir = _ensure_dir(PRIMARY_MODEL_ROOT / REPRESENTATION_TO_MODEL_DIR[rep])
        meta_path = _save_model_metadata(
            model_dir=model_dir,
            experiment_name=PRIMARY_EXPERIMENT_NAME,
            representation=rep,
            settings=settings,
            sc_train_subjects=sc_train,
            sc_val_subjects=sc_val,
            st_test_subjects=st_subjects,
        )
        if train_result.checkpoint_path is not None:
            # Mirror the checkpoint into the canonical model dir as well.
            ckpt_target = model_dir / "best_model.pt"
            ckpt_target.write_bytes(Path(train_result.checkpoint_path).read_bytes())
        pred_dir, rep_dir, fig_dir = _artifact_paths(model_dir, rep)
        # save predictions and metrics using existing output helpers
        if train_result.test_metrics is not None:
            onsets, y_true, y_pred, pred_records = _predict_epoch_dataset(model_impl, test_ds, recipe=recipe, device=device)
            pred_csv = pred_dir / f"{rep}_predictions.csv"
            save_predictions_csv(pred_records, pred_csv)
            plot_hypnogram(
                onsets_sec=onsets,
                y_true=y_true,
                y_pred=y_pred,
                out_path=fig_dir / f"{rep}_hypnogram.png",
                title=f"{rep} - SC→ST external test",
            )
        results["models"][rep] = {
            "model_dir": str(model_dir),
            "metadata_path": str(meta_path),
            "checkpoint_path": str(train_result.checkpoint_path) if train_result.checkpoint_path else None,
            "test_metrics": None if train_result.test_metrics is None else train_result.test_metrics.as_dict(),
            "device": train_result.device,
        }
    return results


def write_primary_inventory_report(report: Mapping[str, object], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
