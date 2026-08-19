"""Primary SC→ST external-generalization experiment orchestration.

Naming
------
``SC`` and ``ST`` are the two Sleep-EDF Expanded cohorts:

``SC`` = **Sleep Cassette** — 153 recordings from healthy subjects studied in
their own homes (the training corpus here).
``ST`` = **Sleep Telemetry** — 44 recordings from subjects with mild
difficulty falling asleep, studied in hospital (the external test corpus).

"SC→ST" therefore means *train on Sleep Cassette, test on Sleep Telemetry*:
a deliberately harder external-validation protocol than a random split,
because the test cohort differs in both population and recording setting.

Protocol
--------
1. Discover SC and ST recordings independently.
2. Build a subject-wise train/validation split over **SC only**
   (55 train / 12 validation subjects; no subject spans both).
3. Train and select checkpoints using SC train/validation exclusively.
4. Evaluate once on the **entire ST cohort** (22 subjects) as a clean,
   never-tuned-on external test set.

This module keeps the primary experiment separate from the legacy SC-only
artifacts, so neither can contaminate the other.
"""

from __future__ import annotations

from dataclasses import asdict
import dataclasses
import gc
import json
import pickle
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import torch

from sleep_staging import __version__ as PACKAGE_VERSION
from sleep_staging.acquisition.loader import discover_recordings
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.common.logging_utils import get_logger
from sleep_staging.config import load_settings
from sleep_staging.evaluation.output import (
    EpochPredictionRecord,
    build_epoch_predictions,
    plot_hypnogram,
    plot_loss_curves,
    save_predictions_csv,
    save_training_history,
)
from sleep_staging.models.factory import build_baseline_model
from sleep_staging.representations.types import EncodedDatasetCollection
from sleep_staging.training import sc_to_st_cache
from sleep_staging.training.classical import train_bandpower_logistic_regression
from sleep_staging.training.dataset import EpochDataset, summarize_partition
from sleep_staging.training.split import assert_no_subject_leakage, subject_wise_split
from sleep_staging.training.trainer import (
    TrainRecipe,
    autocast_for_device,
    make_loader,
    select_device,
    train_baseline,
)


logger = get_logger(__name__)

PRIMARY_EXPERIMENT_NAME = "sc_to_st_external_generalization"
PRIMARY_ARTIFACT_ROOT = Path("artifacts")
PRIMARY_MODEL_ROOT = Path("models")

REPRESENTATION_TO_MODEL_DIR = {
    "raw": "raw_cnn",
    "bandpower": "bandpower_mlp",
    "time_frequency": "stft_cnn",
}


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def discover_sc_st_recordings(dataset_root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    files = discover_recordings(dataset_root)
    sc = tuple(p for p in files if parse_psg_filename(p).study == "SC")
    st = tuple(p for p in files if parse_psg_filename(p).study == "ST")
    return sc, st


def _subject_key(path: Path) -> str:
    """Subject grouping key for splitting -- MUST match ``EncodedDataset.subject_id``.

    ``RecordingMetadata``/``EncodedDataset.subject_id`` is set (in
    ``acquisition/metadata.py::extract_metadata``) to the bare 2-digit
    ``SleepEDFFileIds.subject_id`` (e.g. ``"00"``), *not*
    ``sleep_edf_subject_key`` (``study+series+subject``, e.g. ``"SC400"``).
    ``EpochDataset``/``filter_collection_by_subjects`` filter encoded
    collections by exact string equality against ``item.subject_id``, so a
    split key built any other way silently matches zero recordings.

    The bare subject code is safe to use as the grouping key here because
    ``sc_recordings``/``st_recordings`` are already filtered to a single
    study each (SC or ST) with a constant ``series`` digit, so it remains
    unique per real subject within each cohort -- it just needs to be the
    *same string* the encoder actually stored.
    """
    return parse_psg_filename(path).subject_id


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
    """Encode recordings for one representation, resuming from per-recording cache.

    Delegates to :mod:`sleep_staging.training.sc_to_st_cache`, which checks
    ``artifacts/cache/sc_to_st/<representation>/`` for each recording before
    falling back to the production preprocessing + encoding pipeline. This is
    the single choke point every caller in this module goes through, so the
    cache applies uniformly to SC and ST recordings and to all four models.
    """
    return sc_to_st_cache.encode_recordings_with_cache(
        recording_paths,
        settings=settings,
        representation=representation,
    )


def build_primary_collections(
    *,
    config_path: Path,
    dataset_root: Path | None = None,
    max_sc_recordings: int | None = None,
    max_st_recordings: int | None = None,
) -> tuple[object, tuple[Path, ...], tuple[Path, ...]]:
    """Resolve settings and discover SC/ST recording paths.

    Deliberately does NOT encode any representation here. Measured from the
    on-disk cache: raw ~8.0 GB (SC 6.57 + ST 1.43), time_frequency ~5.24 GB
    (SC 4.31 + ST 0.93), bandpower ~tiny -- holding all three fully-decoded
    collections in memory simultaneously (~12.4 GB+, before Python/torch/mne
    process overhead) exceeds this machine's ~15.65 GB total RAM. Callers
    must encode one representation at a time via
    ``_preprocess_and_encode_recordings`` and release it before moving to
    the next -- see ``run_primary_experiment``, which processes
    representations sequentially for exactly this reason.
    """
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
    return settings, sc_recordings, st_recordings


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
            with autocast_for_device(device):
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


def _class_distribution(dataset: EpochDataset) -> dict[int, int]:
    counts: dict[int, int] = {}
    for example in dataset.examples:
        counts[int(example.label)] = counts.get(int(example.label), 0) + 1
    return counts


def _log_split_diagnostics(
    *,
    rep: str,
    collection: EncodedDatasetCollection,
    sc_train: tuple[str, ...],
    sc_val: tuple[str, ...],
    train_ds: EpochDataset,
    val_ds: EpochDataset,
    test_ds: EpochDataset,
) -> None:
    """Log concrete counts at every filtering stage and fail loudly if a
    split ends up empty, instead of surfacing as a cryptic
    ``num_samples=0`` deep inside torch.utils.data.DataLoader/RandomSampler.
    """
    available_subjects = tuple(sorted({item.subject_id for item in collection.items}))
    train_stats = summarize_partition(collection, sc_train)
    val_stats = summarize_partition(collection, sc_val)
    logger.info(
        "[%s] encoded collection: %d recording(s), %d distinct subject key(s): %s",
        rep,
        len(collection.items),
        len(available_subjects),
        available_subjects,
    )
    logger.info(
        "[%s] sc_train: requested %d subject key(s) -> matched %d recording(s), "
        "%d epoch(s) (supervised=%d, ignore=%d) -> train_dataset len=%d",
        rep,
        len(sc_train),
        train_stats.n_subjects,
        train_stats.n_epochs,
        train_stats.n_epochs_supervised,
        train_stats.n_epochs_ignore,
        len(train_ds),
    )
    logger.info(
        "[%s] sc_val: requested %d subject key(s) -> matched %d recording(s), "
        "%d epoch(s) (supervised=%d, ignore=%d) -> val_dataset len=%d",
        rep,
        len(sc_val),
        val_stats.n_subjects,
        val_stats.n_epochs,
        val_stats.n_epochs_supervised,
        val_stats.n_epochs_ignore,
        len(val_ds),
    )
    logger.info("[%s] test_dataset (full ST cohort, unfiltered) len=%d", rep, len(test_ds))
    logger.info(
        "[%s] class distribution: train=%s val=%s",
        rep,
        _class_distribution(train_ds),
        _class_distribution(val_ds),
    )
    if len(train_ds) == 0:
        raise ValueError(
            f"[{rep}] train_dataset has 0 examples after subject filtering. "
            f"Requested sc_train subject key(s) {sc_train!r} matched "
            f"{train_stats.n_subjects} recording(s) out of {len(available_subjects)} "
            f"available subject key(s) in the encoded collection: {available_subjects!r}. "
            "This almost always means the split-builder's subject-key format "
            "does not match EncodedDataset.subject_id (see _subject_key)."
        )
    if len(val_ds) == 0:
        raise ValueError(
            f"[{rep}] val_dataset has 0 examples after subject filtering. "
            f"Requested sc_val subject key(s) {sc_val!r} matched {val_stats.n_subjects} "
            f"recording(s) out of {len(available_subjects)} available: {available_subjects!r}."
        )


def run_primary_experiment(
    *,
    config_path: Path,
    dataset_root: Path | None = None,
    model: str = "all",
    max_sc_recordings: int | None = None,
    max_st_recordings: int | None = None,
    smoke: bool = False,
) -> dict[str, object]:
    settings, sc_recordings, st_recordings = build_primary_collections(
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
        if device.type == "cuda":
            # Release any cached-but-unused blocks from the previous
            # representation's training before starting the next one.
            torch.cuda.empty_cache()
        if rep == "classical":
            bp_col = _preprocess_and_encode_recordings(sc_recordings, settings=settings, representation="bandpower")
            train_ds = EpochDataset(bp_col, subject_ids=sc_train, drop_ignore=False)
            val_ds = EpochDataset(bp_col, subject_ids=sc_val, drop_ignore=False)
            st_bp = _preprocess_and_encode_recordings(st_recordings, settings=settings, representation="bandpower")
            test_ds = EpochDataset(st_bp, drop_ignore=False)
            _log_split_diagnostics(
                rep="classical(bandpower)",
                collection=bp_col,
                sc_train=sc_train,
                sc_val=sc_val,
                train_ds=train_ds,
                val_ds=val_ds,
                test_ds=test_ds,
            )
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
            # Release this representation's data before the next one starts;
            # see build_primary_collections's docstring for why this matters
            # (~12+ GB to hold all three representations at once vs. ~16 GB
            # total system RAM on this machine).
            del bp_col, train_ds, val_ds, st_bp, test_ds, res
            gc.collect()
            continue

        col = _preprocess_and_encode_recordings(sc_recordings, settings=settings, representation=rep)
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
            block_size=settings.experiment.train.block_size,
        )
        train_ds = EpochDataset(col, subject_ids=sc_train, drop_ignore=False)
        val_ds = EpochDataset(col, subject_ids=sc_val, drop_ignore=False)
        test_ds = EpochDataset(
            _preprocess_and_encode_recordings(st_recordings, settings=settings, representation=rep),
            drop_ignore=False,
        )
        _log_split_diagnostics(
            rep=rep,
            collection=col,
            sc_train=sc_train,
            sc_val=sc_val,
            train_ds=train_ds,
            val_ds=val_ds,
            test_ds=test_ds,
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

        # Persist the training dynamics the trainer already tracks, so
        # train-vs-validation behaviour (and the overfitting gap) is auditable
        # after the run instead of living only in stdout. Purely additive --
        # nothing here feeds back into training or metrics.
        loss_curve_path = fig_dir / f"{rep}_loss_curves.png"
        history_path = rep_dir / f"{rep}_training_history.json"
        if train_result.history:
            save_training_history(
                train_result.history, history_path, best_epoch=train_result.best_epoch
            )
            plot_loss_curves(
                train_result.history,
                loss_curve_path,
                title=f"{rep} — train vs validation loss (SC train/validation)",
                best_epoch=train_result.best_epoch,
            )
            logger.info(
                "[%s] wrote loss curves -> %s (best epoch=%d)",
                rep,
                loss_curve_path,
                train_result.best_epoch,
            )

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
            "loss_curve_path": str(loss_curve_path) if train_result.history else None,
            "training_history_path": str(history_path) if train_result.history else None,
            "best_epoch": train_result.best_epoch,
            "stopped_early": train_result.stopped_early,
        }
        # Release this representation's data before the next one starts;
        # see build_primary_collections's docstring for why this matters
        # (~12+ GB to hold all three representations at once vs. ~16 GB
        # total system RAM on this machine).
        del col, train_ds, val_ds, test_ds, model_impl, train_result
        gc.collect()
    return results


def write_primary_inventory_report(report: Mapping[str, object], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return out_path
