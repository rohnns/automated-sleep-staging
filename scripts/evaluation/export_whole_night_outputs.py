#!/usr/bin/env python3
"""Export whole-night predictions and sleep statistics for all held-out test recordings.

Uses the existing Phase 4a trained checkpoints and the exact recovered held-out
subject split. Does not retrain or modify preprocessing / encoders / models.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from sleep_staging.acquisition.loader import discover_recordings, load_recordings
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.config.settings import load_settings
from sleep_staging.representations.factory import build_encoder
from sleep_staging.representations.types import EncodedDatasetCollection, LabelVocabulary
from sleep_staging.models.factory import build_baseline_model
from sleep_staging.evaluation.output import (
    build_epoch_predictions,
    compute_sleep_statistics,
    plot_hypnogram,
    save_predictions_csv,
)
from sleep_staging.preprocessing.pipeline import preprocess_recording
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.sc_counts import summarize_sc_subject_split
from sleep_staging.training.split import sleep_edf_subject_key
from sleep_staging.training.trainer import evaluate, make_loader


def _load_model_for_representation(rep: str, collection: EncodedDatasetCollection, checkpoint_dir: Path, device: torch.device) -> tuple[torch.nn.Module, object]:
    checkpoint_path = checkpoint_dir / rep / "best_model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    in_channels = collection.items[0].metadata.feature_shape[0]
    n_band_features = collection.items[0].metadata.feature_shape[-1] if rep == "bandpower" else 10
    model = build_baseline_model(rep, in_channels=in_channels, n_band_features=n_band_features)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device), checkpoint["recipe"]


def _run_representation_export(
    *,
    rep: str,
    encoded,
    device: torch.device,
    checkpoint_dir: Path,
    out_dir: Path,
    settings,
    selected_psg: Path,
    test_subjects: list[str],
    expected_test_supervised_epochs: int,
) -> dict[str, object]:
    collection = EncodedDatasetCollection((encoded,))
    model, recipe = _load_model_for_representation(rep, collection, checkpoint_dir, device)
    ds = EpochDataset(collection, drop_ignore=False)
    loader = make_loader(
        ds,
        batch_size=recipe.batch_size,
        shuffle=False,
        seed=recipe.seed,
        num_workers=recipe.num_workers,
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=recipe.ignore_index)
    result = evaluate(model, loader, device=device, ignore_index=recipe.ignore_index, criterion=criterion)

    y_true = np.asarray([ex.label for ex in ds.examples], dtype=np.int64)
    preds: list[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["features"].to(device))
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
    y_pred = np.asarray(preds, dtype=np.int64)
    onsets = [float(x) for x in encoded.onsets_sec]
    records = build_epoch_predictions(
        subject_id=encoded.subject_id,
        recording_id=encoded.recording_id,
        onsets_sec=onsets,
        duration_sec=encoded.metadata.epoch_duration_sec,
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
    )

    rep_dir = out_dir / rep
    rep_dir.mkdir(parents=True, exist_ok=True)
    save_predictions_csv(records, rep_dir / "predictions.csv")
    plot_hypnogram(
        onsets_sec=onsets,
        y_true=y_true.tolist(),
        y_pred=y_pred.tolist(),
        out_path=rep_dir / "hypnogram.png",
        title=f"{rep} - {selected_psg.stem}",
        epoch_duration_sec=encoded.metadata.epoch_duration_sec,
    )
    stats_true = compute_sleep_statistics(y_true.tolist(), epoch_duration_sec=encoded.metadata.epoch_duration_sec)
    stats_pred = compute_sleep_statistics(y_pred.tolist(), epoch_duration_sec=encoded.metadata.epoch_duration_sec)
    summary = {
        "selected_psg": str(selected_psg),
        "selected_subject": encoded.subject_id,
        "test_subjects": test_subjects,
        "expected_test_supervised_epochs": expected_test_supervised_epochs,
        "representation": rep,
        "metrics": result.metrics.as_dict(),
        "sleep_statistics_ground_truth": dataclasses.asdict(stats_true),
        "sleep_statistics_predicted": dataclasses.asdict(stats_pred),
        "n_epochs": int(encoded.n_epochs),
        "n_ignored": int(np.sum(y_true == settings.encodings.ignore_index)),
        "n_supervised": int(np.sum(y_true != settings.encodings.ignore_index)),
    }
    (rep_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    settings = load_settings("configs/default.yaml")
    data_root = settings.acquisition.data_root
    split, stats = summarize_sc_subject_split(
        data_root,
        ratios=settings.experiment.split.ratios,
        seed=settings.experiment.split.seed,
        wake_buffer_sec=settings.preprocessing.wake_crop.buffer_sec,
    )

    psg_files = discover_recordings(data_root)
    test_files = []
    for f in psg_files:
        ids = parse_psg_filename(f)
        if ids.study == "SC" and sleep_edf_subject_key(ids) in split.test:
            test_files.append(f)
    if not test_files:
        raise RuntimeError("Recovered test split contains no PSG files")

    vocab = LabelVocabulary(ignore_label=settings.encodings.ignore_label, ignore_index=settings.encodings.ignore_index)

    reps = ("raw", "bandpower", "time_frequency")
    encoders = {
        "raw": build_encoder(dataclasses.replace(settings.encodings, representation="raw")),
        "bandpower": build_encoder(dataclasses.replace(settings.encodings, representation="bandpower")),
        "time_frequency": build_encoder(dataclasses.replace(settings.encodings, representation="time_frequency")),
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path("outputs") / "phase4_outputs"
    out_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "test_subjects": list(split.test),
        "expected_test_supervised_epochs": stats["test"].n_epochs_supervised,
        "device": str(device),
        "recordings": {},
    }

    for selected_psg in test_files:
        recording = load_recordings([selected_psg], settings=settings.acquisition, preload=settings.acquisition.preload)[0]
        preprocessed = preprocess_recording(recording, settings.preprocessing, copy=False)
        recording_key = selected_psg.stem
        recording_dir = out_root / recording_key
        recording_dir.mkdir(parents=True, exist_ok=True)

        per_rep: dict[str, object] = {}
        for rep in reps:
            encoded = encoders[rep](preprocessed, vocabulary=vocab)
            per_rep[rep] = _run_representation_export(
                rep=rep,
                encoded=encoded,
                device=device,
                checkpoint_dir=Path(settings.experiment.train.checkpoint_dir),
                out_dir=recording_dir,
                settings=settings,
                selected_psg=selected_psg,
                test_subjects=list(split.test),
                expected_test_supervised_epochs=stats["test"].n_epochs_supervised,
            )

        (recording_dir / "manifest.json").write_text(
            json.dumps({"selected_psg": str(selected_psg), "representations": per_rep}, indent=2),
            encoding="utf-8",
        )
        manifest["recordings"][recording_key] = {"selected_psg": str(selected_psg), "representations": per_rep}

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
