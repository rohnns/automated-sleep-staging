import sys
import json
from pathlib import Path

import torch
import numpy as np

from sleep_staging.config.settings import load_settings
from sleep_staging.acquisition.loader import load_recording, discover_recordings
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.preprocessing.pipeline import preprocess_recording
from sleep_staging.representations.factory import build_encoder
from sleep_staging.representations.types import EncodedDatasetCollection, LabelVocabulary
from sleep_staging.training.experiment import run_controlled_pytorch_experiments
import dataclasses

def main():
    print("Loading settings...")
    settings = load_settings("configs/default.yaml")

    data_root = settings.acquisition.data_root
    print(f"Discovering SC recordings in {data_root}...")
    psg_files = discover_recordings(data_root)
    sc_files = [f for f in psg_files if parse_psg_filename(f).study == "SC"]
    print(f"Found {len(sc_files)} SC recordings.")

    raw_enc_settings = dataclasses.replace(settings.encodings, representation="raw")
    bp_enc_settings = dataclasses.replace(settings.encodings, representation="bandpower")
    tf_enc_settings = dataclasses.replace(settings.encodings, representation="time_frequency")

    raw_encoder = build_encoder(raw_enc_settings)
    bp_encoder = build_encoder(bp_enc_settings)
    tf_encoder = build_encoder(tf_enc_settings)

    vocab = LabelVocabulary(
        ignore_label=settings.encodings.ignore_label,
        ignore_index=settings.encodings.ignore_index
    )

    raw_datasets = []
    bp_datasets = []
    tf_datasets = []

    for idx, psg_file in enumerate(sc_files):
        print(f"[{idx+1}/{len(sc_files)}] Processing {psg_file.name}...")
        try:
            recording = load_recording(
                psg_file,
                settings=settings.acquisition,
                preload=settings.acquisition.preload
            )
            preprocessed = preprocess_recording(recording, settings.preprocessing, copy=False)
            
            raw_ds = raw_encoder(preprocessed, vocabulary=vocab)
            bp_ds = bp_encoder(preprocessed, vocabulary=vocab)
            tf_ds = tf_encoder(preprocessed, vocabulary=vocab)
            
            raw_datasets.append(raw_ds)
            bp_datasets.append(bp_ds)
            tf_datasets.append(tf_ds)
            
            # Explicitly free memory if needed, though GC should handle it
            del recording
            del preprocessed
        except Exception as e:
            print(f"Failed to process {psg_file.name}: {e}")

    collections = {
        "raw": EncodedDatasetCollection(tuple(raw_datasets)),
        "bandpower": EncodedDatasetCollection(tuple(bp_datasets)),
        "time_frequency": EncodedDatasetCollection(tuple(tf_datasets))
    }

    print("Running controlled PyTorch experiments...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    from sleep_staging.training.trainer import TrainRecipe
    
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
        checkpoint_dir=settings.experiment.train.checkpoint_dir
    )

    import time
    start_time = time.time()
    
    experiment_result = run_controlled_pytorch_experiments(
        collections,
        split=None,
        ratios=settings.experiment.split.ratios,
        seed=settings.experiment.split.seed,
        recipe=recipe,
        checkpoint_root=settings.experiment.train.checkpoint_dir,
        evaluate_test=True,
        device=device
    )

    end_time = time.time()
    total_train_time = end_time - start_time

    print("\n--- RESULTS ---")
    for rep, res in experiment_result.results.items():
        tr = res.train_result
        metrics = tr.test_metrics
        print(f"\nRepresentation: {rep}")
        print(f"Total Experiment Train Time (all 3 models): {total_train_time:.1f} s")
        print(f"Best Val Epoch: {tr.best_epoch}")
        if metrics:
            print(f"Test Accuracy: {metrics.accuracy:.4f}")
            print(f"Test Macro-F1: {metrics.macro_f1:.4f}")
            print(f"Cohen's Kappa: {metrics.cohen_kappa:.4f}")
            print("Per-class metrics (W, N1, N2, N3, REM):")
            print(f"  Precision: {metrics.precision}")
            print(f"  Recall:    {metrics.recall}")
            print(f"  F1:        {metrics.f1_score}")
            print("Confusion Matrix:")
            print(np.array(metrics.confusion_matrix))

if __name__ == "__main__":
    main()
