import sys
import json
import dataclasses
import torch
import numpy as np
from pathlib import Path
from sleep_staging.config.settings import load_settings
from sleep_staging.training.sc_counts import summarize_sc_subject_split
from sleep_staging.acquisition.loader import load_recordings, discover_recordings
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.preprocessing.pipeline import preprocess_recording
from sleep_staging.representations.factory import build_encoder
from sleep_staging.representations.types import EncodedDatasetCollection, LabelVocabulary
from sleep_staging.models.factory import build_baseline_model
from sleep_staging.training.dataset import EpochDataset
from sleep_staging.training.trainer import make_loader, evaluate
import logging
logger = logging.getLogger(__name__)
import dataclasses

def main():
    print("Loading settings...")
    settings = load_settings("configs/default.yaml")

    data_root = settings.acquisition.data_root
    ratios = settings.experiment.split.ratios
    seed = settings.experiment.split.seed
    
    # 1. Recover the exact split
    split, stats = summarize_sc_subject_split(
        data_root, 
        ratios=ratios, 
        seed=seed, 
        wake_buffer_sec=settings.preprocessing.wake_crop.buffer_sec
    )
    print(f"Test split subjects ({len(split.test)}): {split.test}")
    print(f"Expected test epochs (supervised): {stats['test'].n_epochs_supervised}")

    # 2. Load ONLY the test recordings to save time and memory
    from sleep_staging.training.split import sleep_edf_subject_key
    psg_files = discover_recordings(data_root)
    test_files = []
    print(f"Found {len(psg_files)} PSG files total.")
    for f in psg_files:
        ids = parse_psg_filename(f)
        if ids.study == "SC":
            subj_key = sleep_edf_subject_key(ids)
            if subj_key in split.test:
                test_files.append(f)
    
    print(f"Found {len(test_files)} files for test split.")

    raw_enc_settings = dataclasses.replace(settings.encodings, representation="raw")
    bp_enc_settings = dataclasses.replace(settings.encodings, representation="bandpower")
    tf_enc_settings = dataclasses.replace(settings.encodings, representation="time_frequency")

    encoders = {
        "raw": build_encoder(raw_enc_settings),
        "bandpower": build_encoder(bp_enc_settings),
        "time_frequency": build_encoder(tf_enc_settings)
    }

    vocab = LabelVocabulary(
        ignore_label=settings.encodings.ignore_label,
        ignore_index=settings.encodings.ignore_index
    )

    test_datasets = {"raw": [], "bandpower": [], "time_frequency": []}

    print("Preprocessing and encoding test set...")
    for idx, f in enumerate(test_files):
        recording = load_recordings([f], settings=settings.acquisition, preload=settings.acquisition.preload)[0]
        preprocessed = preprocess_recording(recording, settings.preprocessing, copy=False)
        for rep, enc in encoders.items():
            ds = enc(preprocessed, vocabulary=vocab)
            test_datasets[rep].append(ds)
        del recording
        del preprocessed

    collections = {
        rep: EncodedDatasetCollection(tuple(dsets))
        for rep, dsets in test_datasets.items()
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}...")
    
    report = {}

    for rep in ["raw", "bandpower", "time_frequency"]:
        print(f"\n--- Evaluating {rep} ---")
        checkpoint_dir = settings.experiment.train.checkpoint_dir
        if checkpoint_dir is None:
            raise ValueError("checkpoint_dir is None in settings")
        
        chk_path = Path(checkpoint_dir) / rep / "best_model.pt"
        if not chk_path.exists():
            print(f"Checkpoint not found for {rep} at {chk_path}")
            continue
            
        checkpoint = torch.load(chk_path, map_location=device, weights_only=False)
        
        collection = collections[rep]
        in_channels = collection.items[0].metadata.feature_shape[0]
        n_band_features = collection.items[0].metadata.feature_shape[-1] if rep == "bandpower" else 10
        
        model = build_baseline_model(rep, in_channels=in_channels, n_band_features=n_band_features)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)
        
        test_ds = EpochDataset(collection, drop_ignore=False)
        recipe = checkpoint["recipe"]

        # Sanity check: ensure test dataset has supervised examples
        if len(test_ds) == 0:
            raise RuntimeError("Test dataset is empty after filtering. Check split and subject IDs.")
        # Count supervised vs ignore
        n_total = len(test_ds)
        n_ignore = sum(1 for ex in test_ds.examples if ex.label == recipe.ignore_index)
        n_supervised = n_total - n_ignore
        logger.info(f"Test dataset: {n_total} epochs, {n_ignore} IGNORE, {n_supervised} supervised")
        if n_supervised == 0:
            raise RuntimeError("No supervised test epochs found. Verify split and encoding settings.")
        
        # Count total epochs and ignored epochs in test_ds
        total_epochs = sum(len(ex.label.shape) if hasattr(ex.label, 'shape') else 1 for ex in test_ds.examples)
        ignored_epochs = sum(1 for ex in test_ds.examples if int(ex.label) == recipe.ignore_index)
        evaluated_epochs = total_epochs - ignored_epochs
        
        test_loader = make_loader(
            test_ds,
            batch_size=recipe.batch_size,
            shuffle=False,
            seed=recipe.seed,
            num_workers=recipe.num_workers
        )
        
        criterion = torch.nn.CrossEntropyLoss(ignore_index=recipe.ignore_index)
        eval_res = evaluate(model, test_loader, device=device, ignore_index=recipe.ignore_index, criterion=criterion)
        metrics = eval_res.metrics
        
        rep_results = {
            "Accuracy": float(metrics.accuracy),
            "Macro-F1": float(metrics.macro_f1),
            "Cohen's Kappa": float(metrics.cohen_kappa),
            "Per-Class (W, N1, N2, N3, REM)": {
                "Precision": [float(x) for x in metrics.per_class_precision],
                "Recall": [float(x) for x in metrics.per_class_recall],
                "F1": [float(x) for x in metrics.per_class_f1]
            },
            "Confusion Matrix": [list(int(x) for x in row) for row in metrics.confusion_matrix],
            "Total Test Epochs": total_epochs,
            "Ignored Epochs": ignored_epochs,
            "Evaluated Epochs": evaluated_epochs
        }
        report[rep] = rep_results
        
        print(f"Accuracy: {metrics.accuracy:.4f}")
        print(f"Macro-F1: {metrics.macro_f1:.4f}")
        print(f"Cohen's Kappa: {metrics.cohen_kappa:.4f}")

    # Write report to markdown artifact
    report_path = Path("C:/Users/hp/.gemini/antigravity/brain/1e63826a-219a-4376-a52f-53d4f9a3b9ca/evaluation_report.md")
    with open(report_path, "w") as f:
        f.write("# Phase 4a Baseline Evaluation Report\n\n")
        f.write("This evaluation uses the exact held-out test split from the training run.\n\n")
        for rep, res in report.items():
            f.write(f"## Representation: {rep}\n")
            f.write(f"- **Accuracy**: {res['Accuracy']:.4f}\n")
            f.write(f"- **Macro-F1**: {res['Macro-F1']:.4f}\n")
            f.write(f"- **Cohen's Kappa**: {res['Cohen\'s Kappa']:.4f}\n")
            f.write(f"- **Total Test Epochs**: {res['Total Test Epochs']} (Evaluated: {res['Evaluated Epochs']}, Ignored: {res['Ignored Epochs']})\n\n")
            
            f.write("### Per-Class Metrics (W, N1, N2, N3, REM)\n")
            f.write("| Metric | W | N1 | N2 | N3 | REM |\n")
            f.write("|---|---|---|---|---|---|\n")
            p = res["Per-Class (W, N1, N2, N3, REM)"]["Precision"]
            r = res["Per-Class (W, N1, N2, N3, REM)"]["Recall"]
            f1 = res["Per-Class (W, N1, N2, N3, REM)"]["F1"]
            f.write(f"| Precision | {p[0]:.4f} | {p[1]:.4f} | {p[2]:.4f} | {p[3]:.4f} | {p[4]:.4f} |\n")
            f.write(f"| Recall | {r[0]:.4f} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.4f} |\n")
            f.write(f"| F1 | {f1[0]:.4f} | {f1[1]:.4f} | {f1[2]:.4f} | {f1[3]:.4f} | {f1[4]:.4f} |\n\n")
            
            f.write("### Confusion Matrix\n")
            f.write("Rows: True, Columns: Predicted\n")
            f.write("```\n")
            cm = np.array(res["Confusion Matrix"])
            f.write(np.array2string(cm))
            f.write("\n```\n\n")

if __name__ == "__main__":
    main()
