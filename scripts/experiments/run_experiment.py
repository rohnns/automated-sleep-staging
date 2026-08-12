#!/usr/bin/env python
"""
Phase 4a end-to-end experiment runner
======================================
Preprocess all Sleep-EDF SC recordings → encode (Raw, BandPower, STFT) →
train the three baseline models on a shared 70/15/15 subject-wise split →
evaluate on held-out test set → write a Markdown comparison report.

Usage (from repo root):
    $env:PYTHONPATH = 'src'
    python scripts/run_experiment.py

Optional flags (edit the CONFIG block below):
    MAX_RECORDINGS  – None = all, int = first N (for smoke tests)
    MAX_EPOCHS      – PyTorch training epochs per model
    BATCH_SIZE      – dataloader batch size
    EVALUATE_TEST   – True to compute test metrics at the end
"""

from __future__ import annotations

import pathlib
import time
from collections import defaultdict
from typing import Any

import mne
import numpy as np
import torch

mne.set_log_level("ERROR")

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from sleep_staging.config.settings import load_settings
from sleep_staging.acquisition.loader import SleepEDFLoader
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.preprocessing.pipeline import build_default_pipeline, preprocess_recording
from sleep_staging.representations.factory import build_encoder
from sleep_staging.representations.types import EncodedDatasetCollection
from sleep_staging.training.split import (
    SubjectSplit,
    assert_no_subject_leakage,
    sleep_edf_subject_key,
    subject_wise_split,
)
from sleep_staging.training.experiment import run_controlled_pytorch_experiments
from sleep_staging.training.trainer import TrainRecipe, select_device
from sleep_staging.training.metrics import STAGE_NAMES

# ---------------------------------------------------------------------------
# CONFIG – edit here for smoke tests or full runs
# ---------------------------------------------------------------------------
DATA_ROOT = pathlib.Path(r"D:\SleepEDFX")
CONFIG_PATH = pathlib.Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
REPORT_PATH = pathlib.Path(__file__).resolve().parents[2] / "outputs" / "reports" / "experiment_results.md"

import os

MAX_RECORDINGS: int | None = (
    int(os.environ["MAX_RECORDINGS"]) if "MAX_RECORDINGS" in os.environ else None
)  # None = all 197; set env var for smoke tests
MAX_EPOCHS: int = int(os.environ.get("MAX_EPOCHS", "20"))
BATCH_SIZE: int = int(os.environ.get("BATCH_SIZE", "64"))
EARLY_STOPPING_PATIENCE: int = 5
SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
SPLIT_SEED: int = 42
EVALUATE_TEST: bool = True

REPRESENTATIONS = ("raw", "bandpower", "time_frequency")
REPRESENTATION_LABELS = {
    "raw": "Raw CNN-1D",
    "bandpower": "BandPower MLP",
    "time_frequency": "STFT CNN-2D",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hms(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _subject_key(psg_path: pathlib.Path) -> str:
    ids = parse_psg_filename(psg_path)
    return sleep_edf_subject_key(ids)


# ---------------------------------------------------------------------------
# Step 1: Discover PSG paths and build subject-wise split
# ---------------------------------------------------------------------------
print("=" * 70)
print("Phase 4a Sleep-EDF SC Experiment")
print("=" * 70)

settings = load_settings(CONFIG_PATH)
# Override data root
from sleep_staging.config.settings import AcquisitionSettings
acq_settings = AcquisitionSettings(
    data_root=DATA_ROOT,
    preload=settings.acquisition.preload,
    stim_channel=settings.acquisition.stim_channel,
    infer_types=settings.acquisition.infer_types,
    mne_verbose=settings.acquisition.mne_verbose,
)

from sleep_staging.acquisition.utils import discover_psg_files
all_psg_paths = discover_psg_files(DATA_ROOT)
if MAX_RECORDINGS is not None:
    all_psg_paths = all_psg_paths[:MAX_RECORDINGS]

print(f"\n[1/5] Discovered {len(all_psg_paths)} PSG files under {DATA_ROOT}")
# Note: subject split is deferred until after encoding so IDs match EncodedDataset.subject_id

# ---------------------------------------------------------------------------
# Step 2: Build preprocessing pipeline & encoders
# ---------------------------------------------------------------------------
import dataclasses
from sleep_staging.config.settings import EncodingSettings

print("[2/5] Building preprocessing pipeline and encoders ...")
pipeline = build_default_pipeline(settings.preprocessing)

encoders = {}
for rep in REPRESENTATIONS:
    enc_settings = dataclasses.replace(settings.encodings, representation=rep)
    encoders[rep] = build_encoder(enc_settings)
    print(f"       Encoder ready: {rep}")

# ---------------------------------------------------------------------------
# Step 3: Preprocess and encode all recordings
# ---------------------------------------------------------------------------
print(f"\n[3/5] Preprocessing + encoding {len(all_psg_paths)} recordings …")

loader = SleepEDFLoader(settings=acq_settings)

# collections[representation] → list[EncodedDataset]
raw_items: list[Any] = []
bp_items: list[Any] = []
tf_items: list[Any] = []

t0 = time.time()
errors: list[str] = []

for rec_idx, psg_path in enumerate(all_psg_paths, start=1):
    psg_name = psg_path.stem
    elapsed = time.time() - t0
    eta = (elapsed / rec_idx) * (len(all_psg_paths) - rec_idx) if rec_idx > 1 else 0
    print(
        f"  [{rec_idx:3d}/{len(all_psg_paths)}] {psg_name}"
        f"  elapsed={_hms(elapsed)}  ETA={_hms(eta)}",
        flush=True,
    )

    try:
        recording = loader.load_recording(psg_path)
        preprocessed = preprocess_recording(recording, settings.preprocessing)

        for rep, encoder in encoders.items():
            encoded = encoder.encode_recording(preprocessed)
            if rep == "raw":
                raw_items.append(encoded)
            elif rep == "bandpower":
                bp_items.append(encoded)
            else:
                tf_items.append(encoded)

    except Exception as exc:
        msg = f"SKIP {psg_name}: {exc}"
        print(f"    WARNING: {msg}")
        errors.append(msg)

total_time = time.time() - t0
print(f"\n  Done in {_hms(total_time)}. Skipped {len(errors)} recording(s).")

collections = {
    "raw": EncodedDatasetCollection(items=tuple(raw_items)),
    "bandpower": EncodedDatasetCollection(items=tuple(bp_items)),
    "time_frequency": EncodedDatasetCollection(items=tuple(tf_items)),
}
for rep, col in collections.items():
    print(f"  {rep:15s}: {len(col)} recordings encoded")

# Build subject-wise split from the subject IDs that are actually stored in the
# EncodedDataset objects. These are metadata.subject_id (e.g. '00', '01', ...)
# and are what EpochDataset.filter_collection_by_subjects matches against.
# Both nights of a subject share the same subject_id, so grouping is correct.
unique_subjects = sorted({item.subject_id for item in collections["raw"].items})
print(f"       {len(unique_subjects)} unique subjects (from encoded data)")

split: SubjectSplit = subject_wise_split(unique_subjects, ratios=SPLIT_RATIOS, seed=SPLIT_SEED)
assert_no_subject_leakage(split)
print(
    f"       Split (seed={SPLIT_SEED}): "
    f"train={len(split.train)} | val={len(split.val)} | test={len(split.test)} subjects"
)

# ---------------------------------------------------------------------------
# Step 4: Train baseline models
# ---------------------------------------------------------------------------
print("\n[4/5] Training baseline models …")

device = select_device()
print(f"  Device: {device}")

recipe = TrainRecipe(
    seed=SPLIT_SEED,
    batch_size=BATCH_SIZE,
    max_epochs=MAX_EPOCHS,
    learning_rate=1e-3,
    weight_decay=0.0,
    class_weighting="balanced",
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    checkpoint_dir=pathlib.Path(__file__).resolve().parents[2] / "checkpoints",
)

t1 = time.time()
experiment_result = run_controlled_pytorch_experiments(
    collections,
    split=split,
    recipe=recipe,
    checkpoint_root=recipe.checkpoint_dir,
    evaluate_test=EVALUATE_TEST,
    device=device,
)
train_time = time.time() - t1
print(f"  Training complete in {_hms(train_time)}.")

# ---------------------------------------------------------------------------
# Step 5: Build report
# ---------------------------------------------------------------------------
print("\n[5/5] Building results report …")

lines: list[str] = []
lines += [
    "# Phase 4a Experiment Results: Raw / BandPower / STFT",
    "",
    f"**Dataset**: Sleep-EDF SC (Cassette)  ",
    f"**Recordings processed**: {len(all_psg_paths) - len(errors)} / {len(all_psg_paths)}  ",
    f"**Split** (seed={SPLIT_SEED}, 70/15/15 subject-wise): "
    f"train={len(split.train)} | val={len(split.val)} | test={len(split.test)} subjects  ",
    f"**Device**: {device}  ",
    f"**Training epochs (max)**: {MAX_EPOCHS}  ",
    f"**Batch size**: {BATCH_SIZE}  ",
    "",
]

# --- Summary table ---
lines += [
    "## Summary — Test-set metrics",
    "",
    "| Model | Accuracy | Macro-F1 | Cohen's κ |",
    "|-------|----------|----------|-----------|",
]
for rep in REPRESENTATIONS:
    label = REPRESENTATION_LABELS[rep]
    run = experiment_result.results[rep]
    tm = run.train_result.test_metrics
    if tm is not None:
        lines.append(
            f"| {label} | {tm.accuracy:.4f} | {tm.macro_f1:.4f} | {tm.cohen_kappa:.4f} |"
        )
    else:
        lines.append(f"| {label} | — | — | — |")

lines.append("")

# --- Per-class F1 table ---
lines += [
    "## Per-class F1 scores (test set)",
    "",
    "| Model | W | N1 | N2 | N3 | REM |",
    "|-------|---|----|----|-----|-----|",
]
for rep in REPRESENTATIONS:
    label = REPRESENTATION_LABELS[rep]
    run = experiment_result.results[rep]
    tm = run.train_result.test_metrics
    if tm is not None:
        f1s = [f"{f:.4f}" for f in tm.per_class_f1]
        lines.append(f"| {label} | {' | '.join(f1s)} |")
    else:
        lines.append(f"| {label} | — | — | — | — | — |")

lines.append("")

# --- Training details per model ---
lines += ["## Training details", ""]
for rep in REPRESENTATIONS:
    label = REPRESENTATION_LABELS[rep]
    run = experiment_result.results[rep]
    tr = run.train_result
    vm_best = tr.best_val_macro_f1
    lines += [
        f"### {label}",
        f"- Best epoch: {tr.best_epoch}",
        f"- Best val macro-F1: {vm_best:.4f}",
        f"- Early stopping: {'yes' if tr.stopped_early else 'no'}",
        f"- Device: {tr.device}",
        "",
    ]
    # Training history table (last 5 epochs)
    history = tr.history[-10:]
    lines += [
        "| Epoch | Train Loss | Val Loss | Val Macro-F1 |",
        "|-------|-----------|---------|-------------|",
    ]
    for row in history:
        lines.append(
            f"| {int(row['epoch'])} | {row['train_loss']:.4f} | {row['val_loss']:.4f}"
            f" | {row['val_macro_f1']:.4f} |"
        )
    lines.append("")

# --- Errors ---
if errors:
    lines += ["## Skipped recordings", ""]
    for e in errors:
        lines.append(f"- {e}")
    lines.append("")

REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport written to: {REPORT_PATH}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("RESULTS SUMMARY")
print("=" * 70)
print(f"{'Model':<20} {'Accuracy':>10} {'Macro-F1':>10} {'Cohen K':>10}")
print("-" * 54)
for rep in REPRESENTATIONS:
    label = REPRESENTATION_LABELS[rep]
    tm = experiment_result.results[rep].train_result.test_metrics
    if tm:
        print(f"{label:<20} {tm.accuracy:>10.4f} {tm.macro_f1:>10.4f} {tm.cohen_kappa:>10.4f}")
    else:
        print(f"{label:<20} {'—':>10} {'—':>10} {'—':>10}")
print("=" * 70)
print(f"Total wall time: {_hms(time.time() - t0)}")
