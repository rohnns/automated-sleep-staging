#!/usr/bin/env python
"""
Efficient class‑distribution analysis for Sleep‑EDF SC.

- Uses the acquisition loader only (no full preprocessing pipeline).
- Computes **pre‑rejection** stage counts directly from the hypnogram.
- Computes **post‑rejection** counts after applying the validated amplitude‑rejection
  rules (EEG > 500 µV p‑t‑p, EOG > 1000 µV p‑t‑p) on the raw signals **without** filtering.
- Reports ignored‑epoch numbers and percentages (overall and per stage).

The script writes a markdown report `class_distribution_with_rejection.md` in the same
 directory.
"""
import pathlib
from collections import defaultdict

import numpy as np
import mne  # added for log control

mne.set_log_level('ERROR')  # suppress edge‑artifact warnings
from sleep_staging.config.settings import load_settings
from sleep_staging.acquisition.loader import SleepEDFLoader

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATA_ROOT = pathlib.Path(r'D:\\SleepEDFX')
# Load default settings (we only need the amplitude‑rejection thresholds)
# Resolve the config file relative to this script’s location (repo root)
config_path = pathlib.Path(__file__).resolve().parents[2] / 'configs' / 'default.yaml'
pipeline_settings = load_settings(config_path)
# Override data_root for local dataset
pipeline_settings = pipeline_settings.__class__(
    acquisition=pipeline_settings.acquisition.__class__(
        data_root=DATA_ROOT,
        preload=True,
        stim_channel=pipeline_settings.acquisition.stim_channel,
        infer_types=pipeline_settings.acquisition.infer_types,
        mne_verbose=pipeline_settings.acquisition.mne_verbose,
    ),
    preprocessing=pipeline_settings.preprocessing,
    encodings=pipeline_settings.encodings,
    experiment=pipeline_settings.experiment,
    logging=pipeline_settings.logging,
    project_root=pipeline_settings.project_root,
)
settings = pipeline_settings

# ---------------------------------------------------------------------------
# Helper to map raw channel types to EEG / EOG indices
# ---------------------------------------------------------------------------
def get_channel_indices(raw):
    ch_types = raw.get_channel_types()
    eeg_idx = [i for i, t in enumerate(ch_types) if t == 'eeg']
    eog_idx = [i for i, t in enumerate(ch_types) if t == 'eog']
    return eeg_idx, eog_idx

# ---------------------------------------------------------------------------
# Load recordings
# ---------------------------------------------------------------------------
# Set a small test limit for initial verification (set to None for full run)
MAX_RECORDS = None  # process all recordings

# Load recordings (default preload=False). Warnings are suppressed via mne.set_log_level.
loader = SleepEDFLoader(settings=settings.acquisition)
all_psg_paths = loader.discover()
# Apply test limit if defined
psg_paths = all_psg_paths[:MAX_RECORDS] if MAX_RECORDS is not None else all_psg_paths
print(f"Processing {len(psg_paths)} recordings (out of {len(all_psg_paths)} total)")

# ---------------------------------------------------------------------------
# Accumulators
# ---------------------------------------------------------------------------
pre_counts = defaultdict(int)   # stage -> epoch count (before rejection)
post_counts = defaultdict(int)  # stage -> epoch count (after rejection)
ignored_counts = defaultdict(int)  # stage -> ignored epochs due to amplitude

# Thresholds (values are in volts; raw data from MNE is in Volts)
EEG_THRESH_V = settings.preprocessing.amplitude_reject.eeg_peak_to_peak
EOG_THRESH_V = settings.preprocessing.amplitude_reject.eog_peak_to_peak

for idx, psg_path in enumerate(psg_paths, start=1):
    rec = loader.load_recording(psg_path)
    # Progress output
    rec_id = getattr(rec.metadata, 'recording_id', 'unknown')
    print(f"Processing recording {idx}/{len(psg_paths)}: {psg_path.name}")
    raw = rec.raw  # raw is already loaded by the loader (preload=True)
    sfreq = raw.info['sfreq']
    epoch_len = int(settings.preprocessing.epoch_duration_sec * sfreq)  # 30 s
    eeg_idx, eog_idx = get_channel_indices(raw)

    # Annotations contain the hypnogram labels aligned to the data
    ann = raw.annotations
    # Compute number of full 30‑s epochs; ignore any trailing partial epoch.
    n_epochs = int(np.floor(raw.n_times / epoch_len))
    for ep in range(n_epochs):
        start = ep * epoch_len
        end = start + epoch_len
        start_sec = start / sfreq
        stage = None
        for onset, duration, description in zip(ann.onset, ann.duration, ann.description):
            if onset <= start_sec < onset + duration:
                stage = description.strip()
                break
        if stage is None:
            continue
        pre_counts[stage] += 1

        # --- Amplitude rejection check (raw, no filter) ---
        epoch_data = raw.get_data(start=start, stop=end)  # shape (n_channels, epoch_len)
        eeg_ptp = np.ptp(epoch_data[eeg_idx, :], axis=1) if eeg_idx else np.array([])
        eog_ptp = np.ptp(epoch_data[eog_idx, :], axis=1) if eog_idx else np.array([])
        reject = False
        if (eeg_ptp > EEG_THRESH_V).any():
            reject = True
        if (eog_ptp > EOG_THRESH_V).any():
            reject = True
        if reject:
            ignored_counts[stage] += 1
        else:
            post_counts[stage] += 1

# ---------------------------------------------------------------------------
# Build markdown report
# ---------------------------------------------------------------------------
lines = ["# Sleep‑EDF SC class‑distribution & amplitude‑rejection impact", ""]
order = ["W", "N1", "N2", "N3", "R", "REM"]

lines.append("## Pre‑rejection distribution (all epochs)")
lines.append("| Stage | Epochs |\n|---|---|")
for s in order:
    if s in pre_counts:
        lines.append(f"| {s} | {pre_counts[s]} |")
for s, cnt in pre_counts.items():
    if s not in order:
        lines.append(f"| {s} | {cnt} |")

lines.append("\n## Post‑rejection distribution (epochs kept after amplitude check)")
lines.append("| Stage | Epochs |\n|---|---|")
for s in order:
    if s in post_counts:
        lines.append(f"| {s} | {post_counts[s]} |")
for s, cnt in post_counts.items():
    if s not in order:
        lines.append(f"| {s} | {cnt} |")

# Summary of ignored epochs
total_epochs = sum(pre_counts.values())
ignored_total = sum(ignored_counts.values())
kept_total = sum(post_counts.values())
lines.append("\n## Amplitude‑rejection summary")
lines.append(f"- Total epochs (pre‑rejection): {total_epochs}")
lines.append(f"- Epochs ignored due to amplitude: {ignored_total} ({ignored_total/total_epochs*100:.2f}% )")
lines.append(f"- Epochs kept after rejection: {kept_total} ({kept_total/total_epochs*100:.2f}% )")
lines.append("\n### Ignored epochs per stage")
lines.append("| Stage | Ignored | % of stage |\n|---|---|---|")
for s in order:
    if s in ignored_counts:
        pct = ignored_counts[s] / pre_counts.get(s, 1) * 100
        lines.append(f"| {s} | {ignored_counts[s]} | {pct:.2f}% |")
for s, cnt in ignored_counts.items():
    if s not in order:
        pct = cnt / pre_counts.get(s, 1) * 100
        lines.append(f"| {s} | {cnt} | {pct:.2f}% |")

report_path = pathlib.Path(__file__).with_name('class_distribution_with_rejection.md')
report_path.write_text("\n".join(lines), encoding='utf-8')
print("Report written to", report_path)
