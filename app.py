"""Streamlit dashboard for exported sleep-staging results.

This app is read-only: it reads the artifacts written by ``main.py`` (and the
legacy whole-night exports) plus the existing preprocessing helpers, to
visualize one held-out test recording at a time. It never trains or refits.

Artifact roots, in priority order:

``artifacts/predictions/sc_to_st/``  primary, written by ``main.py``
``outputs/model_outputs/``           legacy per-recording whole-night exports
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from sleep_staging.acquisition.loader import load_recordings
from sleep_staging.acquisition.utils import parse_psg_filename
from sleep_staging.config import load_settings
from sleep_staging.evaluation.output import STAGE_COLORS, STAGE_NAMES, stage_index_to_name
from sleep_staging.preprocessing.pipeline import preprocess_recording

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
PRIMARY_OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "predictions" / "sc_to_st"
LEGACY_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "model_outputs"
#: Roots searched for artifacts, in priority order. Referenced in user-facing
#: messages so the reader can see exactly where the app looked.
OUTPUT_ROOTS = (PRIMARY_OUTPUT_ROOT, LEGACY_OUTPUT_ROOT)


@dataclass(frozen=True, slots=True)
class ArtifactSet:
    recording_key: str
    psg_path: Path
    subject_id: str
    root: Path
    """Which output root this recording's artifacts were found under."""


@st.cache_data(show_spinner=False)
def _load_artifact_sets() -> list[ArtifactSet]:
    items: list[ArtifactSet] = []
    for root in OUTPUT_ROOTS:
        if not root.exists():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            manifest = folder / "manifest.json"
            if not manifest.exists():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                psg = Path(data["selected_psg"])
                subj = str(data.get("selected_subject") or parse_psg_filename(psg).subject_id)
                items.append(ArtifactSet(folder.name, psg, subj, root))
            except Exception:
                continue
    return items


@st.cache_data(show_spinner=False)
def _load_settings_cached(config_path: str):
    return load_settings(Path(config_path))


@st.cache_data(show_spinner=False)
def load_recording_artifacts(recording_key: str) -> dict[str, object]:
    folder = PRIMARY_OUTPUT_ROOT / recording_key
    if not folder.exists():
        folder = LEGACY_OUTPUT_ROOT / recording_key
    out: dict[str, object] = {"recording_key": recording_key}
    for rep in ("raw", "bandpower", "time_frequency"):
        rep_dir = folder / rep
        with (rep_dir / "summary.json").open("r", encoding="utf-8") as f:
            out[rep] = json.load(f)
        with (rep_dir / "predictions.csv").open("r", encoding="utf-8") as f:
            out[f"{rep}_rows"] = list(csv.DictReader(f))
        out[f"{rep}_image"] = rep_dir / "hypnogram.png"
    return out


@st.cache_resource(show_spinner=False)
def load_selected_recording(psg_path_str: str):
    settings = _load_settings_cached(str(DEFAULT_CONFIG))
    psg_path = Path(psg_path_str)
    recording = load_recordings([psg_path], settings=settings.acquisition, preload=settings.acquisition.preload)[0]
    return preprocess_recording(recording, settings.preprocessing, copy=False)


def _rows_to_arrays(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    epoch_index = np.asarray([int(r["epoch_index"]) for r in rows], dtype=int)
    y_true = np.asarray([int(r["target"]) for r in rows], dtype=int)
    y_pred = np.asarray([int(r["prediction"]) for r in rows], dtype=int)
    onsets = np.asarray([float(r["onset_sec"]) for r in rows], dtype=float)
    return epoch_index, onsets, y_true, y_pred


def _plot_hypnogram_side_by_side(onsets: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray):
    stage_order = ["W", "N1", "N2", "N3", "REM"]
    stage_to_idx = {stage: idx for idx, stage in enumerate(stage_order)}
    idx_to_stage = {idx: stage for stage, idx in stage_to_idx.items()}
    stage_colors = [STAGE_COLORS.get(stage, "#999999") for stage in stage_order]

    def _values(vals: np.ndarray) -> np.ndarray:
        return np.asarray([stage_to_idx.get(stage_index_to_name(int(v)), 0) for v in vals], dtype=int)

    # append a final point so step plots extend through the last epoch
    x = np.asarray(onsets, dtype=float) / 3600.0
    if x.size:
        x = np.r_[x, x[-1] + 30.0 / 3600.0]
    expert = _values(y_true)
    predicted = _values(y_pred)
    if expert.size:
        expert = np.r_[expert, expert[-1]]
    if predicted.size:
        predicted = np.r_[predicted, predicted[-1]]

    fig, axes = plt.subplots(2, 1, figsize=(14, 5), sharex=True, constrained_layout=True)
    for ax, vals, title in zip(axes, [expert, predicted], ["Expert", "Predicted"], strict=True):
        ax.step(x, vals, where="post", color="#1f1f1f", linewidth=1.0, label=title)
        ax.scatter(x[:-1], vals[:-1], c=[stage_colors[v] for v in vals[:-1]], s=18, zorder=3)
        ax.set_yticks(range(len(stage_order)), stage_order)
        ax.set_ylim(-0.5, len(stage_order) - 0.5)
        ax.set_ylabel(title)
        ax.grid(True, axis="x", alpha=0.2)
    axes[-1].set_xlabel("Time (hours)")

    from matplotlib.lines import Line2D

    stage_handles = [Line2D([0], [0], color=stage_colors[i], marker="s", linestyle="None", markersize=8, label=stage)
                     for i, stage in enumerate(stage_order)]
    line_handles = [
        Line2D([0], [0], color="#1f1f1f", linewidth=1.5, label="Expert"),
        Line2D([0], [0], color="#1f1f1f", linewidth=1.5, linestyle="--", label="Predicted"),
    ]
    axes[0].legend(handles=line_handles + stage_handles, loc="upper right", fontsize=8, ncol=2)
    axes[1].legend(handles=stage_handles, loc="upper right", fontsize=8, ncol=2)
    return fig


def _plot_epoch_signals(raw, epoch_index: int):
    sfreq = float(raw.info["sfreq"])
    start = epoch_index * 30.0
    stop = start + 30.0
    start_samp = int(round(start * sfreq))
    stop_samp = int(round(stop * sfreq))
    data = raw.get_data(start=start_samp, stop=stop_samp)
    times = np.arange(data.shape[1], dtype=float) / sfreq
    fig, ax = plt.subplots(figsize=(14, 4), constrained_layout=True)
    offset = 0.0
    for ch, signal in zip(raw.ch_names, data, strict=True):
        signal = signal - np.mean(signal)
        scale = np.std(signal) or 1.0
        ax.plot(times, signal / scale + offset, lw=0.8, label=ch)
        offset += 5.0
    ax.set_title(f"Selected 30 s epoch - epoch {epoch_index}")
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=8)
    return fig


def _plot_confusion_matrices(raw_matrix: Iterable[Iterable[float]], norm_matrix: Iterable[Iterable[float]]):
    labels = list(STAGE_NAMES)
    raw_arr = np.asarray([list(row) for row in raw_matrix], dtype=float)
    norm_arr = np.asarray([list(row) for row in norm_matrix], dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    for ax, matrix, title, cmap in (
        (axes[0], raw_arr, "Confusion matrix (counts)", "Blues"),
        (axes[1], norm_arr, "Confusion matrix (row-normalized)", "Greens"),
    ):
        im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        threshold = float(np.nanmax(matrix)) / 2.0 if matrix.size else 0.0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = f"{value:.0f}" if title.endswith("counts)") else f"{value:.2f}"
                ax.text(j, i, text, ha="center", va="center", color="white" if value > threshold else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig


# Band edges match BandPowerEncoder's five bands (no gamma -- the signal is
# low-passed to 30 Hz upstream, so there is no valid gamma content to report).
SPATIAL_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 0.5, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("sigma", 12.0, 16.0),
    ("beta", 16.0, 30.0),
)

# Where each recorded EEG derivation is drawn. A bipolar derivation has no
# single scalp coordinate, so the marker sits at the MIDPOINT of its electrode
# pair purely as a placement convention -- it labels the whole derivation and
# must not be read as a point measurement at that location.
DERIVATION_MIDPOINTS: dict[str, tuple[float, float]] = {
    "Fpz-Cz": (5.0, 3.95),
    "Pz-Oz": (5.0, 1.60),
}


def _band_power(psd_row: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    """Mean PSD within a closed frequency band, in dB."""
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return float("nan")
    return float(10.0 * np.log10(float(np.mean(np.asarray(psd_row)[mask])) + 1e-20))


def _plot_spatial_power(
    ch_names: list[str], psd: np.ndarray, freqs: np.ndarray, band: str, stage: str
) -> plt.Figure | None:
    """Plot per-derivation band power as discrete markers on a head outline.

    This is deliberately *not* an interpolated topomap. Sleep-EDF gives two
    bipolar EEG derivations, which is far too sparse to interpolate a scalp
    field from without inventing structure. Instead each derivation is drawn as
    one discrete marker coloured by its band power, so real spatial variation
    (frontal vs. posterior) is visible while nothing is filled in between.
    """
    lo, hi = next((l, h) for name, l, h in SPATIAL_BANDS if name == band)
    points = [
        (name, DERIVATION_MIDPOINTS[name], _band_power(row, freqs, lo, hi))
        for name, row in zip(ch_names, psd, strict=True)
        if name in DERIVATION_MIDPOINTS
    ]
    if not points:
        return None

    fig, ax = plt.subplots(figsize=(6.4, 5.2), constrained_layout=True)
    ax.set_xlim(1.5, 8.5)
    ax.set_ylim(0.3, 5.9)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((5, 3), 2.1, fill=False, lw=2, color="#444444"))
    ax.plot([4.72, 5, 5.28], [5.05, 5.5, 5.05], color="#444444", lw=2)  # nose
    ax.text(5, 0.5, "Posterior", ha="center", va="center", fontsize=9, color="#666666")

    values = [v for _, _, v in points]
    vmin, vmax = (min(values), max(values)) if len(set(values)) > 1 else (min(values) - 1, max(values) + 1)
    scatter = ax.scatter(
        [p[1][0] for p in points],
        [p[1][1] for p in points],
        c=values,
        s=2600,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        edgecolor="#222222",
        linewidth=1.6,
        zorder=3,
    )
    for name, (x, y), value in points:
        ax.text(x, y + 0.02, f"{value:.1f}", ha="center", va="center", fontsize=10,
                fontweight="bold", color="white", zorder=4)
        ax.text(x + 0.95, y, name, ha="left", va="center", fontsize=10, zorder=4)

    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label=f"{band} power (dB)")
    ax.set_title(f"{stage} — {band} power per derivation ({lo:g}–{hi:g} Hz)", fontsize=11)
    return fig


def _plot_10_20_montage(recording) -> plt.Figure:
    """Render a minimal sagittal 10-20 schematic for the recorded bipolar derivations.

    This is intentionally a teaching / documentation view only: Sleep-EDF
    recordings provide bipolar derivations (Fpz-Cz, Pz-Oz) plus horizontal EOG,
    not a dense scalp montage suitable for a topomap.
    """
    fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
    ax.set_title("Recorded channels on a simplified 10-20 midline schematic")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Head outline
    head = plt.Circle((5, 3), 2.1, fill=False, lw=2, color="#444444")
    ax.add_patch(head)
    ax.plot([5, 5], [1.0, 5.0], color="#dddddd", lw=1, alpha=0.4)
    ax.text(5, 5.55, "Anterior", ha="center", va="bottom", fontsize=10, color="#555555")
    ax.text(5, 0.35, "Posterior", ha="center", va="top", fontsize=10, color="#555555")

    # Approximate midline electrode positions.
    positions = {
        "Fpz": (5.0, 4.85),
        "Cz": (5.0, 3.0),
        "Pz": (5.0, 2.0),
        "Oz": (5.0, 1.2),
    }
    for label, (x, y) in positions.items():
        ax.scatter([x], [y], s=160, color="#1f77b4", edgecolor="white", linewidth=1.5, zorder=3)
        ax.text(x + 0.18, y, label, va="center", ha="left", fontsize=11, fontweight="bold")

    # Recorded derivations and EOG trace labels.
    derivations = [
        ("Fpz-Cz", (5.0, 4.15), (5.0, 3.55), "EEG bipolar derivation"),
        ("Pz-Oz", (5.0, 1.65), (5.0, 1.02), "EEG bipolar derivation"),
        ("horizontal EOG", (7.6, 2.7), None, "EOG channel"),
    ]
    for name, (x, y), tail, subtitle in derivations:
        if tail is not None:
            ax.annotate(
                "",
                xy=(x, y),
                xytext=tail,
                arrowprops=dict(arrowstyle="<->", lw=2.2, color="#d62728"),
            )
            ax.text(
                6.0,
                y,
                name,
                va="center",
                ha="left",
                fontsize=11,
                color="#d62728",
                fontweight="bold",
            )
            ax.text(6.0, y - 0.28, subtitle, va="top", ha="left", fontsize=9, color="#7f7f7f")
        else:
            ax.annotate(
                "",
                xy=(x - 0.5, y),
                xytext=(6.3, y),
                arrowprops=dict(arrowstyle="->", lw=2.2, color="#2ca02c"),
            )
            ax.text(6.35, y, name, va="center", ha="left", fontsize=11, color="#2ca02c", fontweight="bold")
            ax.text(6.35, y - 0.28, subtitle, va="top", ha="left", fontsize=9, color="#7f7f7f")

    # Channel summary explicitly tied to the actual recording.
    channel_text = "Recorded channels: " + ", ".join(recording.channel_names)
    ax.text(0.5, 0.5, channel_text, ha="left", va="center", fontsize=10, color="#333333")
    ax.text(
        0.5,
        0.18,
        "Note: Fpz-Cz and Pz-Oz are bipolar derivations, not independent scalp electrodes.",
        ha="left",
        va="center",
        fontsize=9,
        color="#555555",
    )
    return fig


st.set_page_config(page_title="Sleep Staging Dashboard", layout="wide")
st.title("Sleep Staging Dashboard")
st.caption("Read-only view of exported model results. Run `python main.py` to (re)generate them.")

artifacts = _load_artifact_sets()
if not artifacts:
    searched = "\n".join(f"- `{root}`" for root in OUTPUT_ROOTS)
    st.error(f"No exported results found. Searched:\n{searched}\n\nRun `python main.py` first.")
    st.stop()

recording_labels = [f"{a.recording_key} | subject {a.subject_id}" for a in artifacts]
selected_idx = st.sidebar.selectbox("Recording", list(range(len(artifacts))), format_func=lambda i: recording_labels[i])
selected = artifacts[selected_idx]
model = st.sidebar.radio("Model", ["raw", "bandpower", "time_frequency"], horizontal=False)

art = load_recording_artifacts(selected.recording_key)
summary = art[model]
rows = art[f"{model}_rows"]
_, onsets, y_true, y_pred = _rows_to_arrays(rows)
recording = load_selected_recording(str(selected.psg_path))

st.subheader(f"{selected.recording_key} - {model}")
col1, col2 = st.columns([1.2, 1])
with col1:
    st.markdown("### Expert vs predicted hypnogram")
    st.pyplot(_plot_hypnogram_side_by_side(onsets, y_true, y_pred), clear_figure=True)
with col2:
    st.markdown("### Confusion matrix")
    st.pyplot(
        _plot_confusion_matrices(
            summary["metrics"]["confusion_matrix"],
            summary["metrics"]["normalized_confusion_matrix"],
        ),
        clear_figure=True,
    )
    st.markdown("### Per-class metrics")
    metric_rows = []
    for stage in STAGE_NAMES:
        metric_rows.append(
            {
                "stage": stage,
                "precision": summary["metrics"]["per_class_precision"][stage],
                "recall": summary["metrics"]["per_class_recall"][stage],
                "f1": summary["metrics"]["per_class_f1"][stage],
            }
        )
    st.table(metric_rows)

st.markdown("### Sleep statistics")
ground_truth_stats = summary["sleep_statistics_ground_truth"]
predicted_stats = summary["sleep_statistics_predicted"]
st.table(
    [
        {
            "stat": "Total sleep time (s)",
            "Expert": ground_truth_stats["total_sleep_time_sec"],
            "Predicted": predicted_stats["total_sleep_time_sec"],
        },
        {
            "stat": "Sleep efficiency",
            "Expert": ground_truth_stats["sleep_efficiency"],
            "Predicted": predicted_stats["sleep_efficiency"],
        },
        {
            "stat": "Sleep onset latency (s)",
            "Expert": ground_truth_stats["sleep_onset_latency_sec"],
            "Predicted": predicted_stats["sleep_onset_latency_sec"],
        },
        {
            "stat": "REM latency (s)",
            "Expert": ground_truth_stats["rem_latency_sec"],
            "Predicted": predicted_stats["rem_latency_sec"],
        },
        {"stat": "W (s)", "Expert": ground_truth_stats["time_in_stage_sec"]["W"], "Predicted": predicted_stats["time_in_stage_sec"]["W"]},
        {"stat": "N1 (s)", "Expert": ground_truth_stats["time_in_stage_sec"]["N1"], "Predicted": predicted_stats["time_in_stage_sec"]["N1"]},
        {"stat": "N2 (s)", "Expert": ground_truth_stats["time_in_stage_sec"]["N2"], "Predicted": predicted_stats["time_in_stage_sec"]["N2"]},
        {"stat": "N3 (s)", "Expert": ground_truth_stats["time_in_stage_sec"]["N3"], "Predicted": predicted_stats["time_in_stage_sec"]["N3"]},
        {"stat": "REM (s)", "Expert": ground_truth_stats["time_in_stage_sec"]["REM"], "Predicted": predicted_stats["time_in_stage_sec"]["REM"]},
    ]
)

st.markdown("### Montage / 10-20 electrode-position visualization")
st.caption(
    "Illustrative schematic of the recorded bipolar derivations; not an MNE scalp "
    "montage and not a true topographic map. Electrode positions are drawn for "
    "orientation only — no digitized coordinates exist for these derivations."
)
st.pyplot(_plot_10_20_montage(recording), clear_figure=True)

st.markdown("### Selected 30-second epoch viewer")
epoch_index = st.slider("Epoch index", 0, max(len(y_true) - 1, 0), 0)
if hasattr(recording, "raw"):
    st.pyplot(_plot_epoch_signals(recording.raw, int(epoch_index)), clear_figure=True)

st.markdown("### PSD for selected epoch")
from mne.time_frequency import psd_array_welch

start_samp = int(round(epoch_index * 30.0 * recording.sampling_frequency))
stop_samp = int(round((epoch_index + 1) * 30.0 * recording.sampling_frequency))
segment = recording.raw.get_data(start=start_samp, stop=stop_samp)
psd, freqs = psd_array_welch(segment, sfreq=recording.sampling_frequency, fmin=0.5, fmax=30.0)
fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
for ch_name, row in zip(recording.raw.ch_names, psd, strict=True):
    ax.plot(freqs, 10 * np.log10(np.asarray(row) + 1e-20), label=ch_name)
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Power (dB)")
ax.set_title("Epoch PSD")
ax.legend(fontsize=8)
st.pyplot(fig, clear_figure=True)

st.markdown("### PSD for selected sleep stage")
st.caption(
    "Aggregate spectrum across every expert-scored epoch of the chosen stage in this "
    "recording. Welch PSD is computed per 30 s epoch, then averaged across epochs "
    "(mean of per-epoch PSDs) before conversion to dB."
)

stage_choice = st.selectbox("Sleep stage", list(STAGE_NAMES), index=2, key="stage_psd")
stage_idx = STAGE_NAMES.index(stage_choice)
stage_epochs = [i for i, lab in enumerate(y_true) if int(lab) == stage_idx]

# Reused by the spatial power panel below; stays None when the stage has no
# usable epochs so that panel can explain itself instead of erroring.
stage_mean_psd: np.ndarray | None = None
stage_freqs: np.ndarray | None = None

if not stage_epochs:
    st.info(
        f"No expert-scored **{stage_choice}** epochs in this recording, so no stage-level "
        "PSD can be computed. Pick another stage."
    )
else:
    sfreq = recording.sampling_frequency
    n_use = min(len(stage_epochs), 200)  # cap work for UI responsiveness
    used = stage_epochs[:n_use]
    stack = []
    for ei in used:
        a = int(round(ei * 30.0 * sfreq))
        b = int(round((ei + 1) * 30.0 * sfreq))
        seg = recording.raw.get_data(start=a, stop=b)
        if seg.shape[1] == 0:
            continue
        p, f_ax = psd_array_welch(seg, sfreq=sfreq, fmin=0.5, fmax=30.0, verbose=False)
        stack.append(p)

    if not stack:
        st.info(f"Could not extract signal for the selected {stage_choice} epochs.")
    else:
        mean_psd = np.mean(np.stack(stack, axis=0), axis=0)
        stage_mean_psd, stage_freqs = mean_psd, f_ax
        fig_s, ax_s = plt.subplots(figsize=(10, 4), constrained_layout=True)
        for ch_name, row in zip(recording.raw.ch_names, mean_psd, strict=True):
            ax_s.plot(f_ax, 10 * np.log10(np.asarray(row) + 1e-20), label=ch_name)
        ax_s.set_xlabel("Frequency (Hz)")
        ax_s.set_ylabel("Power (dB)")
        ax_s.set_title(
            f"{stage_choice} — mean PSD over {len(stack)} epoch(s)"
            + (f" (of {len(stage_epochs)} scored)" if len(stack) < len(stage_epochs) else "")
        )
        ax_s.legend(fontsize=8)
        ax_s.grid(True, alpha=0.25)
        st.pyplot(fig_s, clear_figure=True)
        st.caption(
            f"{len(stack)} of {len(stage_epochs)} expert-scored {stage_choice} epochs used "
            f"(capped at {n_use} for responsiveness)."
        )

st.markdown("### Spatial power distribution for selected stage")
st.caption(
    "Band power at each recorded EEG derivation for the stage selected above, drawn as "
    "**discrete markers — not an interpolated topomap.** Each marker is placed at the midpoint "
    "of its electrode pair as a placement convention only; it represents the whole bipolar "
    "derivation, not a point measurement at that spot. Nothing is interpolated between markers."
)

if stage_mean_psd is None or stage_freqs is None:
    st.info(
        f"No usable **{stage_choice}** epochs in this recording, so no spatial power map can be "
        "computed. Pick another stage above."
    )
else:
    band_choice = st.selectbox(
        "Frequency band", [name for name, _, _ in SPATIAL_BANDS], index=0, key="spatial_band"
    )
    fig_sp = _plot_spatial_power(
        list(recording.raw.ch_names), stage_mean_psd, stage_freqs, band_choice, stage_choice
    )
    if fig_sp is None:
        st.info(
            "None of this recording's channels are recognized EEG derivations "
            f"({', '.join(DERIVATION_MIDPOINTS)}), so no spatial map is shown."
        )
    else:
        st.pyplot(fig_sp, clear_figure=True)
        st.caption(
            f"Computed from the same mean {stage_choice} PSD shown above. Compare δ between N3 "
            "and REM, or across the frontal (Fpz-Cz) and posterior (Pz-Oz) derivations."
        )

st.markdown("### Why there is no interpolated topographic map")
st.info(
    "**Dataset/channel limitation, not a missing feature.** A scalp topomap interpolates power "
    "across many electrodes at known 10-20 positions. Sleep-EDF provides only two bipolar EEG "
    "derivations (Fpz-Cz, Pz-Oz) plus one horizontal EOG. A bipolar derivation measures a "
    "*difference* between two sites and has no single scalp coordinate, so there is nothing "
    "valid to interpolate over. Rendering a filled topomap would require inventing electrode "
    "positions and fabricating spatial structure the recording does not contain. The discrete "
    "per-derivation map above shows the spatial information that genuinely is available, "
    "without filling in what is not."
)

st.caption(f"Artifact source: {selected.root}")
