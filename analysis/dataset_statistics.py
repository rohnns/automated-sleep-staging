#!/usr/bin/env python3
"""Offline Sleep-EDF Expanded dataset analysis for Phase 2 design.

This script is **not** part of the runtime pipeline. It loads recordings via
the public ``sleep_staging.acquisition`` API and summarizes statistics that
inform preprocessing choices (epoching windows, wake trimming, channel
selection, class imbalance, etc.).

It does **not** implement preprocessing, AASM remapping for training, or
models.

Usage
-----
    py -3.13 analysis/dataset_statistics.py
    py -3.13 analysis/dataset_statistics.py --data-root D:/SleepEDFX
    py -3.13 analysis/dataset_statistics.py --limit 10   # smoke test
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

if sys.version_info < (3, 12):
    sys.stderr.write(
        f"ERROR: Python 3.12+ is required (found {sys.version.split()[0]}).\n"
        "Use: py -3.13 analysis/dataset_statistics.py\n"
    )
    raise SystemExit(2)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from sleep_staging.acquisition import (
        AcquisitionError,
        SleepEDFLoader,
        SleepRecording,
        discover_recordings,
    )
    from sleep_staging.common import configure_logging, get_logger
    from sleep_staging.config import load_settings
except ModuleNotFoundError as exc:
    sys.stderr.write(
        f"ERROR: missing dependency ({exc.name}).\n"
        "Install with: py -3.13 -m pip install -e \".[dev]\"\n"
    )
    raise SystemExit(2) from exc

logger = get_logger(__name__)

# Raw Sleep-EDF hypnogram labels (Rechtschaffen & Kales style in EDF+).
_WAKE_LABELS = frozenset({"Sleep stage W"})
_SLEEP_LABELS = frozenset(
    {
        "Sleep stage 1",
        "Sleep stage 2",
        "Sleep stage 3",
        "Sleep stage 4",
        "Sleep stage R",
    }
)

# Informative AASM-oriented buckets for class-balance planning only.
_AASM_BUCKETS: dict[str, str] = {
    "Sleep stage W": "W",
    "Sleep stage 1": "N1",
    "Sleep stage 2": "N2",
    "Sleep stage 3": "N3",
    "Sleep stage 4": "N3",
    "Sleep stage R": "REM",
    "Movement time": "OTHER",
    "Sleep stage ?": "OTHER",
}


@dataclass(slots=True)
class AnnotationSegment:
    """One hypnogram segment."""

    onset: float
    duration: float
    label: str


@dataclass(slots=True)
class SleepBoundaryStats:
    """Sleep period boundaries derived from hypnogram labels."""

    sleep_onset_sec: float | None
    sleep_offset_sec: float | None
    wake_before_sleep_sec: float
    wake_after_sleep_sec: float
    wake_during_sleep_sec: float
    sleep_period_duration_sec: float | None
    has_scored_sleep: bool


@dataclass(slots=True)
class RecordingStats:
    """Per-recording statistics used for dataset aggregation."""

    psg_name: str
    study: str
    subject_id: str
    recording_id: str
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    sampling_frequency: float
    duration_sec: float
    n_annotations: int
    label_segment_counts: dict[str, int]
    label_duration_sec: dict[str, float]
    annotation_durations_sec: list[float]
    boundaries: SleepBoundaryStats
    reference: str | None


@dataclass
class DatasetAnalysis:
    """Aggregated offline analysis results."""

    data_root: str
    n_psg_discovered: int = 0
    n_loaded: int = 0
    n_failed: int = 0
    failures: list[dict[str, str]] = field(default_factory=list)
    recordings: list[RecordingStats] = field(default_factory=list)

    def to_jsonable(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict (plus derived summaries)."""
        return {
            "data_root": self.data_root,
            "n_psg_discovered": self.n_psg_discovered,
            "n_loaded": self.n_loaded,
            "n_failed": self.n_failed,
            "failures": self.failures,
            "summary": summarize_dataset(self),
            "recordings": [_recording_to_dict(rec) for rec in self.recordings],
        }


def _percentile(sorted_values: list[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = (len(sorted_values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_values[lo]
    weight = pos - lo
    return sorted_values[lo] * (1.0 - weight) + sorted_values[hi] * weight


def _fmt_hours(seconds: float) -> str:
    return f"{seconds:.1f} s ({seconds / 3600.0:.2f} h)"


def _fmt_minutes(seconds: float) -> str:
    return f"{seconds:.1f} s ({seconds / 60.0:.1f} min)"


def extract_segments(recording: SleepRecording) -> list[AnnotationSegment]:
    """Extract annotation segments from the authoritative Raw annotations."""
    annot = recording.annotations
    return [
        AnnotationSegment(onset=float(onset), duration=float(duration), label=str(label))
        for onset, duration, label in zip(
            annot.onset, annot.duration, annot.description, strict=True
        )
    ]


def compute_sleep_boundaries(
    segments: list[AnnotationSegment],
    recording_duration_sec: float,
) -> SleepBoundaryStats:
    """Estimate sleep onset/offset and wake mass around the sleep period.

    Definitions (analysis-only, for preprocessing design):
    - Sleep onset: onset of the first segment labeled as sleep (1/2/3/4/R)
    - Sleep offset: end time of the last sleep-labeled segment
    - Wake before / after: wake duration before onset / after offset
    - Wake during: wake duration strictly inside [onset, offset]
    """
    sleep_segments = [seg for seg in segments if seg.label in _SLEEP_LABELS]
    if not sleep_segments:
        wake_total = sum(seg.duration for seg in segments if seg.label in _WAKE_LABELS)
        return SleepBoundaryStats(
            sleep_onset_sec=None,
            sleep_offset_sec=None,
            wake_before_sleep_sec=wake_total,
            wake_after_sleep_sec=0.0,
            wake_during_sleep_sec=0.0,
            sleep_period_duration_sec=None,
            has_scored_sleep=False,
        )

    onset = min(seg.onset for seg in sleep_segments)
    offset = max(seg.onset + seg.duration for seg in sleep_segments)
    offset = min(offset, recording_duration_sec)

    wake_before = 0.0
    wake_after = 0.0
    wake_during = 0.0
    for seg in segments:
        if seg.label not in _WAKE_LABELS:
            continue
        start = seg.onset
        end = seg.onset + seg.duration
        # Overlap helpers
        before = max(0.0, min(end, onset) - start)
        after = max(0.0, end - max(start, offset))
        during = max(0.0, min(end, offset) - max(start, onset))
        wake_before += before
        wake_after += after
        wake_during += during

    return SleepBoundaryStats(
        sleep_onset_sec=onset,
        sleep_offset_sec=offset,
        wake_before_sleep_sec=wake_before,
        wake_after_sleep_sec=wake_after,
        wake_during_sleep_sec=wake_during,
        sleep_period_duration_sec=max(0.0, offset - onset),
        has_scored_sleep=True,
    )


def analyze_recording(recording: SleepRecording) -> RecordingStats:
    """Compute per-recording statistics from a loaded ``SleepRecording``."""
    segments = extract_segments(recording)
    label_segments: Counter[str] = Counter()
    label_duration: Counter[str] = Counter()
    durations: list[float] = []

    for seg in segments:
        label_segments[seg.label] += 1
        label_duration[seg.label] += seg.duration
        durations.append(seg.duration)

    meta = recording.metadata
    boundaries = compute_sleep_boundaries(segments, meta.duration_sec)
    return RecordingStats(
        psg_name=meta.psg_path.name,
        study=meta.study,
        subject_id=meta.subject_id,
        recording_id=meta.recording_id,
        channel_names=meta.channel_names,
        channel_types=meta.channel_types,
        sampling_frequency=meta.sampling_frequency,
        duration_sec=meta.duration_sec,
        n_annotations=meta.n_annotations,
        label_segment_counts=dict(label_segments),
        label_duration_sec={k: float(v) for k, v in label_duration.items()},
        annotation_durations_sec=durations,
        boundaries=boundaries,
        reference=meta.reference,
    )


def _recording_to_dict(rec: RecordingStats) -> dict[str, Any]:
    payload = asdict(rec)
    # Keep JSON compact: annotation duration list can be large; store summary only.
    durations = sorted(rec.annotation_durations_sec)
    payload["annotation_duration_summary"] = {
        "n": len(durations),
        "min_sec": durations[0] if durations else None,
        "max_sec": durations[-1] if durations else None,
        "mean_sec": (sum(durations) / len(durations)) if durations else None,
        "unique_values": sorted(set(round(d, 6) for d in durations)),
    }
    del payload["annotation_durations_sec"]
    return payload


def summarize_dataset(analysis: DatasetAnalysis) -> dict[str, Any]:
    """Build aggregate summary statistics across successful loads."""
    recs = analysis.recordings
    if not recs:
        return {"message": "No successful loads."}

    study_counts = Counter(rec.study for rec in recs)
    subject_keys = {(rec.study, rec.subject_id) for rec in recs}
    channel_layouts = Counter(rec.channel_names for rec in recs)
    type_layouts = Counter(rec.channel_types for rec in recs)
    sfreqs = Counter(round(rec.sampling_frequency, 6) for rec in recs)

    durations = sorted(rec.duration_sec for rec in recs)
    n_annots = sorted(rec.n_annotations for rec in recs)

    label_segments: Counter[str] = Counter()
    label_duration: Counter[str] = Counter()
    aasm_segments: Counter[str] = Counter()
    aasm_duration: Counter[str] = Counter()
    all_ann_durs: list[float] = []

    for rec in recs:
        label_segments.update(rec.label_segment_counts)
        for label, dur in rec.label_duration_sec.items():
            label_duration[label] += dur
            bucket = _AASM_BUCKETS.get(label, "OTHER")
            aasm_segments[bucket] += rec.label_segment_counts.get(label, 0)
            aasm_duration[bucket] += dur
        all_ann_durs.extend(rec.annotation_durations_sec)

    all_ann_durs_sorted = sorted(all_ann_durs)
    unique_ann_durs = sorted({round(d, 6) for d in all_ann_durs})

    with_sleep = [rec for rec in recs if rec.boundaries.has_scored_sleep]
    onset_vals = sorted(
        rec.boundaries.sleep_onset_sec
        for rec in with_sleep
        if rec.boundaries.sleep_onset_sec is not None
    )
    offset_vals = sorted(
        rec.boundaries.sleep_offset_sec
        for rec in with_sleep
        if rec.boundaries.sleep_offset_sec is not None
    )
    wake_before = sorted(rec.boundaries.wake_before_sleep_sec for rec in with_sleep)
    wake_after = sorted(rec.boundaries.wake_after_sleep_sec for rec in with_sleep)
    wake_during = sorted(rec.boundaries.wake_during_sleep_sec for rec in with_sleep)
    sleep_period = sorted(
        rec.boundaries.sleep_period_duration_sec
        for rec in with_sleep
        if rec.boundaries.sleep_period_duration_sec is not None
    )

    total_label_duration = sum(label_duration.values()) or 1.0
    total_aasm_duration = sum(aasm_duration.values()) or 1.0

    def _dist_block(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"n": 0, "min": None, "p25": None, "median": None, "p75": None, "max": None, "mean": None}
        return {
            "n": len(values),
            "min": values[0],
            "p25": _percentile(values, 0.25),
            "median": _percentile(values, 0.50),
            "p75": _percentile(values, 0.75),
            "max": values[-1],
            "mean": sum(values) / len(values),
        }

    channel_layout_rows = []
    for names, count in channel_layouts.most_common():
        # Find a matching type tuple from any recording with these names.
        types = next(rec.channel_types for rec in recs if rec.channel_names == names)
        channel_layout_rows.append(
            {
                "count": count,
                "n_channels": len(names),
                "channels": [
                    {"name": n, "type": t} for n, t in zip(names, types, strict=True)
                ],
            }
        )

    return {
        "n_recordings": len(recs),
        "n_subjects": len(subject_keys),
        "study_counts": dict(study_counts),
        "sampling_frequency_hz": {str(k): v for k, v in sorted(sfreqs.items())},
        "recording_duration_sec": _dist_block(durations),
        "annotations_per_recording": _dist_block([float(n) for n in n_annots]),
        "channel_layouts": channel_layout_rows,
        "channel_type_layout_counts": {
            str(types): count for types, count in type_layouts.most_common()
        },
        "annotation_segment_duration_sec": {
            "unique_values": unique_ann_durs,
            "distribution": _dist_block(all_ann_durs_sorted),
            "fraction_exactly_30s": (
                sum(1 for d in all_ann_durs if abs(d - 30.0) < 1e-6) / len(all_ann_durs)
                if all_ann_durs
                else None
            ),
        },
        "raw_label_segment_counts": dict(label_segments.most_common()),
        "raw_label_duration_sec": dict(label_duration.most_common()),
        "raw_label_duration_fraction": {
            label: dur / total_label_duration for label, dur in label_duration.most_common()
        },
        "aasm_oriented_segment_counts": dict(aasm_segments.most_common()),
        "aasm_oriented_duration_sec": dict(aasm_duration.most_common()),
        "aasm_oriented_duration_fraction": {
            label: dur / total_aasm_duration for label, dur in aasm_duration.most_common()
        },
        "sleep_boundaries": {
            "n_with_scored_sleep": len(with_sleep),
            "n_without_scored_sleep": len(recs) - len(with_sleep),
            "sleep_onset_sec": _dist_block(onset_vals),
            "sleep_offset_sec": _dist_block(offset_vals),
            "sleep_period_duration_sec": _dist_block(sleep_period),
            "wake_before_sleep_sec": _dist_block(wake_before),
            "wake_after_sleep_sec": _dist_block(wake_after),
            "wake_during_sleep_sec": _dist_block(wake_during),
        },
        "design_notes": [
            "Most Sleep-EDF Expanded recordings include long Wake tails before/after sleep; "
            "MNE tutorials often crop ~30 min before first sleep and after last sleep.",
            "Stages 3 and 4 are separate in R&K hypnograms; AASM merges them into N3.",
            "Hypnogram annotations are often contiguous multi-epoch stage bouts (not always "
            "exactly 30 s). Clinical staging still targets fixed 30 s epochs.",
            "Cassette (SC) and telemetry (ST) cohorts can differ in channel extras "
            "(e.g. rectal temperature / respiration).",
        ],
    }


def run_analysis(
    data_root: Path,
    *,
    preload: bool = False,
    limit: int | None = None,
) -> DatasetAnalysis:
    """Load recordings and compute dataset statistics."""
    analysis = DatasetAnalysis(data_root=str(data_root.resolve()))
    loader = SleepEDFLoader(data_root=data_root, preload=preload)

    try:
        psg_files = discover_recordings(data_root)
    except AcquisitionError as exc:
        analysis.failures.append({"path": str(data_root), "error": f"discovery: {exc}"})
        analysis.n_failed += 1
        return analysis

    analysis.n_psg_discovered = len(psg_files)
    if limit is not None:
        psg_files = psg_files[: max(0, limit)]

    logger.info(
        "Analyzing %d / %d PSG file(s) under %s",
        len(psg_files),
        analysis.n_psg_discovered,
        data_root,
    )

    for index, psg_path in enumerate(psg_files, start=1):
        try:
            recording = loader.load_recording(psg_path)
            stats = analyze_recording(recording)
            analysis.recordings.append(stats)
            analysis.n_loaded += 1
            logger.info(
                "[%d/%d] %s onset=%s offset=%s",
                index,
                len(psg_files),
                stats.psg_name,
                None
                if stats.boundaries.sleep_onset_sec is None
                else f"{stats.boundaries.sleep_onset_sec / 60.0:.1f}m",
                None
                if stats.boundaries.sleep_offset_sec is None
                else f"{stats.boundaries.sleep_offset_sec / 60.0:.1f}m",
            )
        except Exception as exc:  # noqa: BLE001 — collect and continue
            analysis.n_failed += 1
            analysis.failures.append({"path": str(psg_path), "error": str(exc)})
            logger.error("[%d/%d] FAIL %s: %s", index, len(psg_files), psg_path.name, exc)

    return analysis


def print_report(analysis: DatasetAnalysis) -> None:
    """Print a concise human-readable analysis report."""
    summary = summarize_dataset(analysis)
    print()
    print("=" * 72)
    print("Sleep-EDF Expanded - offline dataset analysis (Phase 2 design)")
    print("=" * 72)
    print(f"Data root     : {analysis.data_root}")
    print(f"PSG discovered: {analysis.n_psg_discovered}")
    print(f"Loaded        : {analysis.n_loaded}")
    print(f"Failed        : {analysis.n_failed}")
    print()

    if analysis.failures:
        print("--- Failures ---")
        for item in analysis.failures[:20]:
            print(f"  - {Path(item['path']).name}: {item['error']}")
        if len(analysis.failures) > 20:
            print(f"  ... and {len(analysis.failures) - 20} more")
        print()

    if analysis.n_loaded == 0:
        print("No recordings loaded; nothing to summarize.")
        print("=" * 72)
        return

    print("--- Cohort ---")
    print(f"  subjects   : {summary['n_subjects']}")
    print(f"  studies    : {summary['study_counts']}")
    print(f"  sfreq (Hz) : {summary['sampling_frequency_hz']}")
    dur = summary["recording_duration_sec"]
    print(
        "  duration   : "
        f"min={_fmt_hours(dur['min'])}, median={_fmt_hours(dur['median'])}, "
        f"max={_fmt_hours(dur['max'])}"
    )
    print()

    print("--- Channel layouts ---")
    for row in summary["channel_layouts"]:
        chans = ", ".join(f"{c['name']}[{c['type']}]" for c in row["channels"])
        print(f"  [{row['count']} rec] {chans}")
    print()

    print("--- Annotation segment duration ---")
    ann = summary["annotation_segment_duration_sec"]
    print(f"  unique durations (s): {ann['unique_values']}")
    frac30 = ann["fraction_exactly_30s"]
    if frac30 is not None:
        print(f"  fraction exactly 30 s: {frac30:.3%}")
    print()

    print("--- Raw hypnogram labels (duration share) ---")
    for label, frac in summary["raw_label_duration_fraction"].items():
        dur_s = summary["raw_label_duration_sec"][label]
        segs = summary["raw_label_segment_counts"][label]
        print(f"  {label:20s}  {frac:6.2%}  {_fmt_hours(dur_s):>22s}  ({segs} segments)")
    print()

    print("--- AASM-oriented buckets (planning only) ---")
    for label, frac in summary["aasm_oriented_duration_fraction"].items():
        dur_s = summary["aasm_oriented_duration_sec"][label]
        segs = summary["aasm_oriented_segment_counts"][label]
        print(f"  {label:6s}  {frac:6.2%}  {_fmt_hours(dur_s):>22s}  ({segs} segments)")
    print()

    print("--- Sleep onset / offset / wake ---")
    sb = summary["sleep_boundaries"]
    print(f"  recordings with scored sleep: {sb['n_with_scored_sleep']}")
    print(f"  recordings without sleep    : {sb['n_without_scored_sleep']}")

    def _print_dist(title: str, block: dict[str, Any], formatter=_fmt_minutes) -> None:
        if not block["n"]:
            print(f"  {title}: n/a")
            return
        print(
            f"  {title}: "
            f"min={formatter(block['min'])}, "
            f"p25={formatter(block['p25'])}, "
            f"median={formatter(block['median'])}, "
            f"p75={formatter(block['p75'])}, "
            f"max={formatter(block['max'])}"
        )

    _print_dist("sleep onset", sb["sleep_onset_sec"])
    _print_dist("sleep offset", sb["sleep_offset_sec"])
    _print_dist("sleep period", sb["sleep_period_duration_sec"], _fmt_hours)
    _print_dist("wake before sleep", sb["wake_before_sleep_sec"])
    _print_dist("wake after sleep", sb["wake_after_sleep_sec"])
    _print_dist("wake during sleep period", sb["wake_during_sleep_sec"])
    print()

    print("--- Design notes for Phase 2 ---")
    for note in summary["design_notes"]:
        print(f"  - {note}")
    print()
    print("=" * 72)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Sleep-EDF dataset statistics for Phase 2 preprocessing design. "
            "Not part of the runtime pipeline."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_PROJECT_ROOT / "configs" / "default.yaml",
        help="Pipeline YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override acquisition.data_root.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_PROJECT_ROOT / "analysis" / "outputs" / "dataset_statistics.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of recordings (smoke tests).",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload signal data (usually unnecessary for annotation stats).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (default: INFO).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    settings = load_settings(args.config, project_root=_PROJECT_ROOT)
    data_root = (
        args.data_root.expanduser().resolve()
        if args.data_root
        else settings.acquisition.data_root
    )
    if not data_root.exists():
        print(f"ERROR: data root does not exist: {data_root}", file=sys.stderr)
        return 2

    analysis = run_analysis(
        data_root,
        preload=args.preload or settings.acquisition.preload,
        limit=args.limit,
    )
    print_report(analysis)

    output_path: Path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(analysis.to_jsonable(), handle, indent=2)
    print(f"Wrote JSON report: {output_path}")

    return 0 if analysis.n_loaded > 0 and analysis.n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
