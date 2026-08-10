#!/usr/bin/env python3
"""Phase 1 acquisition validation for Sleep-EDF Expanded.

Loads every discoverable PSG + hypnogram pair through the public
``sleep_staging.acquisition`` API and prints a concise validation report.

Usage
-----
    python verify_acquisition.py
    python verify_acquisition.py --config configs/default.yaml
    python verify_acquisition.py --data-root /path/to/sleep-edf

Does not modify the acquisition module.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info < (3, 12):
    sys.stderr.write(
        f"ERROR: Python 3.12+ is required (found {sys.version.split()[0]}).\n"
        "On this machine, use:\n"
        "  py -3.13 verify_acquisition.py\n"
        "or install the package into a 3.12+ environment:\n"
        "  py -3.13 -m pip install -e \".[dev]\"\n"
    )
    raise SystemExit(2)

# Allow running without an editable install: python verify_acquisition.py
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from sleep_staging.acquisition import (
        AcquisitionError,
        SleepEDFLoader,
        SleepRecording,
        discover_recordings,
        resolve_hypnogram_path,
    )
    from sleep_staging.common import configure_logging, get_logger
    from sleep_staging.config import load_settings
except ModuleNotFoundError as exc:
    missing = exc.name or "dependency"
    sys.stderr.write(
        f"ERROR: missing dependency ({missing}).\n"
        "Install the project into your active Python environment:\n"
        "  py -3.13 -m pip install -e \".[dev]\"\n"
        "Then re-run:\n"
        "  py -3.13 verify_acquisition.py\n"
    )
    raise SystemExit(2) from exc

logger = get_logger(__name__)


@dataclass
class RecordingSummary:
    """Compact per-recording facts collected after a successful load."""

    psg_name: str
    subject_id: str
    recording_id: str
    study: str
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    sampling_frequency: float
    duration_sec: float
    n_annotations: int
    annotation_labels: tuple[str, ...]
    hypnogram_name: str


@dataclass
class VerificationReport:
    """Aggregated Phase 1 validation results."""

    data_root: Path
    psg_files: list[Path] = field(default_factory=list)
    missing_hypnograms: list[Path] = field(default_factory=list)
    load_failures: list[tuple[Path, str]] = field(default_factory=list)
    successes: list[RecordingSummary] = field(default_factory=list)

    @property
    def n_psg(self) -> int:
        return len(self.psg_files)

    @property
    def n_ok(self) -> int:
        return len(self.successes)

    @property
    def n_failed(self) -> int:
        return len(self.missing_hypnograms) + len(self.load_failures)

    @property
    def ok(self) -> bool:
        return self.n_psg > 0 and self.n_failed == 0


def _project_root() -> Path:
    return Path(__file__).resolve().parent


def _summarize_recording(recording: SleepRecording) -> RecordingSummary:
    meta = recording.metadata
    labels = tuple(str(desc) for desc in recording.annotations.description)
    return RecordingSummary(
        psg_name=meta.psg_path.name,
        subject_id=meta.subject_id,
        recording_id=meta.recording_id,
        study=meta.study,
        channel_names=meta.channel_names,
        channel_types=meta.channel_types,
        sampling_frequency=meta.sampling_frequency,
        duration_sec=meta.duration_sec,
        n_annotations=meta.n_annotations,
        annotation_labels=labels,
        hypnogram_name=meta.hypnogram_path.name,
    )


def run_verification(data_root: Path, *, preload: bool = False) -> VerificationReport:
    """Discover, pair-check, and load every Sleep-EDF recording under ``data_root``."""
    report = VerificationReport(data_root=data_root.resolve())
    loader = SleepEDFLoader(data_root=data_root, preload=preload)

    try:
        report.psg_files = discover_recordings(data_root)
    except AcquisitionError as exc:
        logger.error("Discovery failed: %s", exc)
        report.load_failures.append((data_root, f"discovery failed: {exc}"))
        return report

    if not report.psg_files:
        logger.warning("No PSG files found under %s", data_root)
        return report

    logger.info("Found %d PSG file(s) under %s", len(report.psg_files), data_root)

    for psg_path in report.psg_files:
        try:
            hyp_path = resolve_hypnogram_path(psg_path)
        except AcquisitionError as exc:
            report.missing_hypnograms.append(psg_path)
            logger.error("Missing hypnogram for %s: %s", psg_path.name, exc)
            continue

        try:
            recording = loader.load_recording(psg_path, hyp_path)
            report.successes.append(_summarize_recording(recording))
            logger.info("OK  %s", recording.summary())
        except AcquisitionError as exc:
            report.load_failures.append((psg_path, str(exc)))
            logger.error("FAIL %s: %s", psg_path.name, exc)
        except Exception as exc:  # noqa: BLE001 — report unexpected load errors
            report.load_failures.append((psg_path, f"unexpected: {exc}"))
            logger.error("FAIL %s: unexpected error: %s", psg_path.name, exc)

    return report


def _fmt_duration(seconds: float) -> str:
    hours = seconds / 3600.0
    return f"{seconds:.1f} s ({hours:.2f} h)"


def _fmt_channel_set(names: tuple[str, ...], types: tuple[str, ...]) -> str:
    paired = ", ".join(f"{n} [{t}]" for n, t in zip(names, types, strict=True))
    return paired


def print_report(report: VerificationReport) -> None:
    """Print a concise Phase 1 validation report to stdout."""
    print()
    print("=" * 72)
    print("Phase 1 acquisition verification report")
    print("=" * 72)
    print(f"Data root          : {report.data_root}")
    print(f"PSG files found    : {report.n_psg}")
    print(f"Loaded successfully: {report.n_ok}")
    print(f"Failures           : {report.n_failed}")
    print()

    # --- Hypnogram pairing ---
    print("--- Hypnogram pairing ---")
    if report.n_psg == 0:
        print("No PSG files to check.")
    elif not report.missing_hypnograms:
        print(f"All {report.n_psg} PSG file(s) have a matching *-Hypnogram.edf.")
    else:
        print(f"Missing hypnogram for {len(report.missing_hypnograms)} PSG file(s):")
        for path in report.missing_hypnograms:
            print(f"  - {path.name}")
    print()

    # --- Load failures ---
    print("--- Load failures ---")
    if not report.load_failures:
        print("None.")
    else:
        for path, message in report.load_failures:
            print(f"  - {path.name if path.is_file() or path.suffix else path}: {message}")
    print()

    if not report.successes:
        print("--- Summary statistics ---")
        print("No successful loads; nothing to summarize.")
        print()
        print("RESULT: FAIL")
        print("=" * 72)
        return

    # --- Channels ---
    print("--- Channels ---")
    channel_layouts: Counter[tuple[str, ...]] = Counter()
    type_layouts: dict[tuple[str, ...], tuple[str, ...]] = {}
    for item in report.successes:
        channel_layouts[item.channel_names] += 1
        type_layouts[item.channel_names] = item.channel_types
    for names, count in channel_layouts.most_common():
        types = type_layouts[names]
        print(f"  [{count} recording(s)] {_fmt_channel_set(names, types)}")
    print()

    # --- Sampling frequency ---
    print("--- Sampling frequency ---")
    sfreqs = Counter(round(item.sampling_frequency, 6) for item in report.successes)
    for sfreq, count in sorted(sfreqs.items()):
        print(f"  {sfreq:g} Hz  ({count} recording(s))")
    print()

    # --- Duration ---
    print("--- Recording duration ---")
    durations = [item.duration_sec for item in report.successes]
    print(f"  min    : {_fmt_duration(min(durations))}")
    print(f"  max    : {_fmt_duration(max(durations))}")
    print(f"  mean   : {_fmt_duration(sum(durations) / len(durations))}")
    print(f"  total  : {_fmt_duration(sum(durations))}")
    print()

    # --- Annotation labels ---
    print("--- Annotation labels ---")
    label_counts: Counter[str] = Counter()
    total_segments = 0
    for item in report.successes:
        label_counts.update(item.annotation_labels)
        total_segments += item.n_annotations
    print(f"  total annotation segments: {total_segments}")
    print(f"  unique labels ({len(label_counts)}):")
    for label, count in label_counts.most_common():
        print(f"    {label!r:40s}  {count}")
    print()

    # --- Per-recording one-liners (compact) ---
    print("--- Recordings ---")
    for item in report.successes:
        print(
            f"  {item.psg_name}: study={item.study} "
            f"subject={item.subject_id} night={item.recording_id} "
            f"sfreq={item.sampling_frequency:g} Hz "
            f"dur={item.duration_sec / 3600.0:.2f} h "
            f"ch={len(item.channel_names)} "
            f"annot={item.n_annotations} "
            f"hyp={item.hypnogram_name}"
        )
    print()

    print(f"RESULT: {'PASS' if report.ok else 'FAIL'}")
    print("=" * 72)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate Phase 1 Sleep-EDF acquisition (public API only).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_project_root() / "configs" / "default.yaml",
        help="Path to pipeline YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override acquisition.data_root from the config file.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Preload signal data into memory (slower, more RAM).",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        help="Logging level (default: WARNING so the report stays readable).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(level=args.log_level)

    settings = load_settings(args.config, project_root=_project_root())
    data_root = args.data_root.expanduser().resolve() if args.data_root else settings.acquisition.data_root
    preload = True if args.preload else settings.acquisition.preload

    if not data_root.exists():
        print(f"ERROR: data root does not exist: {data_root}", file=sys.stderr)
        print("Set acquisition.data_root in configs/default.yaml or pass --data-root.", file=sys.stderr)
        return 2

    report = run_verification(data_root, preload=preload)
    print_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
