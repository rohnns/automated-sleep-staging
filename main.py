#!/usr/bin/env python
"""Sleep staging pipeline CLI entry point.

Usage:
    python main.py                    # show help
    python main.py --stage preprocess # run preprocessing only
    python main.py --stage train      # run training only
    python main.py --stage evaluate   # run evaluation only
    python main.py --stage all        # run full pipeline
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sleep-staging-pipeline",
        description="Modular end-to-end clinical EEG sleep staging pipeline.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "default.yaml",
        help="Pipeline YAML config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--stage",
        choices=["preprocess", "train", "evaluate", "all"],
        default=None,
        help="Pipeline stage to run. Omit to show this help.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Override acquisition.data_root from config.",
    )
    parser.add_argument(
        "--max-recordings",
        type=int,
        default=None,
        help="Limit number of recordings (for smoke tests).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    if args.stage is None:
        _parse_args(["--help"])
        return 0

    from sleep_staging.config import load_settings
    from sleep_staging.common import configure_logging

    settings = load_settings(args.config, project_root=PROJECT_ROOT)
    configure_logging(level=settings.logging.level)

    if args.stage in ("preprocess", "all"):
        print("[main] Preprocessing stage — use scripts/experiments/run_experiment.py for full runs")
        print("[main] Or: python -m scripts.experiments.run_experiment")

    if args.stage in ("train", "all"):
        print("[main] Training stage — use scripts/experiments/run_experiment.py for full runs")

    if args.stage in ("evaluate", "all"):
        print("[main] Evaluation stage — use scripts/evaluation/eval_checkpoints.py")

    print(f"[main] Config loaded from: {args.config}")
    print(f"[main] Data root: {settings.acquisition.data_root}")
    print(f"[main] Stage: {args.stage}")
    print("[main] Pipeline entry point ready. Use specific scripts for full execution.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
