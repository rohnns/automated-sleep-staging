#!/usr/bin/env python
"""Primary experiment CLI.

Usage examples:
    python main.py --model all
    python main.py --model raw --smoke
    python main.py --model classical --dataset-root D:/SleepEDFX
"""

from __future__ import annotations

import os

# Must be set before the first CUDA call (i.e. before `torch` is imported by
# anything below) -- mitigates CUDA allocator fragmentation on long training
# loops (thousands of small/varying allocations), which is what caused
# `torch.AcceleratorError: CUDA error: out of memory` deep into epoch 0 on
# this machine's 6 GB laptop GPU despite the per-step model footprint being
# tiny. Purely an allocator behavior change -- no effect on training math,
# model architecture, or results.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sleep_staging.common.logging_utils import configure_logging
from sleep_staging.training.sc_to_st import run_primary_experiment, write_primary_inventory_report


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="sleep-staging-pipeline")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "default.yaml")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--model", choices=["raw", "bandpower", "time_frequency", "classical", "all"], default="all")
    parser.add_argument("--max-sc-recordings", type=int, default=None)
    parser.add_argument("--max-st-recordings", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--inventory-report", type=Path, default=PROJECT_ROOT / "artifacts" / "reports" / "sc_to_st" / "inventory.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = _parse_args(argv)
    report = run_primary_experiment(
        config_path=args.config,
        dataset_root=args.dataset_root,
        model=args.model,
        max_sc_recordings=args.max_sc_recordings,
        max_st_recordings=args.max_st_recordings,
        smoke=args.smoke,
    )
    out = write_primary_inventory_report(report, args.inventory_report)
    print(f"WROTE {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
