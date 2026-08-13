#!/usr/bin/env python
"""Thin wrapper for the primary SC→ST experiment."""
from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from main import main

if __name__ == '__main__':
    raise SystemExit(main())
