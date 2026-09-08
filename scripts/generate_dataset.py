#!/usr/bin/env python3
"""Command-line entry point for Micro-Transformer dataset generation and validation."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from micro_transformer.data.generator import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
