#!/usr/bin/env python3
"""Run the local Micro-World browser inspector."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "micro-world"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def main() -> int:
    try:
        import uvicorn
    except ImportError:
        print(
            "Browser dependencies are missing. Run: "
            "python3 -m pip install -e '.[web]'",
            file=sys.stderr,
        )
        return 2
    uvicorn.run(
        "micro_world.service.api:create_app",
        factory=True,
        host="127.0.0.1",
        port=8000,
        reload=False,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
