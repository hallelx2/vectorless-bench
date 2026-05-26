"""Make `vectorless_bench` importable from src/ without an install, so the suite
can be tested straight from a checkout (CI runs `pytest` with no build step)."""

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
