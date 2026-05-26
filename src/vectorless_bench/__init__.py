"""vectorless-bench: an advanced benchmarking suite for the Vectorless engine.

Public surface for programmatic use:

    from vectorless_bench import RunConfig, run, write_reports
    run_dir = run(RunConfig(dataset="fixtures", systems=["mock"], repeats=3))
    write_reports(run_dir, k=5)

Everything reachable from here is dependency-light; heavy retriever/dataset
dependencies load lazily only when those systems are actually used.
"""

from __future__ import annotations

from .config import RunConfig, load_config
from .report import write_reports
from .runner import run

__all__ = ["RunConfig", "load_config", "run", "write_reports", "__version__"]
__version__ = "0.1.0"
