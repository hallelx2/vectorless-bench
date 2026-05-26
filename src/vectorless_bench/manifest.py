"""Reproducibility manifest.

A benchmark number is only meaningful if you can say exactly how it was produced.
Every run writes a manifest.json capturing the things that move results: model
versions, the price book, the cache mode, dataset + sample seed, code version, and
environment. If you can't reproduce a number later, the manifest tells you what
changed.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import pricing


def _git_sha(path: Optional[str] = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=path, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _pkg_version(name: str) -> Optional[str]:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:
        return None


@dataclass
class Manifest:
    created_at: str
    bench_git_sha: str
    dataset: str
    dataset_size: int
    systems: List[str]
    k: int
    repeats: int
    cold: bool
    model: str
    judge_model: Optional[str]
    embedding_model: Optional[str]
    sample_seed: int
    pricing_fingerprint: str
    python: str
    platform: str
    package_versions: Dict[str, Optional[str]]
    warnings: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(cls, *, dataset: str, dataset_size: int, systems: List[str], k: int,
              repeats: int, cold: bool, model: str, judge_model: Optional[str],
              embedding_model: Optional[str], sample_seed: int,
              warnings: Optional[List[str]] = None,
              extra: Optional[Dict[str, Any]] = None) -> "Manifest":
        return cls(
            created_at=datetime.now(timezone.utc).isoformat(),
            bench_git_sha=_git_sha(str(Path(__file__).resolve().parents[2])),
            dataset=dataset,
            dataset_size=dataset_size,
            systems=systems,
            k=k,
            repeats=repeats,
            cold=cold,
            model=model,
            judge_model=judge_model,
            embedding_model=embedding_model,
            sample_seed=sample_seed,
            pricing_fingerprint=pricing.pricing_fingerprint(),
            python=sys.version.split()[0],
            platform=platform.platform(),
            package_versions={
                name: _pkg_version(name)
                for name in ("vectorless-sdk", "openai", "anthropic",
                             "datasets", "rank-bm25", "psycopg", "tiktoken")
            },
            warnings=warnings or [],
            extra=extra or {},
        )

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
