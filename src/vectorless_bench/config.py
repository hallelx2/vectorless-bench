"""Run configuration.

A RunConfig fully specifies a benchmark run. It can be built directly (the smoke
test does this, needing no YAML) or loaded from a config file. Per-system knobs
live in `system_params[name]` and are splatted into the retriever factory, so new
retriever options never require touching this dataclass.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RunConfig:
    dataset: str = "fixtures"
    systems: List[str] = field(default_factory=lambda: ["mock"])
    k: int = 5
    repeats: int = 1  # >1 enables the determinism axis
    cold: bool = True  # intent to bypass caches (see retriever notes)
    limit: Optional[int] = None  # sample this many questions (stratified)
    sample_seed: int = 0
    out_dir: str = "runs"

    # models
    model: str = "claude-haiku-4-5"  # retrieval model for vectorless/full_context
    embedding_model: str = "text-embedding-3-small"
    judge: bool = False
    judge_model: str = "gpt-4o"
    gen_model: str = "gpt-4o-mini"  # answer generator for the judge axis

    # dataset-specific options (e.g. docs_dir for financebench)
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    # per-system overrides, keyed by system name
    system_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def params_for(self, system: str) -> Dict[str, Any]:
        """Factory kwargs for a system: shared models + its own overrides."""
        base: Dict[str, Any] = {}
        if system in ("vectorless", "full_context"):
            base["model"] = self.model
        if system == "vector_rag":
            base["embedding_model"] = self.embedding_model
        if system == "pageindex":
            # reasoning/selection runs on our priced model; tree-build model can
            # be overridden in system_params (PageIndex's own default otherwise).
            base["retrieve_model"] = self.model
        base.update(self.system_params.get(system, {}))
        return base


def load_config(path: str) -> RunConfig:
    import yaml  # lazy: smoke test/CI builds RunConfig directly, no yaml needed

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = RunConfig().__dict__.keys()
    kwargs = {k: v for k, v in data.items() if k in known}
    return RunConfig(**kwargs)
