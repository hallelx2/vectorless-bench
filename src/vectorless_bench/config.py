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
    # Every generative call is Gemini on Vertex; every embedding is Gemini on
    # Vertex or a local BGE model (see _llm.py). validate() enforces it.
    model: str = "gemini-3.5-flash-lite"  # the answer model, shared by every system (HAL-3048)
    embedding_model: str = "gemini-embedding-2"
    judge: bool = False
    judge_model: str = "jev-latest"  # graded on Jev, a closed-set verdict (HAL-3047)  # HAL-3047 moves grading to Jev
    gen_model: str = "gemini-3.5-flash-lite"  # answer generator for the judge axis

    # How the run is timed (HAL-3032). "latency": one question at a time, every
    # sample bracketed by host evidence and rerun if the machine was busy.
    # "throughput": `concurrency` questions in flight, reported separately.
    phase: str = "latency"
    concurrency: int = 1
    # Speed rig settings (see rig.RigConfig): probe agent URLs per remote host,
    # steal / foreign-process thresholds, reruns, provider RTT targets.
    rig: Dict[str, Any] = field(default_factory=dict)
    # Stop the run once recorded spend reaches this many USD (None: no cap).
    max_spend_usd: Optional[float] = None

    # dataset-specific options (e.g. docs_dir for financebench)
    dataset_params: Dict[str, Any] = field(default_factory=dict)
    # per-system overrides, keyed by system name
    system_params: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def params_for(self, system: str) -> Dict[str, Any]:
        """Factory kwargs for a system: shared models + its own overrides."""
        base: Dict[str, Any] = {}
        if system in ("vectorless", "full_context", "agentic_grep", "memwalker", "readagent", "raptor"):
            base["model"] = self.model
        if system == "vector_rag":
            base["embedding_model"] = self.embedding_model
        if system == "pageindex":
            # PageIndex indexes and chats on the run's shared model unless
            # system_params override either.
            base["model"] = self.model
            base["retrieve_model"] = self.model
        base.update(self.system_params.get(system, {}))
        return base


def _is_vertex_or_local(model: str) -> bool:
    return model.startswith("gemini") or model.startswith("jev") or "/" in model


def validate(cfg: "RunConfig") -> None:
    """Refuse a run that would send any call outside Vertex Gemini, a local
    model or Jev. Checked before anything is indexed, so a bad config costs
    nothing."""
    named = {"model": cfg.model, "embedding_model": cfg.embedding_model, "gen_model": cfg.gen_model}
    if cfg.judge:
        named["judge_model"] = cfg.judge_model
    for system, params in cfg.system_params.items():
        for k, v in params.items():
            if isinstance(v, str) and (k == "model" or k.endswith("_model")) and v:
                named[f"system_params.{system}.{k}"] = v
    if cfg.phase not in ("latency", "throughput", "accuracy"):
        raise ValueError(f"phase must be latency, throughput or accuracy, got {cfg.phase!r}")
    if cfg.phase == "throughput" and cfg.concurrency < 1:
        raise ValueError("throughput needs concurrency >= 1")
    bad = {k: v for k, v in named.items() if not _is_vertex_or_local(v)}
    if bad:
        raise ValueError(
            "models outside Vertex Gemini / local BGE / Jev: "
            + ", ".join(f"{k}={v!r}" for k, v in sorted(bad.items()))
        )


def load_config(path: str) -> RunConfig:
    import yaml  # lazy: smoke test/CI builds RunConfig directly, no yaml needed

    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    known = RunConfig().__dict__.keys()
    unknown = sorted(set(data) - set(known))
    if unknown:
        # A misspelt key (max_spend for max_spend_usd) used to be dropped in
        # silence, and the run went ahead without the setting.
        raise ValueError(f"{path}: unknown config keys {unknown}; known keys are {sorted(known)}")
    return RunConfig(**data)
