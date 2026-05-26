"""Corpus-level aggregation: latency percentiles, cost roll-ups, determinism,
and the efficiency-frontier numbers that are the headline for an
LLM-as-retriever engine.
"""

from __future__ import annotations

import statistics
from typing import Dict, List, Sequence, Set


def percentiles(values: Sequence[float]) -> Dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    xs = sorted(values)

    def pct(p: float) -> float:
        if len(xs) == 1:
            return xs[0]
        idx = p / 100.0 * (len(xs) - 1)
        lo = int(idx)
        frac = idx - lo
        hi = min(lo + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * frac

    return {
        "p50": pct(50),
        "p95": pct(95),
        "p99": pct(99),
        "mean": statistics.fmean(xs),
        "max": xs[-1],
    }


def cost_summary(
    costs: Sequence[float],
    qualities: Sequence[float],
    total_tokens: Sequence[int],
    llm_calls: Sequence[int],
) -> Dict[str, float]:
    """Cost reported ALWAYS alongside quality — never in isolation.

    `cost_per_correct` divides total spend by the number of questions the system
    actually got right (quality >= 0.5), which is the metric that exposes a
    cheap-but-wrong system and a right-but-ruinous one alike."""
    n = max(1, len(costs))
    total_cost = sum(costs)
    n_correct = sum(1 for q in qualities if q >= 0.5)
    mean_quality = statistics.fmean(qualities) if qualities else 0.0
    mean_cost = total_cost / n
    return {
        "mean_cost_usd": mean_cost,
        "total_cost_usd": total_cost,
        "mean_tokens": (sum(total_tokens) / n) if total_tokens else 0.0,
        "mean_llm_calls": (sum(llm_calls) / n) if llm_calls else 0.0,
        "cost_per_correct_usd": (total_cost / n_correct) if n_correct else float("inf"),
        # quality per dollar: the efficiency frontier's y/x. Scaled to $1k of
        # spend so the number is human-readable across cheap/expensive systems.
        "quality_per_1k_usd": (mean_quality / mean_cost / 1000.0)
        if mean_cost > 0
        else float("inf"),
    }


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 1.0


def determinism(repeat_id_sets: Sequence[Sequence[str]]) -> Dict[str, float]:
    """Given the selected-section-id sets from N reruns of the SAME query,
    measure stability. Temp=0 should make this ~1.0; anything less quantifies
    provider nondeterminism, which the engine's determinism claim depends on.

    - exact_match: all reruns returned the identical set.
    - mean_jaccard: average pairwise set overlap.
    """
    runs = [set(s) for s in repeat_id_sets]
    if len(runs) < 2:
        return {"exact_match": 1.0, "mean_jaccard": 1.0}
    exact = all(r == runs[0] for r in runs[1:])
    pairs: List[float] = [
        jaccard(runs[i], runs[j])
        for i in range(len(runs))
        for j in range(i + 1, len(runs))
    ]
    return {
        "exact_match": 1.0 if exact else 0.0,
        "mean_jaccard": statistics.fmean(pairs) if pairs else 1.0,
    }


def bootstrap_ci(
    values: Sequence[float], iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Dict[str, float]:
    """Nonparametric bootstrap confidence interval for a mean — so system A vs
    system B differences can be reported with uncertainty, not as bare points."""
    import random

    if not values:
        return {"mean": 0.0, "lo": 0.0, "hi": 0.0}
    rng = random.Random(seed)
    n = len(values)
    means: List[float] = []
    for _ in range(iters):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(statistics.fmean(sample))
    means.sort()
    lo = means[int((alpha / 2) * iters)]
    hi = means[int((1 - alpha / 2) * iters)]
    return {"mean": statistics.fmean(values), "lo": lo, "hi": hi}
