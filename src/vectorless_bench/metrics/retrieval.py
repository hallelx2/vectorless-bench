"""Ranking / retrieval-quality metrics over a list of per-rank relevance labels.

All functions take `labels: list[bool]` (relevance at each returned rank, in
order) plus `num_gold` (how many distinct relevant units exist) so recall is
well defined. They are pure and dependency-free.
"""

from __future__ import annotations

import math
from typing import Dict, List


def precision_at_k(labels: List[bool], k: int) -> float:
    top = labels[:k]
    return (sum(top) / len(top)) if top else 0.0


def recall_at_k(labels: List[bool], k: int, num_gold: int) -> float:
    if num_gold <= 0:
        return 1.0 if not any(labels[:k]) else 0.0  # no-answer handled upstream
    return min(sum(labels[:k]), num_gold) / num_gold


def f1_at_k(labels: List[bool], k: int, num_gold: int) -> float:
    p = precision_at_k(labels, k)
    r = recall_at_k(labels, k, num_gold)
    return (2 * p * r / (p + r)) if (p + r) > 0 else 0.0


def hit_at_k(labels: List[bool], k: int) -> float:
    return 1.0 if any(labels[:k]) else 0.0


def mrr(labels: List[bool]) -> float:
    for i, rel in enumerate(labels, start=1):
        if rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(labels: List[bool], k: int, num_gold: int) -> float:
    """Binary-gain nDCG@k. IDCG places min(num_gold, k) relevant items first."""
    dcg = sum(
        (1.0 / math.log2(i + 2)) for i, rel in enumerate(labels[:k]) if rel
    )
    ideal_hits = min(num_gold if num_gold > 0 else sum(labels), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return (dcg / idcg) if idcg > 0 else 0.0


def score_ranking(labels: List[bool], num_gold: int, k: int) -> Dict[str, float]:
    """The standard bundle reported per question for answerable items."""
    return {
        f"precision@{k}": precision_at_k(labels, k),
        f"recall@{k}": recall_at_k(labels, k, num_gold),
        f"f1@{k}": f1_at_k(labels, k, num_gold),
        f"hit@{k}": hit_at_k(labels, k),
        "mrr": mrr(labels),
        f"ndcg@{k}": ndcg_at_k(labels, k, num_gold),
    }
