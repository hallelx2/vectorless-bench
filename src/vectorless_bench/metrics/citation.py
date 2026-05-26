"""Citation-exactness, near-miss, and abstention metrics.

These are the metrics that test Vectorless's *differentiating* claims rather
than generic retrieval quality:

- citation exactness: does the top-ranked unit actually contain the answer, and
  does its structural path match? ("the answer has a page number")
- near-miss rate: did the system grab a sibling of the right section (right
  neighbourhood, wrong leaf)? This is the exact vector-RAG failure the
  whitepaper argues against, so we measure it head-on.
- abstention: on NO_ANSWER questions, did the system correctly return nothing?
"""

from __future__ import annotations

from typing import Dict, List, Sequence

from ..anchors import is_sibling_near_miss, path_matches, span_present
from ..schema import GoldAnchor, Question, RetrievedSection


def citation_metrics(
    sections: Sequence[RetrievedSection], golds: Sequence[GoldAnchor]
) -> Dict[str, float]:
    if not sections:
        return {
            "span_in_top1": 0.0,
            "span_in_any": 0.0,
            "path_correct_top1": 0.0,
            "near_miss_rate": 0.0,
        }

    top1 = sections[0]
    span_top1 = any(
        sp and span_present(sp, top1.content)
        for g in golds
        for sp in g.answer_spans
    )
    span_any = any(
        sp and span_present(sp, s.content)
        for s in sections
        for g in golds
        for sp in g.answer_spans
    )
    # path-correctness is STRUCTURAL: it requires an actual heading-path match,
    # so chunk systems (no paths) correctly score 0 here — that gap is the point.
    path_top1 = any(
        top1.title_path and g.title_path and path_matches(g.title_path, top1.title_path)
        for g in golds
    )
    near = sum(1 for s in sections if is_sibling_near_miss(s, golds))
    return {
        "span_in_top1": 1.0 if span_top1 else 0.0,
        "span_in_any": 1.0 if span_any else 0.0,
        "path_correct_top1": 1.0 if path_top1 else 0.0,
        "near_miss_rate": near / len(sections),
    }


def abstention_metrics(
    question: Question, sections: Sequence[RetrievedSection]
) -> Dict[str, float]:
    """For NO_ANSWER questions, correct behaviour is to return (almost) nothing.

    We treat <=0 returned sections as a correct abstention. A system that hands
    back confident-but-wrong sections here is over-retrieving — the failure mode
    that makes RAG unsafe in regulated domains."""
    if not question.is_no_answer:
        return {}
    abstained = 1.0 if len(sections) == 0 else 0.0
    return {
        "abstained": abstained,
        "over_retrieved": 0.0 if abstained else float(len(sections)),
    }


def primary_quality(metrics: Dict[str, float], k: int) -> float:
    """The single quality number used for Pareto / quality-per-dollar.

    f1@k for answerable questions; the abstention flag for no-answer questions.
    Centralised so every downstream summary agrees on what "quality" means."""
    if "abstained" in metrics:
        return metrics["abstained"]
    return metrics.get(f"f1@{k}", 0.0)
