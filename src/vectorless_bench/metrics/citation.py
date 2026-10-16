"""Citation-exactness, near-miss, and abstention metrics.

These are the metrics that test Vectorless's *differentiating* claims rather
than generic retrieval quality:

- citation exactness: does the top-ranked unit actually contain the answer, and
  does its structural path match? ("the answer has a page number")
- near-miss rate: did the system grab a sibling of the right section (right
  neighbourhood, wrong leaf)? This is the exact vector-RAG failure the
  whitepaper argues against, so we measure it head-on.
- path coverage / gold_has_path: whether the two metrics above could be
  computed at all. They are silent about the difference between "measured,
  and the answer is zero" and "could not be measured", and both failures
  look identical in a report — a 0.000 that reads as a real score.

  There are two independent ways they become unmeasurable:

  * the SYSTEM returned no heading paths (`path_coverage == 0`). A chunk
    system has no sibling relation, so its near-miss rate is undefined, not
    0.0 — and 0.0 reads as a perfect score for the very baseline these
    metrics exist to contrast against.
  * the DATASET carries no gold heading paths (`gold_has_path == 0`).
    FinanceBench ships evidence text and page numbers only, so
    `path_correct_top1` is 0.000 there for EVERY system including a perfect
    one. Claiming path-correctness from a FinanceBench run is claiming a
    number the corpus cannot produce. Use a dataset whose gold carries
    `title_path` (the fixtures corpus does) or annotate one.
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
            # Nothing returned: a real miss on path-correctness when the
            # question has a gold path; near-miss has no units to rate.
            **({"path_correct_top1": 0.0} if any(g.title_path for g in golds) else {}),
            "path_coverage": 0.0,
            "gold_has_path": 1.0 if any(g.title_path for g in golds) else 0.0,
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
    # Path metrics are STRUCTURAL and only exist where both sides have a
    # path. A metric that cannot be measured for this question is left OUT
    # of the dict, never written as 0.0: averages then run over the
    # questions that could be scored, and a system or corpus with no paths
    # reads "n/a" instead of a score of zero (HAL-1564).
    gold_has_path = any(g.title_path for g in golds)
    with_path = [s for s in sections if s.title_path]
    out = {
        "span_in_top1": 1.0 if span_top1 else 0.0,
        "span_in_any": 1.0 if span_any else 0.0,
        "path_coverage": len(with_path) / len(sections),
        "gold_has_path": 1.0 if gold_has_path else 0.0,
    }
    if gold_has_path and top1.title_path:
        out["path_correct_top1"] = 1.0 if any(
            g.title_path and path_matches(g.title_path, top1.title_path) for g in golds
        ) else 0.0
    if gold_has_path and with_path:
        # Over units that carry a path: a unit with none cannot be a sibling.
        out["near_miss_rate"] = sum(1 for s in with_path if is_sibling_near_miss(s, golds)) / len(with_path)
    return out


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
