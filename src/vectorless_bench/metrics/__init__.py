"""Per-question scoring entry point + re-exports.

`score_question` is the one function the runner calls; it turns a RetrievalResult
into the metric dict stored on a RunRecord. It picks the right metric family for
the question's answer_type (answerable vs no-answer) so abstention questions are
never scored as if they had a gold passage.
"""

from __future__ import annotations

from typing import Dict

from ..anchors import covered_anchors, relevance_labels
from ..schema import Question, RetrievalResult
from . import aggregate, citation, retrieval

__all__ = ["score_question", "retrieval", "citation", "aggregate", "primary_quality"]

from .citation import primary_quality  # noqa: E402  (re-export)


def score_question(
    question: Question, result: RetrievalResult, k: int
) -> Dict[str, float]:
    """Compute the full metric bundle for one (question, retrieval) pair."""
    sections = result.sections

    # No-answer items are scored purely on abstention; ranking metrics don't
    # apply because there is no gold passage to find.
    if question.is_no_answer:
        m = citation.abstention_metrics(question, sections)
        return m

    labels = relevance_labels(sections, question.gold)
    num_gold = max(1, len(question.gold))
    m: Dict[str, float] = {}
    m.update(retrieval.score_ranking(labels, num_gold, k))
    m.update(citation.citation_metrics(sections, question.gold))
    m["gold_covered"] = covered_anchors(sections, question.gold) / num_gold
    return m
