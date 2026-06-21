"""Judge rubric: correctness is gold-recall, not subset-equivalence (HAL-328).

A more-complete answer that conveys every gold fact and contradicts none must
score correct=1 even when it adds detail beyond the (often partial) gold. Only
a MISSING gold fact or a CONTRADICTION makes it incorrect. These tests pin that
behaviour by stubbing the LLM call, so the rubric can't silently regress.
"""

from __future__ import annotations

import json

import vectorless_bench.judge as judge_mod
from vectorless_bench.judge import Judge
from vectorless_bench.schema import AnswerType, Question


class _Resp:
    def __init__(self, text):
        self.text = text
        from vectorless_bench.schema import Usage
        self.usage = Usage()


def _stub_complete(verdict: dict):
    def _fn(model, system, user, **kwargs):
        return _Resp(json.dumps(verdict))
    return _fn


def _q():
    return Question(qid="q1", doc_id="d1", question="What acquisitions?",
                    answer="Company A; Company B", answer_type=AnswerType.LOOKUP)


def test_extra_detail_stays_correct(monkeypatch):
    # Candidate covers both gold facts and adds more — no missing, no conflict.
    monkeypatch.setattr(judge_mod, "complete", _stub_complete({
        "correct": False,  # a conservative model might say False for "extra info"
        "faithful": True, "missing_facts": [], "contradicting_facts": [],
        "reason": "adds dates beyond reference",
    }))
    jr = Judge("m", "m").evaluate(_q(), [], native_answer="A on Jan 1, B on Feb 2, plus C")
    assert jr.correct == 1.0  # structural rule overrides the conservative model call
    assert jr.faithful == 1.0


def test_missing_gold_fact_is_incorrect(monkeypatch):
    monkeypatch.setattr(judge_mod, "complete", _stub_complete({
        "correct": True, "faithful": True,
        "missing_facts": ["Company B"], "contradicting_facts": [],
        "reason": "only mentions A",
    }))
    jr = Judge("m", "m").evaluate(_q(), [], native_answer="Only Company A")
    assert jr.correct == 0.0
    assert jr.missing_facts == ["Company B"]


def test_contradiction_is_incorrect_and_unfaithful(monkeypatch):
    monkeypatch.setattr(judge_mod, "complete", _stub_complete({
        "correct": True, "faithful": True,
        "missing_facts": [], "contradicting_facts": ["says Company X not A"],
        "reason": "wrong entity",
    }))
    jr = Judge("m", "m").evaluate(_q(), [], native_answer="Company X and Company B")
    assert jr.correct == 0.0
    assert jr.faithful == 0.0
    assert jr.contradicting_facts


def test_review_fields_populated(monkeypatch):
    monkeypatch.setattr(judge_mod, "complete", _stub_complete({
        "correct": True, "faithful": True, "missing_facts": [],
        "contradicting_facts": [], "reason": "matches",
    }))
    jr = Judge("m", "m").evaluate(_q(), [], native_answer="Company A and Company B")
    assert jr.candidate == "Company A and Company B"
    assert jr.reference == "Company A; Company B"
    assert jr.reason == "matches"
