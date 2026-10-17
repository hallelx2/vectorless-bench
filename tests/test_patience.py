"""HAL-3284: a refusal for load costs time, never a question."""

import json
from types import SimpleNamespace

import pytest

from vectorless_bench import _llm


class Busy(Exception):
    code = 429


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(_llm, "_cooldown", lambda s: None)
    monkeypatch.setattr("time.sleep", lambda s: None)


def test_patience_runs_on_a_time_budget_not_a_count(monkeypatch):
    calls = []

    def flaky():
        calls.append(1)
        if len(calls) < 9:
            raise Busy("429 RESOURCE_EXHAUSTED")
        return "ok"

    assert _llm._patiently(flaky, pause=0.0) == "ok" and len(calls) == 9


def test_patience_gives_up_when_the_budget_is_spent(monkeypatch):
    def always():
        raise Busy("429 RESOURCE_EXHAUSTED")

    monkeypatch.setenv("VLBENCH_PATIENCE_SECONDS", "0")
    with pytest.raises(Busy):
        _llm._patiently(always, pause=0.0)


def test_patience_never_retries_a_permanent_error():
    calls = []

    def bad():
        calls.append(1)
        raise ValueError("400 invalid argument")

    with pytest.raises(ValueError):
        _llm._patiently(bad, pause=0.0)
    assert len(calls) == 1


def test_memwalker_question_time_call_rides_out_refusals(monkeypatch):
    from vectorless_bench.retrievers import memwalker as mw
    from vectorless_bench.schema import Usage

    calls = []

    def fake(model, system, user, **kw):
        calls.append(1)
        if len(calls) <= 2:
            raise Busy("429 RESOURCE_EXHAUSTED")
        return SimpleNamespace(text=json.dumps({"action": "answer"}), usage=Usage())

    monkeypatch.setattr(mw, "complete", fake)
    r = mw.MemWalkerRetriever.__new__(mw.MemWalkerRetriever)
    r.model = "gemini-2.5-flash-lite"
    assert r._ask("s", "u", Usage()) == {"action": "answer"} and len(calls) == 3


def test_pageindex_restarts_a_refused_question(monkeypatch):
    from vectorless_bench.retrievers import pageindex as pim
    from vectorless_bench.schema import Question, RetrievalResult, Usage

    r = pim.PageIndexRetriever.__new__(pim.PageIndexRetriever)
    r.name = "pageindex"
    attempts = []

    def ask_once(question, k, cold):
        attempts.append(1)
        if len(attempts) <= 2:
            err = Exception("The model backend failed: litellm.MidStreamFallbackError: litellm.RateLimitError")
            return RetrievalResult(qid=question.qid, system="pageindex", query=question.question,
                                   usage=Usage(cost_usd=0.001), error=str(err)), err
        return RetrievalResult(qid=question.qid, system="pageindex", query=question.question,
                               usage=Usage(cost_usd=0.002), latency_ms=1200.0, trace={}), None

    monkeypatch.setattr(r, "_ask_once", ask_once)
    res = r.retrieve(Question(qid="q", doc_id="d", question="?"), k=5)
    assert res.error is None and len(attempts) == 3
    assert res.latency_ms == 1200.0  # the final attempt's, not the waits
    assert res.trace["throttled_retries"] == 2
    assert res.usage.cost_usd == pytest.approx(0.004)  # the cut-short attempts' spend is kept


def test_pageindex_does_not_retry_a_non_refusal(monkeypatch):
    from vectorless_bench.retrievers import pageindex as pim
    from vectorless_bench.schema import Question, RetrievalResult

    r = pim.PageIndexRetriever.__new__(pim.PageIndexRetriever)
    r.name = "pageindex"
    attempts = []

    def ask_once(question, k, cold):
        attempts.append(1)
        err = ValueError("tool schema mismatch")
        return RetrievalResult(qid=question.qid, system="pageindex", query=question.question, error=str(err)), err

    monkeypatch.setattr(r, "_ask_once", ask_once)
    assert r.retrieve(Question(qid="q", doc_id="d", question="?"), k=5).error and len(attempts) == 1


def test_a_cell_with_questions_lost_to_refusals_is_withheld(tmp_path):
    from vectorless_bench.scoreboard import build, throttled

    assert throttled("litellm.RateLimitError: 429") and throttled("429 RESOURCE_EXHAUSTED")
    assert not throttled("document 'x' was not indexed") and not throttled(None)
    d = tmp_path / "run"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({"dataset": "qasper", "k": 5, "created_at": "2026-10-17T10:00:00",
                                                 "model": "gemini-2.5-flash-lite", "extra": {}}))
    base = {"repeat": 0, "domain": "d", "answer_type": "lookup", "difficulty": "medium", "latency_ms": 900,
            "wall_ms": 900, "phase": "accuracy", "flagged": False, "attempts": 1,
            "usage": {"cost_usd": 0.001, "total_tokens": 10, "llm_calls": 1}}
    m = {"hit@5": 1.0, "recall@5": 1.0, "f1@5": 1.0, "mrr": 1.0, "ndcg@5": 1.0}
    rows = [dict(base, qid="q1", system="pageindex", metrics=m, error=None),
            dict(base, qid="q2", system="pageindex", metrics={}, error="litellm.RateLimitError: 429"),
            dict(base, qid="q1", system="bm25", metrics=m, error=None)]
    (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    build([d], tmp_path / "out")
    out = {r["system"]: r for r in json.loads((tmp_path / "out" / "scoreboard.json").read_text())["rows"]}
    assert out["pageindex"]["publishable"] is False and out["pageindex"]["throttled_errors"] == 1
    assert out["pageindex"]["quality"] is None and "withheld" in out["pageindex"]["label"]
    assert out["bm25"]["publishable"] is True and out["bm25"]["quality"] is not None
