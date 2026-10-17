"""The shared answer step (HAL-3203): every system's excerpts are numbered and
labelled with what the system said about where they came from, and every
graded row records which prompt wrote its answer."""

from types import SimpleNamespace

from vectorless_bench import judge as jmod
from vectorless_bench.schema import Question, RetrievedSection, Usage


def _gen(text):
    return SimpleNamespace(text=text, usage=Usage())


def test_excerpts_carry_title_page_and_document():
    secs = [
        RetrievedSection(content="Net revenue was $86,392 million.", title="Income statement · p.62", page=62),
        RetrievedSection(content="Capex was $5,207 million.", title_path=["Part II", "Item 8", "Cash flows"], doc_id="PEP_2022_10K"),
        RetrievedSection(content="bare chunk text"),
    ]
    text = jmod.excerpts(secs)
    assert "[1] Income statement · p.62\nNet revenue was $86,392 million." in text
    assert "[2] PEP_2022_10K — Part II › Item 8 › Cash flows\nCapex was $5,207 million." in text
    # a chunk retriever that returns bare text gets a bare number, nothing invented
    assert "[3]\nbare chunk text" in text
    assert jmod.excerpts([]) == "(no excerpts)"


def test_answer_step_reads_labelled_excerpts_and_may_derive(monkeypatch):
    seen = {}

    def fake_complete(model, system, prompt, **kw):
        seen.update(system=system, prompt=prompt)
        return _gen("INSUFFICIENT")

    monkeypatch.setattr(jmod, "complete", fake_complete)
    j = jmod.Judge("gemini-2.5-flash-lite", "not-jev")
    res = j.evaluate(Question(qid="q", doc_id="d", question="What is the operating margin?", answer="13.3%"),
                     [RetrievedSection(content="Operating profit $11,512m on revenue $86,392m.", page=62)])
    assert seen["prompt"].startswith("EXCERPTS:\n[1], p.62\nOperating profit")
    assert "<<expression>>" in seen["system"] and "INSUFFICIENT" in seen["system"]
    assert res.verdict == "abstained" and res.prompt == jmod.ANSWER_PROMPT


def test_a_graded_record_names_its_answer_prompt():
    from vectorless_bench import runner
    from vectorless_bench.config import RunConfig
    from vectorless_bench.judge import JudgeResult
    from vectorless_bench.schema import GoldAnchor, RetrievalResult

    class _Ret:
        def retrieve(self, q, k, cold=True):
            return RetrievalResult(qid=q.qid, system="x", query=q.question, sections=[RetrievedSection(content="a")])

    class _Judge:
        def evaluate(self, q, sections, answer_parts=None):
            return JudgeResult(correct=1.0, faithful=1.0, answered=1.0, usage=Usage(), verdict="correct", candidate="a")

    q = Question(qid="q", doc_id="d", question="?", gold=[GoldAnchor(doc_id="d", answer_spans=["a"])], answer="a")
    rec = runner._eval_one("x", q, _Ret(), RunConfig(dataset="fixtures"), _Judge(), 0, None)
    assert rec.trace["answer_prompt"] == jmod.ANSWER_PROMPT == "excerpts-calc-v3"
    assert rec.trace["calculations"] == []


def test_calculations_are_evaluated_before_grading(monkeypatch):
    seen = {}
    monkeypatch.setattr(jmod, "complete", lambda *a, **k: _gen(
        "DPO = <<365 * ((1,174 + 1,587) / 2) / (7,772 + (2,438 - 2,320))>> days [1][2]"))

    def fake_legacy(self, question, candidate, usage):
        seen["candidate"] = candidate
        return jmod.JudgeResult(correct=1.0, faithful=1.0, answered=1.0, usage=usage, verdict="legacy", candidate=candidate)

    monkeypatch.setattr(jmod.Judge, "_legacy", fake_legacy)
    res = jmod.Judge("gemini-2.5-flash-lite", "not-jev").evaluate(
        Question(qid="q", doc_id="d", question="Corning FY2020 DPO?", answer="63.86"), [RetrievedSection(content="x")])
    # the grader sees the value code computed, not the model's arithmetic
    assert seen["candidate"].startswith("DPO = 63.8634 (= 365 * ((1,174 + 1,587) / 2)")
    assert res.calculations[0]["value"] == __import__("pytest").approx(63.8634, abs=1e-4)


def test_v3_is_the_default_and_v4_stays_selectable(monkeypatch):
    seen = []
    monkeypatch.setattr(jmod, "complete", lambda model, system, prompt, **k: seen.append(system) or _gen("INSUFFICIENT"))
    q = Question(qid="q", doc_id="d", question="What acquisitions did Ulta make?", answer="None")
    default = jmod.Judge("gemini-2.5-flash-lite", "not-jev").evaluate(q, [RetrievedSection(content="x")])
    v4 = jmod.Judge("gemini-2.5-flash-lite", "not-jev", prompt="excerpts-calc-v4")
    res = v4.evaluate(q, [RetrievedSection(content="x")])
    assert default.prompt == "excerpts-calc-v3" and '"None" is an answer' not in seen[0]
    assert '"None" is an answer' in seen[1] and res.prompt == "excerpts-calc-v4"


def test_an_unknown_prompt_is_refused():
    import pytest
    with pytest.raises(ValueError, match="unknown answer prompt"):
        jmod.Judge("gemini-2.5-flash-lite", "not-jev", prompt="excerpts-v9")


def test_the_run_config_names_the_prompt_the_judge_uses():
    from vectorless_bench.config import RunConfig
    from vectorless_bench.runner import answer_setup

    cfg = RunConfig(judge=True, answer_prompt="excerpts-calc-v3")
    assert answer_setup(cfg)["answer_prompt"] == "excerpts-calc-v3"
    assert RunConfig().answer_prompt == jmod.ANSWER_PROMPT


def test_a_refusal_spell_outlasting_the_first_retries_still_gets_an_answer(monkeypatch):
    calls = []

    class Busy(Exception):
        code = 429

    def flaky(model, system, user, **kw):
        calls.append(1)
        if len(calls) <= 2:
            raise Busy("429 RESOURCE_EXHAUSTED")
        return _gen("Revenue was $86,392 million [1]")

    monkeypatch.setattr(jmod, "complete", flaky)
    monkeypatch.setattr("vectorless_bench._llm._cooldown", lambda s: None)
    j = jmod.Judge("gemini-2.5-flash-lite", "not-jev")
    j.patience = 0.0
    monkeypatch.setattr(jmod.Judge, "_legacy", lambda self, q, c, u: jmod.JudgeResult(
        correct=1.0, faithful=1.0, answered=1.0, usage=u, verdict="legacy", candidate=c))
    res = j.evaluate(Question(qid="q", doc_id="d", question="Revenue?", answer="$86,392m"), [RetrievedSection(content="x")])
    assert len(calls) == 3 and res.candidate.startswith("Revenue was")


def test_a_permanent_error_is_not_retried(monkeypatch):
    import pytest

    calls = []

    def bad(model, system, user, **kw):
        calls.append(1)
        raise ValueError("400 invalid argument")

    monkeypatch.setattr(jmod, "complete", bad)
    j = jmod.Judge("gemini-2.5-flash-lite", "not-jev")
    j.patience = 0.0
    with pytest.raises(ValueError):
        j.evaluate(Question(qid="q", doc_id="d", question="?", answer="a"), [RetrievedSection(content="x")])
    assert len(calls) == 1
