"""Rival systems, without the network: the grep agent's tools and full
context's not-applicable rule."""

from vectorless_bench.datasets.base import CORPUS
from vectorless_bench.retrievers.agentic_grep import _Corpus
from vectorless_bench.schema import Doc, Question


def test_grep_and_read():
    c = _Corpus([Doc(doc_id="d", title="d", content="Intro\nRevenue was $9.1bn\fCapital expenditure: $1,577m\nOther")])
    hits = c.grep(["d"], r"capital\s+expenditure")
    assert "d | p.2 | L.1 |" in hits
    assert "no matches" == c.grep(["d"], "zzz")
    assert c.grep(["d"], "[unclosed") == "no matches", "an invalid regex falls back to a literal search"
    text, shown = c.read("d", 2, 1, 2)
    assert text.startswith("Capital expenditure") and "L.2 | Other" in shown
    assert c.read("d", 9, 1, 1)[0] == ""


def test_full_context_is_not_applicable_on_corpus_questions():
    from vectorless_bench.retrievers.full_context import FullContextRetriever

    r = FullContextRetriever()
    r.setup([])
    out = r.retrieve(Question(qid="q", doc_id=CORPUS, question="?"), k=5)
    assert out.error.startswith("n/a:")
