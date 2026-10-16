"""The path index, which silently returned nothing for a year of runs.

`get_document_tree` hands back a FLAT `DocumentTree` — `.sections` is a list
of `SectionView` whose `children` are id strings. The adapter used to walk it
as a nested node tree, so every path came back empty and `path_correct_top1`
read 0.000 for every system in every run. These tests pin the real shape.
"""

from __future__ import annotations

import pytest

from vectorless_bench.retrievers.vectorless import _PathIndex, _norm


class _View:
    """The SDK's SectionView, as the wire actually delivers it."""

    def __init__(self, id, parent_id, title, children=()):
        self.id = id
        self.parent_id = parent_id
        self.title = title
        self.children = list(children)  # ids, NOT nodes


class _Tree:
    def __init__(self, sections):
        self.sections = sections
        # deliberately no .root and no .children — that is the bug's origin


def _flat_tree():
    return _Tree([
        _View("sec_a", "", "Item 7. MD&A", ["sec_a1", "sec_a2"]),
        _View("sec_a1", "sec_a", "Liquidity", []),
        _View("sec_a2", "sec_a", "Long-term Debt", []),
        _View("sec_b", "", "Item 8. Financial Statements", ["sec_b1"]),
        _View("sec_b1", "sec_b", "Liquidity", []),  # same leaf title as sec_a1
    ])


def _index(tree):
    from vectorless_bench.retrievers.vectorless import VectorlessRetriever as R
    idx = _PathIndex()
    R._index_flat(list(tree.sections), idx)
    idx.finalise()
    return idx


def test_flat_tree_yields_paths_not_silence():
    idx = _index(_flat_tree())
    assert idx.by_id["sec_a2"] == ["Item 7. MD&A", "Long-term Debt"]
    assert idx.by_id["sec_a"] == ["Item 7. MD&A"]
    # the regression itself: this map must not be empty
    assert idx.by_id, "path index came back empty — the nested-walk bug is back"


def test_section_unit_resolves_by_id():
    idx = _index(_flat_tree())
    assert idx.resolve("sec_a2", "Long-term Debt") == [
        "Item 7. MD&A", "Long-term Debt",
    ]


def test_page_unit_resolves_via_its_owning_section_title():
    idx = _index(_flat_tree())
    # the engine's shape: "<owning section> · p.<n>"
    assert idx.resolve("page_84", "Long-term Debt · p.84") == [
        "Item 7. MD&A", "Long-term Debt",
    ]


def test_ambiguous_title_resolves_to_nothing_rather_than_a_guess():
    idx = _index(_flat_tree())
    # "Liquidity" is a leaf under both Item 7 and Item 8
    assert idx.resolve("page_12", "Liquidity · p.12") == []
    assert idx.ambiguous_titles == 1


def test_unknown_unit_is_empty_not_an_error():
    idx = _index(_flat_tree())
    assert idx.resolve("page_999", "Nowhere · p.999") == []
    assert idx.resolve("sec_missing", "") == []


@pytest.mark.parametrize("title,expected", [
    ("Long-term Debt · p.84", ["Item 7. MD&A", "Long-term Debt"]),
    ("Long-term Debt  ·  p. 84", ["Item 7. MD&A", "Long-term Debt"]),
    ("LONG-TERM DEBT · p.84", ["Item 7. MD&A", "Long-term Debt"]),
])
def test_page_title_parsing_tolerates_spacing_and_case(title, expected):
    assert _index(_flat_tree()).resolve("page_84", title) == expected


def test_parent_cycle_terminates():
    tree = _Tree([
        _View("sec_x", "sec_y", "X"),
        _View("sec_y", "sec_x", "Y"),
    ])
    _index(tree)  # must return rather than recurse forever


def test_norm_collapses_whitespace_and_case():
    assert _norm("  Long-Term   Debt ") == _norm("long-term debt")


# ── the third layer: the dataset itself ────────────────────────────────────
# path_correct_top1 read 0.000 for three independent reasons stacked on top of
# one another. The first two are fixed above. This is the third, and it is not
# fixable in code: FinanceBench ships evidence text and page numbers, never
# heading paths, so the metric is undefined there for EVERY system — a perfect
# retriever included. The harness must say so rather than print a zero.

from vectorless_bench.metrics.citation import citation_metrics
from vectorless_bench.schema import GoldAnchor, RetrievedSection


def _sec(path, content="Total long-term debt was $12,000 million."):
    return RetrievedSection(content=content, section_id="page_84",
                            title="Long-term Debt · p.84", title_path=list(path))


def test_gold_without_paths_marks_the_metric_unmeasurable():
    # FinanceBench's shape: evidence text and a page, no title_path
    gold = [GoldAnchor(doc_id="AMZN_2022_10K",
                       answer_spans=["Total long-term debt was $12,000 million."],
                       page=84)]
    m = citation_metrics([_sec(["Item 7. MD&A", "Long-term Debt"])], gold)
    assert m["gold_has_path"] == 0.0, "gold carries no path; the run must say so"
    assert m["path_coverage"] == 1.0, "our side DID produce a path"
    # unmeasurable, so absent: averages skip it and the report says n/a
    assert "path_correct_top1" not in m


def test_gold_with_paths_is_measurable_and_can_succeed():
    gold = [GoldAnchor(doc_id="d", answer_spans=["x"],
                       title_path=["Item 7. MD&A", "Long-term Debt"])]
    m = citation_metrics([_sec(["Item 7. MD&A", "Long-term Debt"], "x")], gold)
    assert m["gold_has_path"] == 1.0
    assert m["path_correct_top1"] == 1.0


def test_system_without_paths_is_distinguishable_from_a_wrong_path():
    gold = [GoldAnchor(doc_id="d", answer_spans=["x"],
                       title_path=["Item 7. MD&A", "Long-term Debt"])]
    m = citation_metrics([_sec([], "x")], gold)   # a chunk system
    assert m["gold_has_path"] == 1.0              # the corpus could score it
    assert m["path_coverage"] == 0.0              # the system could not be scored


def test_corpus_question_goes_to_corpus_mode_without_its_gold_documents():
    """A MultiHop question is asked over every ingested document; the gold
    document ids (which say where the answer is) are never sent."""
    from types import SimpleNamespace as NS

    from vectorless_bench.datasets.base import CORPUS
    from vectorless_bench.retrievers.vectorless import VectorlessRetriever
    from vectorless_bench.schema import Question

    sent = {}

    class _Client:
        def query_corpus(self, query, **kw):
            sent.update(kw, query=query)
            return NS(
                evidence=[NS(document_id="srv_b", document_title="BizWire", section_title="", page=1,
                             text="Acme revenue was 10 billion", confidence=0.93, hop=1)],
                hops=[NS(read=["srv_b", "srv_a"])], coverage=0.81, documents_considered=3,
                judge_requests=7, errors=None, strategy="corpus", elapsed_ms=4100, timing=None,
                usage=NS(input_tokens=1000, output_tokens=0, total_tokens=1000, cost_usd=0.00004, llm_calls=7),
            )

    r = VectorlessRetriever.__new__(VectorlessRetriever)
    r.name, r._client, r.max_hops, r.server_cache_disabled = "vectorless", _Client(), 2, True
    r.corpus_abstain, r.corpus_card_chars = True, 400
    r._doc_ids = {"a": "srv_a", "b": "srv_b", "c": "srv_c"}
    q = Question(qid="q1", doc_id=CORPUS, question="Did both outlets report 10 billion?",
                 meta={"doc_ids": ["a", "b"]})
    res = r.retrieve(q, k=5)

    assert sorted(sent["document_ids"]) == ["srv_a", "srv_b", "srv_c"], "every ingested document, not the gold ones"
    assert sent["max_evidence"] == 5 and sent["max_hops"] == 2
    assert sent["abstain"] is True and sent["card_chars"] == 400
    assert res.error is None and res.sections[0].section_id == "b#p1" and res.sections[0].score == 0.93
    assert res.trace["mode"] == "corpus" and res.trace["coverage"] == 0.81
    assert res.trace["documents_read"] == ["b", "a"]
    assert res.usage.cost_usd == 0.00004 and res.latency_ms == 4100.0


def test_setup_ingests_concurrently_and_counts_reused_documents():
    import threading
    import time
    from types import SimpleNamespace as NS

    from vectorless_bench.retrievers.vectorless import VectorlessRetriever
    from vectorless_bench.schema import Doc

    lock = threading.Lock()
    state = {"in_flight": 0, "peak": 0, "keys": []}

    class _Client:
        def version(self):
            return NS(version="dev-test")

        def ingest_document(self, source, filename, content_type, idempotency_key=None):
            with lock:
                state["keys"].append(idempotency_key)
                state["in_flight"] += 1
                state["peak"] = max(state["peak"], state["in_flight"])
            time.sleep(0.05)
            with lock:
                state["in_flight"] -= 1
            return NS(document_id="srv_" + filename)

        def wait_for_ready(self, doc_id, timeout):
            old = doc_id.endswith("d0.md")  # one document an earlier run ingested
            return NS(created_at="2020-01-01T00:00:00Z" if old else "2999-01-01T00:00:00Z")

        def get_document_tree(self, doc_id):
            return NS(sections=[])

    r = VectorlessRetriever.__new__(VectorlessRetriever)
    r._client, r.ingest_concurrency, r.reuse_documents, r.ingest_timeout = _Client(), 4, True, 5
    r._doc_ids, r._paths, r.setup_usage = {}, {}, None
    r.setup([Doc(doc_id=f"d{i}", title="t", content=f"body {i}") for i in range(8)])
    assert state["peak"] > 1, "ingest must run concurrently"
    assert len(r._doc_ids) == 8 and r._doc_ids["d3"] == "srv_d3.md"
    assert all(k and k.startswith("vlbench-") for k in state["keys"]) and len(set(state["keys"])) == 8
    assert r.setup_meta["reused_documents"] == 1 and r.setup_meta["ingest_concurrency"] == 4
