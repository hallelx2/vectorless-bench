"""Unit tests for the vectorless_treewalk retriever's response mapping.

These don't touch the network: we stub the retriever's httpx client with a
fake that returns a canned /v1/answer/treewalk payload, then assert that
citations are projected onto the bench's RetrievedSection shape correctly
(content from quote, page from start_page, path from the section-id index)
and that usage is copied through faithfully.

Constructing the retriever needs the `vectorless` SDK (the parent's __init__
builds a client), so the whole module is skipped if it isn't installed.
"""

from __future__ import annotations

import pytest

pytest.importorskip("vectorless")

from vectorless_bench.retrievers.vectorless_treewalk import (  # noqa: E402
    VectorlessTreewalkRetriever,
)
from vectorless_bench.schema import AnswerType, Question  # noqa: E402


class _FakeResponse:
    def __init__(self, status_code: int, payload: dict):
        self.status_code = status_code
        self._payload = payload
        self.text = "fake"

    def json(self) -> dict:
        return self._payload


class _FakeHTTP:
    """Minimal httpx.Client stand-in: returns a pre-set response and records
    the last call so tests can assert on the request body / headers / url."""

    def __init__(self, response: _FakeResponse):
        self._response = response
        self.calls: list[dict] = []

    def post(self, url, json=None, headers=None):  # noqa: A002 - mirror httpx
        self.calls.append({"url": url, "json": json, "headers": headers})
        return self._response

    def close(self):
        pass


def _build(monkeypatch) -> VectorlessTreewalkRetriever:
    # Avoid hitting a real server during construction: the parent __init__
    # builds a VectorlessClient but never calls it until setup(), so a dummy
    # base_url is fine here.
    r = VectorlessTreewalkRetriever(
        api_key="k",
        base_url="http://localhost:9",
        model="gemini-2.5-flash",
        org="vlbench",
        max_hops=6,
        page_content_limit=16000,
    )
    return r


def _payload() -> dict:
    return {
        "document_id": "doc_x",
        "query": "What was net revenue?",
        "answer": "Net revenue was $1,234M.",
        "citations": [
            {
                "start_page": 42,
                "end_page": 42,
                "section_ids": ["sec_a", "sec_b"],
                "quote": "Net revenue was $1,234M for the period.",
                "quote_start": 0,
                "quote_end": 38,
            },
            {
                "start_page": 7,
                "end_page": 8,
                "section_ids": ["sec_c"],
                "quote": "",  # extractor declined a quote — graceful empty
            },
        ],
        "strategy": "pageindex",
        "model": "gemini-2.5-flash",
        "hops_taken": 3,
        "usage": {
            "input_tokens": 1000,
            "output_tokens": 200,
            "total_tokens": 1200,
            "cost_usd": 0.00027,
            "llm_calls": 5,
        },
        "elapsed_ms": 4200,
        "trace_token": "tok_abc",
        "pages_read": [{"start_page": 42, "end_page": 42, "section_ids": ["sec_a"]}],
    }


def test_citations_map_to_sections(monkeypatch):
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, _payload()))
    r._doc_ids["AMZN"] = "doc_x"
    # path index: sec_a has a heading path, sec_c doesn't
    r._paths["AMZN"] = {"sec_a": ["Part II", "Item 8", "Balance Sheet"]}

    q = Question(qid="q1", doc_id="AMZN", question="What was net revenue?",
                 answer_type=AnswerType.NUMERIC)
    res = r.retrieve(q, k=5, cold=True)

    assert res.error is None
    assert res.strategy == "pageindex"
    assert res.latency_ms == 4200.0  # taken from the engine's elapsed_ms
    assert len(res.sections) == 2

    s0 = res.sections[0]
    assert s0.section_id == "sec_a"                  # first section_id anchors
    assert s0.page == 42                              # tagged with start_page
    assert s0.content.startswith("Net revenue was")  # quote becomes content
    assert s0.title_path == ["Part II", "Item 8", "Balance Sheet"]

    s1 = res.sections[1]
    assert s1.section_id == "sec_c"
    assert s1.page == 7
    assert s1.content == ""           # empty quote is preserved, not crashed
    assert s1.title_path == []        # unknown sec id -> no path

    # usage copied straight from the engine's accounting
    assert res.usage.input_tokens == 1000
    assert res.usage.output_tokens == 200
    assert res.usage.cost_usd == pytest.approx(0.00027)
    assert res.usage.llm_calls == 5


def test_engine_title_path_preferred_over_local_index(monkeypatch):
    # When the engine emits a citation `title_path` (HAL-70) it wins over the
    # locally reconstructed path — that's the whole point of the engine now
    # owning the structural path instead of the consumer guessing it.
    p = _payload()
    p["citations"] = [
        {
            "start_page": 42,
            "end_page": 42,
            "section_ids": ["sec_a"],
            "title_path": ["Part II", "Item 8", "Statements of Operations"],
            "quote": "Net revenue was $1,234M.",
        },
    ]
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, p))
    r._doc_ids["AMZN"] = "doc_x"
    # Local index deliberately disagrees — engine path must take precedence.
    r._paths["AMZN"] = {"sec_a": ["WRONG", "LOCAL", "PATH"]}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    res = r.retrieve(q, k=5, cold=True)
    assert res.sections[0].title_path == ["Part II", "Item 8", "Statements of Operations"]


def test_engine_title_path_absent_falls_back_to_local(monkeypatch):
    # No engine-supplied title_path → fall back to the local reconstructed
    # index (older server, or a document with no persisted TOC).
    p = _payload()
    p["citations"] = [
        {"start_page": 42, "end_page": 42, "section_ids": ["sec_a"], "quote": "x"},
    ]
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, p))
    r._doc_ids["AMZN"] = "doc_x"
    r._paths["AMZN"] = {"sec_a": ["Part II", "Item 8", "Balance Sheet"]}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    res = r.retrieve(q, k=5, cold=True)
    assert res.sections[0].title_path == ["Part II", "Item 8", "Balance Sheet"]


def test_request_body_carries_knobs(monkeypatch):
    r = _build(monkeypatch)
    fake = _FakeHTTP(_FakeResponse(200, _payload()))
    r._http = fake
    r._doc_ids["AMZN"] = "doc_x"
    r._paths["AMZN"] = {}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    r.retrieve(q, k=5, cold=True)

    sent = fake.calls[-1]
    assert sent["url"].endswith("/v1/answer/treewalk")
    assert sent["json"]["document_id"] == "doc_x"
    assert sent["json"]["model"] == "gemini-2.5-flash"
    assert sent["json"]["max_hops"] == 6
    assert sent["json"]["max_pages_per_fetch"] == 16000
    assert sent["headers"]["X-Vectorless-Org"] == "vlbench"
    assert sent["headers"]["Authorization"] == "Bearer k"


def test_k_caps_returned_sections(monkeypatch):
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, _payload()))
    r._doc_ids["AMZN"] = "doc_x"
    r._paths["AMZN"] = {}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    res = r.retrieve(q, k=1, cold=True)
    assert len(res.sections) == 1  # two citations available, k=1 caps to one


def _payload_with_duplicate_section() -> dict:
    """A response shaped like the FinanceBench 'sec_363... ×5' miss: several
    citations whose distinct page ranges all anchor to the SAME first
    section_id. Without dedup these became 5 identical RetrievedSections and
    deflated precision@5."""
    p = _payload()
    p["citations"] = [
        {"start_page": 100, "end_page": 101, "section_ids": ["sec_363"], "quote": "a"},
        {"start_page": 102, "end_page": 103, "section_ids": ["sec_363"], "quote": "b"},
        {"start_page": 104, "end_page": 105, "section_ids": ["sec_363"], "quote": "c"},
        {"start_page": 106, "end_page": 107, "section_ids": ["sec_363"], "quote": "d"},
        {"start_page": 108, "end_page": 109, "section_ids": ["sec_363"], "quote": "e"},
        {"start_page": 5, "end_page": 6, "section_ids": ["sec_50"], "quote": "f"},
    ]
    return p


def test_duplicate_section_ids_are_deduped(monkeypatch):
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, _payload_with_duplicate_section()))
    r._doc_ids["AMZN"] = "doc_x"
    r._paths["AMZN"] = {}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    res = r.retrieve(q, k=5, cold=True)

    # sec_363 cited five times collapses to ONE section; sec_50 is the other.
    ids = [s.section_id for s in res.sections]
    assert ids == ["sec_363", "sec_50"], ids
    # No section id repeats.
    assert len(ids) == len(set(ids))
    # The kept sec_363 row is the FIRST occurrence (page 100), not a later one.
    assert res.sections[0].page == 100
    assert res.sections[0].content == "a"


def test_dedup_runs_before_k_cap(monkeypatch):
    # Five duplicate sec_363 citations + one distinct sec_50. With k=2 the
    # duplicates must not crowd out sec_50: dedup first, THEN cap.
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, _payload_with_duplicate_section()))
    r._doc_ids["AMZN"] = "doc_x"
    r._paths["AMZN"] = {}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    res = r.retrieve(q, k=2, cold=True)
    ids = [s.section_id for s in res.sections]
    assert ids == ["sec_363", "sec_50"], ids


def test_empty_section_ids_not_collapsed(monkeypatch):
    # Citations with no section_id keep their own row — they still carry a
    # distinct page anchor — so they are NOT deduped together.
    p = _payload()
    p["citations"] = [
        {"start_page": 10, "end_page": 11, "section_ids": [], "quote": "x"},
        {"start_page": 20, "end_page": 21, "section_ids": [], "quote": "y"},
    ]
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, p))
    r._doc_ids["AMZN"] = "doc_x"
    r._paths["AMZN"] = {}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    res = r.retrieve(q, k=5, cold=True)
    assert len(res.sections) == 2
    assert [s.page for s in res.sections] == [10, 20]


def test_http_error_is_recorded_not_raised(monkeypatch):
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(404, {}))
    r._doc_ids["AMZN"] = "doc_x"
    r._paths["AMZN"] = {}

    q = Question(qid="q1", doc_id="AMZN", question="q?")
    res = r.retrieve(q, k=5, cold=True)
    assert res.error is not None
    assert "404" in res.error
    assert res.sections == []


def test_missing_document_short_circuits(monkeypatch):
    r = _build(monkeypatch)
    r._http = _FakeHTTP(_FakeResponse(200, _payload()))
    # no doc ingested for this logical id
    q = Question(qid="q1", doc_id="UNKNOWN", question="q?")
    res = r.retrieve(q, k=5, cold=True)
    assert res.error is not None
    assert "was not ingested" in res.error
