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


class _Out:
    def __init__(self, text):
        from vectorless_bench.schema import Usage
        self.text, self.usage = text, Usage(input_tokens=10, output_tokens=2, total_tokens=12, cost_usd=1e-6, llm_calls=1)


def test_memwalker_walks_to_the_answer(monkeypatch):
    import json as _j
    from vectorless_bench.retrievers import memwalker as mw

    def fake(model, system, user, **kw):
        if system == mw._SUM:
            return _Out("capex section" if ("capital expenditure" in user or "capex" in user) else "other material")
        if system == mw._NAV:
            lines = [l for l in user.splitlines() if l[:1].isdigit()]
            pick = next((int(l.split(".")[0]) for l in lines if "capex" in l), 1)
            return _Out(_j.dumps({"action": "open", "child": pick}))
        return _Out(_j.dumps({"action": "accept" if "capital expenditure" in user else "back"}))

    monkeypatch.setattr(mw, "complete", fake)
    body = ("filler words " * 400 + "\n") * 8 + "capital expenditure was 1,577 million.\n" + ("tail words " * 400 + "\n") * 8
    r = mw.MemWalkerRetriever(segment_tokens=300, fanout=3)
    r.setup([Doc(doc_id="d", title="d", content=body)])
    out = r.retrieve(Question(qid="q", doc_id="d", question="capex?"), k=3)
    assert out.trace["accepted"] and "1,577" in out.sections[0].content
    assert r.setup_usage.llm_calls > 0, "the tree build is costed as setup"


def test_readagent_rereads_the_chosen_page(monkeypatch):
    import json as _j
    from vectorless_bench.retrievers import readagent as ra

    def fake(model, system, user, **kw):
        if system == ra._PAGINATE:
            marks = [int(t.strip("<>")) for t in user.split() if t.startswith("<") and t.endswith(">") and t.strip("<>").isdigit()]
            return _Out(_j.dumps({"break": marks[0]}))
        if system == ra._GIST:
            return _Out("about capex" if "capital expenditure" in user else "about other things")
        gists = [l for l in user.splitlines() if l.startswith("<Page")]
        pages = [int(l.split()[1].rstrip(">")) for l in gists if "capex" in l]
        return _Out(_j.dumps({"pages": pages}))

    monkeypatch.setattr(ra, "complete", fake)
    paras = ["word " * 150 for _ in range(12)]
    paras[7] = "capital expenditure was 1,577 million. " + "word " * 140
    r = ra.ReadAgentRetriever(min_words=280, max_words=600)
    r.setup([Doc(doc_id="d", title="d", content="\n\n".join(paras))])
    out = r.retrieve(Question(qid="q", doc_id="d", question="capex?"), k=5)
    assert out.sections and "1,577" in out.sections[0].content
    assert len(r._pages["d"]) > 1


def test_jev_grader_uses_a_closed_choice(monkeypatch):
    import io, json as _j
    from vectorless_bench import jev as jevmod
    from vectorless_bench import judge as jmod

    seen = {}

    class _Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): pass

    def fake_urlopen(req, timeout=0):
        seen["body"] = _j.loads(req.data)
        return _Resp(_j.dumps({"model": "jev-1.13.0", "answers": {"verdict": {"type": "choice", "choice": "correct",
                     "probabilities": {"correct": 0.9}, "confidence": 0.9}}, "usage": {"input_tokens": 200, "output_tokens": 0}}).encode())

    monkeypatch.setenv("TYPESAFE_API_KEY", "test")
    monkeypatch.setattr(jevmod.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(jmod, "complete", lambda *a, **k: _Out("$1,577 million"))
    from vectorless_bench.schema import RetrievedSection
    j = jmod.Judge("gemini-3.5-flash-lite", "jev-latest")
    res = j.evaluate(Question(qid="q", doc_id="d", question="capex?", answer="$1577.00"), [RetrievedSection(content="x")])
    q = seen["body"]["questions"]["verdict"]
    assert q["type"] == "choice" and set(q["criteria"]) == {"correct", "partially_correct", "incorrect", "abstained"}
    assert res.correct == 1.0 and res.verdict == "correct" and res.usage.cost_usd > 0


def test_vector_memory_backend_scopes_and_spans_corpus(monkeypatch):
    from vectorless_bench.retrievers.vector_rag import VectorRagRetriever
    from vectorless_bench.datasets.base import CORPUS

    vocab = ["capex", "revenue", "churn"]

    def fake_embed(self, texts):
        return [[float(w in t.lower()) for w in vocab] for t in texts]

    monkeypatch.setattr(VectorRagRetriever, "_embed", fake_embed)
    monkeypatch.chdir(__import__("tempfile").mkdtemp())
    r = VectorRagRetriever(embedding_model="BAAI/bge-small-en-v1.5", backend="memory", chunk_tokens=20, overlap_tokens=0)
    r.setup([Doc(doc_id="a", title="a", content="capex was high"), Doc(doc_id="b", title="b", content="revenue grew; capex fell")])
    assert r.dim == 3
    scoped = r.retrieve(Question(qid="q", doc_id="b", question="capex"), k=2)
    assert scoped.sections and all("revenue" in s.content for s in scoped.sections)  # only document b
    every = r.retrieve(Question(qid="q2", doc_id=CORPUS, question="capex"), k=5)
    assert len(every.sections) == 2
    assert r._query_text("x").startswith("Represent this sentence")


def test_memwalker_root_never_reverts(monkeypatch):
    import json as _j
    from vectorless_bench.retrievers import memwalker as mw

    def fake(model, system, user, **kw):
        if system == mw._SUM:
            return _Out("summary")
        if system == mw._NAV_ROOT:
            return _Out(_j.dumps({"action": "back"}))  # a model that refuses at the root
        if system == mw._NAV:
            return _Out(_j.dumps({"action": "open", "child": "1"}))  # a string number still opens
        return _Out(_j.dumps({"action": "accept"}))

    monkeypatch.setattr(mw, "complete", fake)
    r = mw.MemWalkerRetriever(segment_tokens=300, fanout=3)
    r.setup([Doc(doc_id="d", title="d", content=("words " * 400 + "\n") * 12)])
    out = r.retrieve(Question(qid="q", doc_id="d", question="?"), k=3)
    assert out.sections and out.trace["root_choices_forced"] >= 1 and out.trace["accepted"]


def test_embedding_cache_is_per_document_and_keeps_measured_seconds(monkeypatch, tmp_path):
    from vectorless_bench.retrievers.vector_rag import VectorRagRetriever

    calls = []

    def fake_embed(self, texts):
        calls.append(len(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(VectorRagRetriever, "_embed", fake_embed)
    monkeypatch.setenv("VLBENCH_EMBED_CACHE", str(tmp_path))
    docs = [Doc(doc_id="a", title="a", content="alpha " * 50), Doc(doc_id="b", title="b", content="beta " * 50)]
    r1 = VectorRagRetriever(embedding_model="gemini-embedding-2", backend="memory", chunk_tokens=20, overlap_tokens=0)
    r1.setup(docs)
    first = len(calls)
    r2 = VectorRagRetriever(embedding_model="gemini-embedding-2", backend="memory", chunk_tokens=20, overlap_tokens=0)
    r2.setup(docs[:1])  # a subset hits the same per-document cache
    assert len(calls) == first and r2.setup_meta["embedding_cache_hits"] == 1
    assert r2.setup_usage.cost_usd > 0, "a cache hit still prices the ingest"
    assert "embedding_seconds_measured" in r2.setup_meta


def test_pageindex_pages_from_tool_arguments():
    from vectorless_bench.retrievers.pageindex import _pages_from

    assert _pages_from({"pages": "5-7"}) == [5, 6, 7]
    assert _pages_from({"page": 12}) == [12]
    assert _pages_from({"start_page": 3, "end_page": 4}) == [3, 4]
    assert _pages_from({"page_numbers": [9, 2, 9]}) == [2, 9]
    assert _pages_from({"node_id": "0007"}) == []


def test_pageindex_times_retrieval_to_the_first_answer_token(monkeypatch):
    import time as _t
    from vectorless_bench.retrievers import pageindex as pi

    events = [
        {"type": "tool_call", "call_id": "1", "name": "get_document_structure", "arguments": {}},
        {"type": "tool_result", "call_id": "1", "name": "get_document_structure", "output": "tree"},
        {"type": "tool_call", "call_id": "2", "name": "get_page_content", "arguments": {"pages": "40-41"}},
        {"type": "tool_result", "call_id": "2", "name": "get_page_content", "output": "capex was 1,577"},
        {"type": "answer", "delta": "Capex was "},
        {"type": "answer", "delta": "$1,577 million."},
    ]

    class _Stream:
        @property
        def events(self):
            for i, e in enumerate(events):
                if e["type"] == "tool_call":
                    pi._SINK.add("vertex_ai/gemini-3.5-flash-lite", 100, 10, 0, 0.1)
                if i == 5:
                    pi._SINK.add("vertex_ai/gemini-3.5-flash-lite", 50, 20, 0, 0.1)  # the answer's own call
                    _t.sleep(0.02)
                yield e

    class _Client:
        def chat(self, q, doc_id, stream):
            return _Stream()

    r = pi.PageIndexRetriever.__new__(pi.PageIndexRetriever)
    r._client, r._doc_ids = _Client(), {"d": "pi-1"}
    res = r.retrieve(Question(qid="q", doc_id="d", question="capex?"), k=5)
    assert res.error is None and len(res.sections) == 1 and res.sections[0].page == 40
    assert res.trace["pages_read"] == [40, 41]
    assert res.trace["total_ms"] >= res.trace["retrieval_ms"] and abs(res.latency_ms - res.trace["retrieval_ms"]) < 0.1
    assert res.usage.llm_calls == 2 and res.trace["llm_calls"] == 3, "the answer call is kept apart"
    assert res.trace["answer_cost_usd"] > 0


def test_pageindex_usage_is_charged_per_call_even_late_and_concurrent():
    import threading
    import time as _t
    from vectorless_bench.retrievers import pageindex as pi
    from vectorless_bench.schema import Usage

    sink = pi._UsageSink()
    a, b = Usage(), Usage()

    def operation(target, call_id, delay):
        token = sink.enter(target)
        sink.begin(call_id)  # the pre-call hook, on the caller's thread

        def late():  # LiteLLM's success callback, on a background thread
            _t.sleep(delay)
            sink.add("vertex_ai/gemini-3.5-flash-lite", 100, 10, 0, 0.5, call_id=call_id)

        threading.Thread(target=late).start()
        assert sink.exit(target, token), "exit waits for the call to report"

    ta = threading.Thread(target=operation, args=(a, "call-a", 0.05))
    tb = threading.Thread(target=operation, args=(b, "call-b", 0.01))
    ta.start(); tb.start(); ta.join(); tb.join()
    assert a.llm_calls == 1 and b.llm_calls == 1, (a, b)
    assert sink.orphans == 0


def test_pageindex_does_not_sleep_after_the_last_attempt(monkeypatch):
    from vectorless_bench.retrievers import pageindex as pi

    assert pi._rate_limited(type("E", (Exception,), {"status_code": 429})())
    assert not pi._rate_limited(type("E", (Exception,), {"status_code": 400})("tokens 429 in text"))
    assert pi._rate_limited(Exception("RESOURCE_EXHAUSTED: quota"))


def test_pageindex_reports_corpus_questions_as_not_applicable():
    from vectorless_bench.datasets.base import CORPUS
    from vectorless_bench.retrievers import pageindex as pi

    r = pi.PageIndexRetriever.__new__(pi.PageIndexRetriever)
    r._doc_ids = {}
    res = r.retrieve(Question(qid="q", doc_id=CORPUS, question="?"), k=5)
    assert res.error.startswith("n/a:")
