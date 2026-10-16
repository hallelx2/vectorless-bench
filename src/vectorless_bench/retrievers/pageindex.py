"""PageIndex — VectifyAI's open-source SDK, running its OWN indexer and its
OWN chat agent (HAL-2477), on Gemini through Vertex via LiteLLM.

What runs is theirs, end to end: `PageIndexClient.submit_document` builds the
tree with their prompts, and `chat(..., stream=True)` runs their agent, which
calls their tools (`get_document_structure`, then `get_page_content` and the
like) until it is ready to answer. Earlier versions of this adapter drove our
own selector over their tree, which measured our prompt, not PageIndex.

Measurement boundary, the same as every system's: retrieval ends when the
agent stops calling tools and starts answering (its first answer token), so
the comparison times finding the evidence, not writing prose. What the agent
read through its content tools, in reading order, is its retrieval.

Usage: every LiteLLM call PageIndex makes, indexing and chat, is captured by
a callback and priced from the shared price book, so PageIndex's ingest cost
(missing before) and query cost are both on the bill.

Model: the run's shared model as a LiteLLM Vertex route
("vertex_ai/<model>"), Application Default Credentials, global location.
Indexing concurrency is held at 4: at higher fan-out Vertex's shared pool
refused 2.5 Flash-Lite outright (HAL-3048).
"""

from __future__ import annotations

import contextvars
import dataclasses
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._llm import DEFAULT_MODEL, project, require_vertex_model
from ..datasets.base import CORPUS
from ..pricing import compute
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

_CONTENT_TOOL = re.compile(r"content|page|read|node|section", re.I)


# The Usage a PageIndex operation's model calls are charged to. It is a
# context variable, so concurrent questions (the throughput phase) each see
# their own, and LiteLLM's pre-call hook, which runs on the calling thread,
# reads it to bind each call to its owner.
_ACTIVE: "contextvars.ContextVar[Optional[Usage]]" = contextvars.ContextVar("vlbench_pageindex_usage", default=None)


class _UsageSink:
    """Charges every LiteLLM call PageIndex makes to the operation that made it.

    LiteLLM runs synchronous success callbacks on a background thread, so a
    call's usage can arrive after the operation returned, and on a thread
    that cannot see the caller's context. Each call is therefore bound to
    its owner when it starts (log_pre_api_call, inline on the caller's
    thread, keyed by litellm_call_id), and an operation waits for its own
    outstanding calls to report before it closes its books (drain)."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.cond = threading.Condition(self.lock)
        self.calls: Dict[str, Usage] = {}
        self.pending: Dict[int, int] = {}
        self.seconds: Dict[int, float] = {}
        self.orphans = 0
        self.open: Dict[int, Usage] = {}  # operations in flight

    def enter(self, target: Usage) -> "contextvars.Token":
        with self.lock:
            self.open[id(target)] = target
        return _ACTIVE.set(target)

    def exit(self, target: Usage, token: "contextvars.Token") -> bool:
        done = self.drain(target)
        with self.lock:
            self.open.pop(id(target), None)
        _ACTIVE.reset(token)
        return done

    def _owner(self) -> Optional[Usage]:
        # The caller's context when it reaches here; otherwise (PageIndex's
        # agent loop on a thread of its own) the one operation in flight.
        # With several in flight and no context, nobody is guessed at.
        target = _ACTIVE.get()
        if target is None and len(self.open) == 1:
            target = next(iter(self.open.values()))
        return target

    def begin(self, call_id: Optional[str]) -> None:
        with self.lock:
            target = self._owner()
            if not call_id or target is None:
                return
            self.calls[call_id] = target
            self.pending[id(target)] = self.pending.get(id(target), 0) + 1

    def _finish(self, call_id: Optional[str]) -> Optional[Usage]:
        target = self.calls.pop(call_id, None) if call_id else None
        if target is not None:
            self.pending[id(target)] = max(0, self.pending.get(id(target), 0) - 1)
            self.cond.notify_all()
        return target

    def add(self, model: str, prompt: int, completion: int, cached: int, seconds: float = 0.0,
            call_id: Optional[str] = None) -> None:
        with self.lock:
            target = self._finish(call_id) or self._owner()
            if target is None:
                self.orphans += 1  # a call nobody owns: counted, never charged to a stranger
                return
            m = model.split("/", 1)[-1]
            target.input_tokens += prompt
            target.output_tokens += completion
            target.total_tokens += prompt + completion
            target.cost_usd += compute(m, max(0, prompt - cached), completion, cached_tokens=cached)
            target.llm_calls += 1
            self.seconds[id(target)] = self.seconds.get(id(target), 0.0) + max(0.0, seconds)

    def failed(self, call_id: Optional[str]) -> None:
        with self.lock:
            self._finish(call_id)

    def drain(self, target: Usage, timeout: float = 15.0) -> bool:
        """Wait until every call bound to target has reported."""
        deadline = time.monotonic() + timeout
        with self.lock:
            while self.pending.get(id(target), 0) > 0:
                left = deadline - time.monotonic()
                if left <= 0:
                    return False
                self.cond.wait(left)
            self.pending.pop(id(target), None)
            return True

    def call_seconds(self, target: Usage) -> float:
        with self.lock:
            return self.seconds.pop(id(target), 0.0)


_SINK = _UsageSink()
_INSTALLED = False


def _install_callback() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import litellm  # type: ignore
    from litellm.integrations.custom_logger import CustomLogger  # type: ignore

    class _Logger(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            _SINK.begin((kwargs or {}).get("litellm_call_id"))

        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            _SINK.failed((kwargs or {}).get("litellm_call_id"))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            self.log_failure_event(kwargs, response_obj, start_time, end_time)

        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            u = getattr(response_obj, "usage", None)
            if u is None:
                _SINK.failed((kwargs or {}).get("litellm_call_id"))
                return
            details = getattr(u, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
            try:
                seconds = (end_time - start_time).total_seconds()
            except Exception:
                seconds = 0.0
            _SINK.add(str(kwargs.get("model", "")), int(getattr(u, "prompt_tokens", 0) or 0),
                      int(getattr(u, "completion_tokens", 0) or 0), cached, seconds,
                      call_id=kwargs.get("litellm_call_id"))

        async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
            self.log_success_event(kwargs, response_obj, start_time, end_time)

    litellm.callbacks = list(getattr(litellm, "callbacks", []) or []) + [_Logger()]
    litellm.num_retries = max(int(getattr(litellm, "num_retries", 0) or 0), 6)
    _INSTALLED = True


_PAGE_RANGE = re.compile(r"(\d+)\s*(?:-|to|–)\s*(\d+)")


def _pages_from(args: Any) -> List[int]:
    """The pages a content-tool call asked for, from its arguments: a page
    number, a list of them, a "5-7" range, or start/end page fields."""
    out: List[int] = []

    def take(v: Any) -> None:
        if isinstance(v, bool):
            return
        if isinstance(v, int):
            out.append(v)
        elif isinstance(v, str):
            for a, b in _PAGE_RANGE.findall(v):
                lo, hi = int(a), int(b)
                if 0 < hi - lo < 500:
                    out.extend(range(lo, hi + 1))
            if not _PAGE_RANGE.search(v):
                out.extend(int(x) for x in re.findall(r"\d+", v))
        elif isinstance(v, (list, tuple)):
            for x in v:
                take(x)

    if isinstance(args, dict):
        start, end = args.get("start_page"), args.get("end_page")
        if isinstance(start, int) and isinstance(end, int) and 0 <= end - start < 500:
            out.extend(range(start, end + 1))
        for key, v in args.items():
            if "page" in key.lower() and key not in ("start_page", "end_page"):
                take(v)
    else:
        take(args)
    return sorted(dict.fromkeys(p for p in out if p > 0))


def _rate_limited(e: Exception) -> bool:
    """A refusal for load, decided by type or status code; the text match is
    only the fallback for an error that carries neither."""
    try:
        import litellm  # type: ignore

        if isinstance(e, getattr(litellm, "RateLimitError", ())):
            return True
    except ImportError:  # pragma: no cover
        pass
    code = getattr(e, "status_code", None) or getattr(e, "code", None)
    if isinstance(code, int):
        return code in (429, 503)
    return "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e).split()


def _output_text(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        return "\n".join(_output_text(x) for x in out)
    if isinstance(out, dict):
        for k in ("text", "content", "output"):
            if k in out:
                return _output_text(out[k])
        return str(out)
    return str(out)


class PageIndexRetriever:
    name = "pageindex"

    def __init__(self, model: str = DEFAULT_MODEL, retrieve_model: Optional[str] = None,
                 storage_path: Optional[str] = None, summary_concurrency: int = 2,
                 **_: object) -> None:
        require_vertex_model(model)
        self.model = model
        self.retrieve_model = retrieve_model or model
        require_vertex_model(self.retrieve_model)
        self.storage = storage_path or tempfile.mkdtemp(prefix="vlbench-pageindex-")
        self.summary_concurrency = summary_concurrency
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._client = None
        self._doc_ids: Dict[str, str] = {}
        self.setup_meta: Dict[str, Any] = {}

    def _route(self, m: str) -> str:
        return f"vertex_ai/{m}"

    def setup(self, corpus: List[Doc]) -> None:
        os.environ.setdefault("VERTEXAI_PROJECT", project())
        os.environ.setdefault("VERTEXAI_LOCATION", os.environ.get("GOOGLE_CLOUD_LOCATION", "global"))
        _install_callback()
        from pageindex import PageIndexClient  # type: ignore
        import importlib.metadata as md

        self._client = PageIndexClient(
            index={"model": self._route(self.model), "storage_path": self.storage,
                   "summary_concurrency": self.summary_concurrency},
            chat={"model": self._route(self.retrieve_model)},
        )
        self.setup_meta = {"pageindex_version": md.version("pageindex"), "failed": []}
        t0 = time.perf_counter()
        token = _SINK.enter(self.setup_usage)
        try:
            for d in corpus:
                path = self._source_path(d)
                # Vertex shared capacity refuses bursts at times (429); a
                # document is retried after a pause before it counts as failed.
                last = None
                for attempt in range(4):
                    try:
                        res = self._client.submit_document(path)
                        self._doc_ids[d.doc_id] = res["doc_id"]
                        last = None
                        break
                    except Exception as e:
                        last = e
                        if not _rate_limited(e) or attempt == 3:
                            break  # not retryable, or the last attempt: no dead time on the clock
                        time.sleep(60 * (attempt + 1))
                if last is not None:  # one bad document must not void the run
                    self.setup_meta["failed"].append({"doc": d.doc_id, "error": str(last)[:300]})
        finally:
            drained = _SINK.exit(self.setup_usage, token)
        self.setup_seconds = time.perf_counter() - t0
        self.setup_meta["usage_complete"] = drained
        # Indexing is serial per document, so the model-call seconds are the
        # LLM phase and the rest is PageIndex's local parse and bookkeeping
        # (summary calls overlap at summary_concurrency, so the LLM figure is
        # call time, and can exceed its share of wall time).
        llm_seconds = _SINK.call_seconds(self.setup_usage)
        self.setup_meta["llm_call_seconds"] = round(llm_seconds, 3)
        self.setup_meta["local_seconds"] = round(max(0.0, self.setup_seconds - llm_seconds), 3)

    def _source_path(self, d: Doc) -> str:
        """PageIndex's own parser reads the original: a PDF as is, anything
        else as Markdown (its other supported input)."""
        if d.path and d.path.lower().endswith((".pdf", ".md")):
            return d.path
        p = Path(self.storage) / "src" / f"{re.sub(r'[^A-Za-z0-9_.-]+', '_', d.doc_id)[:150]}.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(d.content, encoding="utf-8")
        return str(p)

    def teardown(self) -> None:
        pass

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        if question.doc_id == CORPUS:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, cold=cold,
                                   error="n/a: PageIndex's agent reads one indexed document; this question spans the corpus")
        pid = self._doc_ids.get(question.doc_id)
        if pid is None:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, cold=cold,
                                   error=f"document {question.doc_id!r} was not indexed by PageIndex")
        usage = Usage()
        sections: List[RetrievedSection] = []
        tools: List[str] = []
        args: Dict[str, Any] = {}
        pages_read: List[int] = []
        retrieval_ms: Optional[float] = None
        retrieval_usage: Optional[Usage] = None
        answer_chars = 0
        t0 = time.perf_counter()
        token = _SINK.enter(usage)
        try:
            stream = self._client.chat(question.question, doc_id=pid, stream=True)
            for ev in stream.events:
                kind = ev.get("type")
                if kind == "tool_call":
                    name = str(ev.get("name"))
                    tools.append(name)
                    args[str(ev.get("call_id") or len(tools))] = ev.get("arguments")
                    if name != "get_document_structure" and _CONTENT_TOOL.search(name):
                        pages_read.extend(p for p in _pages_from(ev.get("arguments")) if p not in pages_read)
                elif kind == "tool_result":
                    name = str(ev.get("name") or "")
                    if name != "get_document_structure" and _CONTENT_TOOL.search(name):
                        text = _output_text(ev.get("output"))
                        if text.strip():
                            call_pages = _pages_from(args.get(str(ev.get("call_id")), None))
                            sections.append(RetrievedSection(
                                content=text, section_id=f"pi:{len(sections)}", title=name,
                                page=call_pages[0] if call_pages else None, doc_id=question.doc_id))
                elif kind == "answer":
                    if retrieval_ms is None:
                        # Retrieval ends where the agent starts answering: the
                        # same boundary as every system. Its answer is PageIndex's
                        # own and is not scored, so its cost is kept apart.
                        retrieval_ms = (time.perf_counter() - t0) * 1000.0
                        with _SINK.lock:
                            retrieval_usage = dataclasses.replace(usage)
                    answer_chars += len(str(ev.get("delta") or ""))
        except Exception as e:  # pragma: no cover - network
            _SINK.exit(usage, token)
            _SINK.call_seconds(usage)
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, usage=usage,
                                   error=str(e)[:500], cold=cold, trace={"tools": tools})
        total_ms = (time.perf_counter() - t0) * 1000.0
        usage_complete = _SINK.exit(usage, token)
        _SINK.call_seconds(usage)
        if retrieval_ms is None:  # the agent never answered: all of it was retrieval
            retrieval_ms, retrieval_usage = total_ms, usage
        return RetrievalResult(
            qid=question.qid, system=self.name, query=question.question,
            sections=sections[:k], usage=retrieval_usage, latency_ms=retrieval_ms,
            strategy="pageindex-agent", cold=cold,
            trace={"tools": tools, "pages_read": pages_read, "retrieval_ms": round(retrieval_ms, 1),
                   "total_ms": round(total_ms, 1), "llm_calls": usage.llm_calls,
                   "retrieval_llm_calls": retrieval_usage.llm_calls,
                   "answer_cost_usd": round(usage.cost_usd - retrieval_usage.cost_usd, 8),
                   "answer_chars": answer_chars, "usage_complete": usage_complete},
        )
