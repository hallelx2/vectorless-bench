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

import dataclasses
import os
import re
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .._llm import DEFAULT_MODEL, project, require_vertex_model
from ..pricing import compute
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

_CONTENT_TOOL = re.compile(r"content|page|read|node|section", re.I)


class _UsageSink:
    """Collects LiteLLM usage from whichever thread made the call."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.active: Optional[Usage] = None
        # Seconds spent inside model calls while active: for ingest, the LLM
        # phase, as against PageIndex's own local parse.
        self.call_seconds = 0.0

    def add(self, model: str, prompt: int, completion: int, cached: int, seconds: float = 0.0) -> None:
        with self.lock:
            if self.active is None:
                return
            self.call_seconds += max(0.0, seconds)
            m = model.split("/", 1)[-1]
            self.active.input_tokens += prompt
            self.active.output_tokens += completion
            self.active.total_tokens += prompt + completion
            self.active.cost_usd += compute(m, max(0, prompt - cached), completion, cached_tokens=cached)
            self.active.llm_calls += 1


_SINK = _UsageSink()
_INSTALLED = False


def _install_callback() -> None:
    global _INSTALLED
    if _INSTALLED:
        return
    import litellm  # type: ignore
    from litellm.integrations.custom_logger import CustomLogger  # type: ignore

    class _Logger(CustomLogger):
        def log_success_event(self, kwargs, response_obj, start_time, end_time):
            u = getattr(response_obj, "usage", None)
            if u is None:
                return
            details = getattr(u, "prompt_tokens_details", None)
            cached = int(getattr(details, "cached_tokens", 0) or 0) if details else 0
            try:
                seconds = (end_time - start_time).total_seconds()
            except Exception:
                seconds = 0.0
            _SINK.add(str(kwargs.get("model", "")), int(getattr(u, "prompt_tokens", 0) or 0),
                      int(getattr(u, "completion_tokens", 0) or 0), cached, seconds)

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
        _SINK.call_seconds = 0.0
        _SINK.active = self.setup_usage
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
                        if "429" not in str(e) and "RESOURCE_EXHAUSTED" not in str(e) and "Rate" not in str(e):
                            break
                        time.sleep(60 * (attempt + 1))
                if last is not None:  # one bad document must not void the run
                    self.setup_meta["failed"].append({"doc": d.doc_id, "error": str(last)[:300]})
        finally:
            _SINK.active = None
        self.setup_seconds = time.perf_counter() - t0
        # Indexing is serial per document, so the model-call seconds are the
        # LLM phase and the rest is PageIndex's local parse and bookkeeping
        # (summary calls overlap at summary_concurrency, so the LLM figure is
        # call time, and can exceed its share of wall time).
        self.setup_meta["llm_call_seconds"] = round(_SINK.call_seconds, 3)
        self.setup_meta["local_seconds"] = round(max(0.0, self.setup_seconds - _SINK.call_seconds), 3)

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
        _SINK.active = usage
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
            _SINK.active = None
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, usage=usage,
                                   error=str(e)[:500], cold=cold, trace={"tools": tools})
        finally:
            _SINK.active = None
        total_ms = (time.perf_counter() - t0) * 1000.0
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
                   "answer_chars": answer_chars},
        )
