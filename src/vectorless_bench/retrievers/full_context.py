"""Full-context baseline: put the WHOLE document in the prompt and ask the model
to return the passages that answer the query.

This is the quality *ceiling* and the cost *worst case* — no retrieval step can
beat seeing everything, and nothing is more expensive than paying for the whole
document on every query. It frames the efficiency frontier: Vectorless's pitch
is "most of this quality at a fraction of this cost", and this baseline makes
that claim measurable instead of rhetorical.

Run fairly (HAL-3046):
- Each document is put in a Vertex context cache the first time a question
  needs it and reused for its other questions, the way anyone running this in
  production would. A cached prompt bills at a tenth of the input rate, and
  the cache's storage (token-hours) is added to the system's setup cost, so
  the saving is not free money.
- A document larger than the model's window is NOT truncated: the question is
  reported not applicable ("n/a:"), so a ceiling is never quietly lowered.
  Corpus-level questions (MultiHop-RAG) are not applicable for the same
  reason.
- `pdf_input=True` sends the PDF itself (Gemini reads pages as images and
  text), which makes this the multimodal ceiling for MMLongBench-Doc.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .._llm import DEFAULT_MODEL, count_tokens, genai_client, generate, require_vertex_model
from ..datasets.base import CORPUS
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

_SYSTEM = (
    "You are a retrieval oracle. You are given a full document and a question. "
    "Return ONLY the verbatim passage(s) from the document that contain the "
    "answer, most relevant first. Respond as JSON: {\"passages\": [{\"text\": \"<verbatim excerpt>\", "
    "\"page\": <page number or null>, \"heading_path\": [\"<section>\", ...]}]}. If the document does not "
    "contain the answer, return {\"passages\": []}."
)

# Context-cache storage, USD per 1M tokens per hour (Vertex list, 2026-10-16):
# Flash and Flash-Lite models $1.00, Pro models $4.50.
_STORAGE_PER_MTOK_HOUR = {"pro": 4.50, "default": 1.00}
# Below this a cache is not worth creating (and Vertex refuses tiny caches).
_MIN_CACHE_TOKENS = 4096


class FullContextRetriever:
    name = "full_context"

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        max_context_tokens: int = 1_000_000,
        max_output_tokens: int = 1500,
        cache: bool = True,
        cache_ttl_seconds: int = 3600,
        pdf_input: bool = False,
        **_: object,
    ) -> None:
        require_vertex_model(model)
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.cache = cache
        self.ttl = cache_ttl_seconds
        self.pdf_input = pdf_input
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._docs: Dict[str, Doc] = {}
        self._tokens: Dict[str, int] = {}
        self._caches: Dict[str, Dict[str, object]] = {}
        self._cache_lock = threading.Lock()
        # Creation is serialised per document, so two questions that miss at
        # once share one cache; a replaced cache is deleted only once no
        # retrieval still holds its name.
        self._doc_locks: Dict[str, threading.Lock] = {}
        self._refs: Dict[str, int] = {}
        self._retired: Dict[str, bool] = {}

    def setup(self, corpus: List[Doc]) -> None:
        self._docs = {d.doc_id: d for d in corpus}

    def _doc_part(self, doc: Doc):
        from google.genai import types  # type: ignore

        if self.pdf_input and doc.path and doc.path.lower().endswith(".pdf"):
            return types.Part.from_bytes(data=Path(doc.path).read_bytes(), mime_type="application/pdf")
        return types.Part(text=f"DOCUMENT ({doc.title}):\n{doc.content}")

    def _doc_tokens(self, doc: Doc) -> int:
        if doc.doc_id not in self._tokens:
            from google.genai import types  # type: ignore

            r = genai_client().models.count_tokens(
                model=self.model, contents=[types.Content(role="user", parts=[self._doc_part(doc)])])
            self._tokens[doc.doc_id] = int(r.total_tokens or 0)
        return self._tokens[doc.doc_id]

    def _cache_for(self, doc: Doc) -> Optional[str]:
        """A live cache for the document, held for the caller: release it
        with _release once the request that uses it is done."""
        if not self.cache or self._doc_tokens(doc) < _MIN_CACHE_TOKENS:
            return None
        with self._cache_lock:
            lock = self._doc_locks.setdefault(doc.doc_id, threading.Lock())
        with lock:
            with self._cache_lock:
                c = self._caches.get(doc.doc_id)
                if c and time.time() < float(c["expires"]) - 60:
                    name = str(c["name"])
                    self._refs[name] = self._refs.get(name, 0) + 1
                    return name
            from google.genai import types  # type: ignore

            created = genai_client().caches.create(model=self.model, config=types.CreateCachedContentConfig(
                contents=[types.Content(role="user", parts=[self._doc_part(doc)])],
                system_instruction=_SYSTEM, ttl=f"{self.ttl}s", display_name=f"vlbench-{doc.doc_id[:40]}"))
            entry = {"name": created.name, "created": time.time(), "expires": time.time() + self.ttl,
                     "tokens": self._doc_tokens(doc)}
            with self._cache_lock:
                old = self._caches.get(doc.doc_id)
                self._caches[doc.doc_id] = entry
                self._refs[created.name] = self._refs.get(created.name, 0) + 1
            if old is not None:
                self._retire(old)  # replaced near expiry: charged now, deleted when unused
            return created.name

    def _release(self, name: Optional[str]) -> None:
        if not name:
            return
        with self._cache_lock:
            self._refs[name] = max(0, self._refs.get(name, 0) - 1)
            delete = self._refs[name] == 0 and self._retired.pop(name, False)
        if delete:
            self._delete(name)

    def _delete(self, name: str) -> None:
        try:
            genai_client().caches.delete(name=name)
        except Exception:
            pass

    def _retire(self, c: Dict[str, object]) -> None:
        """Charge a cache's storage for the time it existed (to expiry at
        most), and delete it now if no retrieval holds it, else on release."""
        rate = _STORAGE_PER_MTOK_HOUR["pro" if "pro" in self.model else "default"]
        end = min(time.time(), float(c["expires"]))
        hours = max(0.0, end - float(c["created"])) / 3600.0
        name = str(c["name"])
        with self._cache_lock:
            self.setup_usage.cost_usd += int(c["tokens"]) * hours * rate / 1_000_000.0
            busy = self._refs.get(name, 0) > 0
            if busy:
                self._retired[name] = True
        if not busy:
            self._delete(name)

    def teardown(self) -> None:
        """Delete the caches and charge their storage to setup cost."""
        with self._cache_lock:
            caches = list(self._caches.values())
            self._caches.clear()
            leftover = [n for n, flagged in self._retired.items() if flagged]
            self._retired.clear()
        for c in caches:
            self._retire(c)
        for name in leftover:  # still held at teardown: the run is over, delete anyway
            self._delete(name)

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        from google.genai import types  # type: ignore

        if question.doc_id == CORPUS:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, cold=cold,
                                   error="n/a: corpus-level question; the corpus exceeds one context window")
        doc = self._docs.get(question.doc_id)
        if doc is None:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question,
                                   error="doc not in corpus", cold=cold)
        t0 = time.perf_counter()
        try:
            ntok = self._doc_tokens(doc)
            if ntok > self.max_context_tokens:
                return RetrievalResult(qid=question.qid, system=self.name, query=question.question, cold=cold,
                                       error=f"n/a: document is {ntok} tokens, past the {self.max_context_tokens}-token window")
            cache = self._cache_for(doc)
            q = types.Part(text=f"QUESTION:\n{question.question}")
            contents = [types.Content(role="user", parts=[q] if cache else [self._doc_part(doc), q])]
            try:
                out = generate(self.model, system=_SYSTEM, contents=contents, max_tokens=self.max_output_tokens,
                               json_mode=True, cached_content=cache)
            finally:
                self._release(cache)
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, error=str(e), cold=cold)
        sections = self._parse(out.text)[:k]
        return RetrievalResult(
            qid=question.qid, system=self.name, query=question.question,
            sections=sections, usage=out.usage, latency_ms=(time.perf_counter() - t0) * 1000.0,
            strategy="full-context" + ("-pdf" if self.pdf_input else ""), cold=cold,
            trace={"model": self.model, "doc_tokens": ntok, "cached": bool(cache)},
            # The ceiling answers from everything it read, not from its quotes.
            answer_parts=[self._doc_part(doc)],
        )

    def _parse(self, text: str) -> List[RetrievedSection]:
        try:
            data = json.loads(text)
            passages = data.get("passages", [])
        except (json.JSONDecodeError, AttributeError):
            # model didn't return clean JSON; treat the whole thing as one passage
            return [RetrievedSection(content=text)] if text.strip() else []
        out: List[RetrievedSection] = []
        for p in passages:
            if isinstance(p, dict):
                page = p.get("page")
                out.append(RetrievedSection(content=p.get("text", ""), title_path=p.get("heading_path", []) or [],
                                            page=int(page) if isinstance(page, (int, float)) else None))
            elif isinstance(p, str):
                out.append(RetrievedSection(content=p))
        return out
