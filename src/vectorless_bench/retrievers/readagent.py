"""ReadAgent (Lee et al. 2024, "A Human-Inspired Reading Agent with Gist
Memory of Very Long Contexts"), REIMPLEMENTED FROM THE PAPER: Google DeepMind
published a demo notebook, not a library (searched 2026-10-16), so this
follows the paper's three steps and is labelled as a reimplementation.

1. Episode pagination (ingest): the model reads the text a window at a time
   and chooses where a natural pause falls, among numbered paragraph breaks
   between `min_words` and `max_words`; each pause ends a page.
2. Memory gisting (ingest): each page is compressed into a short gist,
   tagged with its page number. All gists together are the gist memory.
3. Interactive look-up (query), the paper's ReadAgent-P variant: the model
   sees the question and the whole gist memory and chooses up to
   `max_lookups` pages to reread in full, at once. Those pages, in the order
   it chose them, are the retrieval.

Ingest calls (pagination and gisting) are timed and costed as setup; the
look-up call is the query cost.
"""

from __future__ import annotations

import json
import re
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

from .._llm import DEFAULT_MODEL, complete, complete_patient, require_vertex_model
from ..datasets.base import CORPUS
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

_PAGINATE = (
    "You are reading a long document and choosing where to pause. Numbered break points <N> are placed "
    "between paragraphs. Choose the break point after which there is the most natural pause: the end of a "
    'section, a topic change, or a scene change. Reply as JSON: {"break": <N>}.'
)
_GIST = ("Shorten the passage into a gist of at most 60 words that keeps what a reader would need to "
         "find it again: its topic, names, numbers and defined terms. Reply with the gist only.")
_LOOKUP = (
    "You have a gist memory of a long document: one short gist per page, tagged <Page N>. "
    "To answer the question you may reread up to {n} pages in full. Choose the pages most likely to hold "
    'the answer, most relevant first. Reply as JSON: {{"pages": [N, ...]}}; use [] if none can.'
)


def _paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n|\f", text) if p.strip()]
    out: List[str] = []
    for p in paras:  # very long paragraphs are cut so a page can still end
        words = p.split()
        for i in range(0, len(words), 300):
            out.append(" ".join(words[i : i + 300]))
    return out


class ReadAgentRetriever:
    name = "readagent"

    def __init__(self, model: str = DEFAULT_MODEL, min_words: int = 280, max_words: int = 600,
                 max_lookups: int = 5, build_concurrency: int = 8, **_: object) -> None:
        require_vertex_model(model)
        self.model, self.min_words, self.max_words = model, min_words, max_words
        self.max_lookups, self.build_concurrency = max_lookups, build_concurrency
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._lock = threading.Lock()
        self.setup_meta = {"implementation": "reimplemented from the paper (ReadAgent-P)", "pagination_fallbacks": 0}
        self._pages: Dict[str, List[str]] = {}
        self._gists: Dict[str, List[str]] = {}

    def _paginate(self, text: str) -> List[str]:
        paras = _paragraphs(text)
        pages: List[str] = []
        i = 0
        while i < len(paras):
            window, words = [], 0
            j = i
            while j < len(paras) and words < self.max_words:
                window.append(paras[j])
                words += len(paras[j].split())
                j += 1
            if j >= len(paras) and words <= self.max_words:
                pages.append("\n\n".join(window))
                break
            # break points after paragraphs once min_words is reached
            cands, acc = [], 0
            for n, p in enumerate(window[:-1]):
                acc += len(p.split())
                if acc >= self.min_words:
                    cands.append(n + 1)
            if not cands:
                cut = len(window)
            else:
                marked = []
                for n, p in enumerate(window):
                    marked.append(p)
                    if (n + 1) in cands:
                        marked.append(f"<{n + 1}>")
                out = complete_patient(self.model, _PAGINATE, "\n\n".join(marked), max_tokens=20, json_mode=True)
                self.setup_usage.add(out.usage)
                try:
                    cut = int(json.loads(out.text).get("break"))
                except Exception:
                    cut = -1
                if cut not in cands:  # unparseable or off the list: the latest break, counted
                    cut = cands[-1]
                    self.setup_meta["pagination_fallbacks"] += 1
            pages.append("\n\n".join(window[:cut]))
            i += cut
        return pages

    def _gist(self, pages: List[str]) -> List[str]:
        def one(p: str) -> str:
            out = complete_patient(self.model, _GIST, p, max_tokens=120)
            with self._lock:
                self.setup_usage.add(out.usage)
            return out.text.strip()

        with ThreadPoolExecutor(self.build_concurrency) as pool:
            return list(pool.map(one, pages))

    def setup(self, corpus: List[Doc]) -> None:
        t0 = time.perf_counter()
        self.setup_meta["failed"] = []
        for d in corpus:
            # One document that cannot be read is recorded, and its questions
            # error on their own; it does not void the rest.
            try:
                pages = self._paginate(d.content)
                gists = self._gist(pages)
            except Exception as e:  # pragma: no cover - network
                self.setup_meta["failed"].append({"doc": d.doc_id, "error": str(e)[:300]})
                continue
            self._pages[d.doc_id] = pages
            self._gists[d.doc_id] = gists
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:
        pass

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        if question.doc_id == CORPUS:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, cold=cold,
                                   error="n/a: ReadAgent reads one document's gist memory; this question spans the corpus")
        gists = self._gists.get(question.doc_id)
        if gists is None:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question,
                                   error="document not read", cold=cold)
        usage = Usage()
        t0 = time.perf_counter()
        memory = "\n".join(f"<Page {i}> {g}" for i, g in enumerate(gists))
        try:
            out = complete(self.model, _LOOKUP.format(n=self.max_lookups),
                           f"QUESTION: {question.question}\n\nGIST MEMORY:\n{memory}", max_tokens=80, json_mode=True)
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, error=str(e)[:500], cold=cold)
        usage.add(out.usage)
        try:
            chosen = [int(p) for p in json.loads(out.text).get("pages", [])]
        except Exception:
            chosen = []
        pages = self._pages[question.doc_id]
        seen, sections = set(), []
        for p in chosen[: self.max_lookups]:
            if 0 <= p < len(pages) and p not in seen:
                seen.add(p)
                sections.append(RetrievedSection(content=pages[p], section_id=f"page{p}"))
        return RetrievalResult(qid=question.qid, system=self.name, query=question.question, sections=sections[:k],
                               usage=usage, latency_ms=(time.perf_counter() - t0) * 1000.0, strategy="readagent-p",
                               cold=cold, trace={"pages_total": len(pages), "chosen": chosen})
