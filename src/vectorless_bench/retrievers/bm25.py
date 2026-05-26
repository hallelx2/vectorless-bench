"""BM25 lexical baseline — the cheap, structure-blind floor.

Often surprisingly strong on exact-term finance/legal lookups (ticker symbols,
statute numbers, line-item names), and effectively free (no API, no GPU). Having
it in the suite keeps the story honest: if a $0 lexical method matches a paid
LLM retriever on a question class, that's worth knowing.
"""

from __future__ import annotations

from typing import List, Optional

from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage
from ._chunk import Chunk, chunk_doc


def _tokenize(s: str) -> List[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in s).split()]


class BM25Retriever:
    name = "bm25"

    def __init__(
        self,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        per_doc: bool = True,
        **_: object,
    ) -> None:
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.per_doc = per_doc  # restrict ranking to the question's doc
        self.setup_seconds = 0.0
        self.setup_usage = Usage()  # lexical: no token cost
        self._chunks: List[Chunk] = []
        self._by_doc: dict[str, List[int]] = {}
        self._bm25 = None
        self._corpus_tokens: List[List[str]] = []

    def setup(self, corpus: List[Doc]) -> None:
        import time

        from rank_bm25 import BM25Okapi  # type: ignore

        t0 = time.perf_counter()
        for d in corpus:
            for ch in chunk_doc(d, self.chunk_tokens, self.overlap_tokens):
                self._by_doc.setdefault(ch.doc_id, []).append(len(self._chunks))
                self._chunks.append(ch)
        self._corpus_tokens = [_tokenize(c.text) for c in self._chunks]
        self._bm25 = BM25Okapi(self._corpus_tokens)
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:
        pass

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        if self._bm25 is None:
            return RetrievalResult(
                qid=question.qid, system=self.name,
                query=question.question, error="setup() not run",
            )
        with _Timer() as t:
            scores = self._bm25.get_scores(_tokenize(question.question))
            candidate_idx: List[int]
            if self.per_doc and question.doc_id in self._by_doc:
                candidate_idx = self._by_doc[question.doc_id]
            else:
                candidate_idx = list(range(len(self._chunks)))
            ranked = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)[:k]

        sections = [
            RetrievedSection(
                content=self._chunks[i].text,
                section_id=self._chunks[i].chunk_id,
                score=float(scores[i]),
            )
            for i in ranked
        ]
        return RetrievalResult(
            qid=question.qid,
            system=self.name,
            query=question.question,
            sections=sections,
            usage=Usage(),  # free
            latency_ms=t.ms,
            strategy="bm25-top-k",
            cold=cold,
        )


class _Timer:
    def __enter__(self) -> "_Timer":
        import time

        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        import time

        self.ms = (time.perf_counter() - self._t0) * 1000.0
