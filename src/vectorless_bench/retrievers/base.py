"""The Retriever contract every system implements.

A benchmark "system" is anything that, given a corpus and a query, returns a
ranked list of passages plus honest cost/latency. Vectorless and all baselines
satisfy the same interface so the runner and scorer treat them identically.

Lifecycle: build(cfg) -> setup(corpus) [index/ingest, timed] -> retrieve(...) per
query -> teardown(). `retrieve` must NEVER raise for an expected failure; it sets
RetrievalResult.error and returns, so one bad query can't abort a multi-hour run.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List

from ..schema import Doc, Question, RetrievalResult, Usage


class Retriever(ABC):
    #: short stable id used on the CLI, in configs, and in reports
    name: str = "base"

    def __init__(self, **cfg: Any) -> None:
        self.cfg = cfg
        # populated by setup(); doc_id -> whatever the backend needs to query
        self._docs: Dict[str, Doc] = {}
        self.setup_seconds: float = 0.0
        self.setup_usage: Usage = Usage()

    # -- lifecycle ---------------------------------------------------------
    def setup(self, corpus: List[Doc]) -> None:
        """Index/ingest the corpus. Times itself and records ingest usage so the
        report can show one-time indexing cost (which dominates vector RAG)."""
        self._docs = {d.doc_id: d for d in corpus}
        t0 = time.perf_counter()
        self._index(corpus)
        self.setup_seconds = time.perf_counter() - t0

    @abstractmethod
    def _index(self, corpus: List[Doc]) -> None:
        """Backend-specific indexing. Accumulate ingest cost into self.setup_usage."""

    @abstractmethod
    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        """Return up to k ranked passages for the question."""

    def teardown(self) -> None:  # pragma: no cover - optional
        """Release resources (drop tables, close clients). Default no-op."""

    # -- helpers for subclasses -------------------------------------------
    def _doc(self, doc_id: str) -> Doc:
        return self._docs[doc_id]

    @contextmanager
    def _timed(self) -> Iterator[Dict[str, float]]:
        """`with self._timed() as t: ...; latency = t['ms']` — wall-clock ms."""
        box: Dict[str, float] = {}
        t0 = time.perf_counter()
        try:
            yield box
        finally:
            box["ms"] = (time.perf_counter() - t0) * 1000.0
