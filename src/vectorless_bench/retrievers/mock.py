"""A deterministic in-memory fake retriever.

It exists to exercise the *harness* end-to-end (scoring, aggregation, reporting,
determinism) with zero external services — the proof that the pipeline runs
today. It "cheats" by reading the question's gold anchors and fabricating
plausible sections at a configurable accuracy, plus synthetic-but-priced usage
so cost/latency code paths are covered too.

`accuracy=1.0` -> always returns the correct section first (a perfect system).
Lower values inject sibling near-misses so the near-miss metric is observable.
Output is seeded from qid only (NOT the repeat index), so reruns are identical
and the determinism metric reads ~1.0 — mimicking a temp=0 engine.
"""

from __future__ import annotations

import hashlib
from typing import List

from ..pricing import compute, count_tokens
from ..schema import (
    Doc,
    GoldAnchor,
    Question,
    RetrievalResult,
    RetrievedSection,
    Usage,
)


class MockRetriever:
    name = "mock"

    def __init__(
        self,
        accuracy: float = 1.0,
        model: str = "claude-haiku-4-5",
        base_latency_ms: float = 35.0,
        **_: object,
    ) -> None:
        self.accuracy = accuracy
        self.model = model
        self.base_latency_ms = base_latency_ms
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._docs: dict[str, Doc] = {}

    # Matches the Retriever protocol without inheriting (keeps the mock dep-free).
    def setup(self, corpus: List[Doc]) -> None:
        self._docs = {d.doc_id: d for d in corpus}

    def teardown(self) -> None:
        pass

    def _rand(self, qid: str) -> float:
        h = hashlib.sha256(qid.encode()).hexdigest()
        return int(h[:8], 16) / 0xFFFFFFFF

    def _section_for(self, g: GoldAnchor, correct: bool) -> RetrievedSection:
        span = g.answer_spans[0] if g.answer_spans else "the relevant figure"
        if correct:
            return RetrievedSection(
                content=f"... context establishing {span} within scope ...",
                section_id="sec_mock_hit",
                title=g.title_path[-1] if g.title_path else "Section",
                title_path=list(g.title_path),
                page=g.page,
                score=0.91,
            )
        # sibling near-miss: same parent, wrong leaf, answer span absent
        sib = list(g.parent_path()) + ["Prior Year Comparative"]
        return RetrievedSection(
            content="... superficially similar figure from a different period ...",
            section_id="sec_mock_near",
            title="Prior Year Comparative",
            title_path=sib,
            page=(g.page + 1) if g.page else None,
            score=0.88,
        )

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        if question.is_no_answer:
            # A good system abstains; the mock abstains at `accuracy` rate.
            sections: List[RetrievedSection] = []
            if self._rand(question.qid) > self.accuracy:
                sections = [
                    RetrievedSection(content="plausible but irrelevant", score=0.4)
                ]
        else:
            correct = self._rand(question.qid) <= self.accuracy
            sections = [self._section_for(g, correct) for g in question.gold][:k]

        in_tok = count_tokens(question.question, self.model) + 1200  # tree view
        out_tok = 40
        usage = Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=in_tok + out_tok,
            cost_usd=compute(self.model, in_tok, out_tok),
            llm_calls=1,
        )
        latency = self.base_latency_ms + 100.0 * self._rand(question.qid)
        return RetrievalResult(
            qid=question.qid,
            system=self.name,
            query=question.question,
            sections=sections,
            usage=usage,
            latency_ms=latency,
            strategy="mock",
            cold=cold,
        )
