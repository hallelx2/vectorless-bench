"""Full-context baseline: put the WHOLE document in the prompt and ask the model
to return the passages that answer the query.

This is the quality *ceiling* and the cost *worst case* — no retrieval step can
beat seeing everything, and nothing is more expensive than paying for the whole
document on every query. It frames the efficiency frontier: Vectorless's pitch
is "most of this quality at a fraction of this cost", and this baseline makes
that claim measurable instead of rhetorical.

Documents that exceed the model's context window are truncated (recorded in the
trace) so the run still completes; that truncation is itself a finding.
"""

from __future__ import annotations

import json
from typing import List, Optional

from .._llm import complete
from ..pricing import count_tokens
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

_SYSTEM = (
    "You are a retrieval oracle. You are given a full document and a question. "
    "Return ONLY the verbatim passage(s) from the document that contain the "
    "answer. Respond as JSON: {\"passages\": [{\"text\": \"<verbatim excerpt>\", "
    "\"heading_path\": [\"<section>\", ...]}]}. If the document does not contain "
    "the answer, return {\"passages\": []}."
)


class FullContextRetriever:
    name = "full_context"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        max_context_tokens: int = 120_000,
        max_output_tokens: int = 1500,
        **_: object,
    ) -> None:
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_output_tokens = max_output_tokens
        self.setup_seconds = 0.0
        self.setup_usage = Usage()  # no indexing
        self._docs: dict[str, Doc] = {}

    def setup(self, corpus: List[Doc]) -> None:
        self._docs = {d.doc_id: d for d in corpus}

    def teardown(self) -> None:
        pass

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        import time

        doc = self._docs.get(question.doc_id)
        if doc is None:
            return RetrievalResult(
                qid=question.qid, system=self.name,
                query=question.question, error="doc not in corpus",
            )
        content, truncated = self._fit(doc.content)
        user = f"QUESTION:\n{question.question}\n\nDOCUMENT:\n{content}"
        t0 = time.perf_counter()
        try:
            out = complete(
                self.model, _SYSTEM, user,
                max_tokens=self.max_output_tokens, json_mode=True,
            )
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(
                qid=question.qid, system=self.name,
                query=question.question, error=str(e),
            )
        latency = (time.perf_counter() - t0) * 1000.0
        sections = self._parse(out.text)[:k]
        return RetrievalResult(
            qid=question.qid,
            system=self.name,
            query=question.question,
            sections=sections,
            usage=out.usage,
            latency_ms=latency,
            strategy="full-context",
            cold=cold,
            trace={"truncated": truncated, "model": self.model},
        )

    def _fit(self, content: str) -> tuple[str, bool]:
        if count_tokens(content, self.model) <= self.max_context_tokens:
            return content, False
        # crude char-budget truncation; enough to keep the run going + flag it
        approx_chars = self.max_context_tokens * max(1, len(content) // max(1, count_tokens(content, self.model)))
        return content[:approx_chars], True

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
                out.append(
                    RetrievedSection(
                        content=p.get("text", ""),
                        title_path=p.get("heading_path", []) or [],
                    )
                )
            elif isinstance(p, str):
                out.append(RetrievedSection(content=p))
        return out
