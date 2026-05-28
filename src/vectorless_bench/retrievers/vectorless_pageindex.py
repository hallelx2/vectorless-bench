"""Vectorless on the page-based agentic strategy — wired through the engine's
dedicated `POST /v1/answer/pageindex` endpoint.

This is the same engine and the same priced model as the default `vectorless`
retriever, but a different *retrieval strategy*: the model navigates page ranges
agentically (read_page / read_pages / done) instead of walking a section tree.
The endpoint also owns the full RAG round-trip — answer + page-grounded
citations come back in one request, so there's no separate /v1/query then
synthesise step.

We post directly with httpx rather than going through the SDK because:
  - the SDK's `client.query` targets /v1/query (the existing strategy);
  - the new endpoint is intentionally distinct on the engine — different
    request shape (max_hops, max_pages_per_fetch), different response
    shape (citations[], hops_taken, pages_read).

The bench's RetrievedSection shape is section-centric, so we map the page
endpoint's citations[] into sections as follows:
  - one bench section per citation (one per cited page range);
  - section content = the citation's `quote` (the answer-span the engine
    extracted), which is what the scorer's span-match metrics care about;
  - section id = the FIRST section_id on the citation, tagged with the
    citation's page range so the metrics can still match by anchor.

Ingest is identical to the default retriever (it uses the same engine), so we
inherit it; only `retrieve()` differs.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

import httpx

from ..pricing import compute
from ..schema import Question, RetrievalResult, RetrievedSection, Usage
from .vectorless import VectorlessRetriever


class VectorlessPageIndexRetriever(VectorlessRetriever):
    name = "vectorless_pageindex"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_hops: Optional[int] = None,
        page_content_limit: Optional[int] = None,
        query_timeout: float = 240.0,
        org: str = "vlbench",
        **kwargs: object,
    ) -> None:
        # Parent handles SDK construction, X-Vectorless-Org injection, ingest
        # bookkeeping, and the path index. We just need our own raw HTTP
        # client for the answer endpoint, which the SDK doesn't yet expose.
        super().__init__(
            api_key=api_key,
            base_url=base_url,
            model=model,
            query_timeout=query_timeout,
            org=org,
            **kwargs,
        )
        # Resolve base URL + API key with the same env fallback pattern the
        # SDK uses, so behaviour matches when the caller doesn't pass them.
        self._base_url = (
            base_url
            or os.environ.get("VECTORLESS_BASE_URL")
            or "http://localhost:8080"
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("VECTORLESS_API_KEY") or ""
        self._http = httpx.Client(timeout=query_timeout)
        self.max_hops = max_hops
        self.page_content_limit = page_content_limit

    def teardown(self) -> None:  # pragma: no cover - network
        try:
            self._http.close()
        except Exception:
            pass
        super().teardown()

    # -- query -------------------------------------------------------------
    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        server_doc = self._doc_ids.get(question.doc_id)
        if server_doc is None:
            return RetrievalResult(
                qid=question.qid,
                system=self.name,
                query=question.question,
                error=f"document {question.doc_id!r} was not ingested",
                cold=cold,
            )

        body: Dict[str, Any] = {
            "document_id": server_doc,
            "query": question.question,
        }
        if self.model:
            body["model"] = self.model
        if self.max_hops:
            body["max_hops"] = int(self.max_hops)
        if self.page_content_limit:
            body["max_pages_per_fetch"] = int(self.page_content_limit)

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "X-Vectorless-Org": self.org,
            "Content-Type": "application/json",
        }
        url = f"{self._base_url}/v1/answer/pageindex"

        t0 = time.perf_counter()
        try:
            resp = self._http.post(url, json=body, headers=headers)
            # Treat non-2xx as an error string so a single bad query doesn't
            # crash the run. Match the existing vectorless retriever's
            # "swallow + record" stance.
            if resp.status_code >= 400:
                return RetrievalResult(
                    qid=question.qid,
                    system=self.name,
                    query=question.question,
                    error=f"HTTP {resp.status_code}: {resp.text[:300]}",
                    latency_ms=(time.perf_counter() - t0) * 1000.0,
                    cold=cold,
                )
            data = resp.json()
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(
                qid=question.qid,
                system=self.name,
                query=question.question,
                error=str(e),
                latency_ms=(time.perf_counter() - t0) * 1000.0,
                cold=cold,
            )

        path_index = self._paths.get(question.doc_id, {})
        sections = self._sections_from_citations(data.get("citations") or [], path_index, k)
        usage = self._usage_from_dict(data.get("usage") or {}, data.get("model") or "")

        # elapsed_ms comes from the engine; fall back to local wall-clock if
        # the server didn't fill it in (older revisions, transient errors).
        latency = float(data.get("elapsed_ms") or 0)
        if latency <= 0:
            latency = (time.perf_counter() - t0) * 1000.0

        return RetrievalResult(
            qid=question.qid,
            system=self.name,
            query=question.question,
            sections=sections,
            usage=usage,
            latency_ms=latency,
            strategy=str(data.get("strategy") or "pageindex"),
            cold=cold,
            trace={
                "server_cache_disabled": self.server_cache_disabled,
                "model": data.get("model") or "",
                "hops_taken": int(data.get("hops_taken") or 0),
                "trace_token": data.get("trace_token") or "",
                "answer": (data.get("answer") or "")[:2000],  # truncate for log size
                "citation_count": len(data.get("citations") or []),
                "pages_read": data.get("pages_read") or [],
            },
        )

    # -- helpers -----------------------------------------------------------
    def _sections_from_citations(
        self,
        citations: List[Dict[str, Any]],
        path_index: Dict[str, List[str]],
        k: int,
    ) -> List[RetrievedSection]:
        """Project the page endpoint's citation list onto the bench's section
        shape. Order is preserved (engine sorts by start_page) and capped at k
        so head-to-head fairness against k=5 baselines is intact.
        """
        out: List[RetrievedSection] = []
        for c in citations[:k]:
            sec_ids = c.get("section_ids") or []
            # First section_id anchors the citation in the structural tree
            # for path matching; the page range stays in `page` so the
            # bench's per-page anchors still hit.
            sid = str(sec_ids[0]) if sec_ids else ""
            start_page = c.get("start_page")
            try:
                start_page_int = int(start_page) if start_page is not None else None
            except (TypeError, ValueError):
                start_page_int = None
            quote = (c.get("quote") or "").strip()
            out.append(
                RetrievedSection(
                    content=quote,
                    section_id=sid,
                    title="",
                    title_path=path_index.get(sid, []) if sid else [],
                    page=start_page_int,
                    score=None,
                )
            )
        return out

    def _usage_from_dict(self, u: Dict[str, Any], model: str) -> Usage:
        in_tok = int(u.get("input_tokens") or 0)
        out_tok = int(u.get("output_tokens") or 0)
        cost = float(u.get("cost_usd") or 0.0)
        if cost == 0.0:  # legacy server might omit cost; reprice locally
            cost = compute(model, in_tok, out_tok)
        total = int(u.get("total_tokens") or (in_tok + out_tok))
        return Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=total,
            cost_usd=cost,
            llm_calls=int(u.get("llm_calls") or 1),
        )
