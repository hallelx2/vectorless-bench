"""The system under test: Vectorless, via the official Python SDK.

Two engine-specific details handled here:

1. Section IDs are random `sec_<uuid>`s, so after ingest we fetch the document
   tree and build an id -> title_path map. Returned sections are tagged with
   their structural path, which is what the scorer's path matching needs.
2. Caching would zero out cost and latency. For a fair cold-cache measurement
   the SERVER must run with `retrieval.cache.enabled=false` and the llmgate
   cache middleware off. We record the declared cache mode in the trace and, if
   `cold` is requested but the server cache cannot be guaranteed off, the
   manifest will carry a warning. We do NOT salt queries by default because that
   changes the model's input and contaminates the comparison.

Cost comes straight from the engine's `usage` field (already priced with the
same table), so Vectorless numbers need no re-estimation.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ..pricing import compute
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage


class VectorlessRetriever:
    name = "vectorless"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        max_tokens: Optional[int] = None,
        max_parallel_calls: Optional[int] = None,
        server_cache_disabled: bool = False,
        ingest_timeout: float = 600.0,
        query_timeout: float = 120.0,
        org: Optional[str] = None,
        **_: object,
    ) -> None:
        try:
            from vectorless import VectorlessClient  # type: ignore
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "the 'vectorless' SDK is required for the vectorless system: "
                "pip install vectorless-sdk"
            ) from e

        # Default SDK timeout is 30s. Chunked-tree retrieval over a 200-section
        # 10-Q can run ~60-90s in the worst case; bump to 120 by default so
        # large-doc queries don't fail with "read operation timed out".
        self._client = VectorlessClient(
            api_key=api_key, base_url=base_url, timeout=query_timeout
        )
        # The deployed engine is multi-tenant: it requires an X-Vectorless-Org
        # header that the control plane normally injects. Talking to the engine
        # directly, we supply our own org id (a pure storage scope — the engine
        # doesn't validate it against a registry), so the benchmark's documents
        # live in their own isolated namespace. The SDK has no native org option,
        # so set it as a default header on the underlying httpx client.
        import os as _os

        self.org = org or _os.environ.get("VECTORLESS_ORG", "vlbench")
        try:
            self._client._transport._client.headers["X-Vectorless-Org"] = self.org
        except Exception:  # SDK internals changed; fail loudly at first call
            pass
        self.model = model
        self.max_tokens = max_tokens
        self.max_parallel_calls = max_parallel_calls
        self.server_cache_disabled = server_cache_disabled
        self.ingest_timeout = ingest_timeout
        self.setup_seconds = 0.0
        self.setup_usage = Usage()  # ingest cost isn't returned per-doc by API
        self._doc_ids: Dict[str, str] = {}  # logical -> server document_id
        self._paths: Dict[str, Dict[str, List[str]]] = {}  # doc -> {sec_id: path}

    # -- lifecycle ---------------------------------------------------------
    def setup(self, corpus: List[Doc]) -> None:
        import time

        t0 = time.perf_counter()
        for d in corpus:
            # Prefer the ORIGINAL file (e.g. PDF) so the engine's own parser
            # extracts structure. Feeding pre-extracted flat text yields a
            # structureless tree and cripples retrieval — give the engine the
            # same raw input PageIndex gets.
            if d.path and d.path.lower().endswith((".pdf", ".docx", ".html", ".htm")):
                with open(d.path, "rb") as fh:
                    source: bytes = fh.read()
                ext = d.path.rsplit(".", 1)[-1].lower()
                filename = f"{d.doc_id}.{ext}"
                content_type = {
                    "pdf": "application/pdf",
                    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    "html": "text/html",
                    "htm": "text/html",
                }[ext]
            else:
                source = d.content.encode("utf-8")
                filename = f"{d.doc_id}.md"
                content_type = d.content_type or "text/markdown"
            resp = self._client.ingest_document(
                source=source, filename=filename, content_type=content_type
            )
            doc_id = resp.document_id
            self._client.wait_for_ready(doc_id, timeout=self.ingest_timeout)
            self._doc_ids[d.doc_id] = doc_id
            self._paths[d.doc_id] = self._build_path_index(doc_id)
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:  # pragma: no cover - network
        for doc_id in self._doc_ids.values():
            try:
                self._client.delete_document(doc_id)
            except Exception:
                pass
        try:
            self._client.close()
        except Exception:
            pass

    def _build_path_index(self, server_doc_id: str) -> Dict[str, List[str]]:
        """Map each section id to its heading path.

        The SDK's DocumentTree is a FLAT `.sections` list (each carrying id,
        parent_id, title) — there is no `.root`/`.children` nesting — so we
        reconstruct each section's path by walking its parent_id chain. (Falls
        back to a nested-node walk for older/other tree shapes.)"""
        paths: Dict[str, List[str]] = {}
        try:
            tree = self._client.get_document_tree(server_doc_id)
        except Exception:  # pragma: no cover - network
            return paths

        sections = getattr(tree, "sections", None)
        if sections:
            by_id = {}
            for s in sections:
                sid = str(getattr(s, "id", "") or getattr(s, "section_id", "") or "")
                if sid:
                    by_id[sid] = s

            def path_for(sid: str, seen: tuple = ()) -> List[str]:
                s = by_id.get(sid)
                if s is None or sid in seen:
                    return []
                parent = str(getattr(s, "parent_id", "") or "")
                title = getattr(s, "title", "") or ""
                prefix = path_for(parent, seen + (sid,)) if parent in by_id else []
                return prefix + ([title] if title else [])

            for sid in by_id:
                paths[sid] = path_for(sid)
            return paths

        # fallback: nested root.children shape
        def walk(node: object, prefix: List[str]) -> None:
            title = getattr(node, "title", "") or ""
            here = prefix + [title] if title else prefix
            sid = getattr(node, "id", None) or getattr(node, "section_id", None)
            if sid:
                paths[str(sid)] = here
            for child in getattr(node, "children", None) or []:
                walk(child, here)

        root = getattr(tree, "root", None) or tree
        for child in getattr(root, "children", None) or []:
            walk(child, [])
        return paths

    # -- query -------------------------------------------------------------
    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        server_doc = self._doc_ids.get(question.doc_id)
        if server_doc is None:
            return RetrievalResult(
                qid=question.qid,
                system=self.name,
                query=question.question,
                error=f"document {question.doc_id!r} was not ingested",
            )
        opts: Dict[str, object] = {}
        if self.model:
            opts["model"] = self.model
        if self.max_tokens:
            opts["max_tokens"] = self.max_tokens
        if self.max_parallel_calls:
            opts["max_parallel_calls"] = self.max_parallel_calls
        opts["max_sections"] = k

        try:
            resp = self._client.query(server_doc, question.question, **opts)
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(
                qid=question.qid, system=self.name,
                query=question.question, error=str(e),
            )

        path_index = self._paths.get(question.doc_id, {})
        sections: List[RetrievedSection] = []
        for s in resp.sections:
            sid = str(getattr(s, "id", ""))
            sections.append(
                RetrievedSection(
                    content=getattr(s, "content", "") or "",
                    section_id=sid,
                    title=getattr(s, "title", "") or "",
                    title_path=path_index.get(sid, []),
                    score=None,
                )
            )

        usage = self._usage_from(resp)
        return RetrievalResult(
            qid=question.qid,
            system=self.name,
            query=question.question,
            sections=sections,
            usage=usage,
            latency_ms=float(getattr(resp, "elapsed_ms", 0) or 0),
            strategy=getattr(resp, "strategy", "") or "",
            cold=cold,
            trace={
                "server_cache_disabled": self.server_cache_disabled,
                "model": getattr(resp, "model", "") or "",
            },
        )

    def _usage_from(self, resp: object) -> Usage:
        u = getattr(resp, "usage", None)
        if u is None:
            return Usage()
        in_tok = int(getattr(u, "input_tokens", 0) or 0)
        out_tok = int(getattr(u, "output_tokens", 0) or 0)
        cost = float(getattr(u, "cost_usd", 0.0) or 0.0)
        if cost == 0.0:  # older servers may omit cost; reprice locally
            cost = compute(getattr(resp, "model", "") or "", in_tok, out_tok)
        return Usage(
            input_tokens=in_tok,
            output_tokens=out_tok,
            total_tokens=int(getattr(u, "total_tokens", in_tok + out_tok) or 0),
            cost_usd=cost,
            llm_calls=int(getattr(u, "llm_calls", 1) or 1),
        )
