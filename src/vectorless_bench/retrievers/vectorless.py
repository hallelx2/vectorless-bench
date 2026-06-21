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
        query_timeout: float = 180.0,
        org: str = "vlbench",
        **_: object,
    ) -> None:
        try:
            from vectorless import VectorlessClient  # type: ignore
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "the 'vectorless' SDK is required for the vectorless system: "
                "pip install vectorless-sdk"
            ) from e

        self._client = VectorlessClient(
            api_key=api_key, base_url=base_url, timeout=query_timeout
        )
        self.org = org
        # Engine is multi-tenant — every request needs X-Vectorless-Org.
        # The SDK doesn't expose this option, so we inject it on the
        # underlying httpx client.
        try:
            self._client._transport._client.headers["X-Vectorless-Org"] = org
        except AttributeError:
            try:
                self._client._client.headers["X-Vectorless-Org"] = org
            except AttributeError:
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
        # Per-document ingest wall-time (upload → ready), in seconds, keyed by
        # logical doc_id. This is the engine's parse+persist throughput — the
        # speed axis the Go engine exists for. Surfaced in the report.
        self.ingest_times: Dict[str, float] = {}
        self.ingest_bytes: Dict[str, int] = {}

    # -- lifecycle ---------------------------------------------------------
    def setup(self, corpus: List[Doc]) -> None:
        import time
        import pathlib

        t0 = time.perf_counter()
        for d in corpus:
            # Prefer the original PDF when present — that's the path the
            # engine's parser + pdftable table extractor were tuned for.
            # Pre-extracted .txt loses page boundaries and tables before
            # the engine ever sees the document.
            path = getattr(d, "path", None)
            if path and pathlib.Path(path).is_file() and path.lower().endswith(".pdf"):
                source = pathlib.Path(path).read_bytes()
                filename = pathlib.Path(path).name
                content_type = "application/pdf"
            else:
                source = d.content.encode("utf-8")
                filename = f"{d.doc_id}.md"
                content_type = d.content_type

            # Ingest each doc independently. A single document that fails
            # to parse / reach ready (a pathological PDF) must not sink the
            # whole system's run — skip it and query the rest. Questions
            # whose doc didn't ingest are handled by retrieve()'s
            # "was not ingested" guard.
            #
            # Transient network errors (Windows DNS hiccups, brief socket
            # unreachables) get a few retries with backoff before we
            # conclude the doc can't ingest — the local DNS resolver
            # occasionally returns SERVFAIL on Cloud Run's IPv6-only
            # AAAA records under load, and a single retry usually wins.
            transient = (
                "getaddrinfo", "10065", "10054", "10060",
                "Connection reset", "Temporary failure", "Name or service not known",
            )
            # Stable per-corpus-doc idempotency key. The engine dedups on
            # (org, idempotency_key): any retry of this ingest — whether our
            # loop below OR the SDK transport's own retry-on-transient — returns
            # the SAME document instead of creating a duplicate. This is the fix
            # for the HAL-323 "same doc ingested up to 6×" bug; without it a
            # connection reset AFTER the server committed the doc made the SDK
            # re-POST and spawn a fresh doc_id + parse job each time.
            idem_key = f"vlbench:{self.org}:{d.doc_id}"

            ok = False
            last_err: Exception | None = None
            for attempt in range(4):
                try:
                    t_doc = time.perf_counter()
                    resp = self._client.ingest_document(
                        source=source,
                        filename=filename,
                        content_type=content_type,
                        idempotency_key=idem_key,
                    )
                    doc_id = resp.document_id
                    # Poll tightly (0.25s) so the measured upload→ready time
                    # reflects the engine's real parse+persist latency rather
                    # than the SDK's default 2s poll granularity — the ingest
                    # speed number is a headline for the Go engine.
                    self._client.wait_for_ready(
                        doc_id, timeout=self.ingest_timeout, poll_interval=0.25
                    )
                    # Upload → ready wall-time: the engine's end-to-end ingest
                    # speed for this document (parse + tree build + persist).
                    self.ingest_times[d.doc_id] = time.perf_counter() - t_doc
                    self.ingest_bytes[d.doc_id] = len(source)
                    self._doc_ids[d.doc_id] = doc_id
                    self._paths[d.doc_id] = self._build_path_index(doc_id)
                    ok = True
                    break
                except Exception as e:  # pragma: no cover - network/timeout
                    last_err = e
                    msg = str(e)
                    if not any(t in msg for t in transient) or attempt == 3:
                        break
                    import sys as _sys, time as _time
                    print(f"[vectorless] transient on {d.doc_id} attempt {attempt+1}: {msg[:80]} — retrying",
                          file=_sys.stderr)
                    _time.sleep(2 * (attempt + 1))
            if not ok:
                import sys
                print(f"[vectorless] WARN: skipping {d.doc_id} — ingest failed: {last_err}",
                      file=sys.stderr)
                continue
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
        """Walk the document tree to map each section id to its heading path."""
        paths: Dict[str, List[str]] = {}
        try:
            tree = self._client.get_document_tree(server_doc_id)
        except Exception:  # pragma: no cover - network
            return paths

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
