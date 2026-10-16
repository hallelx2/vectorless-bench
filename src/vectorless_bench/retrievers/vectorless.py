"""The system under test: Vectorless, via the official Python SDK.

Two engine-specific details handled here:

1. Section IDs are random `sec_<uuid>`s, so after ingest we fetch the document
   tree and build an id -> title_path map. Returned sections are tagged with
   their structural path, which is what the scorer's path matching needs.
   The tree arrives FLAT: `DocumentTree.sections` is a list of `SectionView`
   whose `children` are id strings, not nodes. Walking it as a nested tree
   silently yields nothing, which is how `path_correct_top1` came to read
   0.000 for every system rather than for none.
1b. Since HAL-1390 the leading units are evidence PAGES (`id: page_<n>`),
   which are not tree nodes and so have no id in that map. The engine does
   name the owning section in the unit's title (`"<section> · p.<n>"`), so
   a page is resolved to its owning leaf's path by that title. Titles are
   not guaranteed unique; an ambiguous one resolves to no path rather than
   to a guess, and the count is reported in the trace.
2. Caching would zero out cost and latency. For a fair cold-cache measurement
   the SERVER must run with `retrieval.cache.enabled=false` and the llmgate
   cache middleware off. We record the declared cache mode in the trace and, if
   `cold` is requested but the server cache cannot be guaranteed off, the
   manifest will carry a warning. We do NOT salt queries by default because that
   changes the model's input and contaminates the comparison.

3. A corpus-level question (MultiHop-RAG: the evidence is in 2-4 of 609
   articles and nobody says which) goes to the engine's corpus mode,
   `query_corpus`, over EVERY document this run ingested. The question's
   own gold document ids are never passed: they would tell the engine where
   the answer is. The merged evidence pages come back tagged with their
   engine document, which is mapped back to the dataset's id; coverage, the
   hops and the documents read are kept in the trace.

Cost comes straight from the engine's `usage` field (already priced with the
same table), so Vectorless numbers need no re-estimation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from ..datasets.base import CORPUS
from ..pricing import compute
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

# The engine titles an evidence page "<owning section> · p.<n>"
# (internal/api/server.go, evidencePageSections).
_PAGE_TITLE = re.compile(r"^(?P<section>.*?)\s*·\s*p\.\s*(?P<page>\d+)\s*$")


def _norm(title: str) -> str:
    """Compare titles the way a reader would: case- and space-insensitively."""
    return " ".join(title.split()).casefold()


@dataclass
class _PathIndex:
    """Heading paths for a document, by section id and by leaf title.

    `by_title` exists only to resolve evidence pages, which are not tree
    nodes and carry their owning section's title rather than its id. A
    title shared by two sections cannot resolve to one path, so it is
    dropped rather than guessed, and `ambiguous_titles` records that it
    happened — an unresolvable page is a measurement gap, not a zero.
    """

    by_id: Dict[str, List[str]] = field(default_factory=dict)
    by_title: Dict[str, List[str]] = field(default_factory=dict)
    _title_hits: Dict[str, int] = field(default_factory=dict)
    ambiguous_titles: int = 0

    def add_title(self, title: str, path: List[str]) -> None:
        key = _norm(title)
        if not key:
            return
        self._title_hits[key] = self._title_hits.get(key, 0) + 1
        self.by_title[key] = path

    def finalise(self) -> None:
        dupes = [k for k, n in self._title_hits.items() if n > 1]
        for k in dupes:
            self.by_title.pop(k, None)
        self.ambiguous_titles = len(dupes)

    def resolve(self, section_id: str, title: str) -> List[str]:
        """The heading path for a returned unit, section or page."""
        direct = self.by_id.get(section_id)
        if direct:
            return list(direct)
        m = _PAGE_TITLE.match(title or "")
        if m:
            owner = self.by_title.get(_norm(m.group("section")))
            if owner:
                return list(owner)
        return []


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
        query_timeout: float = 300.0,
        max_hops: Optional[int] = None,
        **_: object,
    ) -> None:
        try:
            from vectorless import VectorlessClient  # type: ignore
        except ImportError as e:  # pragma: no cover - env dependent
            raise ImportError(
                "the 'vectorless' SDK is required for the vectorless system: "
                "pip install vectorless-sdk"
            ) from e

        # A Judge-navigated query is a few requests to a provider whose
        # latency swings between seconds and a minute; the SDK's default
        # 30 s would time out the slow half and score them as errors.
        # max_retries=0: a retried request would hide a failure inside a
        # latency figure. A failure is recorded as a failure.
        self._client = VectorlessClient(api_key=api_key, base_url=base_url, timeout=query_timeout,
                                        max_retries=0)
        self.engine_version = None
        self.model = model
        self.max_tokens = max_tokens
        self.max_parallel_calls = max_parallel_calls
        self.max_hops = max_hops  # corpus mode only; None takes the engine's default (3)
        self.server_cache_disabled = server_cache_disabled
        self.ingest_timeout = ingest_timeout
        self.setup_seconds = 0.0
        self.setup_usage = Usage()  # ingest cost isn't returned per-doc by API
        self._doc_ids: Dict[str, str] = {}  # logical -> server document_id
        self._paths: Dict[str, _PathIndex] = {}  # logical doc -> heading paths

    # -- lifecycle ---------------------------------------------------------
    def setup(self, corpus: List[Doc]) -> None:
        import time

        # Name the build being measured (HAL-1582 / HAL-3067).
        try:
            v = self._client.version()
            self.engine_version = {"version": getattr(v, "version", None) or str(v)}
        except Exception as e:  # pragma: no cover - network
            self.engine_version = {"version": None, "error": str(e)}

        t0 = time.perf_counter()
        for d in corpus:
            # Hand the engine the original PDF when the dataset has one: its
            # page-based pipeline (parser, table of contents, page
            # resolution, judgewalk) only exists for paged documents. The
            # pre-extracted text is what the OTHER systems chunk; giving it
            # to Vectorless would benchmark a different engine.
            if d.path and d.path.lower().endswith(".pdf"):
                from pathlib import Path as _P
                resp = self._client.ingest_document(
                    source=_P(d.path).read_bytes(),
                    filename=f"{d.doc_id}.pdf",
                    content_type="application/pdf",
                )
            else:
                resp = self._client.ingest_document(
                    source=d.content.encode("utf-8"),
                    filename=f"{d.doc_id}.md",
                    content_type=d.content_type,
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

    def _build_path_index(self, server_doc_id: str) -> "_PathIndex":
        """Map section id -> heading path, and section title -> heading path.

        `get_document_tree` returns a FLAT `DocumentTree`: `.sections` is a
        list of `SectionView` with `id`, `parent_id`, `title`, and `children`
        as id STRINGS. There is no `.root`, and `.children` are not nodes, so
        the obvious recursive walk terminates immediately and leaves every
        path empty. Paths are built by walking `parent_id` upwards instead.

        A nested shape (older servers, or a future tree endpoint) is still
        handled, so this does not depend on which one arrives.
        """
        idx = _PathIndex()
        try:
            tree = self._client.get_document_tree(server_doc_id)
        except Exception:  # pragma: no cover - network
            return idx

        flat = list(getattr(tree, "sections", None) or [])
        if flat:
            self._index_flat(flat, idx)
        else:  # pragma: no cover - shape-dependent
            self._index_nested(tree, idx)
        idx.finalise()
        return idx

    @staticmethod
    def _index_flat(flat: List[object], idx: "_PathIndex") -> None:
        """Resolve each section's path by climbing parent_id to the root."""
        nodes = {}
        for s in flat:
            sid = str(getattr(s, "id", "") or "")
            if sid:
                nodes[sid] = s

        def path_of(sid: str, seen: Optional[set] = None) -> Optional[List[str]]:
            # A malformed tree (a parent cycle, or a parent id that is not in
            # the tree) yields None, and the section stays unresolved: a
            # partial path would match fabricated headings.
            seen = seen if seen is not None else set()
            if sid in seen or sid not in nodes:
                return None
            seen.add(sid)
            node = nodes[sid]
            parent = str(getattr(node, "parent_id", "") or "")
            title = (getattr(node, "title", "") or "").strip()
            if parent:
                prefix = path_of(parent, seen)
                if prefix is None:
                    return None
            else:
                prefix = []
            return prefix + [title] if title else prefix

        for sid in nodes:
            p = path_of(sid)
            if p:
                idx.by_id[sid] = p
                idx.add_title(p[-1], p)

    @staticmethod
    def _index_nested(tree: object, idx: "_PathIndex") -> None:  # pragma: no cover
        """Fallback for a genuinely nested tree shape."""
        def walk(node: object, prefix: List[str]) -> None:
            title = (getattr(node, "title", "") or "").strip()
            here = prefix + [title] if title else prefix
            sid = getattr(node, "id", None) or getattr(node, "section_id", None)
            if sid and here:
                idx.by_id[str(sid)] = here
                idx.add_title(here[-1], here)
            for child in getattr(node, "children", None) or []:
                if isinstance(child, str):
                    continue  # an id, not a node — flat shape, handled above
                walk(child, here)

        root = getattr(tree, "root", None) or tree
        for child in getattr(root, "children", None) or []:
            if not isinstance(child, str):
                walk(child, [])

    # -- query -------------------------------------------------------------
    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        if question.doc_id == CORPUS:
            return self._retrieve_corpus(question, k, cold)
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

        path_index = self._paths.get(question.doc_id) or _PathIndex()
        sections: List[RetrievedSection] = []
        unresolved = 0
        # An abstention is a legitimate answer ("nothing confident enough"),
        # scored as zero sections — not a transport error.
        for s in (resp.sections or []):
            sid = str(getattr(s, "id", ""))
            title = getattr(s, "title", "") or ""
            path = path_index.resolve(sid, title)
            if not path:
                unresolved += 1
            sections.append(
                RetrievedSection(
                    content=getattr(s, "content", "") or "",
                    section_id=sid,
                    title=title,
                    title_path=path,
                    page=getattr(s, "page", None),
                    score=getattr(s, "confidence", None),
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
                "abstained": bool(getattr(resp, "abstained", False)),
                # Path metrics are only meaningful when units resolve to a
                # path. A non-zero count here means path_correct_top1 is
                # understated and must not be read as a quality signal.
                "units_without_path": unresolved,
                "ambiguous_titles": path_index.ambiguous_titles,
                # The engine's own split (engine >= 2026-10-16, SDK with Timing).
                "timing": _dump(getattr(resp, "timing", None)),
            },
        )

    def _retrieve_corpus(self, question: Question, k: int, cold: bool) -> RetrievalResult:
        """A question over the whole collection, through corpus mode."""
        if not self._doc_ids:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question,
                                   error="no documents were ingested", cold=cold)
        logical = {v: k_ for k_, v in self._doc_ids.items()}
        try:
            resp = self._client.query_corpus(
                question.question,
                document_ids=list(self._doc_ids.values()),
                max_evidence=k,
                max_hops=self.max_hops,
            )
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question,
                                   error=str(e), cold=cold)
        sections: List[RetrievedSection] = []
        for ev in resp.evidence or []:
            doc = logical.get(ev.document_id, ev.document_id)
            sections.append(RetrievedSection(
                content=ev.text or "",
                section_id=f"{doc}#p{ev.page}",
                title=ev.document_title or ev.section_title or doc,
                page=ev.page or None,
                score=ev.confidence,
            ))
        read = [logical.get(d, d) for h in (resp.hops or []) for d in (h.read or [])]
        return RetrievalResult(
            qid=question.qid,
            system=self.name,
            query=question.question,
            sections=sections,
            usage=self._usage_from(resp),
            latency_ms=float(resp.elapsed_ms or 0),
            strategy=resp.strategy or "corpus",
            cold=cold,
            trace={
                "server_cache_disabled": self.server_cache_disabled,
                "mode": "corpus",
                "coverage": resp.coverage,
                "hops": len(resp.hops or []),
                "documents_considered": resp.documents_considered,
                "documents_read": read,
                "judge_requests": resp.judge_requests,
                "errors": dict(resp.errors or {}),
                "timing": _dump(getattr(resp, "timing", None)),
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


def _dump(obj: object):
    """A pydantic model as a plain dict, or None."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    return dict(obj) if isinstance(obj, dict) else None
