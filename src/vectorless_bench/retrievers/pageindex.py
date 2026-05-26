"""PageIndex baseline — wired to the REAL upstream code (VectifyAI/PageIndex).

PageIndex is the closest structural-retrieval competitor and the reference
implementation Vectorless is positioned against, so we run *their* code rather
than a reimplementation:

- Tree building uses their actual functions: `page_index(pdf_path, ...)` for PDFs
  and `md_to_tree(md_path, ...)` for markdown (returns {doc_name, structure:[...]}).
- Retrieval follows PageIndex's own reasoning pattern (their agentic demo + the
  `get_document_structure` / `get_page_content` tools): show the LLM the tree
  outline with text stripped, let it select the relevant node(s), then return
  those nodes' text. We drive that selection with the shared, priced LLM client
  so PageIndex's query cost is measured on the exact same basis as Vectorless.

PageIndex is not on PyPI — it's clone + `pip install -r requirements.txt`. Point
this adapter at the checkout via `repo_path=` or `VLBENCH_PAGEINDEX_PATH`; the
Docker bundle clones it to /opt/PageIndex and sets that automatically.

Modes:
- "oss"  (default): self-hosted. Their tree builder + reasoning over the tree.
- "cloud": their hosted PageIndexClient (`index` + `get_document_structure` +
  `get_page_content`) with the same reasoning loop.

Note: PageIndex's tree builders call an LLM internally and don't return token
usage, so PageIndex's *ingest* cost isn't captured (recorded as 0 with a flag);
query cost is fully measured.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from typing import Dict, List, Optional

from .._llm import complete
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

_NAV_SYSTEM = (
    "You are PageIndex's reasoning retriever. You are given a document's tree "
    "outline (node ids, heading paths, summaries — no body text) and a question. "
    "Decide which node(s) to open to answer it, the way an expert skims a table "
    "of contents then jumps to the right section. Prefer the most specific nodes; "
    "return as few as possible. Respond as JSON: {\"node_ids\": [\"...\"]}."
)


class PageIndexRetriever:
    name = "pageindex"

    def __init__(
        self,
        mode: str = "oss",
        model: str = "gpt-4o-mini",            # tree-build model (their funcs)
        retrieve_model: Optional[str] = None,  # reasoning/selection model
        repo_path: Optional[str] = None,
        api_key: Optional[str] = None,
        node_summaries: str = "yes",
        **_: object,
    ) -> None:
        self.mode = mode
        self.model = model
        self.retrieve_model = retrieve_model or model
        self.repo_path = repo_path or os.environ.get("VLBENCH_PAGEINDEX_PATH")
        self.api_key = api_key or os.environ.get("PAGEINDEX_API_KEY")
        self.node_summaries = node_summaries
        self.setup_seconds = 0.0
        self.setup_usage = Usage(llm_calls=0)  # upstream doesn't surface usage
        self._client = None
        self._trees: Dict[str, dict] = {}
        self._nodes: Dict[str, Dict[str, dict]] = {}  # logical -> {node_id: node}
        self._cloud_docs: Dict[str, str] = {}

    # -- lifecycle ---------------------------------------------------------
    def setup(self, corpus: List[Doc]) -> None:
        self._ensure_importable()
        t0 = time.perf_counter()
        if self.mode == "cloud":
            from pageindex import PageIndexClient  # type: ignore

            self._client = PageIndexClient(
                api_key=self.api_key, model=self.model,
                retrieve_model=self.retrieve_model,
            )
        for d in corpus:
            tree = self._build_tree(d)
            self._trees[d.doc_id] = tree
            self._nodes[d.doc_id] = self._flatten(tree.get("structure", []))
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:
        pass

    def _ensure_importable(self) -> None:
        if self.repo_path and self.repo_path not in sys.path:
            sys.path.insert(0, self.repo_path)
        try:
            import pageindex  # noqa: F401  type: ignore
        except ImportError as e:
            raise ImportError(
                "PageIndex isn't importable. Clone VectifyAI/PageIndex and "
                "`pip install -r requirements.txt`, then set repo_path= or "
                "VLBENCH_PAGEINDEX_PATH to the checkout (the Docker bundle does "
                "this for you)."
            ) from e

    # -- tree building (their real functions) ------------------------------
    def _build_tree(self, d: Doc) -> dict:
        if self.mode == "cloud":
            path = d.path or self._write_temp(d)
            doc_id = self._client.index(path, mode="auto")
            self._cloud_docs[d.doc_id] = doc_id
            return json.loads(self._client.get_document_structure(doc_id))

        from pageindex import md_to_tree, page_index  # type: ignore

        if d.path and d.path.lower().endswith(".pdf"):
            return page_index(
                d.path, model=self.model, if_add_node_id="yes",
                if_add_node_summary=self.node_summaries, if_add_node_text="yes",
            )
        # markdown/text: md_to_tree is async and reads from a path on disk
        md_path = (
            d.path if (d.path and d.path.lower().endswith(".md"))
            else self._write_temp(d, ".md")
        )
        return asyncio.run(
            md_to_tree(
                md_path, model=self.model, if_add_node_id="yes",
                if_add_node_summary=self.node_summaries, if_add_node_text="yes",
            )
        )

    def _write_temp(self, d: Doc, suffix: str = ".md") -> str:
        f = tempfile.NamedTemporaryFile("w", suffix=suffix, delete=False, encoding="utf-8")
        f.write(d.content)
        f.close()
        return f.name

    def _flatten(self, structure: list) -> Dict[str, dict]:
        nodes: Dict[str, dict] = {}

        def walk(node: dict, prefix: List[str]) -> None:
            title = node.get("title", "") or ""
            here = prefix + [title] if title else prefix
            nid = str(node.get("node_id", len(nodes)))
            node["_path"] = here
            nodes[nid] = node
            for ch in node.get("nodes", []) or []:
                walk(ch, here)

        for root in structure or []:
            walk(root, [])
        return nodes

    def _node_text(self, doc_id: str, node: dict) -> str:
        if node.get("text"):
            return node["text"]
        # cloud structure has no text -> fetch the node's pages on demand
        if self.mode == "cloud" and self._client is not None:
            start, end = node.get("start_index"), node.get("end_index")
            if start is not None:
                pages = f"{start}-{end}" if end and end != start else str(start)
                try:
                    payload = json.loads(
                        self._client.get_page_content(self._cloud_docs[doc_id], pages)
                    )
                    return "\n".join(p.get("content", "") for p in payload)
                except Exception:
                    pass
        return node.get("summary") or node.get("prefix_summary") or ""

    # -- retrieval (PageIndex's reasoning pattern, priced on our table) -----
    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        nodes = self._nodes.get(question.doc_id)
        if not nodes:
            return RetrievalResult(
                qid=question.qid, system=self.name, query=question.question,
                error=f"no tree for {question.doc_id!r}", cold=cold,
            )
        outline = "\n".join(
            f"{nid}\t{' > '.join(n.get('_path', []))}\t"
            f"{(n.get('summary') or n.get('prefix_summary') or '')[:160]}"
            for nid, n in nodes.items()
        )
        t0 = time.perf_counter()
        try:
            out = complete(
                self.retrieve_model, _NAV_SYSTEM,
                f"QUESTION: {question.question}\n\nOUTLINE (id\\tpath\\tsummary):\n{outline}",
                max_tokens=512, json_mode=True,
            )
        except Exception as e:  # pragma: no cover - network/env dependent
            return RetrievalResult(
                qid=question.qid, system=self.name, query=question.question,
                error=str(e), cold=cold,
            )
        latency = (time.perf_counter() - t0) * 1000.0
        try:
            chosen = json.loads(out.text).get("node_ids", [])[:k]
        except json.JSONDecodeError:
            chosen = []
        sections = [
            RetrievedSection(
                content=self._node_text(question.doc_id, nodes[nid]),
                section_id=str(nid),
                title=nodes[nid].get("title", ""),
                title_path=nodes[nid].get("_path", []),
            )
            for nid in chosen
            if nid in nodes
        ]
        return RetrievalResult(
            qid=question.qid, system=self.name, query=question.question,
            sections=sections, usage=out.usage, latency_ms=latency,
            strategy=f"pageindex-{self.mode}", cold=cold,
            trace={"ingest_cost_measured": False},
        )
