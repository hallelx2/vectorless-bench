"""RAPTOR (Sarthi et al. 2024): recursive clustered summaries built over
embeddings. It is the closest embedding-based rival to a tree design, so it
runs the OFFICIAL code (github.com/parthsarthi03/raptor, cloned at a pinned
commit) with its model hooks pointed at Vertex: Gemini writes the cluster
summaries and gemini-embedding-2 embeds the chunks and summaries.

Retrieval is the paper's recommended collapsed-tree mode: every node, leaf
chunks and summaries alike, is ranked by similarity to the question, and
the top k nodes' text is returned. The tree build (summaries and
embeddings) is timed and costed as setup.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

from .._llm import DEFAULT_MODEL, _busy, complete_patient, genai_client, require_vertex_model
from ..datasets.base import CORPUS
from ..pricing import compute_embedding, count_tokens
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

REPO = "https://github.com/parthsarthi03/raptor.git"
PIN = os.environ.get("VLBENCH_RAPTOR_COMMIT", "7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767")


def _ensure_repo() -> str:
    path = Path(os.environ.get("VLBENCH_RAPTOR_PATH", Path.home() / ".cache" / "vlbench" / "raptor"))
    if not (path / "raptor").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", REPO, str(path)], check=True)
        subprocess.run(["git", "-C", str(path), "checkout", "-q", PIN], check=False)
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return str(path)


def _embed_patient(call, attempts: int = 8):
    """An embedding call retried on a load refusal with growing pauses, the
    way the summaries are (complete_patient): one refused embedding used to
    fail the filing it belonged to."""
    delay = 2.0
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as e:  # pragma: no cover - network
            if not _busy(e) or attempt == attempts:
                raise
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise RuntimeError("unreachable")


class RaptorRetriever:
    name = "raptor"

    def __init__(self, model: str = DEFAULT_MODEL, embedding_model: str = "gemini-embedding-2",
                 tb_max_tokens: int = 100, **_: object) -> None:
        require_vertex_model(model)
        self.model, self.embedding_model, self.tb_max_tokens = model, embedding_model, tb_max_tokens
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self.setup_meta: Dict[str, object] = {}
        self._ras: Dict[str, object] = {}
        self._lock = threading.Lock()
        # Which Usage an embedding is charged to: per thread, because RAPTOR
        # embeds on the calling thread and the throughput phase runs several
        # questions at once. Unset means setup.
        self._target = threading.local()

    def _models(self):
        _ensure_repo()
        from raptor import BaseEmbeddingModel, BaseQAModel, BaseSummarizationModel  # type: ignore

        outer = self

        class GeminiSummary(BaseSummarizationModel):
            def summarize(self, context, max_tokens=150):
                out = complete_patient(outer.model, "Write a summary of the following, including as many key details as possible.",
                               context, max_tokens=max_tokens)
                with outer._lock:
                    outer.setup_usage.add(out.usage)
                return out.text

        class GeminiEmbedding(BaseEmbeddingModel):
            def create_embedding(self, text):
                from google.genai import types  # type: ignore

                r = _embed_patient(lambda: genai_client().models.embed_content(
                    model=outer.embedding_model, contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=768)))
                toks = count_tokens(text)
                with outer._lock:
                    target = getattr(outer._target, "usage", None) or outer.setup_usage
                    target.embedding_tokens += toks
                    target.cost_usd += compute_embedding(outer.embedding_model, toks)
                return list(r.embeddings[0].values)

        class NoQA(BaseQAModel):  # retrieval only: RAPTOR's QA step is never called
            def answer_question(self, context, question):
                return ""

        return GeminiSummary(), GeminiEmbedding(), NoQA()

    def setup(self, corpus: List[Doc]) -> None:
        summ, emb, qa = self._models()
        from raptor import RetrievalAugmentation, RetrievalAugmentationConfig  # type: ignore

        self.setup_meta = {"raptor_commit": PIN, "failed": []}
        t0 = time.perf_counter()
        for d in corpus:
            cfg = RetrievalAugmentationConfig(summarization_model=summ, qa_model=qa, embedding_model=emb,
                                              tb_max_tokens=self.tb_max_tokens)
            ra = RetrievalAugmentation(config=cfg)
            try:
                ra.add_documents(d.content)
                self._ras[d.doc_id] = ra
            except Exception as e:  # one bad document must not void the run
                self.setup_meta["failed"].append({"doc": d.doc_id, "error": str(e)[:300]})
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:
        pass

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        if question.doc_id == CORPUS:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, cold=cold,
                                   error="n/a: RAPTOR's tree is built per document; this question spans the corpus")
        ra = self._ras.get(question.doc_id)
        if ra is None:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question,
                                   error="document not built", cold=cold)
        usage = Usage()
        t0 = time.perf_counter()
        self._target.usage = usage
        try:
            _, layers = ra.retrieve(question.question, top_k=k, collapse_tree=True, return_layer_information=True)
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, usage=usage,
                                   error=str(e)[:500], cold=cold)
        finally:
            self._target.usage = None
        nodes = ra.tree.all_nodes
        sections = []
        for info in layers[:k]:
            node = nodes.get(info.get("node_index")) if isinstance(nodes, dict) else None
            if node is not None:
                sections.append(RetrievedSection(content=node.text, section_id=f"n{info['node_index']}",
                                                 title=f"layer {info.get('layer_number')}"))
        return RetrievalResult(qid=question.qid, system=self.name, query=question.question, sections=sections,
                               usage=usage, latency_ms=(time.perf_counter() - t0) * 1000.0,
                               strategy="raptor-collapsed-tree", cold=cold)
