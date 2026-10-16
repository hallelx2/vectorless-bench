"""Vector-RAG baseline: pgvector + Gemini (Vertex) or local BGE embeddings + cosine top-k.

This is the system Vectorless is positioned against, so it is implemented as a
*fair* representative of standard practice, not a strawman:
- standard token-windowed chunks with overlap (see _chunk.py),
- gemini-embedding-2 on Vertex by default, or a local BGE model (BAAI/bge-m3),
- cosine top-k, with an optional reranker hook,
- ingest embedding cost tracked and reported (the hidden cost of re-indexing).

Two storage backends:
- "pgvector" (default): the real thing — Postgres + the pgvector extension.
- "memory": pure-Python cosine over an in-RAM matrix, so you can get quality
  numbers without standing up Postgres.

Query-time cost is the query-embedding cost (retrieval does no generation); the
one-time index cost is reported separately via setup_usage.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence

from ..datasets.base import CORPUS
from ..pricing import compute_embedding, count_tokens
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage
from ._chunk import Chunk, chunk_doc


class VectorRagRetriever:
    name = "vector_rag"

    def __init__(
        self,
        embedding_model: str = "gemini-embedding-2",
        backend: str = "pgvector",
        dsn: Optional[str] = None,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        dim: int = 768,
        per_doc: bool = True,
        reranker: Optional[str] = None,
        **_: object,
    ) -> None:
        self.embedding_model = embedding_model
        self.backend = backend
        self.dsn = dsn or os.environ.get("VLBENCH_PG_DSN")
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.dim = dim
        self.per_doc = per_doc
        self.reranker = reranker
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._chunks: List[Chunk] = []
        self._vectors: List[List[float]] = []  # memory backend
        self._conn = None
        # one table per embedding model, so two vector systems in one run
        # never read each other's rows
        import re as _re

        self._table = "vlbench_chunks_" + _re.sub(r"[^a-z0-9]+", "_", self.embedding_model.lower()).strip("_")[:40]
        self._matrix = None  # memory backend: normalised numpy matrix

    # -- embeddings --------------------------------------------------------

    def _cached_embed(self, texts: List[str]) -> List[List[float]]:
        import hashlib
        import json as _json
        from pathlib import Path

        key = hashlib.sha256((self.embedding_model + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:24]
        cache = Path("data/cache") / f"emb-{self.embedding_model.replace('/', '_')}-{key}.json"
        if cache.exists():
            self.setup_meta = {"embedding_cache": "hit", "cache_file": str(cache)}
            return _json.loads(cache.read_text())
        batch = 128
        vectors: List[List[float]] = []
        for i in range(0, len(texts), batch):
            vectors.extend(self._embed(texts[i : i + batch]))
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(_json.dumps(vectors))
        self.setup_meta = {"embedding_cache": "miss", "cache_file": str(cache)}
        return vectors

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        toks = sum(count_tokens(t, self.embedding_model) for t in texts)
        self.setup_usage.embedding_tokens += toks
        self.setup_usage.cost_usd += compute_embedding(self.embedding_model, toks)
        if self.embedding_model.startswith("gemini-embedding"):
            return self._embed_gemini(texts)
        if "/" in self.embedding_model:  # a Hugging Face id: run it locally
            return self._embed_local(texts)
        raise ValueError(
            f"embedding model {self.embedding_model!r} is neither Gemini on Vertex "
            "nor a local Hugging Face model; this benchmark uses only those"
        )

    def _embed_local(self, texts: Sequence[str]) -> List[List[float]]:
        """A local sentence-transformers model — BAAI/bge-m3 for the full
        benchmark (1024 dimensions, the strong open multilingual retriever). No
        network in the baseline's numbers, cost zero by construction, setup
        time measured and reported like every other system's."""
        from sentence_transformers import SentenceTransformer  # type: ignore

        if getattr(self, "_local", None) is None:
            self._local = SentenceTransformer(self.embedding_model)
        threads = int(os.environ.get("VLBENCH_TORCH_THREADS", "0"))
        if threads:
            import torch  # type: ignore

            torch.set_num_threads(threads)
        vecs = self._local.encode(list(texts), batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vecs]

    def _embed_gemini(self, texts: Sequence[str]) -> List[List[float]]:
        """Gemini embeddings on Vertex AI. Vertex's embedContent takes ONE
        content per request for gemini-embedding-2 (measured 2026-10-16:
        "only supports one content at a time"; the AI Studio endpoint took
        a list), so texts go one per request, in parallel, retried with
        backoff on a rate limit. 768 dimensions, one of the three Google
        recommends; cosine ranking is unaffected. Order is preserved."""
        import time
        from concurrent.futures import ThreadPoolExecutor

        from google.genai import types  # type: ignore

        from .._llm import genai_client

        client = genai_client()
        cfg = types.EmbedContentConfig(output_dimensionality=768)

        def one(text: str) -> List[float]:
            delay = 1.0
            for attempt in range(8):
                try:
                    r = client.models.embed_content(model=self.embedding_model, contents=text, config=cfg)
                    return list(r.embeddings[0].values)
                except Exception as e:  # pragma: no cover - network
                    msg = str(e)
                    if attempt == 7 or not ("429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg):
                        raise
                    time.sleep(delay)
                    delay = min(delay * 2, 30.0)
            raise RuntimeError("unreachable")

        workers = int(os.environ.get("VLBENCH_EMBED_CONCURRENCY", "16"))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, list(texts)))

    def setup(self, corpus: List[Doc]) -> None:
        import time

        t0 = time.perf_counter()
        for d in corpus:
            self._chunks.extend(
                chunk_doc(d, self.chunk_tokens, self.overlap_tokens)
            )
        # batch embed
        # Embeddings are cached on disk keyed by model and chunk text, so a
        # re-run of the bench does not repeat a CPU-hours ingest. The
        # first, uncached pass is the one whose setup_seconds is reported.
        vectors = self._cached_embed([c.text for c in self._chunks])

        if vectors:
            self.dim = len(vectors[0])  # the model decides: bge-small 384, gemini 768, bge-m3 1024
        if self.backend == "pgvector":
            self._pg_setup(vectors)
        else:
            import numpy as np

            m = np.asarray(vectors, dtype=np.float32)
            norms = np.linalg.norm(m, axis=1, keepdims=True)
            self._matrix = m / np.where(norms == 0, 1, norms)
            self._doc_ids = np.asarray([c.doc_id for c in self._chunks])
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:
        if self._conn is not None:
            try:
                with self._conn.cursor() as cur:
                    cur.execute(f"DROP TABLE IF EXISTS {self._table}")
                self._conn.commit()
                self._conn.close()
            except Exception:
                pass

    def _pg_setup(self, vectors: List[List[float]]) -> None:
        import psycopg  # type: ignore
        from pgvector.psycopg import register_vector  # type: ignore

        if not self.dsn:
            raise RuntimeError(
                "vector_rag pgvector backend needs a DSN (set VLBENCH_PG_DSN "
                "or pass dsn=...), or use backend='memory'"
            )
        self._conn = psycopg.connect(self.dsn)
        with self._conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            register_vector(self._conn)
            cur.execute(f"DROP TABLE IF EXISTS {self._table}")
            cur.execute(
                f"CREATE TABLE {self._table} (id serial primary key, "
                f"doc_id text, chunk_id text, text text, embedding vector({self.dim}))"
            )
            import numpy as np

            with cur.copy(f"COPY {self._table} (doc_id, chunk_id, text, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
                cp.set_types(["text", "text", "text", "vector"])
                for ch, vec in zip(self._chunks, vectors):
                    cp.write_row((ch.doc_id, ch.chunk_id, ch.text, np.asarray(vec, dtype=np.float32)))
            # exact search: a per-document filter leaves a few hundred rows,
            # and an approximate index would cost the baseline recall it
            # would not have in a tuned deployment of this size
            cur.execute(f"CREATE INDEX ON {self._table} (doc_id)")
        self._conn.commit()

    # -- query -------------------------------------------------------------
    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        import time

        t0 = time.perf_counter()
        q_tokens = count_tokens(question.question, self.embedding_model)
        try:
            qvec = self._embed_query(question.question)
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(
                qid=question.qid, system=self.name,
                query=question.question, error=str(e),
            )
        if self.backend == "pgvector":
            rows = self._pg_query(qvec, question.doc_id, k)
        else:
            rows = self._mem_query(qvec, question.doc_id, k)
        latency = (time.perf_counter() - t0) * 1000.0

        usage = Usage(
            embedding_tokens=q_tokens,
            cost_usd=compute_embedding(self.embedding_model, q_tokens),
        )
        sections = [
            RetrievedSection(content=text, section_id=cid, score=score)
            for (cid, text, score) in rows
        ]
        return RetrievalResult(
            qid=question.qid,
            system=self.name,
            query=question.question,
            sections=sections,
            usage=usage,
            latency_ms=latency,
            strategy=f"top-{k}",
            cold=cold,
        )

    def _embed_query(self, q: str) -> List[float]:
        # Same provider as setup — local, Gemini or OpenAI — so a local
        # model never reaches for an OpenAI client at query time. The
        # query's tokens are priced by the caller, not folded into ingest.
        before = self.setup_usage.embedding_tokens, self.setup_usage.cost_usd
        vec = self._embed([self._query_text(q)])[0]
        self.setup_usage.embedding_tokens, self.setup_usage.cost_usd = before
        return vec

    # BGE v1.5 English models are trained with this instruction on the query
    # side only (BAAI's model card); without it their retrieval drops.
    _BGE_QUERY = "Represent this sentence for searching relevant passages: "

    def _query_text(self, q: str) -> str:
        m = self.embedding_model.lower()
        if m.startswith("baai/bge-") and m.endswith("-v1.5") and "-en" in m:
            return self._BGE_QUERY + q
        return q

    def _pg_query(self, qvec, doc_id, k):
        # the cosine operator (<=>) appears in both SELECT and ORDER BY, so the
        # query vector is bound twice; doc filter is bound between them.
        scoped = self.per_doc and doc_id != CORPUS  # a corpus-level question searches every chunk
        where = "WHERE doc_id = %s" if scoped else ""
        params = [qvec] + ([doc_id] if scoped else []) + [qvec, k]
        sql = (
            f"SELECT chunk_id, text, 1 - (embedding <=> %s) AS score "
            f"FROM {self._table} {where} ORDER BY embedding <=> %s LIMIT %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]

    def _mem_query(self, qvec, doc_id, k):
        import numpy as np

        if self._matrix is None or not len(self._matrix):
            return []
        q = np.asarray(qvec, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        idx = np.arange(len(self._chunks))
        if self.per_doc and doc_id != CORPUS:
            idx = idx[self._doc_ids == doc_id]
        scores = self._matrix[idx] @ q
        top = idx[np.argsort(-scores)[:k]]
        return [(self._chunks[i].chunk_id, self._chunks[i].text, float(self._matrix[i] @ q)) for i in top]
