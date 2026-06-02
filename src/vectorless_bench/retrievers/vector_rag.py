"""Vector-RAG baseline: pgvector + OpenAI embeddings + cosine top-k.

This is the system Vectorless is positioned against, so it is implemented as a
*fair* representative of standard practice, not a strawman:
- standard token-windowed chunks with overlap (see _chunk.py),
- text-embedding-3-small by default (cheap, widely used),
- cosine top-k, with an optional reranker hook,
- ingest embedding cost tracked and reported (the hidden cost of re-indexing).

Two storage backends:
- "pgvector" (default): the real thing — Postgres + the pgvector extension.
- "memory": pure-Python cosine over an in-RAM matrix, so you can get quality
  numbers without standing up Postgres. Both still need an OpenAI key to embed.

Query-time cost is the query-embedding cost (retrieval does no generation); the
one-time index cost is reported separately via setup_usage.
"""

from __future__ import annotations

import math
import os
from typing import List, Optional, Sequence

from ..pricing import compute_embedding, count_tokens
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage
from ._chunk import Chunk, chunk_doc


class VectorRagRetriever:
    name = "vector_rag"

    def __init__(
        self,
        embedding_model: str = "text-embedding-3-small",
        backend: str = "pgvector",
        dsn: Optional[str] = None,
        chunk_tokens: int = 512,
        overlap_tokens: int = 64,
        dim: int = 1536,
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
        self._table = "vlbench_chunks"
        self._openai = None
        self._st = None
        # A "st:<hf-model-id>" embedding_model selects a local
        # sentence-transformers embedder (free, deterministic, no API key) —
        # the no-OpenAI path for a fully self-hosted vector-RAG baseline.
        self._is_local = embedding_model.startswith("st:")
        self._st_model_id = embedding_model[3:] if self._is_local else ""

    # -- embeddings --------------------------------------------------------
    def _client(self):
        if self._openai is None:
            from openai import OpenAI  # type: ignore

            self._openai = OpenAI()
        return self._openai

    def _st_model(self):
        if self._st is None:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._st = SentenceTransformer(self._st_model_id)
            # keep self.dim in sync so the pgvector column matches
            self.dim = int(self._st.get_sentence_embedding_dimension())
        return self._st

    def _encode(self, texts: Sequence[str]) -> List[List[float]]:
        """Embed texts via either a local sentence-transformers model or the
        OpenAI-compatible client, depending on embedding_model."""
        if self._is_local:
            vecs = self._st_model().encode(
                list(texts), normalize_embeddings=True, show_progress_bar=False
            )
            return [list(map(float, v)) for v in vecs]
        resp = self._client().embeddings.create(
            model=self.embedding_model, input=list(texts)
        )
        return [d.embedding for d in resp.data]

    def _embed(self, texts: Sequence[str]) -> List[List[float]]:
        out = self._encode(texts)
        toks = sum(count_tokens(t, self.embedding_model) for t in texts)
        self.setup_usage.embedding_tokens += toks
        # local embeddings are free; compute_embedding returns 0 for unpriced
        self.setup_usage.cost_usd += compute_embedding(self.embedding_model, toks)
        return out

    # -- lifecycle ---------------------------------------------------------
    def setup(self, corpus: List[Doc]) -> None:
        import time

        t0 = time.perf_counter()
        for d in corpus:
            self._chunks.extend(
                chunk_doc(d, self.chunk_tokens, self.overlap_tokens)
            )
        # batch embed
        batch = 128
        vectors: List[List[float]] = []
        for i in range(0, len(self._chunks), batch):
            vectors.extend(self._embed([c.text for c in self._chunks[i : i + batch]]))

        if self.backend == "pgvector":
            self._pg_setup(vectors)
        else:
            self._vectors = vectors
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
            for ch, vec in zip(self._chunks, vectors):
                cur.execute(
                    f"INSERT INTO {self._table} (doc_id, chunk_id, text, embedding) "
                    f"VALUES (%s, %s, %s, %s)",
                    (ch.doc_id, ch.chunk_id, ch.text, vec),
                )
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
        # query embedding cost is tiny; don't fold it into ingest usage
        return self._encode([q])[0]

    def _pg_query(self, qvec, doc_id, k):
        # the cosine operator (<=>) appears in both SELECT and ORDER BY, so the
        # query vector is bound twice; doc filter is bound between them.
        where = "WHERE doc_id = %s" if self.per_doc else ""
        params = [qvec] + ([doc_id] if self.per_doc else []) + [qvec, k]
        sql = (
            f"SELECT chunk_id, text, 1 - (embedding <=> %s) AS score "
            f"FROM {self._table} {where} ORDER BY embedding <=> %s LIMIT %s"
        )
        with self._conn.cursor() as cur:
            cur.execute(sql, params)
            return [(r[0], r[1], float(r[2])) for r in cur.fetchall()]

    def _mem_query(self, qvec, doc_id, k):
        def cos(a, b):
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else 0.0

        idxs = range(len(self._chunks))
        if self.per_doc:
            idxs = [i for i in idxs if self._chunks[i].doc_id == doc_id]
        scored = [(i, cos(qvec, self._vectors[i])) for i in idxs]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [
            (self._chunks[i].chunk_id, self._chunks[i].text, s)
            for i, s in scored[:k]
        ]
