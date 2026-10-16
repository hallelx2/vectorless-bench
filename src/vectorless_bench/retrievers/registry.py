"""Name -> Retriever factory.

Retrievers are imported lazily inside the factory so that a missing optional
dependency (psycopg, google-genai, rank_bm25, the vectorless SDK, pageindex) only
fails if you actually ask for that system. Listing or running the mock never
imports a heavy dependency.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List

from .base import Retriever

_FACTORIES: Dict[str, Callable[..., Retriever]] = {}


def register(name: str, factory: Callable[..., Retriever]) -> None:
    _FACTORIES[name] = factory


def available() -> List[str]:
    return sorted(_FACTORIES)


def build(name: str, **cfg: Any) -> Retriever:
    if name not in _FACTORIES:
        raise KeyError(
            f"unknown system {name!r}; available: {', '.join(available())}"
        )
    return _FACTORIES[name](**cfg)


def _mock(**cfg: Any) -> Retriever:
    from .mock import MockRetriever

    return MockRetriever(**cfg)


def _vectorless(**cfg: Any) -> Retriever:
    from .vectorless import VectorlessRetriever

    return VectorlessRetriever(**cfg)


def _vector_rag(**cfg: Any) -> Retriever:
    from .vector_rag import VectorRagRetriever

    return VectorRagRetriever(**cfg)


def _bm25(**cfg: Any) -> Retriever:
    from .bm25 import BM25Retriever

    return BM25Retriever(**cfg)


def _full_context(**cfg: Any) -> Retriever:
    from .full_context import FullContextRetriever

    return FullContextRetriever(**cfg)


def _pageindex(**cfg: Any) -> Retriever:
    from .pageindex import PageIndexRetriever

    return PageIndexRetriever(**cfg)


register("mock", _mock)
register("vectorless", _vectorless)
register("vector_rag", _vector_rag)


def _vector_bge(**cfg: Any) -> Retriever:
    """The open-weights baseline: the same pipeline with a local BGE model on
    CPU (no GPU, no network, zero embedding spend)."""
    from .vector_rag import VectorRagRetriever

    cfg.setdefault("embedding_model", os.environ.get("VLBENCH_BGE_MODEL", "BAAI/bge-base-en-v1.5"))
    r = VectorRagRetriever(**cfg)
    r.name = "vector_bge"
    return r


register("vector_bge", _vector_bge)
register("bm25", _bm25)
register("full_context", _full_context)
register("pageindex", _pageindex)


def _agentic_grep(**cfg: Any) -> Retriever:
    from .agentic_grep import AgenticGrepRetriever

    return AgenticGrepRetriever(**cfg)


register("agentic_grep", _agentic_grep)


def _memwalker(**cfg: Any) -> Retriever:
    from .memwalker import MemWalkerRetriever

    return MemWalkerRetriever(**cfg)


def _readagent(**cfg: Any) -> Retriever:
    from .readagent import ReadAgentRetriever

    return ReadAgentRetriever(**cfg)


def _raptor(**cfg: Any) -> Retriever:
    from .raptor import RaptorRetriever

    return RaptorRetriever(**cfg)


register("memwalker", _memwalker)
register("readagent", _readagent)
register("raptor", _raptor)
