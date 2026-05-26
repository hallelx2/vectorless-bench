"""Name -> Retriever factory.

Retrievers are imported lazily inside the factory so that a missing optional
dependency (psycopg, openai, rank_bm25, the vectorless SDK, pageindex) only
fails if you actually ask for that system. Listing or running the mock never
imports a heavy dependency.
"""

from __future__ import annotations

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
register("bm25", _bm25)
register("full_context", _full_context)
register("pageindex", _pageindex)
