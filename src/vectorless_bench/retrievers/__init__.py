"""Retriever implementations + the name->factory registry.

Import `build`, `available`, `register` from here. Concrete retrievers are
imported lazily by the registry so optional dependencies stay optional.
"""

from __future__ import annotations

from .base import Retriever
from .registry import available, build, register

__all__ = ["Retriever", "build", "available", "register"]
