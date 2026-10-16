"""Datasets: corpus + gold-anchored questions. Use `build_dataset(name, ...)`."""

from __future__ import annotations

from .base import available, Dataset, build_dataset

__all__ = ["Dataset", "build_dataset"]
from .base import CORPUS, available, build_dataset  # noqa: F401,E402
