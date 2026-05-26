"""Dataset abstraction: a Dataset yields a corpus (the docs to index) and a list
of questions (with gold anchors). Concrete datasets live alongside this file.

A Dataset is responsible for producing STABLE gold anchors — title-paths,
answer-spans, pages — never engine section IDs. See schema.GoldAnchor.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..schema import Doc, Question


class Dataset(ABC):
    name: str = "dataset"

    @abstractmethod
    def corpus(self) -> List[Doc]:
        """The documents to index (only those referenced by the questions)."""

    @abstractmethod
    def questions(self) -> List[Question]:
        """The evaluation items."""

    def subset(
        self, questions: List[Question], limit: Optional[int], seed: int = 0
    ) -> List[Question]:
        """Deterministic sample of `limit` questions, stratified by domain so a
        small run still spans the domains it should."""
        if not limit or limit >= len(questions):
            return questions
        import random
        from collections import defaultdict

        by_domain = defaultdict(list)
        for q in questions:
            by_domain[q.domain].append(q)
        rng = random.Random(seed)
        for v in by_domain.values():
            rng.shuffle(v)
        out: List[Question] = []
        domains = sorted(by_domain)
        i = 0
        while len(out) < limit:
            d = domains[i % len(domains)]
            if by_domain[d]:
                out.append(by_domain[d].pop())
            elif all(not by_domain[x] for x in domains):
                break
            i += 1
        return out[:limit]


def build_dataset(name: str, **cfg) -> Dataset:
    if name in ("fixtures", "mini"):
        from .fixtures import FixturesDataset

        return FixturesDataset(**cfg)
    if name == "financebench":
        from .financebench import FinanceBenchDataset

        return FinanceBenchDataset(**cfg)
    raise KeyError(f"unknown dataset {name!r}")
