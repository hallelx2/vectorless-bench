"""Dataset abstraction: a Dataset yields a corpus (the docs to index) and a list
of questions (with gold anchors). Concrete datasets live alongside this file.

A Dataset is responsible for producing STABLE gold anchors — title-paths,
answer-spans, pages — never engine section IDs. See schema.GoldAnchor.

Every dataset also knows how to `fetch` its own data (questions and source
documents) and how to `check` itself: the measured counts and gold-evidence
validity a run's numbers rest on (`vlbench datasets check <name>`).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..anchors import span_present
from ..schema import Doc, Question

# Questions whose evidence spans several documents use this doc_id; the
# documents are listed in `Question.meta["doc_ids"]` and retrieval runs over
# the whole corpus.
CORPUS = "__corpus__"


class Dataset(ABC):
    name: str = "dataset"

    @abstractmethod
    def corpus(self) -> List[Doc]:
        """The documents to index (only those referenced by the questions)."""

    @abstractmethod
    def questions(self) -> List[Question]:
        """The evaluation items."""

    def fetch(self) -> None:
        """Download this dataset's questions and documents. Idempotent."""
        raise NotImplementedError(f"{self.name} has no fetch; see its module docstring")

    def questions_for_available_corpus(self) -> List[Question]:
        """Questions whose documents are present, so a partial download is a
        smaller valid run rather than a run full of errors."""
        docs = {d.doc_id: d for d in self.corpus()}
        out = []
        for q in self.questions():
            if q.doc_id == CORPUS:
                if all(d in docs for d in q.meta.get("doc_ids", [])):
                    out.append(q)
            elif q.doc_id in docs:
                out.append(q)
        for q in out:
            q.meta["gold_reachable"] = gold_reachable(q, docs)
        return out

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

    def check(self) -> Dict[str, Any]:
        """Measured facts about the loaded data: counts, documents present,
        and whether the gold evidence is really where the gold says.

        For every anchor with spans: is a span found anywhere in its document
        (span_found)? For every anchor with a page, on a document with
        page-separated text: is a span on that page (page_ok), or on the page
        before or after (page_off_by_one, the 0- vs 1-based trap)?"""
        from ._files import PAGE_BREAK

        all_q = self.questions()
        docs = {d.doc_id: d for d in self.corpus()}
        avail = self.questions_for_available_corpus()
        types = Counter(q.answer_type.value for q in all_q)
        needed = set()
        for q in all_q:
            needed.update(q.meta.get("doc_ids", []) if q.doc_id == CORPUS else [q.doc_id])
        anchors = span_anchors = span_found = paged = page_ok = off_by_one = 0
        page_text_anchors = page_text_matchable = 0
        for q in avail:
            for g in q.gold:
                anchors += 1
                d = docs.get(g.doc_id)
                if d is None:
                    continue
                if g.answer_spans:
                    span_anchors += 1
                    if any(span_present(sp, d.content) for sp in g.answer_spans):
                        span_found += 1
                if g.page_text or (g.page is not None and not g.answer_spans):
                    page_text_anchors += 1
                    from ..anchors import page_text_overlaps

                    if g.page_text and page_text_overlaps(g.page_text, g.page_text):
                        page_text_matchable += 1
                    continue
                if g.page is not None and PAGE_BREAK in d.content:
                    pages = d.content.split(PAGE_BREAK)
                    paged += 1

                    def on(p: int) -> bool:
                        return 1 <= p <= len(pages) and any(span_present(sp, pages[p - 1]) for sp in g.answer_spans)

                    if on(g.page):
                        page_ok += 1
                    elif on(g.page - 1) or on(g.page + 1):
                        off_by_one += 1
        pct = lambda a, b: round(100.0 * a / b, 1) if b else None  # noqa: E731
        return {
            "dataset": self.name,
            "questions": len(all_q),
            "questions_with_docs": len(avail),
            "documents_needed": len(needed),
            "documents_present": len([d for d in needed if d in docs]),
            "answer_types": dict(types),
            "corpus_level_questions": sum(1 for q in all_q if q.doc_id == CORPUS),
            "gold_anchors": anchors,
            "span_found_pct": pct(span_found, span_anchors),
            "page_anchors_checked": paged,
            "page_ok_pct": pct(page_ok, paged),
            "page_off_by_one_pct": pct(off_by_one, paged),
            "page_text_anchors": page_text_anchors,
            "page_text_matchable_pct": pct(page_text_matchable, page_text_anchors),
            "corpus_chars": sum(len(d.content) for d in docs.values()),
        }


def gold_reachable(q: Question, docs: Dict[str, Doc]) -> bool:
    """Can any system that reads our extracted text find this question's
    gold at all? True when at least one anchor matches its own document's
    text: a span present, a heading path given, or a gold page with text.
    A question whose gold evidence is a figure, or a passage the PDF
    extraction mangled, is a miss for every system alike; the report scores
    such questions in the full set and also gives the reachable subset, so
    the comparison is not diluted by questions no one can win."""
    from ..anchors import page_text_overlaps, span_present

    if q.is_no_answer:
        return True
    for g in q.gold:
        d = docs.get(g.doc_id)
        if d is None:
            continue
        if g.title_path:
            return True
        if g.answer_spans and any(span_present(sp, d.content) for sp in g.answer_spans):
            return True
        if g.page_text and page_text_overlaps(g.page_text, g.page_text):
            return True
    return False


DATASETS = ["fixtures", "financebench", "qasper", "cuad", "legalbench_rag", "mmlongbench_doc", "multihop_rag", "manuals"]


def available() -> List[str]:
    return list(DATASETS)


def build_dataset(name: str, **cfg) -> Dataset:
    if name in ("fixtures", "mini"):
        from .fixtures import FixturesDataset

        return FixturesDataset(**cfg)
    if name == "financebench":
        from .financebench import FinanceBenchDataset

        return FinanceBenchDataset(**cfg)
    if name == "qasper":
        from .qasper import QasperDataset

        return QasperDataset(**cfg)
    if name == "cuad":
        from .cuad import CuadDataset

        return CuadDataset(**cfg)
    if name == "legalbench_rag":
        from .legalbench_rag import LegalBenchRagDataset

        return LegalBenchRagDataset(**cfg)
    if name == "mmlongbench_doc":
        from .mmlongbench import MMLongBenchDocDataset

        return MMLongBenchDocDataset(**cfg)
    if name == "multihop_rag":
        from .multihop_rag import MultiHopRagDataset

        return MultiHopRagDataset(**cfg)
    if name == "manuals":
        from .manuals import ManualsDataset

        return ManualsDataset(**cfg)
    raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
