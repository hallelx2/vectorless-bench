"""A tiny, in-repo curated dataset: two short documents (finance + medicine) and
a handful of hand-written questions with stable gold anchors.

It is the seed of the "curated in-house golden set" and the corpus the smoke
test + CI run against — small enough to run in seconds with zero external
services (via the mock retriever), but shaped exactly like the real datasets
(structural paths, answer spans, a no-answer item, a near-miss trap).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from .base import Dataset

_ROOT = Path(__file__).resolve().parents[3]


class FixturesDataset(Dataset):
    name = "fixtures"

    def __init__(self, data_dir: Optional[str] = None, **_: object) -> None:
        base = Path(data_dir) if data_dir else _ROOT / "data" / "fixtures"
        self.corpus_dir = base / "corpus"
        self.qa_path = base / "qa.jsonl"

    def corpus(self) -> List[Doc]:
        docs: List[Doc] = []
        for md in sorted(self.corpus_dir.glob("*.md")):
            text = md.read_text(encoding="utf-8")
            title = text.lstrip("# ").splitlines()[0] if text.strip() else md.stem
            docs.append(
                Doc(
                    doc_id=md.stem,
                    title=title,
                    content=text,
                    path=str(md),
                    content_type="text/markdown",
                )
            )
        return docs

    def questions(self) -> List[Question]:
        out: List[Question] = []
        for line in self.qa_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            r = json.loads(line)
            gold = [
                GoldAnchor(
                    doc_id=r["doc_id"],
                    title_path=g.get("title_path", []),
                    answer_spans=g.get("answer_spans", []),
                    page=g.get("page"),
                )
                for g in r.get("gold", [])
            ]
            out.append(
                Question(
                    qid=r["qid"],
                    doc_id=r["doc_id"],
                    question=r["question"],
                    gold=gold,
                    answer=r.get("answer"),
                    answer_type=AnswerType(r.get("answer_type", "lookup")),
                    difficulty=r.get("difficulty", "medium"),
                    domain=r.get("domain", "general"),
                )
            )
        return out
