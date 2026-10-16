"""MMLongBench-Doc (Ma et al. 2024): questions over long, visually rich PDFs
(reports, brochures, papers, slides), with evidence pages and the kind of
evidence: pure text, layout, chart, table or image. Measured 2026-10-16 on
the HF dataset: 1,091 questions over 135 PDFs; 244 answers are
"Not answerable".

Gold: the evidence pages (1-based, as the dataset gives them) are page
anchors, and a short answer is also a span on those pages. Scanned and chart
evidence is where text extraction (and so every text-reading system) is
weakest, so results are sliced by evidence source (`domain`), and that slice
is published whatever it shows.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from . import _files
from .base import Dataset

BASE = "https://huggingface.co/datasets/yubo2333/MMLongBench-Doc/resolve/main/"


def _list(v) -> list:
    if isinstance(v, list):
        return v
    try:
        out = ast.literal_eval(str(v))
        return out if isinstance(out, list) else [out]
    except Exception:
        return []


class MMLongBenchDocDataset(Dataset):
    name = "mmlongbench_doc"

    def __init__(self, data_dir: str = "data/mmlongbench_doc", **_: object) -> None:
        self.dir = Path(data_dir)
        self._rows: Optional[List[dict]] = None

    def fetch(self) -> None:
        pq = _files.download(BASE + "data/train-00000-of-00001.parquet", self.dir / "questions.parquet")
        rows = _files.read_parquet_rows(pq)
        docs = sorted({r["doc_id"] for r in rows})
        print(f"mmlongbench_doc: {len(rows)} questions over {len(docs)} PDFs")
        failed = 0
        for d in docs:
            pdf = self.dir / "documents" / d
            try:
                _files.download(BASE + "documents/" + d.replace(" ", "%20"), pdf)
                _files.ensure_text(pdf)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAILED {d}: {e}")
        print(f"mmlongbench_doc: {len(docs) - failed} PDFs ready, {failed} failed")

    def _load(self) -> List[dict]:
        if self._rows is None:
            self._rows = _files.read_parquet_rows(self.dir / "questions.parquet")
        return self._rows

    def corpus(self) -> List[Doc]:
        docs = []
        for d in sorted({r["doc_id"] for r in self._load()}):
            pdf = self.dir / "documents" / d
            txt = pdf.with_suffix(".txt")
            if txt.exists():
                docs.append(Doc(doc_id=d, title=d, content=txt.read_text(encoding="utf-8", errors="ignore"),
                                path=str(pdf), content_type="text/plain"))
        return docs

    def questions(self) -> List[Question]:
        out = []
        for i, r in enumerate(self._load()):
            pages = [int(p) for p in _list(r.get("evidence_pages")) if str(p).isdigit()]
            sources = [str(s) for s in _list(r.get("evidence_sources"))]
            answer = str(r.get("answer", "")).strip()
            na = answer.lower() == "not answerable"
            spans = [answer] if answer and not na and len(answer) <= 120 else []
            gold = [GoldAnchor(doc_id=r["doc_id"], answer_spans=spans, page=p) for p in pages]
            out.append(Question(
                qid=f"mmlb_{i:04d}", doc_id=r["doc_id"], question=r["question"],
                gold=[] if na else gold,
                answer=None if na else answer,
                answer_type=AnswerType.NO_ANSWER if na else (AnswerType.MULTIHOP if len(pages) > 1 else AnswerType.LOOKUP),
                domain=(sources[0].lower().replace(" ", "_") if sources else "unknown"),
                meta={"evidence_sources": sources, "evidence_pages": pages, "doc_type": r.get("doc_type"),
                      "answer_format": r.get("answer_format")},
            ))
        return out
