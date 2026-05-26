"""FinanceBench loader (Islam et al. 2023 / Patronus AI open subset, 150 Qs).

FinanceBench is a near-ideal fit: real questions over real 10-Ks/10-Qs, with
human-curated evidence (text + page + source document). We map each item to:

    answer_spans <- evidence_text     (verbatim, robust span matching)
    page         <- evidence page number
    doc_id       <- doc_name          (logical key; resolved to the source text)
    answer       <- gold answer        (for the optional LLM-judge axis)

The QA rows come from HuggingFace `datasets`. The DOCUMENTS (large PDFs) are not
shipped with the QA set, so run `scripts/download_financebench.py` first to fetch
+ extract them into `docs_dir`. Questions whose document text is missing are
skipped with a warning, so a partial corpus still produces a valid (smaller) run.

Field names in the HF dataset have shifted across releases; the loader probes a
few aliases for each field so it survives minor schema drift.
"""

from __future__ import annotations

import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from .base import Dataset

_NUMERIC = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?%?")


def _first(d: Dict[str, Any], *keys: str, default=None):
    for k in keys:
        if k in d and d[k] not in (None, ""):
            return d[k]
    return default


def _answer_type(answer: str, qtype: str) -> AnswerType:
    qt = (qtype or "").lower()
    if "novel" in qt and "generated" in qt:
        # FinanceBench marks some items as not answerable from the doc alone
        return AnswerType.MULTIHOP
    if answer and _NUMERIC.fullmatch(answer.strip().replace(" ", "")):
        return AnswerType.NUMERIC
    return AnswerType.LOOKUP


class FinanceBenchDataset(Dataset):
    name = "financebench"

    def __init__(
        self,
        hf_name: str = "PatronusAI/financebench",
        split: str = "train",
        docs_dir: Optional[str] = None,
        **_: object,
    ) -> None:
        self.hf_name = hf_name
        self.split = split
        self.docs_dir = Path(
            docs_dir or os.environ.get("VLBENCH_FINANCEBENCH_DOCS", "data/financebench/docs")
        )
        self._rows: Optional[List[Dict[str, Any]]] = None
        self._needed_docs: set[str] = set()

    def _load_rows(self) -> List[Dict[str, Any]]:
        if self._rows is not None:
            return self._rows
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as e:
            raise ImportError(
                "FinanceBench needs the 'datasets' package: pip install datasets"
            ) from e
        ds = load_dataset(self.hf_name, split=self.split)
        self._rows = [dict(r) for r in ds]
        return self._rows

    # -- questions ---------------------------------------------------------
    def questions(self) -> List[Question]:
        rows = self._load_rows()
        out: List[Question] = []
        for r in rows:
            qid = str(_first(r, "financebench_id", "id", default=f"fb_{len(out)}"))
            doc_name = str(_first(r, "doc_name", "document", default="")).strip()
            question = str(_first(r, "question", default="")).strip()
            answer = str(_first(r, "answer", default="")).strip()
            qtype = str(_first(r, "question_type", "question_reasoning", default=""))
            if not (doc_name and question):
                continue
            self._needed_docs.add(doc_name)
            out.append(
                Question(
                    qid=qid,
                    doc_id=doc_name,
                    question=question,
                    gold=self._anchors(r, doc_name, answer),
                    answer=answer or None,
                    answer_type=_answer_type(answer, qtype),
                    domain="finance",
                    difficulty="hard" if "novel" in qtype.lower() else "medium",
                    meta={
                        "company": _first(r, "company", default=""),
                        "question_type": qtype,
                    },
                )
            )
        return out

    def _anchors(self, row: Dict[str, Any], doc_name: str, answer: str) -> List[GoldAnchor]:
        anchors: List[GoldAnchor] = []
        evidence = _first(row, "evidence", "evidence_text", default=[])
        if isinstance(evidence, str):
            evidence = [{"evidence_text": evidence}]
        for ev in evidence or []:
            if isinstance(ev, str):
                ev = {"evidence_text": ev}
            text = _first(ev, "evidence_text", "text", "evidence_text_full_page", default="")
            page = _first(ev, "evidence_page_num", "page_number", "page", default=None)
            spans = [s for s in [text, answer] if s]
            anchors.append(
                GoldAnchor(
                    doc_id=doc_name,
                    answer_spans=spans,
                    page=int(page) if isinstance(page, (int, float)) else None,
                )
            )
        if not anchors:  # fall back to the answer string alone
            anchors.append(GoldAnchor(doc_id=doc_name, answer_spans=[answer] if answer else []))
        return anchors

    # -- corpus ------------------------------------------------------------
    def corpus(self) -> List[Doc]:
        if not self._needed_docs:
            self.questions()  # populate _needed_docs
        docs: List[Doc] = []
        missing: List[str] = []
        for name in sorted(self._needed_docs):
            path = self.docs_dir / f"{name}.txt"
            if not path.exists():
                missing.append(name)
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            # keep the original PDF path if present, so PageIndex can use its own
            # real PDF parser (page_index) rather than our pre-extracted text.
            pdf = self.docs_dir / f"{name}.pdf"
            docs.append(
                Doc(
                    doc_id=name,
                    title=name,
                    content=text,
                    path=str(pdf) if pdf.exists() else None,
                    content_type="text/plain",
                )
            )
        if missing:
            warnings.warn(
                f"{len(missing)} FinanceBench docs missing from {self.docs_dir} "
                f"(run scripts/download_financebench.py). Their questions will be "
                f"skipped. First few: {missing[:3]}",
                stacklevel=2,
            )
        return docs

    def questions_for_available_corpus(self) -> List[Question]:
        """Questions whose source document text is actually present — what the
        runner should evaluate so missing docs don't count as failures."""
        have = {d.doc_id for d in self.corpus()}
        return [q for q in self.questions() if q.doc_id in have]
