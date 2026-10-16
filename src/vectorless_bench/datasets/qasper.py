"""QASPER (Dasigi et al. 2021): questions written by NLP practitioners over
full research papers, answered by other practitioners with evidence
paragraphs. Test split: 416 papers (measured on the HF dataset server,
2026-10-16).

Every system reads the SAME source: the paper's arXiv PDF. QASPER ships the
papers as parsed JSON, but a system that reads PDFs (Vectorless, PageIndex)
and one that reads text must see the same words, so the PDF is fetched and
its text extracted with page breaks for the text systems.

Gold: each annotator's evidence paragraphs become anchors. A paragraph from
the parsed JSON rarely matches its PDF rendering character for character
(hyphenation, line breaks, ligatures), so the anchor's span is a 12-word
window from the middle of the paragraph, and its page is the PDF page that
window is found on. `vlbench datasets check qasper` reports how many windows
were found.

Answers: a question every annotator marked unanswerable is NO_ANSWER; a
yes/no answer is LOOKUP with the gold "yes"/"no"; otherwise the first
annotator's free-form answer or extractive spans are the gold answer. The
dataset's own answer-F1 is reported alongside the graded answer.

Evidence that is a figure or table ("FLOAT SELECTED: ...") keeps its caption
text, which the PDF does contain.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from . import _files
from .base import Dataset

PARQUET = "https://huggingface.co/api/datasets/allenai/qasper/parquet/qasper/{split}/0.parquet"
ARXIV_PDF = "https://arxiv.org/pdf/{id}"


class QasperDataset(Dataset):
    name = "qasper"

    def __init__(self, data_dir: str = "data/qasper", split: str = "test", **_: object) -> None:
        self.dir = Path(data_dir)
        self.split = split
        self._rows: Optional[List[Dict[str, Any]]] = None

    # -- fetch -------------------------------------------------------------
    def fetch(self) -> None:
        pq = _files.download(PARQUET.format(split=self.split), self.dir / f"{self.split}.parquet")
        rows = _files.read_parquet_rows(pq)
        print(f"qasper {self.split}: {len(rows)} papers")
        ok = failed = 0
        for r in rows:
            pid = str(r["id"])
            pdf = self.dir / "pdfs" / f"{pid}.pdf"
            try:
                # arXiv asks automated clients for one request every 3 s.
                _files.download(ARXIV_PDF.format(id=pid), pdf, min_interval=3.0)
                _files.ensure_text(pdf)
                ok += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAILED {pid}: {e}")
        print(f"qasper: {ok} PDFs ready, {failed} failed")

    def _load(self) -> List[Dict[str, Any]]:
        if self._rows is None:
            pq = self.dir / f"{self.split}.parquet"
            if not pq.exists():
                raise FileNotFoundError(f"{pq} missing: run `vlbench datasets fetch qasper`")
            self._rows = _files.read_parquet_rows(pq)
        return self._rows

    # -- corpus ------------------------------------------------------------
    def corpus(self) -> List[Doc]:
        docs = []
        missing = 0
        for r in self._load():
            pid = str(r["id"])
            pdf = self.dir / "pdfs" / f"{pid}.pdf"
            txt = pdf.with_suffix(".txt")
            if not txt.exists():
                missing += 1
                continue
            docs.append(Doc(doc_id=pid, title=str(r.get("title") or pid),
                            content=txt.read_text(encoding="utf-8", errors="ignore"),
                            path=str(pdf), content_type="text/plain"))
        if missing:
            warnings.warn(f"qasper: {missing} papers have no PDF text yet (run fetch)", stacklevel=2)
        return docs

    # -- questions ---------------------------------------------------------
    def questions(self) -> List[Question]:
        out: List[Question] = []
        pages_cache: Dict[str, List[str]] = {}
        for r in self._load():
            pid = str(r["id"])
            txt = self.dir / "pdfs" / f"{pid}.txt"
            if pid not in pages_cache:
                pages_cache[pid] = _files.text_pages(txt) if txt.exists() else []
            pages = pages_cache[pid]
            qas = r.get("qas") or {}
            qs = qas.get("question") or []
            qids = qas.get("question_id") or []
            answers = qas.get("answers") or []
            for i, qtext in enumerate(qs):
                ann = (answers[i] or {}).get("answer") or [] if i < len(answers) else []
                out.append(self._question(pid, str(qids[i]) if i < len(qids) else f"{pid}_{i}", qtext, ann, pages))
        return out

    def _question(self, pid: str, qid: str, qtext: str, ann: List[Dict[str, Any]], pages: List[str]) -> Question:
        unanswerable = bool(ann) and all(a.get("unanswerable") for a in ann)
        first = next((a for a in ann if not a.get("unanswerable")), {})
        if first.get("yes_no") is not None:
            answer = "yes" if first["yes_no"] else "no"
        else:
            answer = (first.get("free_form_answer") or "").strip() or "; ".join(first.get("extractive_spans") or [])
        gold: List[GoldAnchor] = []
        seen = set()
        for a in ann:
            for para in a.get("evidence") or []:
                para = str(para).replace("FLOAT SELECTED:", "").strip()
                key = para[:200]
                if not para or key in seen:
                    continue
                seen.add(key)
                win = _files.probe(para)
                gold.append(GoldAnchor(doc_id=pid, answer_spans=[win], page=_files.page_of(win, pages)))
        return Question(
            qid=qid, doc_id=pid, question=qtext,
            gold=gold,
            answer=None if unanswerable else (answer or None),
            answer_type=AnswerType.NO_ANSWER if unanswerable else AnswerType.LOOKUP,
            domain="science",
            meta={"answers": ann, "answer_type_qasper": _qasper_type(first, unanswerable)},
        )


def _qasper_type(a: Dict[str, Any], unanswerable: bool) -> str:
    if unanswerable:
        return "none"
    if a.get("yes_no") is not None:
        return "boolean"
    if a.get("extractive_spans"):
        return "extractive"
    return "abstractive"
