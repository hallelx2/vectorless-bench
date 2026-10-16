"""CUAD v1 (Hendrycks et al. 2021): 41 kinds of clause a lawyer reviews,
labelled in 510 commercial contracts. Measured 2026-10-16: the official test
split (data.zip, test.json) is 102 contracts and 4,182 questions, of which
2,938 (70%) have no such clause in the contract.

Each (contract, clause type) is one question; the gold is the annotated
clause text. A question with no clause is NO_ANSWER: the right behaviour is
to return nothing confident, and the report scores those apart, as
abstentions, so they neither inflate nor drown the clause-finding numbers.

Source documents: the original contract PDFs from the CUAD repository on
Hugging Face, so PDF-reading systems get the real filing; text systems get
the same PDF's extracted text, page-separated. (CUAD's own context string is
a different extraction; the clause spans are matched leniently against ours,
and `vlbench datasets check cuad` reports how many are found.)
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from . import _files
from .base import Dataset

DATA_ZIP = "https://github.com/TheAtticusProject/cuad/raw/main/data.zip"
HF = "https://huggingface.co/datasets/theatticusproject/cuad"
PDF_TREE = HF.replace("/datasets/", "/api/datasets/") + "/tree/main/CUAD_v1/full_contract_pdf?recursive=true"


class CuadDataset(Dataset):
    name = "cuad"

    def __init__(self, data_dir: str = "data/cuad", split: str = "test", **_: object) -> None:
        self.dir = Path(data_dir)
        self.split = split
        self._data: Optional[dict] = None

    def fetch(self) -> None:
        z = _files.download(DATA_ZIP, self.dir / "data.zip")
        with zipfile.ZipFile(z) as zf:
            (self.dir / f"{self.split}.json").write_bytes(zf.read(f"{self.split}.json"))
        titles = [x["title"] for x in self._load()["data"]]
        with urllib.request.urlopen(urllib.request.Request(PDF_TREE, headers={"User-Agent": _files.UA}), timeout=60) as r:  # noqa: S310
            tree = json.loads(r.read())
        by_stem = {Path(t["path"]).stem: t["path"] for t in tree if t["path"].lower().endswith(".pdf")}
        missing = failed = 0
        for t in titles:
            path = by_stem.get(t)
            if path is None:
                missing += 1
                print(f"  no PDF for {t}")
                continue
            pdf = self.dir / "pdfs" / f"{t}.pdf"
            try:
                _files.download(f"{HF}/resolve/main/{urllib.parse.quote(path)}", pdf)
                _files.ensure_text(pdf)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  FAILED {t}: {e}")
        print(f"cuad {self.split}: {len(titles)} contracts; {missing} without a PDF, {failed} failed")

    def _load(self) -> dict:
        if self._data is None:
            self._data = json.loads((self.dir / f"{self.split}.json").read_text(encoding="utf-8"))
        return self._data

    def corpus(self) -> List[Doc]:
        docs = []
        for x in self._load()["data"]:
            t = x["title"]
            pdf = self.dir / "pdfs" / f"{t}.pdf"
            txt = pdf.with_suffix(".txt")
            if txt.exists():
                docs.append(Doc(doc_id=t, title=t, content=txt.read_text(encoding="utf-8", errors="ignore"),
                                path=str(pdf), content_type="text/plain"))
        return docs

    def questions(self) -> List[Question]:
        out = []
        for x in self._load()["data"]:
            t = x["title"]
            for p in x["paragraphs"]:
                for q in p["qas"]:
                    clause = q["id"].split("__", 1)[-1]
                    spans = sorted({a["text"].strip() for a in q.get("answers", []) if a.get("text", "").strip()})
                    none = bool(q.get("is_impossible")) or not spans
                    out.append(Question(
                        qid=q["id"], doc_id=t, question=q["question"],
                        gold=[] if none else [GoldAnchor(doc_id=t, answer_spans=[s]) for s in spans],
                        answer=None if none else spans[0],
                        answer_type=AnswerType.NO_ANSWER if none else AnswerType.LOOKUP,
                        domain="legal_contract",
                        meta={"clause_type": clause},
                    ))
        return out
