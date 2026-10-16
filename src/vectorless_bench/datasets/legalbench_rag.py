"""LegalBench-RAG (Pipitone & Alami 2024): legal questions whose gold is the
exact character ranges of contract text that answer them. Four
sub-benchmarks (ContractNLI, CUAD, MAUD, PrivacyQA) over 718 plain-text
documents; the data is the authors' Dropbox release (measured 2026-10-16:
corpus/cuad 463 files, maud 151, contractnli 96, privacy_qa 8).

Scope, and how it differs from the paper. Each query names its document
("Consider the Marketing Affiliate Agreement between X and Y; ..."). The
paper retrieves over the whole sub-benchmark corpus; here every system is
given the named document, so the benchmark measures finding the clause, not
finding the contract. Every system gets the same scope, and the report says
so.

Gold: each snippet's character range, cut from its corpus file, is an anchor
span (char ranges kept in meta for an exact-overlap score). `mini=True` takes
the first 194 cases of each sub-benchmark in file order, 776 in all, the
size of the paper's LegalBench-RAG-mini (whose sampling seed is not
published, so this is a reproducible stand-in, not the same 776).

Documents are text, not PDFs: this set exercises every system's non-PDF path,
Vectorless's included.
"""

from __future__ import annotations

import json
import unicodedata
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from . import _files
from .base import Dataset

DROPBOX = ("https://www.dropbox.com/scl/fo/r7xfa5i3hdsbxex1w6amw/AID389Olvtm-ZLTKAPrw6k4"
           "?rlkey=5n8zrbk4c08lbit3iiexofmwg&st=0hu354cq&dl=1")
SUBSETS = ("contractnli", "cuad", "maud", "privacy_qa")


class LegalBenchRagDataset(Dataset):
    name = "legalbench_rag"

    def __init__(self, data_dir: str = "data/legalbench_rag", mini: bool = False,
                 subsets: Optional[List[str]] = None, **_: object) -> None:
        self.dir = Path(data_dir)
        self.mini = mini
        self.subsets = list(subsets or SUBSETS)
        self._tests: Optional[Dict[str, List[dict]]] = None

    def fetch(self) -> None:
        z = _files.download(DROPBOX, self.dir / "legalbench_rag.zip", timeout=600)
        with zipfile.ZipFile(z) as zf:
            for n in zf.namelist():
                if n.endswith("/") or not (n.startswith("corpus/") or n.startswith("benchmarks/")):
                    continue
                dest = self.dir / n
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(zf.read(n))
        t = self._load()
        print("legalbench_rag:", {k: len(v) for k, v in t.items()}, "total", sum(len(v) for v in t.values()))

    def _load(self) -> Dict[str, List[dict]]:
        if self._tests is None:
            self._tests = {}
            for s in self.subsets:
                p = self.dir / "benchmarks" / f"{s}.json"
                tests = json.loads(p.read_text(encoding="utf-8"))["tests"]
                # The benchmark JSON spells some paths in decomposed Unicode
                # (NFD: "E" + combining accent) while the extracted corpus file
                # is composed (NFC). Byte for byte they differ, and the document
                # went missing (measured 2026-10-16: 713 of 714 present, the
                # absent one "LECLANCHÉ S.A. - ..."). Every path is NFC here.
                for t in tests:
                    for sn in t["snippets"]:
                        sn["file_path"] = unicodedata.normalize("NFC", sn["file_path"])
                self._tests[s] = tests[:194] if self.mini else tests
        return self._tests

    def _files_needed(self) -> List[str]:
        return sorted({sn["file_path"] for tests in self._load().values() for t in tests for sn in t["snippets"]})

    def _path(self, fp: str) -> Path:
        """The corpus file for a path, in whichever Unicode form the
        filesystem stored it (macOS keeps NFD, Linux keeps what it was given)."""
        for form in ("NFC", "NFD"):
            p = self.dir / "corpus" / unicodedata.normalize(form, fp)
            if p.exists():
                return p
        return self.dir / "corpus" / fp

    def corpus(self) -> List[Doc]:
        docs = []
        for fp in self._files_needed():
            p = self._path(fp)
            if p.exists():
                docs.append(Doc(doc_id=fp, title=Path(fp).stem, content=p.read_text(encoding="utf-8", errors="ignore"),
                                path=str(p), content_type="text/plain"))
        return docs

    def questions(self) -> List[Question]:
        texts: Dict[str, str] = {}
        out = []
        for s, tests in self._load().items():
            for i, t in enumerate(tests):
                sn = t["snippets"]
                fp = sn[0]["file_path"]
                gold = []
                for x in sn:
                    if x["file_path"] not in texts:
                        p = self._path(x["file_path"])
                        texts[x["file_path"]] = p.read_text(encoding="utf-8", errors="ignore") if p.exists() else ""
                    a, b = x["span"]
                    span = texts[x["file_path"]][a:b].strip()
                    if span:
                        gold.append(GoldAnchor(doc_id=x["file_path"], answer_spans=[span]))
                out.append(Question(
                    qid=f"lbr_{s}_{i:05d}", doc_id=fp, question=t["query"],
                    gold=gold, answer=(sn[0].get("answer") or None),
                    answer_type=AnswerType.LOOKUP if len({x['file_path'] for x in sn}) == 1 else AnswerType.MULTIHOP,
                    domain=f"legal_{s}",
                    meta={"spans": [(x["file_path"], x["span"]) for x in sn], "subset": s},
                ))
        return out
