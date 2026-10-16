"""MultiHop-RAG (Tang & Yang 2024): questions whose answer needs facts from
two to four news articles. Measured 2026-10-16 on the HF dataset: 2,556
queries over a 609-article corpus; 856 comparison, 816 inference, 583
temporal and 301 null queries (null: the corpus does not hold the answer).

Every question is corpus-level: its doc_id is CORPUS and the evidence
articles are listed in meta["doc_ids"]. A system has to find the right
articles as well as the right passages, which is the point of the set.

Gold: each evidence item's `fact` (a sentence quoted from its article) is an
anchor on that article. A null query is NO_ANSWER: the right behaviour is to
return nothing confident.

Documents are plain text (title, source, date, body); there is no PDF, so
this set also exercises every system's non-PDF path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from . import _files
from .base import CORPUS, Dataset

BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main/"


def _doc_id(title: str) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "_", title).strip("_")
    return s[:120] or "untitled"


class MultiHopRagDataset(Dataset):
    name = "multihop_rag"

    def __init__(self, data_dir: str = "data/multihop_rag", **_: object) -> None:
        self.dir = Path(data_dir)
        self._q: Optional[List[dict]] = None
        self._c: Optional[List[dict]] = None

    def fetch(self) -> None:
        for f in ("MultiHopRAG.json", "corpus.json"):
            _files.download(BASE + f, self.dir / f)
        print(f"multihop_rag: {len(self._queries())} queries, {len(self._corpus_rows())} articles")

    def _queries(self) -> List[dict]:
        if self._q is None:
            self._q = json.loads((self.dir / "MultiHopRAG.json").read_text(encoding="utf-8"))
        return self._q

    def _corpus_rows(self) -> List[dict]:
        if self._c is None:
            self._c = json.loads((self.dir / "corpus.json").read_text(encoding="utf-8"))
        return self._c

    def _ids(self) -> Dict[str, str]:
        """Article title -> doc id, made unique."""
        ids: Dict[str, str] = {}
        used: Dict[str, int] = {}
        for a in self._corpus_rows():
            base = _doc_id(a["title"])
            n = used.get(base, 0)
            used[base] = n + 1
            ids[a["title"]] = base if n == 0 else f"{base}_{n}"
        return ids

    def corpus(self) -> List[Doc]:
        ids = self._ids()
        docs = []
        for a in self._corpus_rows():
            content = f"{a['title']}\n{a.get('source', '')} · {a.get('published_at', '')}\n\n{a.get('body', '')}"
            docs.append(Doc(doc_id=ids[a["title"]], title=a["title"], content=content,
                            content_type="text/plain",
                            meta={k: a.get(k) for k in ("source", "category", "published_at", "url")}))
        return docs

    def questions(self) -> List[Question]:
        ids = self._ids()
        out = []
        for i, q in enumerate(self._queries()):
            qtype = q.get("question_type", "")
            ev = q.get("evidence_list") or []
            doc_ids = sorted({ids[e["title"]] for e in ev if e.get("title") in ids})
            gold = [GoldAnchor(doc_id=ids[e["title"]], answer_spans=[e["fact"]])
                    for e in ev if e.get("title") in ids and e.get("fact")]
            null = qtype == "null_query"
            out.append(Question(
                qid=f"mh_{i:04d}", doc_id=CORPUS, question=q["query"],
                gold=[] if null else gold,
                answer=None if null else q.get("answer"),
                answer_type=AnswerType.NO_ANSWER if null else AnswerType.MULTIHOP,
                domain=qtype.replace("_query", "") or "multihop",
                meta={"doc_ids": doc_ids, "question_type": qtype},
            ))
        return out
