"""Technical manuals: E-Manual QA (Nandy et al., EMNLP 2021 Findings), the
Samsung Smart TV and Smart Remote e-manual.

Why this set (chosen 2026-10-16, see HAL-3040): it is the public manuals QA
set whose questions carry the manual's own heading path ("Quick Guides >
Using Bixby > Running Bixby") and an answer quoted from the manual, which is
what a deep-heading-tree benchmark needs. Public release, measured: 409
"normal" and 50 "compound" questions over 261 manual sections (a further
200 paraphrased questions restate the normal ones and are left out so no
question counts twice).

Limit, stated rather than hidden: the release ships the manual as extracted
sections, not the PDF. The document every system reads is rebuilt from those
sections as Markdown, in order, with heading depth recovered from the paths
the questions name (a section that appears third in a path is a level-3
heading). Sections no question names stay at level 2. Finding the original
PDF is a follow-up; until then this set measures structure-following on a
reconstructed document, and the report says so.

Gold: the heading path (title_path) and a 12-word window of the quoted
answer (answer_spans). Some release files are not valid JSON (unescaped
quotes inside answers), so questions are read field by field.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

from ..schema import AnswerType, Doc, GoldAnchor, Question
from . import _files
from .base import Dataset

RAW = "https://raw.githubusercontent.com/abhi1nandy2/EMNLP-2021-Findings/main/data/"
DOC_ID = "samsung_smart_tv_remote_emanual"
_ITEM = re.compile(
    r'"Section Hierarchy"\s*:\s*"(?P<path>.*?)"\s*,\s*"Link"\s*:\s*"(?P<link>.*?)"\s*,\s*'
    r'"Question"\s*:\s*"(?P<q>.*?)"\s*,\s*"Answer"\s*:\s*"(?P<a>.*?)"\s*}', re.S)


class ManualsDataset(Dataset):
    name = "manuals"

    def __init__(self, data_dir: str = "data/manuals", include_compound: bool = True, **_: object) -> None:
        self.dir = Path(data_dir)
        self.include_compound = include_compound
        self._items: Optional[List[dict]] = None

    def fetch(self) -> None:
        _files.download(RAW + "smart_remote_QA.zip", self.dir / "smart_remote_QA.zip")
        _files.download(RAW + "smart_tv_remote_manual_corpus.json", self.dir / "corpus.json")
        print(f"manuals: {len(self._load())} questions, {len(self._sections())} sections")

    def _load(self) -> List[dict]:
        if self._items is None:
            items = []
            with zipfile.ZipFile(self.dir / "smart_remote_QA.zip") as z:
                for n in sorted(z.namelist()):
                    if not n.endswith(".json") or n.startswith("Paraphrased"):
                        continue
                    if n.startswith("Compound") and not self.include_compound:
                        continue
                    raw = z.read(n).decode("utf-8", "ignore")
                    for m in _ITEM.finditer(raw):
                        items.append({"path": m["path"], "link": m["link"], "question": m["q"].strip(),
                                      "answer": m["a"].strip(), "file": n})
            self._items = items
        return self._items

    def _sections(self) -> List[dict]:
        c = json.loads((self.dir / "corpus.json").read_text(encoding="utf-8"))
        return list(c.values()) if isinstance(c, dict) else c

    def corpus(self) -> List[Doc]:
        depth: Dict[str, int] = {}
        for it in self._load():
            for i, h in enumerate(x.strip() for x in it["path"].split(">")):
                depth.setdefault(h.lower(), i + 1)
        out = io.StringIO()
        out.write("# Samsung Smart TV and Smart Remote e-Manual\n\n")
        for s in self._sections():
            title = str(s.get("title", "")).strip()
            level = min(6, 1 + depth.get(title.lower(), 1))
            text = s.get("text") or []
            body = " ".join(text) if isinstance(text, list) else str(text)
            out.write(f"{'#' * level} {title}\n\n{body}\n\n")
        p = self.dir / f"{DOC_ID}.md"
        p.write_text(out.getvalue(), encoding="utf-8")
        return [Doc(doc_id=DOC_ID, title="Samsung Smart TV and Smart Remote e-Manual", content=out.getvalue(),
                    path=str(p), content_type="text/markdown",
                    meta={"reconstructed_from_sections": True})]

    def questions(self) -> List[Question]:
        out = []
        for i, it in enumerate(self._load()):
            path = [h.strip() for h in it["path"].split(">") if h.strip()]
            out.append(Question(
                qid=f"man_{i:04d}", doc_id=DOC_ID, question=it["question"],
                gold=[GoldAnchor(doc_id=DOC_ID, title_path=path, answer_spans=[_files.probe(it["answer"])])],
                answer=it["answer"],
                answer_type=AnswerType.MULTIHOP if it["file"].startswith("Compound") else AnswerType.LOOKUP,
                domain="manual",
                meta={"section_path": path, "source_link": it["link"], "file": it["file"]},
            ))
        return out
