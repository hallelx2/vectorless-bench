"""Each dataset's own published metric, computed beside the harness's
span-anchor retrieval metrics and the Jev answer grade (HAL-3087).

The anchor metrics answer "did the system return the passage?" the same way
for every dataset, which is what makes systems comparable across datasets.
The native metrics answer the question each dataset's paper asks, so a
number can be set beside the published ones.

- QASPER (Dasigi et al. 2021): answer F1, token-level over SQuAD-normalised
  text, the best over the question's annotators; yes/no and "unanswerable"
  are answers like any other. Evidence F1 over the paper's paragraphs: a
  paragraph is retrieved when its text is in a returned unit.
- LegalBench-RAG (Pipitone & Alami 2024): character-level precision and
  recall at k, from the overlap between the returned units, located in their
  source file, and the gold character spans. A unit carries its offset when
  the system cut it (the chunk systems); otherwise its text is searched for,
  and a passage that appears twice is taken at its first occurrence. A unit
  that cannot be located still counts against precision.
- CUAD (Hendrycks et al. 2021): a predicted clause matches when its token
  Jaccard with a gold span is at least 0.5, the CUAD evaluation's rule; a
  no-clause question is right when the answer abstains. CUAD scores an
  extracted span, and here the prediction is the answer the shared model
  wrote from the retrieved passages, so this is that rule applied to a
  generated answer, labelled as such in the report.
- MultiHop-RAG (Tang & Yang 2024): retrieval over the gold evidence
  articles. hits@k is the share of gold articles among the documents of the
  first k units, MAP@k and MRR@k are over those documents in rank order.

Names carry a `native_` prefix so the report gathers them per dataset.
"""

from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ..schema import Doc, Question, RetrievedSection

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)
_WS = re.compile(r"\s+")


def squad_normalize(s: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace (the
    normalisation QASPER's official evaluator uses)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def token_f1(prediction: str, gold: str) -> float:
    p, g = squad_normalize(prediction).split(), squad_normalize(gold).split()
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall)


def jaccard(a: str, b: str) -> float:
    x, y = set(squad_normalize(a).split()), set(squad_normalize(b).split())
    if not x and not y:
        return 1.0
    if not x or not y:
        return 0.0
    return len(x & y) / len(x | y)


def _declined(answer: str) -> bool:
    a = answer.strip().upper()
    return a.startswith("INSUFFICIENT") or a in ("", "UNANSWERABLE")


class _Located:
    """A document's text with a whitespace-collapsed copy and the map back, so
    a unit whose spacing differs from the source can still be placed."""

    def __init__(self, text: str) -> None:
        self.text = text
        chars, index = [], []
        prev_space = True
        for i, ch in enumerate(text):
            if ch.isspace():
                if not prev_space:
                    chars.append(" ")
                    index.append(i)
                prev_space = True
            else:
                chars.append(ch)
                index.append(i)
                prev_space = False
        self.squashed = "".join(chars)
        self.index = index

    def find(self, unit: str) -> Optional[Tuple[int, int]]:
        """Character range of unit in the source text, or None."""
        u = unit.strip()
        if not u:
            return None
        i = self.text.find(u)
        if i >= 0:
            return i, i + len(u)
        q = " ".join(u.split())
        j = self.squashed.find(q)
        if j < 0:
            return None
        start = self.index[j]
        end = self.index[min(j + len(q) - 1, len(self.index) - 1)] + 1
        return start, end


def _merge(ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    out: List[Tuple[int, int]] = []
    for a, b in sorted(r for r in ranges if r[1] > r[0]):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _length(ranges: Sequence[Tuple[int, int]]) -> int:
    return sum(b - a for a, b in ranges)


def _overlap(xs: Sequence[Tuple[int, int]], ys: Sequence[Tuple[int, int]]) -> int:
    total, i, j = 0, 0, 0
    while i < len(xs) and j < len(ys):
        a, b = max(xs[i][0], ys[j][0]), min(xs[i][1], ys[j][1])
        if b > a:
            total += b - a
        if xs[i][1] < ys[j][1]:
            i += 1
        else:
            j += 1
    return total


class NativeScorer:
    """The native metrics for one dataset, over its corpus."""

    def __init__(self, dataset: str, corpus: Sequence[Doc]) -> None:
        self.dataset = dataset
        self._docs: Dict[str, Doc] = {d.doc_id: d for d in corpus}
        self._located: Dict[str, _Located] = {}

    def _loc(self, doc_id: str) -> Optional[_Located]:
        if doc_id not in self._located:
            d = self._docs.get(doc_id)
            if d is None:
                return None
            self._located[doc_id] = _Located(d.content)
        return self._located[doc_id]

    # -- retrieval ---------------------------------------------------------
    def retrieval(self, q: Question, sections: Sequence[RetrievedSection], k: int) -> Dict[str, float]:
        top = list(sections)[:k]
        if self.dataset == "legalbench_rag":
            return self._char_pr(q, top, k)
        if self.dataset == "qasper":
            return self._qasper_evidence(q, top)
        if self.dataset == "multihop_rag":
            return self._doc_ranking(q, top, k)
        return {}

    def _char_pr(self, q: Question, top: Sequence[RetrievedSection], k: int) -> Dict[str, float]:
        spans = q.meta.get("spans") or []
        if not spans:
            return {}
        gold: Dict[str, List[Tuple[int, int]]] = {}
        for fp, (a, b) in spans:
            gold.setdefault(fp, []).append((int(a), int(b)))
        retrieved: Dict[str, List[Tuple[int, int]]] = {}
        retrieved_chars = 0
        unlocated = 0
        for s in top:
            doc = s.doc_id or q.doc_id
            loc = self._loc(doc)
            r = None
            if loc is not None and s.char_start is not None and s.char_start >= 0:
                # the system's own offset, checked against the text it claims
                end = s.char_start + len(s.content)
                if loc.text[s.char_start:end] == s.content:
                    r = (s.char_start, end)
            if r is None and loc is not None:
                r = loc.find(s.content)
            if r is None:
                unlocated += 1
                retrieved_chars += len(s.content.strip())
                continue
            retrieved.setdefault(doc, []).append(r)
        overlap = 0
        gold_chars = 0
        for doc, rs in gold.items():
            g = _merge(rs)
            gold_chars += _length(g)
            overlap += _overlap(_merge(retrieved.get(doc, [])), g)
        for doc, rs in retrieved.items():
            retrieved_chars += _length(_merge(rs))
        return {
            f"native_char_precision@{k}": overlap / retrieved_chars if retrieved_chars else 0.0,
            f"native_char_recall@{k}": overlap / gold_chars if gold_chars else 0.0,
            "native_units_unlocated": float(unlocated),
        }

    def _qasper_evidence(self, q: Question, top: Sequence[RetrievedSection]) -> Dict[str, float]:
        probes: List[str] = q.meta.get("paragraph_probes") or []
        gold = set(q.meta.get("evidence_paragraphs") or [])
        if not probes or not gold:
            return {}
        from ..anchors import span_present

        text = "\n\n".join(s.content for s in top)
        predicted = {i for i, p in enumerate(probes) if p and span_present(p, text)}
        tp = len(predicted & gold)
        precision = tp / len(predicted) if predicted else 0.0
        recall = tp / len(gold)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return {"native_evidence_f1": f1, "native_evidence_precision": precision, "native_evidence_recall": recall}

    def _doc_ranking(self, q: Question, top: Sequence[RetrievedSection], k: int) -> Dict[str, float]:
        gold = list(dict.fromkeys(q.meta.get("doc_ids") or []))
        if not gold:
            return {}
        ranked: List[str] = []
        for s in top:
            if s.doc_id and s.doc_id not in ranked:
                ranked.append(s.doc_id)
        gold_set = set(gold)
        hits = len(gold_set & set(ranked))
        precisions = []
        found = 0
        rr = 0.0
        for i, d in enumerate(ranked, start=1):
            if d in gold_set:
                found += 1
                precisions.append(found / i)
                if rr == 0.0:
                    rr = 1.0 / i
        return {
            f"native_hits@{k}": hits / len(gold),
            f"native_map@{k}": sum(precisions) / len(gold),
            f"native_mrr@{k}": rr,
            "native_units_without_document": float(sum(1 for s in top if not s.doc_id)),
        }

    # -- answers -----------------------------------------------------------
    def answer(self, q: Question, candidate: Optional[str]) -> Dict[str, float]:
        if candidate is None:
            return {}
        if self.dataset == "qasper":
            golds: List[str] = [g for g in (q.meta.get("answer_strings") or []) if g] or (
                [q.answer] if q.answer else ["unanswerable"] if q.is_no_answer else [])
            if not golds:
                return {}
            pred = "unanswerable" if _declined(candidate) else candidate
            return {"native_answer_f1": max(token_f1(pred, g) for g in golds)}
        if self.dataset == "cuad":
            if q.is_no_answer:
                return {"native_clause_match": 1.0 if _declined(candidate) else 0.0}
            spans: List[str] = q.meta.get("answers") or [a for g in q.gold for a in g.answer_spans]
            if not spans:
                return {}
            if _declined(candidate):
                return {"native_clause_match": 0.0}
            return {"native_clause_match": 1.0 if max(jaccard(candidate, s) for s in spans) >= 0.5 else 0.0}
        return {}
