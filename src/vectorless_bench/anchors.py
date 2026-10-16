"""Gold-anchor matching.

The engine assigns every section a random `sec_<uuid>` that changes on every
re-ingest, so gold labels can never be IDs. Instead, gold is a `GoldAnchor`
(stable title-path / answer-span / page) and we decide at scoring time whether a
returned unit satisfies it. This module is the single source of truth for what
"the system retrieved the right thing" means, kept separate from the metrics so
the definition is auditable in one place.

Matching is intentionally lenient on surface form (whitespace, case,
punctuation, currency/number formatting) and strict on substance (the answer
span must actually be present, or the structural path must actually match).
"""

from __future__ import annotations

import re
from typing import List, Sequence

from .schema import GoldAnchor, RetrievedSection

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^\w\s]")
_NUM = re.compile(r"(?<![\w.])\$?\s?-?\d[\d,]*(?:\.\d+)?%?")


_HYPHEN_BREAK = re.compile(r"(\w)-[ \t]*\r?\n[ \t]*(\w)")


def norm_text(s: str) -> str:
    """Lowercase, rejoin words a PDF hyphenated across a line break, strip
    punctuation, collapse whitespace — for span search."""
    s = _HYPHEN_BREAK.sub(r"\1\2", s)
    return _WS.sub(" ", _PUNCT.sub(" ", s.lower())).strip()


def norm_heading(s: str) -> str:
    """Normalise a single heading for path comparison. Drops common ordinal
    noise like 'Item 8.' vs 'Item 8' and 'Note 12 -' prefixes."""
    s = s.lower().strip()
    s = re.sub(r"^(item|note|section|part|chapter|article)\s+", "", s)
    s = _PUNCT.sub(" ", s)
    return _WS.sub(" ", s).strip()


def norm_number(s: str) -> str:
    """Canonicalise a numeric string: '$1,234.50' -> '1234.5', '12%' -> '12'."""
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    try:
        f = float(s)
        return repr(int(f)) if f.is_integer() else repr(f)
    except ValueError:
        return s


def _numbers(s: str) -> List[str]:
    return [norm_number(m.group()) for m in _NUM.finditer(s)]


def span_present(span: str, content: str) -> bool:
    """True if `span` appears in `content`, robust to formatting.

    Strategy: try a normalised substring match first; if the span is
    number-bearing (typical in finance), also accept when every number in the
    span appears in the content (handles '$1,234' vs '1234' vs '1,234.0')."""
    if not span:
        return False
    nspan, ncontent = norm_text(span), norm_text(content)
    if nspan and f" {nspan} " in f" {ncontent} ":  # whole words: '8' is not in '1987'
        return True
    if _long_window_present(nspan.split(), ncontent):
        return True
    span_nums = _numbers(span)
    if span_nums and _numbers_distinctive(span_nums):
        content_nums = set(_numbers(content))
        return all(n in content_nums for n in span_nums)
    return False


def _long_window_present(words: List[str], ncontent: str) -> bool:
    """A long span whose PDF rendering differs in a word or two (a ligature,
    a dropped reference marker, a broken line) still counts when a contiguous
    run of at least 8 words and 60% of it appears verbatim. Short spans get
    no such slack: for them the exact match above is the test."""
    n = len(words)
    if n < 10:
        return False
    need = max(8, int(0.6 * n + 0.999))
    for i in range(0, n - need + 1):
        if " ".join(words[i : i + need]) in ncontent:
            return True
    return False


def _numbers_distinctive(nums: List[str]) -> bool:
    """The number-only fallback is safe for '1,577.0' vs '$1577' but not for
    a bare '8', which most sections contain. Allowed when the span carries two
    or more numbers, or one with at least three significant digits."""
    if len(nums) >= 2:
        return True
    return len(nums[0].replace(".", "").replace("-", "").lstrip("0")) >= 3


def path_matches(gold_path: Sequence[str], cand_path: Sequence[str]) -> bool:
    """True if the candidate's heading path satisfies the gold path.

    A match is a normalised *suffix* match: the gold path must appear as the
    tail of the candidate path (so ["Item 8", "Balance Sheet"] matches a
    candidate ["Part II", "Item 8", "Balance Sheet"]). Empty gold path never
    matches by path (force span/page matching instead)."""
    g = [norm_heading(h) for h in gold_path if h.strip()]
    c = [norm_heading(h) for h in cand_path if h.strip()]
    if not g or len(g) > len(c):
        return False
    return c[len(c) - len(g):] == g


def match_anchor(section: RetrievedSection, anchor: GoldAnchor) -> bool:
    """A section satisfies an anchor if it matches on ANY provided dimension:
    structural path, answer-span containment, or page. Dimensions the anchor
    leaves empty are simply not tested."""
    if anchor.title_path and section.title_path and path_matches(
        anchor.title_path, section.title_path
    ):
        return True
    if anchor.answer_spans and any(
        span_present(sp, section.content) for sp in anchor.answer_spans
    ):
        return True
    if anchor.page is not None and section.page is not None and (
        anchor.page == section.page
    ):
        return True
    if anchor.page_text and page_text_overlaps(section.content, anchor.page_text):
        return True
    return False


_SHINGLE = 6


def _shingles(ntext: str) -> set:
    w = ntext.split()
    return {" ".join(w[i : i + _SHINGLE]) for i in range(max(0, len(w) - _SHINGLE + 1))}


def page_text_overlaps(content: str, page_text: str) -> bool:
    """True when a returned section carries a real part of the gold page:
    it covers 30% of the page's six-word shingles, or (a short section
    lying inside the page) 50% of its own shingles are on the page and it
    has at least 20 of them. A page with fewer than 10 shingles of text
    (a figure, a scanned slide) cannot be matched this way."""
    page = _shingles(norm_text(page_text))
    if len(page) < 10:
        return False
    sec = _shingles(norm_text(content))
    if not sec:
        return False
    shared = len(page & sec)
    return shared >= 0.3 * len(page) or (shared >= 20 and shared >= 0.5 * len(sec))


def relevance_labels(
    sections: Sequence[RetrievedSection], golds: Sequence[GoldAnchor]
) -> List[bool]:
    """Per-rank relevance for the returned list: position i is True if that
    section satisfies at least one gold anchor. Order is preserved so ranking
    metrics (MRR, nDCG) are meaningful."""
    return [any(match_anchor(s, g) for g in golds) for s in sections]


def covered_anchors(
    sections: Sequence[RetrievedSection], golds: Sequence[GoldAnchor]
) -> int:
    """How many distinct gold anchors were satisfied by the returned set —
    the numerator for recall over multi-hop questions."""
    return sum(1 for g in golds if any(match_anchor(s, g) for s in sections))


def is_sibling_near_miss(
    section: RetrievedSection, golds: Sequence[GoldAnchor]
) -> bool:
    """The signature vector-RAG failure the whitepaper calls out: a section
    that lives under the SAME parent as a gold anchor (right neighbourhood) but
    is NOT itself a hit (wrong fiscal year / wrong drug / wrong sub-clause).

    Requires structural paths on both sides; returns False if unavailable."""
    if not section.title_path:
        return False
    for g in golds:
        gp = [norm_heading(h) for h in g.parent_path() if h.strip()]
        if not gp:
            continue
        cp = [norm_heading(h) for h in section.title_path if h.strip()]
        # shares the gold's parent prefix but isn't itself a hit
        if len(cp) >= len(gp) and cp[: len(gp)] == gp and not match_anchor(
            section, g
        ):
            return True
    return False
