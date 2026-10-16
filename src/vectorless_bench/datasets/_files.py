"""Shared plumbing for dataset loaders: polite downloads, PDFs kept with a
page-aware text extraction, and the measured check every loader reports.

Page-aware text: a PDF's extracted text is stored with a form feed (\\f)
between pages, so any system that reads text (BM25, vector RAG, full
context) gets exactly the same words as the PDF-reading systems, and the
check can tell whether a gold page number really holds its gold evidence.
"""

from __future__ import annotations

import io
import json
import re
import time
import urllib.request
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from ..anchors import norm_text

UA = "vectorless-bench/0.2 (+https://github.com/hallelx2/vectorless-bench; benchmark data fetch)"
PAGE_BREAK = "\f"


def download(url: str, dest: Path, *, timeout: float = 120.0, retries: int = 4,
             min_interval: float = 0.0, _last: Dict[str, float] = {}) -> Path:
    """Fetch `url` to `dest` unless it is already there. Retries with
    backoff; `min_interval` spaces requests to one host (arXiv asks for
    one request every 3 seconds)."""
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    host = url.split("/")[2]
    wait = _last.get(host, 0.0) + min_interval - time.monotonic()
    if wait > 0:
        time.sleep(wait)
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - dataset URLs
                data = r.read()
            tmp = dest.with_suffix(dest.suffix + ".part")
            tmp.write_bytes(data)
            tmp.replace(dest)
            _last[host] = time.monotonic()
            return dest
        except Exception:
            _last[host] = time.monotonic()
            if attempt == retries - 1:
                raise
            time.sleep(delay)
            delay *= 2
    return dest


def pdf_pages(pdf: Path) -> List[str]:
    """Per-page text of a PDF (pypdf)."""
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(io.BytesIO(pdf.read_bytes()))
    out = []
    for page in reader.pages:
        try:
            out.append(page.extract_text() or "")
        except Exception:
            out.append("")
    return out


def ensure_text(pdf: Path) -> Path:
    """`<pdf>.txt` beside the PDF, pages separated by form feeds."""
    txt = pdf.with_suffix(".txt")
    if not txt.exists():
        txt.write_text(PAGE_BREAK.join(pdf_pages(pdf)), encoding="utf-8")
    return txt


def text_pages(txt: Path) -> List[str]:
    return txt.read_text(encoding="utf-8", errors="ignore").split(PAGE_BREAK)


_PLACEHOLDER = re.compile(r"^(secref|inlineform|displayform|bibref|tabref|figref|citation)\d*$")


def probes(text: str, words: int = 12, n: int = 2) -> List[str]:
    """Up to `n` short, distinctive windows of a long evidence passage: the
    words a PDF extraction is most likely to reproduce verbatim. Long
    paragraphs differ from their PDF rendering in hyphenation and line
    breaks, which a 12-word window rarely crosses. Windows avoid S2ORC's
    placeholder tokens (SECREF3, INLINEFORM1, BIBREF7), which stand for a
    reference or formula and never appear in the PDF text; a run of text
    without one is preferred, from the middle outward."""
    toks = norm_text(text).split()
    if len(toks) <= words:
        clean = [t for t in toks if not _PLACEHOLDER.match(t)]
        return [" ".join(clean)] if clean else []
    starts = sorted(range(0, len(toks) - words + 1), key=lambda s: abs(s - (len(toks) - words) // 2))
    out: List[str] = []
    used: List[int] = []
    for s in starts:
        win = toks[s : s + words]
        if any(_PLACEHOLDER.match(t) for t in win) or any(abs(s - u) < words for u in used):
            continue
        out.append(" ".join(win))
        used.append(s)
        if len(out) == n:
            break
    if not out:  # every window crosses a placeholder: drop them and take the middle
        clean = [t for t in toks if not _PLACEHOLDER.match(t)]
        mid = max(0, (len(clean) - words) // 2)
        out = [" ".join(clean[mid : mid + words])] if clean else []
    return out


def probe(text: str, words: int = 12) -> str:
    """The single best window (see probes)."""
    p = probes(text, words, 1)
    return p[0] if p else ""


def page_of(span: str, pages: Sequence[str]) -> Optional[int]:
    """1-based page whose text contains `span` (normalised), or None."""
    n = norm_text(span)
    if not n:
        return None
    for i, p in enumerate(pages):
        if n in norm_text(p):
            return i + 1
    return None


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def read_parquet_rows(path: Path) -> List[dict]:
    import pyarrow.parquet as pq  # type: ignore

    return pq.read_table(path).to_pylist()


def batched(xs: Iterable, n: int):
    buf = []
    for x in xs:
        buf.append(x)
        if len(buf) == n:
            yield buf
            buf = []
    if buf:
        yield buf
