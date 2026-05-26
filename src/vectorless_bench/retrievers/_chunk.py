"""Shared fixed-size chunking for the chunk-based baselines (vector RAG, BM25).

Deliberately the *standard* recipe — token-windowed chunks with overlap — so the
baselines represent how RAG is actually built, not a strawman. Chunks carry no
title_path (chunk systems destroy structure, which is the whole point of the
comparison), but they do carry the source doc_id and a char offset for auditing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from ..pricing import count_tokens
from ..schema import Doc


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    text: str
    start: int  # char offset in the source, for audit/citation


def chunk_doc(
    doc: Doc, chunk_tokens: int = 512, overlap_tokens: int = 64, model: str = "gpt-4o"
) -> List[Chunk]:
    """Split a document into overlapping token windows.

    Uses a char-per-token estimate to place window boundaries cheaply, then
    snaps to whitespace so chunks don't cut mid-word. Exact token counts aren't
    needed for boundary placement — retrieval quality is insensitive to a few
    tokens of slack, and we avoid encoding the whole corpus twice."""
    text = doc.content
    if not text.strip():
        return []
    cpt = max(1, len(text) // max(1, count_tokens(text, model)))  # chars/token
    win = chunk_tokens * cpt
    step = max(1, (chunk_tokens - overlap_tokens) * cpt)

    chunks: List[Chunk] = []
    i = 0
    n = len(text)
    idx = 0
    while i < n:
        end = min(n, i + win)
        # snap end to the next whitespace to avoid splitting a word
        if end < n:
            j = text.rfind(" ", i, end)
            if j > i:
                end = j
        piece = text[i:end].strip()
        if piece:
            chunks.append(Chunk(doc.doc_id, f"{doc.doc_id}::ch{idx}", piece, i))
            idx += 1
        if end >= n:
            break
        i += step
    return chunks
