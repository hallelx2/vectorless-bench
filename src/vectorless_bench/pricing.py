"""Token counting and USD cost — a faithful mirror of the engine's pricing.

The table below is copied verbatim from the engine's
`llmgate/pricing/pricing.go` (defaultPrices, "as of April 2026", USD per
million tokens). Keeping it identical is what makes a Vectorless query and a
baseline query comparable on cost: both are priced from the same book.

If you bump prices in the engine, bump them here and the manifest will record
the change (see manifest.pricing_fingerprint).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# Chat/completion models — (input_per_mtok, output_per_mtok), USD.
# Source: llmgate/pricing/pricing.go defaultPrices.
# ---------------------------------------------------------------------------
PRICES: Dict[str, Tuple[float, float]] = {
    # Anthropic
    "claude-sonnet-4-5": (3.00, 15.00),
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-1": (15.00, 75.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-haiku-3-5": (0.80, 4.00),
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "o3": (2.00, 8.00),
    "o3-mini": (1.10, 4.40),
    "o4-mini": (1.10, 4.40),
    # Google
    "gemini-2.5-flash": (0.15, 0.60),
    "gemini-2.5-pro": (1.25, 10.00),
    "gemini-2.0-flash": (0.10, 0.40),
}

# Embedding models (input only), USD per million tokens. Needed to price the
# vector-RAG baseline's ingest + query-embedding cost on the same basis.
EMBEDDING_PRICES: Dict[str, float] = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
    "text-embedding-ada-002": 0.10,
}

# When True, an unknown model raises instead of silently costing $0 — turn on
# in CI so a typo in a model name can't quietly zero out a cost comparison.
STRICT = False


def lookup(model: str) -> Optional[Tuple[float, float]]:
    return PRICES.get(model)


def compute(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for a completion call. Unknown model -> 0.0 (matches engine), or
    raises if STRICT."""
    p = PRICES.get(model)
    if p is None:
        if STRICT:
            raise KeyError(f"no price for model {model!r}; add it to PRICES")
        return 0.0
    return (input_tokens * p[0] + output_tokens * p[1]) / 1_000_000.0


def compute_embedding(model: str, tokens: int) -> float:
    p = EMBEDDING_PRICES.get(model)
    if p is None:
        if STRICT:
            raise KeyError(f"no embedding price for {model!r}")
        return 0.0
    return tokens * p / 1_000_000.0


# ---------------------------------------------------------------------------
# Token counting. Prefer tiktoken; fall back to the engine's own len/4 heuristic
# (tree.Section.TokenCount uses ~4 chars/token) so behaviour degrades gracefully
# with zero hard dependency.
# ---------------------------------------------------------------------------
_ENCODER_CACHE: Dict[str, object] = {}


def _encoding_name(model: str) -> str:
    if model.startswith(("gpt-4o", "gpt-4.1", "o3", "o4")):
        return "o200k_base"
    if model.startswith(("gpt-4", "gpt-3.5", "text-embedding-3")):
        return "cl100k_base"
    # Anthropic/Gemini have no public tiktoken encoder; cl100k is a close proxy
    # for *relative* comparison, which is all the benchmark needs.
    return "cl100k_base"


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Best-effort token count. tiktoken if available, else ~4 chars/token."""
    if not text:
        return 0
    try:
        import tiktoken  # type: ignore

        name = _encoding_name(model)
        enc = _ENCODER_CACHE.get(name)
        if enc is None:
            enc = tiktoken.get_encoding(name)
            _ENCODER_CACHE[name] = enc
        return len(enc.encode(text))  # type: ignore[union-attr]
    except Exception:
        # Engine's approximation: tree.go uses len(content)/4.
        return max(1, len(text) // 4)


def pricing_fingerprint() -> str:
    """Stable hash of the active price book, recorded in the run manifest so a
    silent price change is always attributable."""
    import hashlib
    import json

    blob = json.dumps(
        {"chat": PRICES, "embed": EMBEDDING_PRICES}, sort_keys=True
    ).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
