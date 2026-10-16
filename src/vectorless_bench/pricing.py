"""Token counting and USD cost, mirroring the engine's price book.

Every generative call in this benchmark is Gemini on Vertex AI, so the table
is the Vertex list price for the Gemini models a run may use, copied from
llmgate's `pricing/google.go` (read from cloud.google.com/vertex-ai/
generative-ai/pricing on 2026-10-16): global endpoint, ≤200K-token prompts,
standard on-demand tier, USD per million tokens. Keeping it identical to the
engine's is what makes a Vectorless query and a baseline query comparable on
cost: both are priced from the same book.

Gemini 3.6–3.8 Flash list at $1.50/$7.50 with an introductory 50% credit-back
through 2026-12-31; the invoice line is the list price, so that is what is
recorded. Jev (TypeSafe's System One model) is priced here too, because the
Vectorless system and the Jev grader both spend on it.

If a price changes, change it here and in llmgate; the manifest records a
fingerprint of this table (see pricing_fingerprint).
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

# (input, output, cached input) per 1M tokens, USD.
PRICES: Dict[str, Tuple[float, float, float]] = {
    "gemini-2.5-flash-lite": (0.10, 0.40, 0.01),
    "gemini-2.5-flash": (0.30, 2.50, 0.03),
    "gemini-2.5-pro": (1.25, 10.00, 0.125),
    "gemini-3.1-flash-lite": (0.25, 1.50, 0.025),
    "gemini-3.5-flash-lite": (0.30, 2.50, 0.03),
    "gemini-3.5-flash": (1.50, 9.00, 0.15),
    "gemini-3.6-flash": (1.50, 7.50, 0.15),
    "gemini-3.7-flash": (1.50, 7.50, 0.15),
    "gemini-3.8-flash": (1.50, 7.50, 0.15),
    "gemini-3.1-pro": (2.00, 12.00, 0.20),
    # TypeSafe Jev: input only; output is free (docs.typesafe.ai, 2026-09-17).
    "jev-1.13.0": (0.042, 0.0, 0.042),
    "jev-latest": (0.042, 0.0, 0.042),
}

# Embedding models, USD per 1M input tokens. Vertex list, 2026-10-16:
# gemini-embedding-2 is $0.20 online and $0.10 in batch.
EMBEDDING_PRICES: Dict[str, float] = {
    "gemini-embedding-2": 0.20,
    "gemini-embedding-2-batch": 0.10,
}

# When True, an unknown model raises instead of silently costing $0 — on in the
# runner, so a typo in a model name can't quietly zero out a cost comparison.
STRICT = False


def _resolve(model: str) -> Optional[str]:
    """Exact ID first, then the longest table key the ID starts with, so a
    versioned ID ("gemini-2.5-flash-lite-001") prices as its model and
    "gemini-2.5-flash-lite" never resolves to "gemini-2.5-flash"."""
    if model in PRICES:
        return model
    best = None
    for k in PRICES:
        if model.startswith(k) and (best is None or len(k) > len(best)):
            best = k
    return best


def lookup(model: str) -> Optional[Tuple[float, float, float]]:
    k = _resolve(model)
    return PRICES[k] if k else None


def compute(model: str, input_tokens: int, output_tokens: int, cached_tokens: int = 0) -> float:
    """USD for one call. `input_tokens` excludes `cached_tokens`, which bill at
    the cached rate. Unknown model -> 0.0, or raises if STRICT."""
    p = lookup(model)
    if p is None:
        if STRICT:
            raise KeyError(f"no price for model {model!r}; add it to PRICES")
        return 0.0
    return (input_tokens * p[0] + output_tokens * p[1] + cached_tokens * p[2]) / 1_000_000.0


def compute_embedding(model: str, tokens: int) -> float:
    # A local model (a Hugging Face id such as BAAI/bge-m3) costs nothing per
    # token; its price is the setup time the report already carries.
    if "/" in model:
        return 0.0
    p = EMBEDDING_PRICES.get(model)
    if p is None:
        if STRICT:
            raise KeyError(f"no embedding price for {model!r}")
        return 0.0
    return tokens * p / 1_000_000.0


def count_tokens(text: str, model: str = "") -> int:
    """A local, deterministic token estimate (~4 characters per token, the
    engine's own tree.Section.TokenCount rule) for chunk sizing and budgets.
    Billing never uses it: billed tokens always come from the provider's
    usage. For the model's exact count use `_llm.count_tokens`."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def pricing_fingerprint() -> str:
    """Stable hash of the active price book, recorded in the run manifest so a
    silent price change is always attributable."""
    import hashlib
    import json

    blob = json.dumps({"chat": PRICES, "embed": EMBEDDING_PRICES}, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]
