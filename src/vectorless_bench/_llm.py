"""A tiny multi-provider completion helper for the baselines that actually call
an LLM (full-context retriever, LLM-judge).

It mirrors what llmgate does for the engine — route by model name, pull real
token usage from the provider, price it with the shared table — so an LLM-using
baseline is costed on exactly the same basis as Vectorless. Providers are
imported lazily; only the one you use needs to be installed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

from .pricing import compute
from .schema import Usage

# Per-client SDK retries (anthropic/openai auto-retry APIConnectionError, 408,
# 409, 429, and >=500 with exponential backoff — default is only 2). A long
# judged run (hundreds of LLM calls) WILL hit a transient z.ai/provider blip; a
# generous client retry budget keeps a single hiccup from killing the whole run.
_CLIENT_MAX_RETRIES = 8
_CLIENT_TIMEOUT = 120.0

# Belt-and-suspenders: an outer retry around the whole provider call, in case
# the SDK's retries are exhausted (e.g. a sustained ~minute outage) or the
# provider raises a transient error the SDK doesn't classify as retryable.
_OUTER_RETRIES = 4


@dataclass
class Completion:
    text: str
    usage: Usage


def _is_transient(exc: Exception) -> bool:
    """A network/availability error worth retrying — never an auth/validation
    error (those fail fast). Classified by exception type name + message so we
    don't have to import every provider's exception hierarchy."""
    name = type(exc).__name__.lower()
    transient_types = (
        "connection", "timeout", "apiconnection", "apitimeout",
        "internalserver", "overloaded", "ratelimit", "serviceunavailable",
    )
    if any(t in name for t in transient_types):
        return True
    msg = str(exc).lower()
    transient_msgs = (
        "connection error", "connection reset", "connection aborted",
        "timed out", "timeout", "temporarily unavailable", "overloaded",
        "rate limit", "502", "503", "504", "10054", "10053", "10060",
    )
    return any(m in msg for m in transient_msgs)


def _provider(model: str) -> str:
    if model.startswith("glm"):
        return "glm"  # Zhipu / Z.ai — Anthropic-compatible gateway
    if model.startswith("claude"):
        return "anthropic"
    if model.startswith("gemini"):
        return "gemini"
    return "openai"  # gpt-*, o3/o4-*


def complete(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
) -> Completion:
    p = _provider(model)

    def _once() -> Completion:
        if p == "glm":
            return _glm(model, system, user, max_tokens, temperature)
        if p == "anthropic":
            return _anthropic(model, system, user, max_tokens, temperature)
        if p == "gemini":
            return _gemini(model, system, user, max_tokens, temperature)
        return _openai(model, system, user, max_tokens, temperature, json_mode)

    # Outer retry: survive a transient provider/network blip that outlives the
    # SDK's own retry budget, so one hiccup doesn't abort an hours-long run.
    last: Exception | None = None
    for attempt in range(_OUTER_RETRIES + 1):
        try:
            return _once()
        except Exception as e:  # noqa: BLE001 — re-raised below if not transient
            last = e
            if not _is_transient(e) or attempt == _OUTER_RETRIES:
                raise
            import sys
            delay = min(2.0 * (2 ** attempt), 30.0)
            print(f"[vlbench] transient LLM error ({type(e).__name__}), "
                  f"retry {attempt + 1}/{_OUTER_RETRIES} in {delay:.0f}s: {str(e)[:80]}",
                  file=sys.stderr)
            time.sleep(delay)
    raise last  # type: ignore[misc]


def _glm(model, system, user, max_tokens, temperature) -> Completion:
    """GLM via the Anthropic-compatible gateway (Z.ai). Lets the LLM-judge
    axis run on the same provider the engine uses, with no OpenAI key."""
    import os

    import anthropic  # type: ignore

    base = (
        os.environ.get("VLBENCH_GLM_BASE_URL")
        or os.environ.get("VLE_LLM_ANTHROPIC_BASE_URL")
        or "https://api.z.ai/api/anthropic"
    )
    # The anthropic Python SDK appends "/v1/messages" itself, so the base must
    # NOT end in /v1 (unlike the Go engine's client, whose default base already
    # has /v1). Strip it so the same env var works for both.
    base = base.rstrip("/")
    if base.endswith("/v1"):
        base = base[: -len("/v1")]
    key = (
        os.environ.get("VLBENCH_GLM_API_KEY")
        or os.environ.get("VLE_LLM_ANTHROPIC_API_KEY")
        or os.environ.get("GLM_API_KEY", "")
    )
    client = anthropic.Anthropic(
        base_url=base, api_key=key,
        max_retries=_CLIENT_MAX_RETRIES, timeout=_CLIENT_TIMEOUT,
    )
    r = client.messages.create(
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    in_tok, out_tok = int(r.usage.input_tokens), int(r.usage.output_tokens)
    return Completion(text=text, usage=_usage(model, in_tok, out_tok))


def _openai(model, system, user, max_tokens, temperature, json_mode) -> Completion:
    from openai import OpenAI  # type: ignore

    client = OpenAI(max_retries=_CLIENT_MAX_RETRIES, timeout=_CLIENT_TIMEOUT)
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    r = client.chat.completions.create(**kwargs)
    u = r.usage
    in_tok, out_tok = int(u.prompt_tokens), int(u.completion_tokens)
    return Completion(
        text=r.choices[0].message.content or "",
        usage=_usage(model, in_tok, out_tok),
    )


def _anthropic(model, system, user, max_tokens, temperature) -> Completion:
    import anthropic  # type: ignore

    client = anthropic.Anthropic(
        max_retries=_CLIENT_MAX_RETRIES, timeout=_CLIENT_TIMEOUT,
    )
    r = client.messages.create(
        model=model,
        system=system,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
    in_tok, out_tok = int(r.usage.input_tokens), int(r.usage.output_tokens)
    return Completion(text=text, usage=_usage(model, in_tok, out_tok))


def _gemini(model, system, user, max_tokens, temperature) -> Completion:
    from google import genai  # type: ignore

    client = genai.Client()
    r = client.models.generate_content(
        model=model,
        contents=f"{system}\n\n{user}",
        config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    um = getattr(r, "usage_metadata", None)
    in_tok = int(getattr(um, "prompt_token_count", 0) or 0)
    out_tok = int(getattr(um, "candidates_token_count", 0) or 0)
    return Completion(text=r.text or "", usage=_usage(model, in_tok, out_tok))


def _usage(model: str, in_tok: int, out_tok: int) -> Usage:
    return Usage(
        input_tokens=in_tok,
        output_tokens=out_tok,
        total_tokens=in_tok + out_tok,
        cost_usd=compute(model, in_tok, out_tok),
        llm_calls=1,
    )
