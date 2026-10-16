"""Model calls for the baselines that use a generative model (full-context,
PageIndex, the agentic rivals, the answer step) — all of them on Gemini through
Google Cloud Vertex AI.

Why Vertex only: the benchmark's model spend is paid by Google Cloud credits,
which cover Gemini on Vertex and nothing else, and the comparison is only fair
if every system answers with the SAME model and the SAME thinking cap
(owner decision, 2026-10-16: "everything vertex ai gemini models or bge
models"). `require_vertex_model` enforces it; a config naming anything else
fails before a run starts rather than half-way through.

It mirrors what llmgate's provider/vertex does for the engine — real token
usage from usageMetadata (cached prompt tokens and thinking tokens included),
priced from the same book — so a baseline's cost is computed exactly like
Vectorless's.

Authentication is Application Default Credentials: the benchmark VM's service
account, or `gcloud auth application-default login` on a laptop. The project
and location come from GOOGLE_CLOUD_PROJECT / GOOGLE_CLOUD_LOCATION
(VLBENCH_GCP_PROJECT / VLBENCH_GCP_LOCATION override them).
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .pricing import compute
from .schema import Usage

DEFAULT_MODEL = "gemini-3.5-flash-lite"  # HAL-3048: 2.5 Flash-Lite is refused by Vertex shared capacity at benchmark pace

_client = None
_client_lock = threading.Lock()

# Gemini 2.x/3.x text models on Vertex run on Dynamic Shared Quota: no
# per-project limit to raise, and 429 RESOURCE_EXHAUSTED when the shared pool
# is busy (measured 2026-10-16: 2.5 Flash-Lite refused 46 of 48 calls at 16
# in flight; 3.5 Flash-Lite took 16 cleanly, 32 with 8 of 96 refused). One
# gate bounds calls in flight across every system in the process, and a
# refused call is retried with jittered backoff. Retries are counted on the
# completion so latency the provider caused is visible in the record.
_GATE = threading.BoundedSemaphore(int(os.environ.get("VLBENCH_LLM_CONCURRENCY", "8")))
_MAX_RETRIES = int(os.environ.get("VLBENCH_LLM_RETRIES", "6"))


def project() -> str:
    p = os.environ.get("VLBENCH_GCP_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT") or ""
    if not p:
        raise RuntimeError("set GOOGLE_CLOUD_PROJECT (or VLBENCH_GCP_PROJECT) to the Vertex project to bill")
    return p


def location() -> str:
    return os.environ.get("VLBENCH_GCP_LOCATION") or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"


def thinking_budget() -> Optional[int]:
    """The thinking cap every generative call uses, from VLBENCH_THINKING_BUDGET
    (default 0: no thinking, on the models that allow it). An empty value
    leaves the model's own default, which on Gemini 3 is to think."""
    raw = os.environ.get("VLBENCH_THINKING_BUDGET", "0").strip()
    if raw == "":
        return None
    return int(raw)


def genai_client():
    """One shared google-genai client in Vertex mode."""
    global _client
    with _client_lock:
        if _client is None:
            from google import genai  # type: ignore

            _client = genai.Client(vertexai=True, project=project(), location=location())
        return _client


def require_vertex_model(model: str) -> None:
    """Refuse anything that is not a Gemini model served by Vertex."""
    if not model or not model.startswith("gemini"):
        raise ValueError(
            f"model {model!r} is not a Gemini model on Vertex; this benchmark runs every "
            "generative call on Vertex Gemini (see _llm.py)"
        )


@dataclass
class Completion:
    text: str
    usage: Usage
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    finish_reason: str = ""
    # The model's turn exactly as returned; a tool loop appends it to the
    # conversation unchanged (it carries Gemini 3's thought signatures).
    raw_content: Any = None
    retries: int = 0


def usage_from(model: str, um: Any) -> Usage:
    """Normalise Vertex usageMetadata: cached prompt tokens are part of
    prompt_token_count and bill at the cached rate; thinking tokens are
    reported beside candidate tokens and bill as output (measured 2026-10-16:
    prompt 14 + candidates 3 + thoughts 56 = total 73)."""
    prompt = int(getattr(um, "prompt_token_count", 0) or 0)
    cached = min(int(getattr(um, "cached_content_token_count", 0) or 0), prompt)
    cand = int(getattr(um, "candidates_token_count", 0) or 0)
    thoughts = int(getattr(um, "thoughts_token_count", 0) or 0)
    tool_prompt = int(getattr(um, "tool_use_prompt_token_count", 0) or 0)
    in_tok = prompt - cached + tool_prompt
    out_tok = cand + thoughts
    return Usage(
        input_tokens=in_tok + cached,
        output_tokens=out_tok,
        total_tokens=int(getattr(um, "total_token_count", 0) or (prompt + out_tok + tool_prompt)),
        cost_usd=compute(model, in_tok, out_tok, cached_tokens=cached),
        llm_calls=1,
    )


def complete(
    model: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
    cached_content: Optional[str] = None,
) -> Completion:
    """One generateContent call on Vertex."""
    return generate(
        model,
        system=system,
        contents=[{"role": "user", "parts": [{"text": user}]}],
        max_tokens=max_tokens,
        temperature=temperature,
        json_mode=json_mode,
        cached_content=cached_content,
    )


def generate(
    model: str,
    *,
    system: str = "",
    contents: List[Any],
    max_tokens: int = 1024,
    temperature: float = 0.0,
    json_mode: bool = False,
    tools: Optional[List[Any]] = None,
    cached_content: Optional[str] = None,
) -> Completion:
    """generateContent with full control: multi-turn contents, function tools
    and a context cache. Used directly by the agentic rivals."""
    require_vertex_model(model)
    from google.genai import types  # type: ignore

    cfg: Dict[str, Any] = {"max_output_tokens": max_tokens, "temperature": temperature}
    if system and not cached_content:
        # A cached context carries its own system instruction.
        cfg["system_instruction"] = system
    tb = thinking_budget()
    if tb is not None:
        cfg["thinking_config"] = types.ThinkingConfig(thinking_budget=tb)
    if json_mode:
        cfg["response_mime_type"] = "application/json"
    if tools:
        cfg["tools"] = tools
    if cached_content:
        cfg["cached_content"] = cached_content

    import random
    import time as _time

    retries = 0
    delay = 1.0
    while True:
        try:
            with _GATE:
                r = genai_client().models.generate_content(
                    model=model, contents=contents, config=types.GenerateContentConfig(**cfg)
                )
            break
        except Exception as e:
            msg = str(e)
            busy = "429" in msg or "RESOURCE_EXHAUSTED" in msg or "503" in msg or "UNAVAILABLE" in msg
            if not busy or retries >= _MAX_RETRIES:
                raise
            retries += 1
            _time.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, 30.0)
    calls: List[Dict[str, Any]] = []
    text_parts: List[str] = []
    finish = ""
    raw = None
    if r.candidates:
        cand = r.candidates[0]
        finish = str(getattr(cand, "finish_reason", "") or "")
        raw = cand.content
        for p in (cand.content.parts if cand.content and cand.content.parts else []):
            if getattr(p, "function_call", None) is not None:
                calls.append({"name": p.function_call.name, "args": dict(p.function_call.args or {}), "part": p})
            elif getattr(p, "text", None) and not getattr(p, "thought", False):
                text_parts.append(p.text)
    return Completion(
        text="".join(text_parts),
        usage=usage_from(model, getattr(r, "usage_metadata", None)),
        tool_calls=calls,
        finish_reason=finish,
        raw_content=raw,
        retries=retries,
    )


def count_tokens(model: str, text: str) -> int:
    """The model's own token count, from Vertex's countTokens."""
    require_vertex_model(model)
    r = genai_client().models.count_tokens(model=model, contents=text)
    return int(r.total_tokens or 0)
