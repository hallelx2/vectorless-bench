"""A minimal client for TypeSafe's System One API (Jev), the same endpoint and
wire shape llmgate's judge/typesafe uses: POST {base}/v1/systemone with a
state, a model and named questions; each answer comes back typed.

Only the Choice primitive is used here: the grader asks Jev to pick one of a
closed set of verdicts, never to write one (HAL-3047, and the owner's rule
that every judgement in Vectorless is a closed set on Jev).
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .pricing import compute
from .schema import Usage

DEFAULT_BASE = "https://api.typesafe.ai"
DEFAULT_MODEL = "jev-latest"


@dataclass
class ChoiceAnswer:
    choice: str
    probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: Optional[float] = None


class Jev:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 model: Optional[str] = None, timeout: float = 60.0, retries: int = 4) -> None:
        self.key = api_key or os.environ.get("TYPESAFE_API_KEY") or os.environ.get("VLS_TYPESAFE_API_KEY") or ""
        if not self.key:
            raise RuntimeError("Jev needs TYPESAFE_API_KEY")
        self.base = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.model = model or os.environ.get("TYPESAFE_MODEL") or DEFAULT_MODEL
        self.timeout, self.retries = timeout, retries

    def choose(self, state: object, questions: Dict[str, Tuple[str, List[Tuple[str, str]]]]) -> Tuple[Dict[str, ChoiceAnswer], Usage, str]:
        """questions: id -> (instructions, [(option, description), ...]).
        Returns the answers, the priced usage, and the model that answered."""
        body = {
            "state": state,
            "model": self.model,
            "questions": {qid: {"type": "choice", "instructions": instr, "criteria": {k: v for k, v in opts}}
                          for qid, (instr, opts) in questions.items()},
        }
        raw = json.dumps(body).encode()
        delay = 1.0
        for attempt in range(self.retries + 1):
            req = urllib.request.Request(self.base + "/v1/systemone", data=raw, method="POST",
                                         headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 - fixed provider URL
                    data = json.loads(r.read())
                break
            except urllib.error.HTTPError as e:
                if e.code not in (429, 500, 502, 503, 504) or attempt == self.retries:
                    raise RuntimeError(f"jev {e.code}: {e.read()[:300]!r}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt == self.retries:
                    raise RuntimeError(f"jev transport: {e}") from e
            time.sleep(delay * (0.5 + random.random()))
            delay = min(delay * 2, 20.0)
        answers: Dict[str, ChoiceAnswer] = {}
        got = data.get("answers") or {}
        for qid, (_, opts) in questions.items():
            a = got.get(qid)
            if not isinstance(a, dict):
                raise RuntimeError(f"jev: no answer for {qid!r}")
            choice = str(a.get("choice", ""))
            if choice not in {k for k, _ in opts}:
                raise RuntimeError(f"jev: {qid!r} answered {choice!r}, not one of the offered options")
            answers[qid] = ChoiceAnswer(choice=choice, probabilities=a.get("probabilities") or {},
                                        confidence=a.get("confidence"))
        u = data.get("usage") or {}
        model = str(data.get("model") or self.model)
        in_tok, out_tok = int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0)
        usage = Usage(input_tokens=in_tok, output_tokens=out_tok, total_tokens=in_tok + out_tok,
                      cost_usd=compute(model if model.startswith("jev") else "jev-latest", in_tok, out_tok), llm_calls=1)
        return answers, usage, model
