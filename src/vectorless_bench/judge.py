"""Optional LLM-as-judge for the end-to-end answer-quality axis.

Vectorless is a *retriever*, so retrieval quality (did it fetch the right
section?) is the primary axis and runs without any judge. This module adds the
downstream question: if you generate an answer from ONLY the retrieved context,
is that answer correct and faithful? That rewards a retriever for handing the
generator everything it needs and nothing that misleads it.

Validity controls baked in:
- the same judge model scores every system (set once in config);
- the judge is blind to which system produced the answer (no system name in the
  prompt), removing self-preference bias;
- judging is grounded against the dataset's gold answer, not the judge's own
  world knowledge, so it measures faithfulness to the source, not recall of
  pre-training.

Disabled by default (needs an API key). Costs are tracked and reported under a
separate "judge_usd" line so judging spend never contaminates system cost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional, Sequence

from ._llm import complete
from .schema import Question, RetrievedSection, Usage

_GEN_SYSTEM = (
    "Answer the question using ONLY the provided context. If the context is "
    "insufficient, reply exactly: INSUFFICIENT. Be concise."
)
_JUDGE_SYSTEM = (
    "You are a strict grader. Given a question, the reference answer, and a "
    "candidate answer, decide if the candidate is correct (same factual "
    "content as the reference; formatting differences are fine). Also rate "
    "whether the candidate is supported by the reference. Respond as JSON: "
    "{\"correct\": true|false, \"faithful\": true|false, \"reason\": \"...\"}."
)


def _loads_lenient(text: str) -> dict:
    """Parse a JSON verdict, tolerating markdown code fences and prose around
    the object (some models — e.g. GLM — wrap JSON in ```json fences)."""
    import re

    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
    return {"correct": False, "faithful": False}


@dataclass
class JudgeResult:
    correct: float
    faithful: float
    answered: float  # 0 if the generator said INSUFFICIENT
    usage: Usage


class Judge:
    def __init__(self, gen_model: str, judge_model: str) -> None:
        self.gen_model = gen_model
        self.judge_model = judge_model

    def evaluate(
        self,
        question: Question,
        sections: Sequence[RetrievedSection],
        native_answer: Optional[str] = None,
    ) -> JudgeResult:
        usage = Usage()
        # Answer-first systems (treewalk) emit their own answer; grade THAT
        # directly. Re-generating from the retrieved sections would measure a
        # different pipeline than the one under test and unfairly penalise a
        # system whose value is the answer, not the section selection.
        if native_answer is not None and native_answer.strip():
            candidate = native_answer.strip()
        else:
            context = "\n\n---\n\n".join(s.content for s in sections) or "(no context)"
            gen = complete(
                self.gen_model, _GEN_SYSTEM,
                f"CONTEXT:\n{context}\n\nQUESTION: {question.question}",
                max_tokens=400,
            )
            usage.add(gen.usage)
            candidate = gen.text.strip()

        if question.is_no_answer:
            # correct behaviour is to decline; reward INSUFFICIENT
            answered = 0.0 if candidate.upper().startswith("INSUFFICIENT") else 1.0
            return JudgeResult(
                correct=1.0 - answered, faithful=1.0 - answered,
                answered=answered, usage=usage,
            )
        if candidate.upper().startswith("INSUFFICIENT"):
            return JudgeResult(correct=0.0, faithful=1.0, answered=0.0, usage=usage)

        verdict = complete(
            self.judge_model, _JUDGE_SYSTEM,
            f"QUESTION: {question.question}\nREFERENCE: {question.answer}\n"
            f"CANDIDATE: {candidate}",
            max_tokens=200, json_mode=True,
        )
        usage.add(verdict.usage)
        v = _loads_lenient(verdict.text)
        return JudgeResult(
            correct=1.0 if v.get("correct") else 0.0,
            faithful=1.0 if v.get("faithful") else 0.0,
            answered=1.0,
            usage=usage,
        )
