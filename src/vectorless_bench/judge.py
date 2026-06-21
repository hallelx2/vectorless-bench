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
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from ._llm import complete
from .schema import Question, RetrievedSection, Usage

_GEN_SYSTEM = (
    "Answer the question using ONLY the provided context. If the context is "
    "insufficient, reply exactly: INSUFFICIENT. Be concise."
)
_JUDGE_SYSTEM = (
    "You grade a CANDIDATE answer against a REFERENCE (gold) answer for a "
    "question over a document. The reference is often PARTIAL — it may list only "
    "some of the facts a full answer contains.\n"
    "\n"
    "Grade by FACT RECALL, not by exact string or subset match:\n"
    "- `correct: true` iff the candidate conveys EVERY fact stated in the "
    "reference AND contradicts NONE of them. A number must match within normal "
    "rounding; an entity/date/name must match.\n"
    "- Extra facts in the candidate that go BEYOND the reference but do not "
    "conflict with it MUST NOT lower correctness — a more complete answer is not "
    "wrong. Only a MISSING reference fact or a CONTRADICTION (wrong number, "
    "wrong entity, wrong date, opposite claim) makes it incorrect.\n"
    "- `faithful: true` iff the candidate makes no claim that contradicts the "
    "reference (extra unverifiable-but-non-conflicting detail is still faithful).\n"
    "\n"
    "List the specific reference facts the candidate omitted (`missing_facts`) and "
    "any candidate claims that conflict with the reference (`contradicting_facts`). "
    "Respond as JSON: {\"correct\": true|false, \"faithful\": true|false, "
    "\"missing_facts\": [\"...\"], \"contradicting_facts\": [\"...\"], "
    "\"reason\": \"...\"}."
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
    # Human-review trail: the exact answer that was graded, the gold reference
    # it was graded against, and the judge's own rationale. These let a person
    # re-check any verdict; they're persisted, never used in the scores.
    candidate: str = ""
    reference: str = ""
    reason: str = ""
    # Gold facts the candidate omitted, and candidate claims that conflict with
    # the reference. Drives the recall-based verdict and the human review trail.
    missing_facts: List[str] = field(default_factory=list)
    contradicting_facts: List[str] = field(default_factory=list)


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

        reference = (question.answer or "").strip()

        if question.is_no_answer:
            # correct behaviour is to decline; reward INSUFFICIENT
            answered = 0.0 if candidate.upper().startswith("INSUFFICIENT") else 1.0
            return JudgeResult(
                correct=1.0 - answered, faithful=1.0 - answered,
                answered=answered, usage=usage,
                candidate=candidate, reference=reference,
                reason="no-answer question: correct behaviour is to decline"
                + (" (declined)" if answered == 0.0 else " (over-answered)"),
            )
        if candidate.upper().startswith("INSUFFICIENT"):
            return JudgeResult(
                correct=0.0, faithful=1.0, answered=0.0, usage=usage,
                candidate=candidate, reference=reference,
                reason="generator declined (INSUFFICIENT context)",
            )

        verdict = complete(
            self.judge_model, _JUDGE_SYSTEM,
            f"QUESTION: {question.question}\nREFERENCE: {question.answer}\n"
            f"CANDIDATE: {candidate}",
            max_tokens=200, json_mode=True,
        )
        usage.add(verdict.usage)
        v = _loads_lenient(verdict.text)

        def _facts(key: str) -> List[str]:
            val = v.get(key) or []
            if isinstance(val, str):
                val = [val] if val.strip() else []
            return [str(x).strip() for x in val if str(x).strip()]

        missing = _facts("missing_facts")
        contradicting = _facts("contradicting_facts")
        # Recall-based correctness, enforced structurally from the fact lists so
        # the verdict is auditable and the "penalized for extra detail" bug can't
        # recur: a candidate is correct IFF it omits no reference fact and
        # contradicts none. Extra non-conflicting detail never lands in either
        # list, so a more-complete answer keeps full credit. (review.md shows the
        # exact missing/contradicting facts behind every call.)
        correct = not missing and not contradicting
        return JudgeResult(
            correct=1.0 if correct else 0.0,
            faithful=1.0 if (v.get("faithful") and not contradicting) else 0.0,
            answered=1.0,
            usage=usage,
            candidate=candidate, reference=reference,
            reason=str(v.get("reason", "")).strip(),
            missing_facts=missing, contradicting_facts=contradicting,
        )
