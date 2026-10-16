"""Answer grading: is an answer written from ONLY a system's retrieved
passages correct?

Vectorless is a retriever, so retrieval quality is the primary axis and runs
without any grader. This module adds the downstream question that readers
care about: given what the system found, can the shared answer model answer
right?

1. Generate. The run's shared Gemini model writes an answer from the
   retrieved passages only (or replies INSUFFICIENT). The same model and
   prompt for every system, so the only thing that varies is what was
   retrieved.
2. Grade on Jev (HAL-3047). Jev picks one verdict from a closed set,
   correct / partially_correct / incorrect / abstained, against the
   dataset's gold answer. Nothing is generated at this step: a verdict is a
   choice with probabilities, not prose a model might bend (the owner's rule:
   every judgement is a closed set on Jev). The grader never sees which
   system produced the answer.
3. Repeat a sample. Jev is not deterministic (0.375 exact match across
   repeats, measured 2026-09), so ~10% of questions (chosen by a hash of
   the question id, the same ones for every system) are graded twice and the
   agreement is reported.

A judge_model that is not Jev falls back to a generative grader (the old
path), kept only so a run without a Jev key still works; its numbers are not
the published ones.

Spend (generation + grading) is reported as judge_usd, apart from the
system's own cost.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Sequence

from ._llm import complete
from .schema import Question, RetrievedSection, Usage

_GEN_SYSTEM = (
    "Answer the question using ONLY the provided context. If the context is "
    "insufficient, reply exactly: INSUFFICIENT. Be concise."
)
_VERDICTS = [
    ("correct", "The candidate answer states the same facts as the gold answer. Differences in wording, "
                "formatting, units or rounding that do not change the meaning are fine."),
    ("partially_correct", "The candidate gets part of the gold answer right but misses or misstates another part, "
                          "or is less precise than the gold answer in a way that matters."),
    ("incorrect", "The candidate contradicts the gold answer or answers something else."),
    ("abstained", "The candidate declines to answer or says the information is not available."),
]
_INSTRUCTIONS = ("Grade the candidate answer against the gold answer for the question in the state. "
                 "Judge only whether the candidate's facts match the gold answer.")
_LEGACY_JUDGE = (
    "You are a strict grader. Given a question, the reference answer, and a "
    "candidate answer, decide if the candidate is correct (same factual "
    "content as the reference; formatting differences are fine). Respond as JSON: "
    "{\"correct\": true|false, \"faithful\": true|false}."
)


@dataclass
class JudgeResult:
    correct: float
    faithful: float
    answered: float  # 0 if the generator said INSUFFICIENT
    usage: Usage
    verdict: str = ""
    verdict_repeat: Optional[str] = None
    confidence: Optional[float] = None
    # The answer the shared model wrote from the retrieved passages, kept for
    # the dataset-native answer metrics (QASPER F1, CUAD clause match).
    candidate: Optional[str] = None


def _sampled(qid: str, rate: float = 0.10) -> bool:
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16)
    return (h % 1000) < rate * 1000


class Judge:
    def __init__(self, gen_model: str, judge_model: str, repeat_rate: float = 0.10) -> None:
        self.gen_model = gen_model
        self.judge_model = judge_model
        self.repeat_rate = repeat_rate
        self._jev = None
        if judge_model.startswith("jev"):
            from .jev import Jev

            self._jev = Jev(model=judge_model)

    def evaluate(self, question: Question, sections: Sequence[RetrievedSection]) -> JudgeResult:
        usage = Usage()
        context = "\n\n---\n\n".join(s.content for s in sections) or "(no context)"
        gen = complete(self.gen_model, _GEN_SYSTEM, f"CONTEXT:\n{context}\n\nQUESTION: {question.question}", max_tokens=400)
        usage.add(gen.usage)
        candidate = gen.text.strip()
        declined = candidate.upper().startswith("INSUFFICIENT")

        if question.is_no_answer:
            # correct behaviour is to decline; reward INSUFFICIENT
            ok = 1.0 if declined else 0.0
            return JudgeResult(candidate=candidate, correct=ok, faithful=ok, answered=0.0 if declined else 1.0, usage=usage,
                               verdict="abstained" if declined else "answered_unanswerable")
        if declined:
            return JudgeResult(candidate=candidate, correct=0.0, faithful=1.0, answered=0.0, usage=usage, verdict="abstained")

        if self._jev is None:
            return self._legacy(question, candidate, usage)

        state = {"question": question.question, "gold_answer": question.answer or "", "candidate_answer": candidate}
        q = {"verdict": (_INSTRUCTIONS, _VERDICTS)}
        answers, ju, _ = self._jev.choose(state, q)
        usage.add(ju)
        v = answers["verdict"]
        repeat = None
        if _sampled(question.qid, self.repeat_rate):
            again, ju2, _ = self._jev.choose(state, q)
            usage.add(ju2)
            repeat = again["verdict"].choice
        return JudgeResult(candidate=candidate, 
            correct=1.0 if v.choice == "correct" else (0.5 if v.choice == "partially_correct" else 0.0),
            faithful=1.0 if v.choice in ("correct", "partially_correct") else 0.0,
            answered=0.0 if v.choice == "abstained" else 1.0,
            usage=usage, verdict=v.choice, verdict_repeat=repeat, confidence=v.confidence,
        )

    def _legacy(self, question: Question, candidate: str, usage: Usage) -> JudgeResult:
        verdict = complete(self.judge_model, _LEGACY_JUDGE,
                           f"QUESTION: {question.question}\nREFERENCE: {question.answer}\nCANDIDATE: {candidate}",
                           max_tokens=200, json_mode=True)
        usage.add(verdict.usage)
        try:
            v = json.loads(verdict.text)
        except json.JSONDecodeError:
            v = {"correct": False, "faithful": False}
        return JudgeResult(candidate=candidate, correct=1.0 if v.get("correct") else 0.0, faithful=1.0 if v.get("faithful") else 0.0,
                           answered=1.0, usage=usage, verdict="legacy")
