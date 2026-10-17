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
from dataclasses import asdict, dataclass, field
from typing import Any, List, Optional, Sequence

from ._llm import _patiently, complete
from .calc import resolve
from .schema import Question, RetrievedSection, Usage

# The shared answer step's prompt, versioned: every graded row records the
# version it was written with, so a scoreboard never compares answers from
# two prompts (HAL-3203). v1 joined unlabelled excerpts and declined
# anything not stated outright: on the pilot's Vectorless declines it
# answered 3 of 15 whose gold page it had been given. v3 labels the
# excerpts, lets the model derive and judge, and does the arithmetic in
# code (calc.py): the model sets each calculation up, it does not work it
# out. v4: see _V4.
_V3 = (
    "You answer a question from numbered excerpts of a document. Use only what the excerpts show. "
    "When the question asks for a figure the excerpts let you derive (a ratio, margin, growth rate, change, "
    "average or sum), find each input in the excerpts and set the calculation up, but do not work it out: "
    "write it as <<expression>> using only the numbers from the excerpts, + - * / and parentheses, and it is "
    "computed for you. For example: operating margin = <<11,512 / 86,392 * 100>>%. "
    "When it asks for a judgement (is a business cyclical, is a company growing fast, what drove a change), "
    "answer it from the facts the excerpts state and name those facts. "
    "Cite the excerpts you used by number, like [2]. Be concise. "
    "Reply exactly INSUFFICIENT, and nothing else, only when the excerpts lack an input the answer needs."
)
# v4 (HAL-3247), kept for experiments, not the default: rules for "none"
# answers, a question's own alternative and judgement questions. Measured on
# the 100 FinanceBench questions outside the pilot's sample, it declined 27%
# against v3's 29% but answered 21% wrongly against 18%, so v3 stays. In
# 21 of v3's 29 declines the gold page was in the top five: the prompt is not
# what stops the answer.
_V4 = (
    "You answer a question from numbered excerpts of a document. Use only what the excerpts show.\n"
    "- A figure the excerpts let you derive (a ratio, margin, growth rate, change, average or sum): find each "
    "input in the excerpts and set the calculation up, but do not work it out. Write it as <<expression>> using "
    "only numbers from the excerpts, + - * / and parentheses, and it is computed for you, for example "
    "operating margin = <<11,512 / 86,392 * 100>>%. Apply the question's own definitions term by term; an "
    "average of two values is their sum divided by 2.\n"
    "- A judgement (is a business cyclical, is a company growing fast, what drove a change): answer yes or no, "
    "or name the cause, from the facts and figures the excerpts state, and name them.\n"
    "- \"None\" is an answer: when the excerpts cover what the question asks about (the acquisitions note, the "
    "vote results, the segment table) and show no such item, say there was none and cite where.\n"
    "- When the question offers its own alternative (\"if X is not a useful metric, say so and explain why\"), "
    "that alternative is a full answer when the excerpts support it.\n"
    "Cite the excerpts you used by number, like [2]. Be concise. Reply exactly INSUFFICIENT, and nothing else, "
    "only when the excerpts lack an input the answer needs and none of the above applies."
)
PROMPTS = {"excerpts-calc-v3": _V3, "excerpts-calc-v4": _V4}
ANSWER_PROMPT = "excerpts-calc-v3"
_GEN_SYSTEM = PROMPTS[ANSWER_PROMPT]
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
    prompt: str = ANSWER_PROMPT
    # The calculations the answer set up and code evaluated.
    calculations: List[Any] = field(default_factory=list)


def _label(i: int, s: RetrievedSection) -> str:
    """`[n] document — section, p.N`: whatever the system said about where
    the excerpt is from. A chunk retriever that returns bare text gets a
    bare number; nothing is invented for it."""
    where = s.title or " › ".join(s.title_path)
    parts = [p for p in (s.doc_id, where) if p]
    label = f"[{i}]"
    if parts:
        label += " " + " — ".join(parts)
    if s.page is not None and not where.endswith(f"p.{s.page}"):
        label += f", p.{s.page}"
    return label


def excerpts(sections: Sequence[RetrievedSection]) -> str:
    """The numbered, labelled excerpts the shared answer step reads."""
    if not sections:
        return "(no excerpts)"
    return "\n\n".join(f"{_label(i, s)}\n{s.content}" for i, s in enumerate(sections, 1))


def _sampled(qid: str, rate: float = 0.10) -> bool:
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16)
    return (h % 1000) < rate * 1000


class Judge:
    def __init__(self, gen_model: str, judge_model: str, repeat_rate: float = 0.10,
                 prompt: str = ANSWER_PROMPT) -> None:
        if prompt not in PROMPTS:
            raise ValueError(f"unknown answer prompt {prompt!r}; known: {sorted(PROMPTS)}")
        self.prompt = prompt
        self.system = PROMPTS[prompt]
        self.patience = 90.0  # seconds before the first patient retry; tests shorten it
        self.gen_model = gen_model
        self.judge_model = judge_model
        self.repeat_rate = repeat_rate
        self._jev = None
        if judge_model.startswith("jev"):
            from .jev import Jev

            self._jev = Jev(model=judge_model)

    def evaluate(self, question: Question, sections: Sequence[RetrievedSection],
                 answer_parts: Optional[List[Any]] = None) -> JudgeResult:
        usage = Usage()
        if answer_parts:
            # the system's own context (full context: the whole document it read)
            from google.genai import types  # type: ignore

            from ._llm import generate

            parts = list(answer_parts) + [types.Part(text=f"QUESTION: {question.question}")]
            # patient like the index builds: a 429 spell must not leave the
            # row ungraded (HAL-3262)
            gen = _patiently(generate, self.gen_model, attempts=4, pause=self.patience, system=self.system, contents=[types.Content(role="user", parts=parts)],
                           max_tokens=600)
        else:
            # patient like the index builds: a 429 spell must not leave the
            # row ungraded (HAL-3262)
            gen = _patiently(complete, self.gen_model, self.system,
                             f"EXCERPTS:\n{excerpts(sections)}\n\nQUESTION: {question.question}",
                             attempts=4, pause=self.patience, max_tokens=600)
        usage.add(gen.usage)
        candidate, calcs = resolve(gen.text.strip())
        calculations = [asdict(c) for c in calcs]
        declined = candidate.upper().startswith("INSUFFICIENT")

        if question.is_no_answer:
            # correct behaviour is to decline; reward INSUFFICIENT
            ok = 1.0 if declined else 0.0
            return JudgeResult(prompt=self.prompt, candidate=candidate, calculations=calculations, correct=ok, faithful=ok, answered=0.0 if declined else 1.0, usage=usage,
                               verdict="abstained" if declined else "answered_unanswerable")
        if declined:
            return JudgeResult(prompt=self.prompt, candidate=candidate, calculations=calculations, correct=0.0, faithful=1.0, answered=0.0, usage=usage, verdict="abstained")

        if self._jev is None:
            res = self._legacy(question, candidate, usage)
            res.calculations = calculations
            return res

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
        return JudgeResult(prompt=self.prompt, candidate=candidate, calculations=calculations,
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
        return JudgeResult(prompt=self.prompt, candidate=candidate, correct=1.0 if v.get("correct") else 0.0, faithful=1.0 if v.get("faithful") else 0.0,
                           answered=1.0, usage=usage, verdict="legacy")
