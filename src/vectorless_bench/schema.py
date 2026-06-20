"""Core data model for the vectorless-bench harness.

Everything here is plain dataclasses with no third-party dependencies so the
harness, metrics, and smoke tests run under a bare Python install. The shapes
deliberately mirror the engine's own `retrieval.Usage` / query response so that
Vectorless numbers and baseline numbers are directly comparable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class AnswerType(str, Enum):
    """How a question must be answered. Drives which metrics are meaningful.

    - LOOKUP: the answer lives verbatim in one section (single-hop fact).
    - MULTIHOP: the answer requires combining two or more sections.
    - NUMERIC: the answer is a number; scoring normalises currency/units.
    - NO_ANSWER: the corpus does NOT contain the answer; the correct behaviour
      is to retrieve nothing / abstain. Measures over-retrieval.
    """

    LOOKUP = "lookup"
    MULTIHOP = "multihop"
    NUMERIC = "numeric"
    NO_ANSWER = "no_answer"


@dataclass
class Usage:
    """Token + cost accounting for a single retrieval, mirroring engine Usage.

    `cost_usd` is computed with the SAME pricing table the engine uses
    (see pricing.py) so cross-system comparison is apples-to-apples.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    llm_calls: int = 0
    embedding_tokens: int = 0  # baselines: query-time embedding cost driver

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.total_tokens += other.total_tokens
        self.cost_usd += other.cost_usd
        self.llm_calls += other.llm_calls
        self.embedding_tokens += other.embedding_tokens


@dataclass
class GoldAnchor:
    """A STABLE description of where the answer lives, independent of the
    engine's per-ingest random `sec_<uuid>` IDs.

    A retrieved unit counts as a hit if it matches on ANY provided dimension
    (see anchors.match_anchor). Provide whatever the dataset gives you:
    `title_path` for structural systems, `answer_spans` for chunk systems,
    `page` when the source has stable pagination.
    """

    # Logical (not engine-assigned) document key, e.g. "AMZN_2022_10K".
    doc_id: str
    # Hierarchical heading path, e.g. ["Part II", "Item 8", "Balance Sheet"].
    title_path: List[str] = field(default_factory=list)
    # Verbatim strings that MUST appear in the retrieved content for a hit.
    answer_spans: List[str] = field(default_factory=list)
    # 1-based page number in the source document, when applicable.
    page: Optional[int] = None

    def parent_path(self) -> List[str]:
        return self.title_path[:-1]


@dataclass
class Question:
    """One evaluation item."""

    qid: str
    doc_id: str  # logical doc key; resolved to a live document at setup time
    question: str
    gold: List[GoldAnchor] = field(default_factory=list)
    answer: Optional[str] = None  # gold free-text answer (for LLM-judge axis)
    answer_type: AnswerType = AnswerType.LOOKUP
    difficulty: str = "medium"  # easy | medium | hard
    domain: str = "general"  # finance | legal | medical | ...
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_no_answer(self) -> bool:
        return self.answer_type == AnswerType.NO_ANSWER


@dataclass
class Doc:
    """A source document in a corpus."""

    doc_id: str  # logical key matching Question.doc_id / GoldAnchor.doc_id
    title: str
    content: str  # extracted plain text (parsers live upstream of the bench)
    path: Optional[str] = None  # on-disk original (pdf/docx) if available
    content_type: str = "text/markdown"
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievedSection:
    """A single unit returned by any retriever, normalised across systems.

    Structural systems (Vectorless, PageIndex) populate `title_path`; chunk
    systems (vector RAG, BM25, full-context) typically only set `content`.
    """

    content: str
    section_id: str = ""  # engine UUID or chunk id; opaque, never scored
    title: str = ""
    title_path: List[str] = field(default_factory=list)
    page: Optional[int] = None
    score: Optional[float] = None  # similarity / rank score if the system has one


@dataclass
class RetrievalResult:
    """What a retriever returns for one query — the unit the scorer consumes."""

    qid: str
    system: str
    query: str
    sections: List[RetrievedSection] = field(default_factory=list)
    # Native answer, for systems that produce one (e.g. treewalk's agentic
    # loop emits the answer directly). When set, the LLM-judge axis grades
    # THIS answer rather than re-generating one from the sections — the only
    # fair way to score an answer-first system.
    answer: Optional[str] = None
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0
    strategy: str = ""  # "single-pass" | "chunked-tree" | "top-k" | ...
    cold: bool = True  # whether caches were bypassed for this call
    trace: Dict[str, Any] = field(default_factory=dict)  # audit/repro detail
    error: Optional[str] = None  # set if retrieval failed; record still emitted


@dataclass
class RunRecord:
    """One scored (system, question, repeat) observation — the JSONL row."""

    qid: str
    system: str
    repeat: int
    domain: str
    answer_type: str
    difficulty: str
    latency_ms: float
    usage: Dict[str, Any]
    metrics: Dict[str, float]
    selected_ids: List[str] = field(default_factory=list)  # for determinism
    strategy: str = ""
    cold: bool = True
    error: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def usage_to_dict(u: Usage) -> Dict[str, Any]:
    return asdict(u)
