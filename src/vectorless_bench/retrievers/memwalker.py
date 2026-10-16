"""MemWalker (Chen et al. 2023, "Walking Down the Memory Maze", arXiv
2310.05029), REIMPLEMENTED FROM THE PAPER: Meta released no code (searched
GitHub 2026-10-16), so this follows the paper's method and is labelled as a
reimplementation wherever its numbers appear.

Build (ingest, timed and costed): the document is cut into segments
(~1,000 tokens, the paper's scale); each segment is summarised; summaries are
grouped `fanout` at a time and each group summarised into a parent node,
recursively, until one root remains. That summary tree is the memory.

Navigate (query): from the root, the model sees the children's summaries
and its working memory (what it has read on the way down) and chooses a
child to open, or to go back up when none is relevant. At a leaf it reads
the segment itself and decides whether it holds the answer; if not, it goes
back. The walk stops at the first leaf it accepts, or after `max_steps`.
The accepted leaf (and any it read along the way, most recent first) is the
retrieval.

Deviations, stated: the paper's prompts are used in spirit (choose a child
by number, or revert), with JSON replies instead of free text so a parse
failure cannot end a walk silently; summaries are capped at 120 words.
"""

from __future__ import annotations

import json
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional

from .._llm import DEFAULT_MODEL, complete, require_vertex_model
from ..datasets.base import CORPUS
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage
from ._chunk import chunk_doc

_SUM = "Summarise the text in at most 120 words, keeping names, numbers and section headings. Reply with the summary only."
_NAV = (
    "You are navigating a tree of summaries of a long document to find the part that answers a question. "
    "You see the summaries of the current node's children and your working memory of what you read on the way. "
    'Reply as JSON: {"action": "open", "child": <number>} to open the child most likely to hold the answer, '
    'or {"action": "back"} if none of them can.'
)
_LEAF = (
    "You are reading one segment of a long document. Decide whether it contains the information needed to answer "
    'the question. Reply as JSON: {"action": "accept"} if it does, or {"action": "back"} if it does not.'
)


class _Node:
    __slots__ = ("summary", "children", "text", "idx")

    def __init__(self, summary: str, children: Optional[List["_Node"]] = None, text: str = "", idx: int = -1):
        self.summary, self.children, self.text, self.idx = summary, children or [], text, idx


class MemWalkerRetriever:
    name = "memwalker"

    def __init__(self, model: str = DEFAULT_MODEL, segment_tokens: int = 1000, fanout: int = 8,
                 max_steps: int = 20, build_concurrency: int = 8, **_: object) -> None:
        require_vertex_model(model)
        self.model, self.segment_tokens, self.fanout = model, segment_tokens, fanout
        self.max_steps, self.build_concurrency = max_steps, build_concurrency
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._lock = threading.Lock()
        self.setup_meta = {"implementation": "reimplemented from the paper (arXiv 2310.05029)"}
        self._roots: Dict[str, _Node] = {}

    def _summarise(self, texts: List[str]) -> List[str]:
        def one(t: str) -> str:
            out = complete(self.model, _SUM, t, max_tokens=220)
            with self._lock:
                self.setup_usage.add(out.usage)
            return out.text.strip()

        with ThreadPoolExecutor(self.build_concurrency) as pool:
            return list(pool.map(one, texts))

    def _build(self, doc: Doc) -> _Node:
        segs = [c.text for c in chunk_doc(doc, self.segment_tokens, 0)]
        leaves = [_Node(s, text=t, idx=i) for i, (s, t) in enumerate(zip(self._summarise(segs), segs))]
        level: List[_Node] = leaves
        while len(level) > 1:
            groups = [level[i : i + self.fanout] for i in range(0, len(level), self.fanout)]
            sums = self._summarise(["\n\n".join(f"[{j + 1}] {n.summary}" for j, n in enumerate(g)) for g in groups])
            level = [_Node(s, children=g) for s, g in zip(sums, groups)]
        return level[0] if level else _Node("(empty)")

    def setup(self, corpus: List[Doc]) -> None:
        t0 = time.perf_counter()
        for d in corpus:
            self._roots[d.doc_id] = self._build(d)
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:
        pass

    def _ask(self, system: str, user: str, usage: Usage) -> Dict[str, Any]:
        out = complete(self.model, system, user, max_tokens=60, json_mode=True)
        usage.add(out.usage)
        try:
            return json.loads(out.text)
        except Exception:
            return {"action": "back"}

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        if question.doc_id == CORPUS:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, cold=cold,
                                   error="n/a: MemWalker navigates one document's tree; this question spans the corpus")
        root = self._roots.get(question.doc_id)
        if root is None:
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question,
                                   error="document not built", cold=cold)
        usage = Usage()
        t0 = time.perf_counter()
        stack: List[_Node] = [root]
        tried: Dict[int, set] = {}
        memory: List[str] = []
        read: List[_Node] = []
        accepted: Optional[_Node] = None
        steps = 0
        try:
            while stack and steps < self.max_steps:
                steps += 1
                node = stack[-1]
                if not node.children:  # leaf: read the segment itself
                    read.append(node)
                    v = self._ask(_LEAF, f"QUESTION: {question.question}\n\nSEGMENT:\n{node.text}", usage)
                    if v.get("action") == "accept":
                        accepted = node
                        break
                    stack.pop()
                    continue
                done = tried.setdefault(id(node), set())
                options = [(i, c) for i, c in enumerate(node.children) if i not in done]
                if not options:
                    stack.pop()
                    continue
                listing = "\n".join(f"{i + 1}. {c.summary}" for i, c in options)
                wm = "\n".join(memory[-4:]) or "(nothing yet)"
                v = self._ask(_NAV, f"QUESTION: {question.question}\n\nWORKING MEMORY:\n{wm}\n\nCHILDREN:\n{listing}", usage)
                pick = v.get("child")
                if v.get("action") == "open" and isinstance(pick, int) and (pick - 1) in dict(options):
                    done.add(pick - 1)
                    child = node.children[pick - 1]
                    memory.append(child.summary)
                    stack.append(child)
                else:
                    stack.pop()
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, usage=usage,
                                   error=str(e)[:500], cold=cold)
        order = ([accepted] if accepted else []) + [n for n in reversed(read) if n is not accepted]
        sections = [RetrievedSection(content=n.text, section_id=f"seg{n.idx}") for n in order[:k]]
        return RetrievalResult(qid=question.qid, system=self.name, query=question.question, sections=sections,
                               usage=usage, latency_ms=(time.perf_counter() - t0) * 1000.0, strategy="memwalker",
                               cold=cold, trace={"steps": steps, "accepted": accepted is not None, "leaves_read": len(read)})
