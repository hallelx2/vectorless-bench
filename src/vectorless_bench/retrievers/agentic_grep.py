"""Agentic grep: a Gemini agent that searches the document with grep and reads
what it finds, the way coding agents (Claude Code, Gemini CLI) search a
codebase. A May 2026 study found grep beat vector retrieval inside agent
loops (arXiv 2605.15184), which makes it the strongest simple vectorless
rival and the one a sceptical engineer will ask about.

Tools, over the question's document (every document for a corpus-level
question), each document split into pages at form feeds and pages into
lines:

  grep(pattern, ignore_case=true)  -> up to 25 hits as "doc | p.N | L.M | text"
  read(doc, page, start_line, end_line) -> those lines (at most 80 at once)
  finish(evidence=[{doc, page, start_line, end_line}]) -> ends the search

The agent gets at most `max_turns` model calls; whatever it read and cited
when it calls finish (or runs out) is its retrieval, ranked in the order it
cited it. Same answer model and thinking cap as every other system; usage is
the sum over its turns.
"""

from __future__ import annotations

import re
import time
from typing import Any, Dict, List, Optional, Tuple

from .._llm import DEFAULT_MODEL, generate_patient, require_vertex_model
from ..datasets.base import CORPUS
from ..schema import Doc, Question, RetrievalResult, RetrievedSection, Usage

_SYSTEM = (
    "You find the passages of a document that answer a question. You cannot see the document; "
    "you search it with tools. Use grep with distinctive words or regular expressions from the "
    "question (names, numbers, defined terms, likely headings), read around the best hits, and "
    "refine. When you have found the passages that answer the question, call finish with them, "
    "most relevant first. If nothing in the document answers it, call finish with no evidence. "
    "Do not answer the question yourself."
)
MAX_HITS = 25
MAX_READ = 80
# A model that replies in prose instead of calling a tool is told this, at
# most _MAX_NUDGES times a question.
_NUDGE = ("Keep searching with the tools: read the most promising hits with read, grep again if they miss, "
          "and call finish with the passages that answer the question, or with none if nothing does. "
          "Do not answer in text.")
_MAX_NUDGES = 2

# Lines kept on each side of a cited range the agent never read itself.
_CONTEXT_LINES = 20


class _Corpus:
    def __init__(self, docs: List[Doc]) -> None:
        # doc -> pages -> lines
        self.pages: Dict[str, List[List[str]]] = {
            d.doc_id: [p.splitlines() for p in d.content.split("\f")] for d in docs
        }

    def grep(self, docs: List[str], pattern: str, ignore_case: bool = True) -> str:
        flags = re.IGNORECASE if ignore_case else 0
        try:
            rx = re.compile(pattern, flags)
        except re.error:
            rx = re.compile(re.escape(pattern), flags)
        hits: List[str] = []
        total = 0
        for d in docs:
            for pi, lines in enumerate(self.pages.get(d, [])):
                for li, line in enumerate(lines):
                    if rx.search(line):
                        total += 1
                        if len(hits) < MAX_HITS:
                            hits.append(f"{d} | p.{pi + 1} | L.{li + 1} | {line.strip()[:240]}")
        if not hits:
            return "no matches"
        more = f"\n({total - len(hits)} more matches not shown; refine the pattern)" if total > len(hits) else ""
        return "\n".join(hits) + more

    def read(self, doc: str, page: int, start: int, end: int) -> Tuple[str, str]:
        lines = (self.pages.get(doc) or [[]])
        if not (1 <= page <= len(lines)):
            return "", f"no page {page} in {doc} (it has {len(lines)})"
        pl = lines[page - 1]
        start = max(1, int(start))
        end = min(len(pl), int(end), start + MAX_READ - 1)
        text = "\n".join(pl[start - 1 : end])
        return text, "\n".join(f"L.{i} | {l}" for i, l in enumerate(pl[start - 1 : end], start)) or "(empty)"


def _tools():
    from google.genai import types  # type: ignore

    S = types.Schema
    T = types.Type
    return [types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="grep", description="Search every line of the document(s) for a regular expression.",
            parameters=S(type=T.OBJECT, properties={
                "pattern": S(type=T.STRING, description="Python regular expression or plain words"),
                "ignore_case": S(type=T.BOOLEAN)}, required=["pattern"])),
        types.FunctionDeclaration(
            name="read", description=f"Read lines of one page (at most {MAX_READ} lines).",
            parameters=S(type=T.OBJECT, properties={
                "doc": S(type=T.STRING), "page": S(type=T.INTEGER),
                "start_line": S(type=T.INTEGER), "end_line": S(type=T.INTEGER)},
                required=["doc", "page", "start_line", "end_line"])),
        types.FunctionDeclaration(
            name="finish", description="End the search with the evidence passages, most relevant first.",
            parameters=S(type=T.OBJECT, properties={"evidence": S(type=T.ARRAY, items=S(type=T.OBJECT, properties={
                "doc": S(type=T.STRING), "page": S(type=T.INTEGER),
                "start_line": S(type=T.INTEGER), "end_line": S(type=T.INTEGER)},
                required=["doc", "page", "start_line", "end_line"]))}, required=["evidence"])),
    ])]


class AgenticGrepRetriever:
    name = "agentic_grep"

    def __init__(self, model: str = DEFAULT_MODEL, max_turns: int = 12, context: str = "read", **_: object) -> None:
        require_vertex_model(model)
        if context not in ("read", "cited"):
            raise ValueError(f"context must be 'read' or 'cited', not {context!r}")
        self.model = model
        self.max_turns = max_turns
        self.context = context
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._corpus: Optional[_Corpus] = None

    def _context(self, evidence: List[Dict[str, Any]], read_log: List[Dict[str, Any]], docs: List[str],
                 finished: bool) -> List[RetrievedSection]:
        """What the answer step reads from this search.

        "read" (the default, HAL-3389): the windows the agent read, the ones
        holding its citations first, then the rest of what it read, most
        recent first, overlapping windows on a page merged. The answer step
        sees what the agent saw, as PageIndex's units are the pages its agent
        read. "cited" (the old behaviour): only the line ranges named in
        finish, often one to three lines: in the pilot that averaged about
        356 tokens and 60-70% of answers were declined."""
        if self.context == "cited":
            return self._cited(evidence, read_log, docs, finished)
        windows: List[Dict[str, Any]] = []  # merged, in order of relevance

        def place(doc: str, page: int, start: int, end: int) -> None:
            for w in windows:
                if w["doc"] == doc and w["page"] == page and start <= w["end"] + 1 and end >= w["start"] - 1:
                    w["start"], w["end"] = min(w["start"], start), max(w["end"], end)
                    return
            windows.append({"doc": doc, "page": page, "start": start, "end": end})

        for ev in evidence:
            doc, page = str(ev.get("doc") or docs[0]), int(ev.get("page", 1))
            s0, e0 = int(ev.get("start_line", 1)), int(ev.get("end_line", ev.get("start_line", 1)))
            held = [r for r in read_log if r["doc"] == doc and r["page"] == page and r["start"] <= e0 and r["end"] >= s0]
            if held:
                for r in held:
                    place(doc, page, r["start"], r["end"])
            else:  # cited without reading it: the cited lines with their surroundings
                place(doc, page, max(1, s0 - _CONTEXT_LINES), e0 + _CONTEXT_LINES)
        for r in read_log[::-1]:
            place(r["doc"], r["page"], r["start"], r["end"])
        sections: List[RetrievedSection] = []
        for w in windows:
            text, _ = self._corpus.read(w["doc"], w["page"], w["start"], w["end"]) if w["end"] - w["start"] < MAX_READ \
                else (self._read_long(w), "")
            if text.strip():
                sections.append(RetrievedSection(content=text, section_id=f"{w['doc']}:p{w['page']}:{w['start']}-{w['end']}",
                                                 title=w["doc"], page=w["page"], doc_id=w["doc"]))
        return sections

    def _read_long(self, w: Dict[str, Any]) -> str:
        lines = (self._corpus.pages.get(w["doc"]) or [[]])
        if not (1 <= w["page"] <= len(lines)):
            return ""
        return "\n".join(lines[w["page"] - 1][w["start"] - 1 : w["end"]])

    def _cited(self, evidence: List[Dict[str, Any]], read_log: List[Dict[str, Any]], docs: List[str],
               finished: bool) -> List[RetrievedSection]:
        sections: List[RetrievedSection] = []
        for ev in evidence:
            doc = str(ev.get("doc") or docs[0])
            text, _ = self._corpus.read(doc, int(ev.get("page", 1)), int(ev.get("start_line", 1)), int(ev.get("end_line", 1)))
            if text:
                sections.append(RetrievedSection(content=text, section_id=f"{doc}:p{ev.get('page')}:{ev.get('start_line')}",
                                                 title=doc, page=int(ev.get("page", 1)), doc_id=doc))
        if not sections and not finished:
            # Ran out of turns: what it read is what it found.
            sections = [RetrievedSection(content=r["text"], section_id=f"{r['doc']}:p{r['page']}", title=str(r["doc"]),
                                         page=r["page"], doc_id=str(r["doc"])) for r in read_log[::-1]]
        return sections

    def setup(self, corpus: List[Doc]) -> None:
        t0 = time.perf_counter()
        self._corpus = _Corpus(corpus)  # no index: grep reads the text as it is
        self.setup_seconds = time.perf_counter() - t0

    def teardown(self) -> None:
        pass

    def retrieve(self, question: Question, k: int, cold: bool = True) -> RetrievalResult:
        from google.genai import types  # type: ignore

        assert self._corpus is not None
        docs = question.meta.get("doc_ids_scope") or (
            list(self._corpus.pages) if question.doc_id == CORPUS else [question.doc_id])
        intro = (f"QUESTION: {question.question}\n\nDocuments you can search: "
                 + ", ".join(f"{d} ({len(self._corpus.pages.get(d, []))} pages)" for d in docs[:40])
                 + (f" and {len(docs) - 40} more" if len(docs) > 40 else ""))
        contents: List[Any] = [types.Content(role="user", parts=[types.Part(text=intro)])]
        usage = Usage()
        read_log: List[Dict[str, Any]] = []
        evidence: List[Dict[str, Any]] = []
        turns = 0
        nudges = 0
        finished = False
        t0 = time.perf_counter()
        try:
            while turns < self.max_turns and not finished:
                turns += 1
                # a refusal for load costs time, never the question (HAL-3284)
                out = generate_patient(self.model, system=_SYSTEM, contents=contents, max_tokens=1024, tools=_tools())
                usage.add(out.usage)
                if not out.tool_calls:
                    # A reply in prose is not the end of a search: in the pilot
                    # 2.5 Flash-Lite answered in text after one grep on 26 of 50
                    # questions, read nothing on 21, and the loop stopped there
                    # (HAL-3389). It is told to use its tools, at most
                    # _MAX_NUDGES times, inside the same turn cap.
                    if nudges >= _MAX_NUDGES or turns >= self.max_turns:
                        break
                    nudges += 1
                    if out.raw_content is not None:
                        contents.append(out.raw_content)
                    contents.append(types.Content(role="user", parts=[types.Part(text=_NUDGE)]))
                    continue
                contents.append(out.raw_content)
                responses = []
                for call in out.tool_calls:
                    name, args = call["name"], call["args"]
                    if name == "grep":
                        result = self._corpus.grep(docs, str(args.get("pattern", "")), bool(args.get("ignore_case", True)))
                    elif name == "read":
                        text, result = self._corpus.read(str(args.get("doc", docs[0])), int(args.get("page", 1)),
                                                         int(args.get("start_line", 1)), int(args.get("end_line", 40)))
                        if text:
                            start = max(1, int(args.get("start_line", 1)))
                            read_log.append({"doc": str(args.get("doc", docs[0])), "page": int(args.get("page", 1)),
                                             "start": start, "end": start + text.count("\n"), "text": text})
                    elif name == "finish":
                        evidence = list(args.get("evidence") or [])
                        finished = True
                        result = "done"
                    else:
                        result = f"unknown tool {name}"
                    responses.append(types.Part.from_function_response(name=name, response={"result": result}))
                contents.append(types.Content(role="user", parts=responses))
        except Exception as e:  # pragma: no cover - network
            return RetrievalResult(qid=question.qid, system=self.name, query=question.question, usage=usage,
                                   error=str(e), cold=cold, trace={"turns": turns})
        sections = self._context(evidence, read_log, docs, finished)
        return RetrievalResult(
            qid=question.qid, system=self.name, query=question.question,
            sections=sections[:k], usage=usage, latency_ms=(time.perf_counter() - t0) * 1000.0,
            strategy="agentic-grep", cold=cold,
            trace={"turns": turns, "finished": finished, "reads": len(read_log), "cited": len(evidence), "nudges": nudges,
                   "context": self.context, "llm_calls": usage.llm_calls},
        )
