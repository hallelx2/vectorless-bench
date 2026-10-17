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

    def __init__(self, model: str = DEFAULT_MODEL, max_turns: int = 12, **_: object) -> None:
        require_vertex_model(model)
        self.model = model
        self.max_turns = max_turns
        self.setup_seconds = 0.0
        self.setup_usage = Usage()
        self._corpus: Optional[_Corpus] = None

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
        finished = False
        t0 = time.perf_counter()
        try:
            while turns < self.max_turns and not finished:
                turns += 1
                # a refusal for load costs time, never the question (HAL-3284)
                out = generate_patient(self.model, system=_SYSTEM, contents=contents, max_tokens=1024, tools=_tools())
                usage.add(out.usage)
                if not out.tool_calls:
                    break
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
                            read_log.append({"doc": args.get("doc", docs[0]), "page": int(args.get("page", 1)), "text": text})
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
        return RetrievalResult(
            qid=question.qid, system=self.name, query=question.question,
            sections=sections[:k], usage=usage, latency_ms=(time.perf_counter() - t0) * 1000.0,
            strategy="agentic-grep", cold=cold,
            trace={"turns": turns, "finished": finished, "reads": len(read_log), "llm_calls": usage.llm_calls},
        )
