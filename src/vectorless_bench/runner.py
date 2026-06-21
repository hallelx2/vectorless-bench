"""The orchestrator: dataset -> per-system index -> per-question retrieve x repeats
-> score -> JSONL records + manifest.

Design choices that matter for valid numbers:
- One retriever is set up (indexed) ONCE per run; indexing time/cost is recorded
  separately from query time, because for vector RAG indexing is the hidden cost.
- Retrieval runs sequentially by default so latency isn't distorted by contention
  (the engine already parallelises internally; we measure that, not our own load).
- Every (system, question) failure is caught and recorded as a row with an error,
  so a single bad document can't void a multi-hour run.
- `repeats > 1` reruns the SAME query to feed the determinism metric; we seed
  nothing and change nothing between repeats, which is the whole point.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

from . import metrics
from .config import RunConfig
from .datasets import build_dataset
from .manifest import Manifest
from .retrievers import build as build_retriever
from .schema import Doc, Question, RunRecord, usage_to_dict


def _load_dataset(cfg: RunConfig):
    ds = build_dataset(cfg.dataset, **cfg.dataset_params)
    corpus = ds.corpus()
    if hasattr(ds, "questions_for_available_corpus"):
        questions = ds.questions_for_available_corpus()  # skip missing docs
    else:
        questions = ds.questions()
    questions = ds.subset(questions, cfg.limit, cfg.sample_seed)
    # keep only docs actually referenced by the selected questions
    needed = {q.doc_id for q in questions}
    corpus = [d for d in corpus if d.doc_id in needed]
    return ds, corpus, questions


def run(cfg: RunConfig, progress: bool = True) -> Path:
    captured: List[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        ds, corpus, questions = _load_dataset(cfg)
        captured = [str(w.message) for w in wlist]

    out_dir = Path(cfg.out_dir) / _stamp()
    out_dir.mkdir(parents=True, exist_ok=True)
    records_path = out_dir / "records.jsonl"
    setup_info = {}
    reviews: List[RunRecord] = []

    def log(msg: str) -> None:
        if progress:
            print(msg, flush=True)

    log(f"[vlbench] dataset={cfg.dataset} docs={len(corpus)} "
        f"questions={len(questions)} systems={cfg.systems} "
        f"k={cfg.k} repeats={cfg.repeats}")

    judge = None
    if cfg.judge:
        from .judge import Judge

        judge = Judge(cfg.gen_model, cfg.judge_model)

    with records_path.open("w", encoding="utf-8") as fh:
        for system in cfg.systems:
            log(f"[vlbench] === system: {system} ===")
            try:
                retriever = build_retriever(system, **cfg.params_for(system))
            except Exception as e:
                log(f"[vlbench]   build failed: {e}")
                captured.append(f"{system}: build failed: {e}")
                continue
            try:
                retriever.setup(corpus)
            except Exception as e:
                log(f"[vlbench]   setup failed: {e}")
                captured.append(f"{system}: setup failed: {e}")
                continue
            setup_info[system] = {
                "setup_seconds": getattr(retriever, "setup_seconds", 0.0),
                "setup_usage": usage_to_dict(getattr(retriever, "setup_usage", None))
                if getattr(retriever, "setup_usage", None)
                else {},
            }
            # Per-document ingest timing (the engine's parse+persist speed). Only
            # the engine-backed systems expose this; baselines index in-process.
            ing = getattr(retriever, "ingest_times", None)
            if ing:
                setup_info[system]["ingest_timing"] = _ingest_timing_summary(
                    ing, getattr(retriever, "ingest_bytes", {}) or {}
                )

            for qi, q in enumerate(questions):
                for rep in range(cfg.repeats):
                    rec = _eval_one(system, q, retriever, cfg, judge, rep)
                    fh.write(rec.to_json() + "\n")
                    if rec.review is not None and rep == 0:
                        reviews.append(rec)
                if progress and (qi + 1) % 25 == 0:
                    log(f"[vlbench]   {qi + 1}/{len(questions)} questions")

            try:
                retriever.teardown()
            except Exception:
                pass

    if cfg.persist_answers and reviews:
        _write_review_md(out_dir / "review.md", reviews)
        log(f"[vlbench] review: {out_dir / 'review.md'}")

    (out_dir / "setup.json").write_text(
        json.dumps(setup_info, indent=2), encoding="utf-8"
    )
    Manifest.build(
        dataset=cfg.dataset, dataset_size=len(questions), systems=cfg.systems,
        k=cfg.k, repeats=cfg.repeats, cold=cfg.cold, model=cfg.model,
        judge_model=cfg.judge_model if cfg.judge else None,
        embedding_model=cfg.embedding_model, sample_seed=cfg.sample_seed,
        warnings=captured, extra={"config": asdict(cfg)},
    ).write(out_dir / "manifest.json")

    log(f"[vlbench] wrote {records_path}")
    return out_dir


def _eval_one(system, q: Question, retriever, cfg: RunConfig, judge, repeat: int) -> RunRecord:
    try:
        result = retriever.retrieve(q, k=cfg.k, cold=cfg.cold)
    except Exception as e:  # retriever promised not to raise, but be safe
        return RunRecord(
            qid=q.qid, system=system, repeat=repeat, domain=q.domain,
            answer_type=q.answer_type.value, difficulty=q.difficulty,
            latency_ms=0.0, usage={}, metrics={}, error=str(e),
        )
    m = {} if result.error else metrics.score_question(q, result, cfg.k)
    review = None
    # Judge only on the first repeat: answer quality doesn't need rerunning, and
    # it keeps judge spend from scaling with the determinism repeat count.
    if judge is not None and not result.error and repeat == 0:
        jr = judge.evaluate(q, result.sections, native_answer=result.answer)
        m["answer_correct"] = jr.correct
        m["answer_faithful"] = jr.faithful
        m["answered"] = jr.answered
        m["judge_usd"] = jr.usage.cost_usd  # tracked apart from system cost
        if cfg.persist_answers:
            review = _build_review(q, result, jr)
    elif cfg.persist_answers and not result.error:
        review = _build_review(q, result, None)
    return RunRecord(
        qid=q.qid, system=system, repeat=repeat, domain=q.domain,
        answer_type=q.answer_type.value, difficulty=q.difficulty,
        latency_ms=result.latency_ms, usage=usage_to_dict(result.usage),
        metrics=m, selected_ids=[s.section_id for s in result.sections],
        strategy=result.strategy, cold=result.cold, error=result.error,
        review=review,
    )


def _build_review(q: Question, result, jr) -> dict:
    """Assemble the human-review record: question, gold, the candidate answer
    that was graded, the judge's rationale, and a preview of the retrieved
    context. `jr` is None when the row wasn't judged (still useful to see what
    was retrieved and what the system answered)."""
    # Up to 3 retrieved units, each previewed so the file stays readable.
    ctx = []
    for s in result.sections[:3]:
        body = (s.content or "").strip().replace("\n", " ")
        if len(body) > 300:
            body = body[:300] + "…"
        ctx.append({
            "title": s.title or " / ".join(s.title_path),
            "section_id": s.section_id,
            "preview": body,
        })
    review = {
        "question": q.question,
        "answer_type": q.answer_type.value,
        "gold": q.answer,
        "candidate": (jr.candidate if jr else (result.answer or "")),
        "retrieved": ctx,
    }
    if jr is not None:
        review["verdict"] = {
            "correct": bool(jr.correct),
            "faithful": bool(jr.faithful),
            "answered": bool(jr.answered),
            "judge_reason": jr.reason,
            "missing_facts": jr.missing_facts,
            "contradicting_facts": jr.contradicting_facts,
        }
    return review


def _ingest_timing_summary(times: dict, sizes: dict) -> dict:
    """Summarise per-document ingest wall-times into the speed numbers the
    report needs: count, total, mean/median/p95 seconds, and MB/s throughput
    (the headline for the Go engine). `times` is {doc_id: seconds},
    `sizes` is {doc_id: bytes}."""
    secs = sorted(times.values())
    n = len(secs)
    if n == 0:
        return {}

    def pct(p: float) -> float:
        i = min(n - 1, int(round((p / 100.0) * (n - 1))))
        return round(secs[i], 3)

    total_s = sum(secs)
    total_bytes = sum(sizes.get(k, 0) for k in times)
    mb = total_bytes / (1024 * 1024)
    per_doc = {k: round(v, 3) for k, v in sorted(times.items(), key=lambda kv: kv[1], reverse=True)}
    return {
        "docs": n,
        "total_seconds": round(total_s, 3),
        "mean_seconds": round(total_s / n, 3),
        "median_seconds": pct(50),
        "p95_seconds": pct(95),
        "min_seconds": round(secs[0], 3),
        "max_seconds": round(secs[-1], 3),
        "total_mb": round(mb, 2),
        "mb_per_second": round(mb / total_s, 3) if total_s > 0 else 0.0,
        "per_doc_seconds": per_doc,
    }


def _write_review_md(path: Path, records: List[RunRecord]) -> None:
    """Render the per-question judging trail as readable Markdown, grouped by
    question so a human can compare every system's answer to the gold side by
    side and second-guess the judge."""
    by_q: dict = {}
    order: List[str] = []
    for r in records:
        if r.qid not in by_q:
            by_q[r.qid] = []
            order.append(r.qid)
        by_q[r.qid].append(r)

    def tick(v) -> str:
        return "✅" if v else "❌"

    lines = [
        "# Answer review",
        "",
        "Every judged answer, side by side with the gold reference and the "
        "judge's own rationale — so you can re-check each verdict yourself. "
        "`correct`/`faithful`/`answered` are the judge's calls; the retrieved "
        "context preview shows what each system actually had to work with.",
        "",
    ]
    for qid in order:
        recs = by_q[qid]
        rv0 = recs[0].review or {}
        lines.append(f"## {qid}  ·  {rv0.get('answer_type', '')}")
        lines.append("")
        lines.append(f"**Question:** {rv0.get('question', '')}")
        lines.append("")
        lines.append(f"**Gold answer:** {rv0.get('gold', '')}")
        lines.append("")
        for r in recs:
            rv = r.review or {}
            v = rv.get("verdict")
            lines.append(f"### {r.system}")
            if v:
                lines.append(
                    f"- correct {tick(v['correct'])}  ·  faithful {tick(v['faithful'])}"
                    f"  ·  answered {tick(v['answered'])}"
                )
                if v.get("missing_facts"):
                    lines.append("- missing gold facts: " + "; ".join(v["missing_facts"]))
                if v.get("contradicting_facts"):
                    lines.append("- contradictions: " + "; ".join(v["contradicting_facts"]))
                if v.get("judge_reason"):
                    lines.append(f"- judge: _{v['judge_reason']}_")
            lines.append("")
            lines.append(f"**Answer:** {rv.get('candidate', '') or '(none)'}")
            lines.append("")
            ctx = rv.get("retrieved") or []
            if ctx:
                lines.append("<details><summary>retrieved context</summary>")
                lines.append("")
                for c in ctx:
                    title = c.get("title") or c.get("section_id") or "(section)"
                    lines.append(f"- **{title}** — {c.get('preview', '')}")
                lines.append("")
                lines.append("</details>")
                lines.append("")
        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")
