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

            for qi, q in enumerate(questions):
                for rep in range(cfg.repeats):
                    rec = _eval_one(system, q, retriever, cfg, judge, rep)
                    fh.write(rec.to_json() + "\n")
                if progress and (qi + 1) % 25 == 0:
                    log(f"[vlbench]   {qi + 1}/{len(questions)} questions")

            try:
                retriever.teardown()
            except Exception:
                pass

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
    # Judge only on the first repeat: answer quality doesn't need rerunning, and
    # it keeps judge spend from scaling with the determinism repeat count.
    if judge is not None and not result.error and repeat == 0:
        jr = judge.evaluate(q, result.sections)
        m["answer_correct"] = jr.correct
        m["answer_faithful"] = jr.faithful
        m["answered"] = jr.answered
        m["judge_usd"] = jr.usage.cost_usd  # tracked apart from system cost
    return RunRecord(
        qid=q.qid, system=system, repeat=repeat, domain=q.domain,
        answer_type=q.answer_type.value, difficulty=q.difficulty,
        latency_ms=result.latency_ms, usage=usage_to_dict(result.usage),
        metrics=m, selected_ids=[s.section_id for s in result.sections],
        strategy=result.strategy, cold=result.cold, error=result.error,
    )


def _stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")
