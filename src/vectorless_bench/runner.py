"""The orchestrator: dataset -> per-system index -> per-question retrieve x repeats
-> score -> JSONL records + manifest.

Design choices that matter for valid numbers:
- One retriever is set up (indexed) ONCE per run; indexing time/cost is recorded
  separately from query time, because for vector RAG indexing is the hidden cost.
- Every system is timed the same way: wall-clock around `retrieve()` in this
  process, so a network hop, a client library and the server all count for
  everyone. What a system reports about itself (the engine's elapsed_ms and its
  timing split) is kept beside it, never in place of it.
- Phase "latency" (default) runs one question at a time, one system at a time:
  nothing of ours competes with the call being timed. The rig (rig.py)
  brackets every sample with host evidence from each machine and reruns a
  sample that shows competition (CPU steal, a foreign process); the report
  uses unflagged samples only.
- Phase "throughput" runs `concurrency` questions at once per system and is
  reported separately: latency under load is a different number.
- Every (system, question) failure is caught and recorded as a row with an error,
  so a single bad document can't void a multi-hour run.
- `repeats > 1` reruns the SAME query to feed the determinism metric; we seed
  nothing and change nothing between repeats, which is the whole point.
- `max_spend_usd` stops the run once recorded spend (system calls, index
  builds and grading) reaches it. The counter is the provider-reported usage
  each record carries, the same numbers the bill is built from.
"""

from __future__ import annotations

import json
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import metrics
from .config import RunConfig, validate
from .datasets import build_dataset
from .datasets.base import CORPUS
from .manifest import Manifest
from .retrievers import build as build_retriever
from .rig import Rig, RigConfig, machine_info, remote_machine_info
from .schema import Question, RunRecord, usage_to_dict


class SpendCapReached(Exception):
    pass


class _Spend:
    """Running total of recorded spend; thread-safe for the throughput pass."""

    def __init__(self, cap: Optional[float]) -> None:
        self.cap = cap
        self.total = 0.0
        self._lock = threading.Lock()

    def add(self, usd: float) -> None:
        with self._lock:
            self.total += float(usd or 0.0)
            if self.cap is not None and self.total >= self.cap:
                raise SpendCapReached(f"recorded spend ${self.total:.2f} reached the cap ${self.cap:.2f}")


def _load_dataset(cfg: RunConfig):
    ds = build_dataset(cfg.dataset, **cfg.dataset_params)
    corpus = ds.corpus()
    if hasattr(ds, "questions_for_available_corpus"):
        questions = ds.questions_for_available_corpus()  # skip missing docs
    else:
        questions = ds.questions()
    questions = ds.subset(questions, cfg.limit, cfg.sample_seed)
    # Keep only docs the selected questions are asked over. A per-document
    # question is asked over its own document. A corpus-level question
    # (MultiHop-RAG) is asked over the WHOLE collection: trimming it to the
    # sampled questions' gold documents would hand every system a library
    # made of answers, so the full corpus is kept.
    if any(q.doc_id == CORPUS for q in questions):
        return ds, corpus, questions
    needed = {q.doc_id for q in questions}
    corpus = [d for d in corpus if d.doc_id in needed]
    return ds, corpus, questions


def run(cfg: RunConfig, progress: bool = True) -> Path:
    validate(cfg)
    captured: List[str] = []
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        ds, corpus, questions = _load_dataset(cfg)
        captured = [str(w.message) for w in wlist]

    out_dir = _new_run_dir(Path(cfg.out_dir))
    records_path = out_dir / "records.jsonl"
    setup_info: Dict[str, Any] = {}
    rig = Rig(RigConfig.from_dict(cfg.rig))
    spend = _Spend(cfg.max_spend_usd)
    stopped: Optional[str] = None

    def log(msg: str) -> None:
        if progress:
            print(msg, flush=True)

    log(f"[vlbench] dataset={cfg.dataset} docs={len(corpus)} "
        f"questions={len(questions)} systems={cfg.systems} "
        f"k={cfg.k} repeats={cfg.repeats} phase={cfg.phase}"
        + (f" concurrency={cfg.concurrency}" if cfg.phase == "throughput" else ""))

    network = rig.network_baseline() if rig.cfg.enabled and rig.cfg.rtt_targets else {}

    judge = None
    if cfg.judge:
        from .judge import Judge

        judge = Judge(cfg.gen_model, cfg.judge_model)

    engine_versions: Dict[str, Any] = {}
    write_lock = threading.Lock()
    with records_path.open("w", encoding="utf-8") as fh:
        def emit(rec: RunRecord) -> None:
            with write_lock:
                fh.write(rec.to_json() + "\n")
                fh.flush()
            spend.add((rec.usage or {}).get("cost_usd", 0.0) + rec.metrics.get("judge_usd", 0.0)
                      + (rec.rerun_cost_usd or 0.0))

        for system in cfg.systems:
            if stopped:
                break
            log(f"[vlbench] === system: {system} ===")
            try:
                retriever = build_retriever(system, **cfg.params_for(system))
            except Exception as e:
                log(f"[vlbench]   build failed: {e}")
                captured.append(f"{system}: build failed: {e}")
                continue
            t0 = time.perf_counter()
            try:
                retriever.setup(corpus)
            except Exception as e:
                log(f"[vlbench]   setup failed: {e}")
                captured.append(f"{system}: setup failed: {e}")
                continue
            setup_wall = time.perf_counter() - t0
            su = getattr(retriever, "setup_usage", None)
            setup_info[system] = {
                "setup_seconds": getattr(retriever, "setup_seconds", 0.0) or setup_wall,
                "setup_wall_seconds": setup_wall,
                "setup_usage": usage_to_dict(su) if su else {},
                "setup_meta": getattr(retriever, "setup_meta", None),
            }
            if getattr(retriever, "engine_version", None):
                engine_versions[system] = retriever.engine_version
            try:
                spend.add(getattr(su, "cost_usd", 0.0) if su else 0.0)
                if cfg.phase == "throughput":
                    _throughput(system, questions, retriever, cfg, emit, setup_info, log)
                else:
                    _latency(system, questions, retriever, cfg, judge, rig, emit, log)
            except SpendCapReached as e:
                stopped = str(e)
                log(f"[vlbench] STOPPED: {stopped}")
                captured.append(f"stopped: {stopped}")
            finally:
                try:
                    retriever.teardown()
                except Exception:
                    pass

    (out_dir / "setup.json").write_text(json.dumps(setup_info, indent=2, default=str), encoding="utf-8")

    for name, v in engine_versions.items():
        ver = str(v.get("version", "")) if isinstance(v, dict) else str(v)
        if not ver or ver == "dev" or ver.endswith("-dirty"):
            captured.append(f"{name}: engine build is unattributable ({ver or 'unknown'}); "
                            "do not publish these numbers")

    from . import _llm

    Manifest.build(
        dataset=cfg.dataset, dataset_size=len(questions), systems=cfg.systems,
        k=cfg.k, repeats=cfg.repeats, cold=cfg.cold, model=cfg.model,
        judge_model=cfg.judge_model if cfg.judge else None,
        embedding_model=cfg.embedding_model, sample_seed=cfg.sample_seed,
        warnings=captured,
        extra={
            "config": asdict(cfg),
            "phase": cfg.phase,
            "concurrency": cfg.concurrency,
            "thinking_budget": _safe(_llm.thinking_budget),
            "vertex": {"project": _safe(_llm.project), "location": _safe(_llm.location)},
            "engine_versions": engine_versions,
            "machines": {"bench": machine_info(),
                         **{n: remote_machine_info(u) for n, u in rig.cfg.probes.items()}},
            "network": network,
            "rig": asdict(rig.cfg),
            "spend_usd": round(spend.total, 6),
            "max_spend_usd": cfg.max_spend_usd,
            "stopped": stopped,
        },
    ).write(out_dir / "manifest.json")

    log(f"[vlbench] wrote {records_path} (recorded spend ${spend.total:.4f})")
    return out_dir


def _safe(fn):
    try:
        return fn()
    except Exception as e:
        return f"unavailable: {e}"


def _latency(system, questions: List[Question], retriever, cfg: RunConfig, judge, rig: Rig, emit, log) -> None:
    for qi, q in enumerate(questions):
        for rep in range(cfg.repeats):
            emit(_eval_one(system, q, retriever, cfg, judge, rep, rig))
        if (qi + 1) % 25 == 0:
            log(f"[vlbench]   {qi + 1}/{len(questions)} questions")


def _throughput(system, questions: List[Question], retriever, cfg: RunConfig, emit, setup_info, log) -> None:
    """All questions at `concurrency` in flight; one pass, no repeats, no
    judge: this measures sustained rate and latency under load."""
    t0 = time.perf_counter()

    def one(q: Question) -> None:
        emit(_eval_one(system, q, retriever, cfg, None, 0, None))

    from concurrent.futures import FIRST_COMPLETED, wait

    n = max(1, cfg.concurrency)
    pending = iter(questions)
    capped: Optional[SpendCapReached] = None
    with ThreadPoolExecutor(max_workers=n) as pool:
        inflight = set()
        for q in pending:
            inflight.add(pool.submit(one, q))
            if len(inflight) >= n:
                break
        while inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for f in done:
                try:
                    f.result()
                except SpendCapReached as e:
                    capped = e
            if capped is None:
                for q in pending:
                    inflight.add(pool.submit(one, q))
                    if len(inflight) >= n:
                        break
    if capped is not None:
        raise capped
    wall = time.perf_counter() - t0
    setup_info[system]["throughput"] = {
        "questions": len(questions),
        "concurrency": cfg.concurrency,
        "wall_seconds": round(wall, 3),
        "questions_per_minute": round(60.0 * len(questions) / wall, 2) if wall > 0 else None,
    }
    log(f"[vlbench]   throughput: {setup_info[system]['throughput']}")


def _eval_one(system, q: Question, retriever, cfg: RunConfig, judge, repeat: int,
              rig: Optional[Rig]) -> RunRecord:
    attempts = 0
    err = ""
    host: Dict[str, Any] = {}
    max_attempts = 1 + (rig.cfg.rerun_flagged if rig is not None and rig.cfg.enabled else 0)
    result = None
    wall_ms = 0.0
    # Spend on attempts that were thrown away for a busy machine is still
    # spend: it is kept on the record and counted against the cap.
    rerun_usd = 0.0
    while attempts < max_attempts:
        if result is not None:
            rerun_usd += float(getattr(result.usage, "cost_usd", 0.0) or 0.0)
        attempts += 1
        start = rig.begin() if rig is not None else {}
        t0 = time.perf_counter()
        try:
            result = retriever.retrieve(q, k=cfg.k, cold=cfg.cold)
        except Exception as e:  # retriever promised not to raise, but be safe
            result = None
            err = str(e)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        host = rig.end(start) if rig is not None else {"flagged": False, "reasons": [], "hosts": {}}
        if not host.get("flagged"):
            break
    base = dict(
        qid=q.qid, system=system, repeat=repeat, domain=q.domain,
        answer_type=q.answer_type.value, difficulty=q.difficulty,
        phase=cfg.phase, attempts=attempts, flagged=bool(host.get("flagged")),
        host=host, rerun_cost_usd=rerun_usd,
    )
    if result is None:
        return RunRecord(latency_ms=wall_ms, wall_ms=wall_ms, usage={}, metrics={}, error=err, **base)
    m = {} if result.error else metrics.score_question(q, result, cfg.k)
    if m and "gold_reachable" in q.meta:
        m["gold_reachable"] = 1.0 if q.meta["gold_reachable"] else 0.0
    # Judge only on the first repeat: answer quality doesn't need rerunning, and
    # it keeps judge spend from scaling with the determinism repeat count.
    if judge is not None and not result.error and repeat == 0:
        jr = judge.evaluate(q, result.sections)
        m["answer_correct"] = jr.correct
        m["answer_faithful"] = jr.faithful
        m["answered"] = jr.answered
        m["judge_usd"] = jr.usage.cost_usd  # tracked apart from system cost
        if jr.verdict:
            # every graded row carries all four flags, so each share is over
            # the same denominator (the graded questions)
            for v in ("correct", "partially_correct", "incorrect", "abstained"):
                m[f"verdict_{v}"] = 1.0 if jr.verdict == v else 0.0
        if jr.verdict_repeat is not None:
            m["grade_repeat_agree"] = 1.0 if jr.verdict_repeat == jr.verdict else 0.0
    return RunRecord(
        latency_ms=wall_ms, wall_ms=wall_ms, reported_ms=result.latency_ms,
        usage=usage_to_dict(result.usage), metrics=m,
        selected_ids=[s.section_id for s in result.sections],
        strategy=result.strategy, cold=result.cold, error=result.error,
        trace=dict(result.trace or {}),
        timing=result.trace.get("timing") if isinstance(result.trace, dict) else None,
        **base,
    )


def _new_run_dir(root: Path) -> Path:
    """A fresh run directory, created exclusively: the timestamp keeps runs
    sorted, the random suffix and exist_ok=False keep two runs started in
    the same second from writing into one directory."""
    import secrets

    root.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        d = root / f"{_stamp()}_{secrets.token_hex(3)}"
        try:
            d.mkdir()
            return d
        except FileExistsError:
            continue
    raise RuntimeError(f"could not create a unique run directory under {root}")


def _stamp() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")
