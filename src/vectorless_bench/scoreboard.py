"""The published scoreboard: every dataset and every system, from the run
directories (HAL-3088).

Each run directory holds one dataset's records for some systems. The
scoreboard recomputes every summary from the raw records (never from a
cached results.json, which an older harness may have written), then keeps,
per dataset and system, the latest isolated-latency run, and takes
throughput from the latest throughput run beside it. A cell is empty when it
was not measured, never zero.

Outputs, in the directory given:
- scoreboard.md: the headline table per dataset, the slices each dataset
  needs (MMLongBench by evidence source, MultiHop by query type), and the
  manifest of what was measured, when, on which builds
- scoreboard.csv: one row per dataset x system, every headline column
- scoreboard.json: the full summaries
- quality_vs_cost.svg, quality_vs_latency.svg: one point per dataset x system

Wording is for readers outside the project: the answer grade is "graded
answer accuracy", and no internal model names appear in it.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .report import summarize

# The native metric that heads each dataset's column, as its paper reports it.
_NATIVE_HEADLINE = {
    "qasper": ("answer_f1", "answer F1"),
    "legalbench_rag": ("char_recall@{k}", "char recall@{k}"),
    "cuad": ("clause_match", "clause match (on the generated answer)"),
    "multihop_rag": ("hits@{k}", "hits@{k}"),
}
_SLICES = {
    "mmlongbench_doc": "evidence source",
    "multihop_rag": "query type",
    "cuad": "contract",
    "legalbench_rag": "subset",
}
_LABEL = {
    "vectorless": "Vectorless",
    "vector_rag": "Vector RAG (gemini-embedding-2)",
    "vector_bge": "Vector RAG (BGE, CPU)",
    "bm25": "BM25",
    "full_context": "Full context",
    "pageindex": "PageIndex",
    "agentic_grep": "Agentic grep",
    "memwalker": "MemWalker (reimpl.)",
    "readagent": "ReadAgent (reimpl.)",
    "raptor": "RAPTOR",
}


def _load(run_dir: Path) -> Optional[Dict[str, Any]]:
    if not (run_dir / "records.jsonl").exists():
        return None
    man = {}
    if (run_dir / "manifest.json").exists():
        man = json.loads((run_dir / "manifest.json").read_text())
    k = int(man.get("k") or 5)
    return {"dir": str(run_dir), "manifest": man, "k": k, "results": summarize(run_dir, k)}


def _phase(entry: Dict[str, Any], system: str) -> str:
    r = entry["results"].get(system) or {}
    speed = r.get("speed") or {}
    if speed.get("throughput") and not speed.get("samples"):
        return "throughput"
    return "latency"


class MixedAnswerSetups(ValueError):
    """The runs were answered under different models or answer prompts."""


def setup_of(manifest: Dict[str, Any]) -> str:
    """One run's answer setup as a label: model, answer model, prompt. Runs
    written before the manifest recorded it (bench < #24) are read from
    their config and marked prompt v1."""
    s = _get(manifest, "extra", "answer_setup")
    if not s:
        cfg = _get(manifest, "extra", "config") or {}
        s = {"model": manifest.get("model") or cfg.get("model"),
             "gen_model": cfg.get("gen_model") if cfg.get("judge") else None,
             "answer_prompt": "v1" if cfg.get("judge") else None}
    return f"{s.get('model')} | answers {s.get('gen_model') or '—'} | prompt {s.get('answer_prompt') or '—'}"


def collect(run_dirs: Iterable[Path], allow_mixed: bool = False) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """board[dataset][system] = {"result", "k", "run", "created_at", "manifest", "throughput"}

    Refuses (MixedAnswerSetups) runs answered under different models or
    answer prompts: a later run would silently replace an earlier one's
    cells and the board would compare answers written two ways."""
    runs = [e for e in (_load(Path(d)) for d in run_dirs) if e]
    setups = sorted({setup_of(e["manifest"]) for e in runs})
    if len(setups) > 1 and not allow_mixed:
        raise MixedAnswerSetups("runs were answered under different setups: " + "; ".join(setups)
                                + ". Build one scoreboard per setup.")
    runs.sort(key=lambda e: e["manifest"].get("created_at") or "")
    board: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for e in runs:  # oldest first, so a later run replaces an earlier one
        ds = e["manifest"].get("dataset") or "unknown"
        for system, res in e["results"].items():
            cell = board.setdefault(ds, {}).setdefault(system, {})
            if _phase(e, system) == "throughput":
                cell["throughput"] = (res.get("speed") or {}).get("throughput")
                cell["throughput_run"] = e["dir"]
                continue
            if (res.get("n") or 0) > 0:
                # quality, cost and answers: the latest run that answered questions
                cell.update(result=res, k=e["k"], run=e["dir"], created_at=e["manifest"].get("created_at"),
                            manifest=e["manifest"], setup=setup_of(e["manifest"]))
            if (res.get("speed") or {}).get("samples"):
                # speed: only runs timed on the isolated rig carry latency samples
                # (an accuracy pass records none)
                cell["timing"] = {"latency_ms": res.get("latency_ms"), "speed": res.get("speed"), "run": e["dir"]}
    return board


def _get(d: Any, *path: str) -> Any:
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def _num(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isinf(f) or math.isnan(f) else f


def row(dataset: str, system: str, cell: Dict[str, Any]) -> Dict[str, Any]:
    r = cell.get("result") or {}
    k = cell.get("k", 5)
    native_key, native_label = _NATIVE_HEADLINE.get(dataset, (None, None))
    native = None
    if native_key:
        native = _num(_get(r, "native", native_key.format(k=k)))
    n = r.get("n") or 0
    judge_usd = _num(_get(r, "answer", "judge_usd"))
    graded = _get(r, "answer", "graded") or 0
    tp = cell.get("throughput") or _get(r, "speed", "throughput") or {}
    return {
        "dataset": dataset,
        "system": system,
        "label": _LABEL.get(system, system),
        "n": n,
        "errors": r.get("errors"),
        "not_applicable": r.get("not_applicable"),
        "quality": _num(_get(r, "quality", "primary")),
        "quality_lo": _num(_get(r, "quality", "primary_ci", "lo")),
        "quality_hi": _num(_get(r, "quality", "primary_ci", "hi")),
        f"hit@{k}": _num(_get(r, "quality", f"hit@{k}")),
        "reachable_quality": _num(_get(r, "reachable", "primary")),
        "reachable_share": _num(_get(r, "reachable", "share_of_scored")),
        "native_metric": native_label.format(k=k) if native_label else None,
        "native": native,
        "answer_accuracy": _num(_get(r, "answer", "accuracy")),
        "answer_declined": _num(_get(r, "answer", "abstained")),
        "abstained": _num(_get(r, "abstention", "abstained")),
        "usd_per_question": _num(_get(r, "cost", "mean_cost_usd")),
        "usd_per_correct": _num(_get(r, "cost", "cost_per_correct_usd")),
        "grading_usd_per_question": (judge_usd / graded) if judge_usd is not None and graded else None,
        "context_tokens": _num(_get(r, "quality", "context_tokens")),
        "p50_ms": _num(_get(cell, "timing", "latency_ms", "p50")),
        "p95_ms": _num(_get(cell, "timing", "latency_ms", "p95")),
        "flagged_reruns": _get(cell, "timing", "speed", "reruns"),
        "timing_run": _get(cell, "timing", "run"),
        "questions_per_minute": _num(tp.get("questions_per_minute")) if isinstance(tp, dict) else None,
        "ingest_seconds": _num(_get(r, "ingest", "setup_seconds")),
        "run": cell.get("run"),
        "setup": cell.get("setup"),
    }


def _fmt(x: Any, p: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.{p}f}"
    return str(x)


def _usd(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"${x:.5f}" if x < 0.01 else f"${x:.3f}"


def _ms(x: Optional[float]) -> str:
    if x is None:
        return "—"
    return f"{x / 1000:.1f} s" if x >= 1000 else f"{x:.0f} ms"


def build(run_dirs: Iterable[Path], out: Path, allow_mixed: bool = False) -> Dict[str, Any]:
    board = collect(run_dirs, allow_mixed=allow_mixed)
    out.mkdir(parents=True, exist_ok=True)
    rows = [row(ds, sys_, cell) for ds in sorted(board) for sys_, cell in sorted(board[ds].items()) if cell.get("result")]
    _write_csv(out / "scoreboard.csv", rows)
    (out / "scoreboard.json").write_text(json.dumps({"rows": rows, "board": board}, indent=2, default=str), encoding="utf-8")
    (out / "scoreboard.md").write_text(_markdown(board, rows), encoding="utf-8")
    (out / "quality_vs_cost.svg").write_text(_scatter(rows, "usd_per_question", "$ per question (log)", log=True), encoding="utf-8")
    (out / "quality_vs_latency.svg").write_text(_scatter(rows, "p50_ms", "median latency, ms (log)", log=True), encoding="utf-8")
    return {"datasets": len(board), "rows": len(rows), "out": str(out)}


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for r in rows:
        for key in r:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _markdown(board: Dict[str, Dict[str, Dict[str, Any]]], rows: List[Dict[str, Any]]) -> str:
    lines = ["# Vectorless benchmark scoreboard", ""]
    lines.append("Quality is F1 at k of the units each system returned against the gold evidence (on a "
                 "question with no answer, whether the system abstained), with a 95% bootstrap interval. "
                 "Reachable restricts it to questions whose evidence exists in the extracted "
                 "text. Graded answer accuracy grades an answer written by the same model, from each system's "
                 "units only (full context answers from the whole document it read, as a ceiling should); "
                 "Declined is the share where that model said the units were not enough. Cost is the system's own spend per question; grading is reported apart. Latency "
                 "is from the isolated pass, with contended samples rerun.")
    lines.append("")
    for ds in sorted(board):
        ds_rows = sorted((r for r in rows if r["dataset"] == ds), key=lambda r: -(r["quality"] or -1))
        if not ds_rows:
            continue
        native = next((r["native_metric"] for r in ds_rows if r["native_metric"]), None)
        lines.append(f"## {ds}")
        lines.append("")
        head = ["System", "n", "Quality (95% CI)", "Reachable"] + ([native] if native else []) + [
            "Graded answer accuracy", "Declined", "$ / question", "p50", "p95", "q/min", "Context tokens"]
        lines.append("| " + " | ".join(head) + " |")
        lines.append("|" + "---|" * len(head))
        for r in ds_rows:
            ci = f"{_fmt(r['quality'])} ({_fmt(r['quality_lo'])}–{_fmt(r['quality_hi'])})" if r["quality"] is not None else "—"
            cells = [r["label"], str(r["n"]), ci, _fmt(r["reachable_quality"])] + (
                [_fmt(r["native"])] if native else []) + [
                _fmt(r["answer_accuracy"]), _fmt(r["answer_declined"]), _usd(r["usd_per_question"]),
                _ms(r["p50_ms"]), _ms(r["p95_ms"]),
                _fmt(r["questions_per_minute"], 1), _fmt(r["context_tokens"], 0)]
            lines.append("| " + " | ".join(cells) + " |")
        na = [r for r in ds_rows if r.get("not_applicable")]
        if na:
            lines.append("")
            lines.append("Not applicable: " + "; ".join(f"{r['label']} on {r['not_applicable']} questions" for r in na) + ".")
        if ds in _SLICES:
            slice_lines = _slices(board[ds], _SLICES[ds])
            if slice_lines:
                lines.append("")
                lines.extend(slice_lines)
        lines.append("")
    lines.append("## What was measured")
    lines.append("")
    lines.append("| Dataset | System | Run | Measured | Bench | Model | Engine |")
    lines.append("|---|---|---|---|---|---|---|")
    for ds in sorted(board):
        for system, cell in sorted(board[ds].items()):
            if not cell.get("result"):
                continue
            man = cell.get("manifest") or {}
            ev = (man.get("extra") or {}).get("engine_versions") or {}
            eng = (ev.get(system) or {}).get("version") if isinstance(ev.get(system), dict) else None
            lines.append(f"| {ds} | {_LABEL.get(system, system)} | `{Path(cell['run']).name}` | "
                         f"{(cell.get('created_at') or '')[:19]} | {man.get('bench_git_sha', '—')} | "
                         f"{man.get('model', '—')} | {eng or '—'} |")
    return "\n".join(lines) + "\n"


def _slices(cells: Dict[str, Dict[str, Any]], what: str) -> List[str]:
    systems = [s for s, c in sorted(cells.items()) if (c.get("result") or {}).get("by_domain")]
    if not systems:
        return []
    domains = sorted({d for s in systems for d in cells[s]["result"]["by_domain"]})
    out = [f"By {what} (quality):", "", "| " + " | ".join([what.capitalize()] + [_LABEL.get(s, s) for s in systems]) + " |",
           "|" + "---|" * (len(systems) + 1)]
    for d in domains:
        vals = []
        for s in systems:
            v = cells[s]["result"]["by_domain"].get(d)
            q = v.get("primary") if isinstance(v, dict) else v
            n = v.get("n") if isinstance(v, dict) else None
            vals.append(f"{_fmt(_num(q))}" + (f" (n={n})" if n else ""))
        out.append("| " + " | ".join([d] + vals) + " |")
    return out


_PALETTE = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c", "#65a30d", "#a16207"]


def _scatter(rows: List[Dict[str, Any]], xkey: str, xlabel: str, log: bool = False) -> str:
    pts = [(r, _num(r.get(xkey)), r.get("quality")) for r in rows]
    pts = [(r, x, y) for r, x, y in pts if x is not None and y is not None and (x > 0 or not log)]
    w, h, pad = 760, 460, 64
    if not pts:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}"><text x="20" y="40">no data</text></svg>'
    xs = [math.log10(x) if log else x for _, x, _ in pts]
    x0, x1 = min(xs), max(xs)
    if x1 - x0 < 1e-9:
        x0, x1 = x0 - 1, x1 + 1
    systems = sorted({r["system"] for r, _, _ in pts})
    color = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(systems)}

    def px(v: float) -> float:
        return pad + (v - x0) / (x1 - x0) * (w - 2 * pad)

    def py(v: float) -> float:
        return h - pad - v * (h - 2 * pad)

    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" font-family="sans-serif" font-size="12">',
             f'<rect width="{w}" height="{h}" fill="white"/>',
             f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#555"/>',
             f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#555"/>',
             f'<text x="{w / 2}" y="{h - 20}" text-anchor="middle">{xlabel}</text>',
             f'<text x="18" y="{h / 2}" transform="rotate(-90 18 {h / 2})" text-anchor="middle">quality</text>']
    for t in (0.0, 0.25, 0.5, 0.75, 1.0):
        parts.append(f'<text x="{pad - 8}" y="{py(t) + 4}" text-anchor="end">{t:.2f}</text>')
    for (r, x, y), xv in zip(pts, xs):
        parts.append(f'<circle cx="{px(xv):.1f}" cy="{py(y):.1f}" r="5" fill="{color[r["system"]]}">'
                     f'<title>{r["label"]} · {r["dataset"]}: {y:.3f} at {x:g}</title></circle>')
    for i, s in enumerate(systems):
        parts.append(f'<rect x="{w - pad - 190}" y="{pad + i * 16 - 9}" width="10" height="10" fill="{color[s]}"/>'
                     f'<text x="{w - pad - 175}" y="{pad + i * 16}">{_LABEL.get(s, s)}</text>')
    parts.append("</svg>")
    return "\n".join(parts)
