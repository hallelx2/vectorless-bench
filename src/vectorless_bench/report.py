"""Aggregate a run's records into results.json, report.md, and pareto.csv.

The report is opinionated about one thing: for an LLM-as-retriever engine, the
headline is the efficiency frontier — quality plotted against cost and latency —
never quality alone. A system that wins on F1 while costing 50x is not a win, and
this report makes that visible in the first table.

Quality/cost/latency are computed from the first repeat of each question; the
determinism axis uses ALL repeats (it needs the reruns). Errored rows are counted
but excluded from quality/cost so a flaky network call doesn't read as "0% F1".
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

from .metrics import aggregate
from .metrics.citation import primary_quality


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _mean(xs: List[float]) -> float:
    return statistics.fmean(xs) if xs else 0.0


def summarize(run_dir: Path, k: int) -> Dict[str, Any]:
    records = _read_jsonl(run_dir / "records.jsonl")
    setup = json.loads((run_dir / "setup.json").read_text()) if (run_dir / "setup.json").exists() else {}

    by_system: Dict[str, List[Dict]] = defaultdict(list)
    for r in records:
        by_system[r["system"]].append(r)

    results: Dict[str, Any] = {}
    for system, rows in by_system.items():
        first = [r for r in rows if r["repeat"] == 0]
        # "n/a:" marks a question the system cannot be asked (a corpus past
        # its window): counted apart, neither a failure nor a zero.
        not_applicable = sum(1 for r in first if str(r.get("error") or "").startswith("n/a:"))
        first = [r for r in first if not str(r.get("error") or "").startswith("n/a:")]
        rows = [r for r in rows if not str(r.get("error") or "").startswith("n/a:")]
        ok = [r for r in first if not r.get("error")]
        errors = sum(1 for r in first if r.get("error"))

        qualities = [primary_quality(r["metrics"], k) for r in ok]
        costs = [float(r["usage"].get("cost_usd", 0.0)) for r in ok]
        toks = [int(r["usage"].get("total_tokens", 0)) for r in ok]
        calls = [int(r["usage"].get("llm_calls", 0)) for r in ok]
        # Latency from the latency phase only, and only from samples the rig
        # did not flag: a sample measured on a busy machine is not a speed.
        # First repeat only, like quality and cost: later repeats can run warm.
        lat_rows = [r for r in first if not r.get("error") and r.get("phase", "latency") == "latency"]
        latencies = [float(r["latency_ms"]) for r in lat_rows if not r.get("flagged")]
        flagged = sum(1 for r in lat_rows if r.get("flagged"))
        reruns = sum(max(0, int(r.get("attempts", 1)) - 1) for r in lat_rows)
        load_rows = [r for r in rows if not r.get("error") and r.get("phase") == "throughput"]
        engine_split = _engine_split([r for r in lat_rows if not r.get("flagged")])

        # determinism: group selected_ids by qid across repeats
        by_q: Dict[str, List[List[str]]] = defaultdict(list)
        for r in rows:
            if not r.get("error"):
                by_q[r["qid"]].append(r.get("selected_ids", []))
        det = [aggregate.determinism(v) for v in by_q.values() if len(v) > 1]

        def avg_metric(name: str):
            vals = [r["metrics"][name] for r in ok if name in r["metrics"]]
            # None, not 0.0, when no row measured it: "unmeasured" and "zero"
            # are different findings.
            return _mean(vals) if vals else None

        results[system] = {
            "n": len(ok),
            "errors": errors,
            "not_applicable": not_applicable,
            "quality": {
                "primary": _mean(qualities),
                "primary_ci": aggregate.bootstrap_ci(qualities),
                f"precision@{k}": avg_metric(f"precision@{k}"),
                f"recall@{k}": avg_metric(f"recall@{k}"),
                f"f1@{k}": avg_metric(f"f1@{k}"),
                f"ndcg@{k}": avg_metric(f"ndcg@{k}"),
                "mrr": avg_metric("mrr"),
                f"hit@{k}": avg_metric(f"hit@{k}"),
            },
            "citation": {
                "span_in_top1": avg_metric("span_in_top1"),
                "path_correct_top1": avg_metric("path_correct_top1"),
                "near_miss_rate": avg_metric("near_miss_rate"),
                "path_coverage": avg_metric("path_coverage"),
                "gold_has_path": avg_metric("gold_has_path"),
            },
            "abstention": {
                "abstained": avg_metric("abstained"),
            },
            "cost": aggregate.cost_summary(costs, qualities, toks, calls, corrects=_corrects(ok, k)),
            "latency_ms": aggregate.percentiles(latencies),
            "speed": {
                "samples": len(latencies),
                "flagged": flagged,
                "reruns": reruns,
                "under_load_ms": aggregate.percentiles([float(r["latency_ms"]) for r in load_rows]) if load_rows else None,
                "throughput": (setup.get(system) or {}).get("throughput"),
                "engine_split": engine_split,
            },
            "determinism": {
                "exact_match": _mean([d["exact_match"] for d in det]) if det else None,
                "mean_jaccard": _mean([d["mean_jaccard"] for d in det]) if det else None,
            },
            "ingest": setup.get(system, {}),
            "by_domain": _by_domain(ok, k),
        }
    return results


def _corrects(rows: List[Dict], k: int) -> List[float]:
    """Per question: the graded answer when the run graded answers, else
    whether the gold evidence was in the top k (hit@k)."""
    out = []
    for r in rows:
        m = r["metrics"]
        if "answer_correct" in m:
            out.append(float(m["answer_correct"]))
        elif "abstained" in m:
            out.append(float(m["abstained"]))
        else:
            out.append(float(m.get(f"hit@{k}", 0.0)))
    return out


def _engine_split(rows: List[Dict]) -> Dict[str, Any]:
    """Medians of the engine's own timing split over the given rows (systems
    that report one: Vectorless). Empty when none do."""
    t = [r["timing"] for r in rows if isinstance(r.get("timing"), dict)]
    if not t:
        return {}

    def med(get) -> float:
        vals = [float(get(x) or 0.0) for x in t]
        return round(statistics.median(vals), 2)

    return {
        "n": len(t),
        "total_ms": med(lambda x: x.get("total_ms")),
        "engine_ms": med(lambda x: x.get("engine_ms")),
        "judge_wait_ms": med(lambda x: (x.get("judge") or {}).get("wait_ms")),
        "answer_model_wait_ms": med(lambda x: (x.get("answer_model") or {}).get("wait_ms")),
        "db_wait_ms": med(lambda x: (x.get("db") or {}).get("wait_ms")),
        "storage_wait_ms": med(lambda x: (x.get("storage") or {}).get("wait_ms")),
    }


def _by_domain(rows: List[Dict], k: int) -> Dict[str, Dict[str, float]]:
    dom: Dict[str, List[float]] = defaultdict(list)
    for r in rows:
        dom[r["domain"]].append(primary_quality(r["metrics"], k))
    return {d: {"primary": _mean(v), "n": len(v)} for d, v in dom.items()}


def _fmt(x: Any, p: int = 3) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        if x == float("inf"):
            return "∞"
        return f"{x:.{p}f}"
    return str(x)



def _path_cells(c: dict) -> tuple:
    """Table cells for path_correct_top1 and near_miss_rate. A value is shown
    only where it was measured; otherwise the cell says why not: an older run
    that never recorded gold paths, a corpus with none, or a system that
    returns no paths (chunk systems)."""
    gold = c.get("gold_has_path")
    if gold is None:
        why = "unknown (run predates gold_has_path)"
    elif not gold:
        why = "n/a (no gold paths)"
    else:
        why = "n/a (no paths returned)"
    pc, nm = c.get("path_correct_top1"), c.get("near_miss_rate")
    return (_fmt(pc) if pc is not None else why, _fmt(nm) if nm is not None else why)

def write_reports(run_dir: Path, k: int) -> Dict[str, Any]:
    results = summarize(run_dir, k)
    (run_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    _write_pareto_csv(run_dir, results, k)
    _write_markdown(run_dir, results, k)
    _write_html(run_dir, results, k)
    return results


def _write_pareto_csv(run_dir: Path, results: Dict[str, Any], k: int) -> None:
    with (run_dir / "pareto.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["system", "primary_quality", "mean_cost_usd",
                    "p50_latency_ms", "quality_per_1k_usd", "cost_per_correct_usd"])
        for s, r in results.items():
            w.writerow([
                s, _fmt(r["quality"]["primary"]),
                _fmt(r["cost"]["mean_cost_usd"], 6),
                _fmt(r["latency_ms"]["p50"], 1),
                _fmt(r["cost"]["quality_per_1k_usd"], 2),
                _fmt(r["cost"]["cost_per_correct_usd"], 6),
            ])


def _write_markdown(run_dir: Path, results: Dict[str, Any], k: int) -> None:
    lines: List[str] = []
    lines.append(f"# vectorless-bench results\n")
    lines.append(f"Run: `{run_dir.name}` · k={k} · see `manifest.json` for full config.\n")

    lines.append("## Efficiency frontier (the headline)\n")
    lines.append("Quality is meaningless without its price. `quality_per_1k_usd` = "
                 "primary quality bought per $1,000 of query spend (higher is better); "
                 "`$/correct` = spend per correct answer (lower is better).\n")
    lines.append("| System | Quality | $/query | p50 ms | p95 ms | qual/$1k | $/correct | errors |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for s, r in _sorted(results):
        lines.append(
            f"| {s} | {_fmt(r['quality']['primary'])} | "
            f"{_fmt(r['cost']['mean_cost_usd'], 6)} | {_fmt(r['latency_ms']['p50'], 1)} | "
            f"{_fmt(r['latency_ms']['p95'], 1)} | {_fmt(r['cost']['quality_per_1k_usd'], 1)} | "
            f"{_fmt(r['cost']['cost_per_correct_usd'], 5)} | {r['errors']} |"
        )

    lines.extend(_speed_markdown(run_dir, results))
    lines.append("\n## Retrieval quality\n")
    lines.append(f"| System | P@{k} | R@{k} | F1@{k} | nDCG@{k} | MRR | hit@{k} |")
    lines.append("|---|---|---|---|---|---|---|")
    for s, r in _sorted(results):
        q = r["quality"]
        lines.append(
            f"| {s} | {_fmt(q[f'precision@{k}'])} | {_fmt(q[f'recall@{k}'])} | "
            f"{_fmt(q[f'f1@{k}'])} | {_fmt(q[f'ndcg@{k}'])} | {_fmt(q['mrr'])} | "
            f"{_fmt(q[f'hit@{k}'])} |"
        )

    lines.append("\n## Citation exactness & the near-miss test\n")
    lines.append("`near_miss_rate` is the share of returned sections that are a "
                 "*sibling* of the right one (right neighbourhood, wrong leaf) — the "
                 "vector-RAG failure Vectorless is built to avoid. Lower is better.\n")
    lines.append("`path_coverage` is the share of returned units that carried a "
                 "heading path at all. The two path metrics are **undefined, not "
                 "zero**, when either side lacks paths: the system returned none "
                 "(coverage 0), or the DATASET has no gold heading paths at all — "
                 "FinanceBench ships evidence text and pages only, so "
                 "`path_correct_top1` there is 0.000 for every system including a "
                 "perfect one. Those cells print `n/a`, and a `n/a (no gold paths)` "
                 "note means the claim cannot be made from this corpus.\n")
    lines.append("| System | span_in_top1 | path_correct_top1 | near_miss_rate | path_coverage | abstained(no-answer) |")
    lines.append("|---|---|---|---|---|---|")
    for s, r in _sorted(results):
        c = r["citation"]
        path_cell, near_cell = _path_cells(c)
        lines.append(
            f"| {s} | {_fmt(c['span_in_top1'])} | {path_cell} | "
            f"{near_cell} | {_fmt(c.get('path_coverage'))} | {_fmt(r['abstention']['abstained'])} |"
        )

    lines.append("\n## Determinism & ingest cost\n")
    lines.append("| System | exact_match | mean_jaccard | ingest s | ingest $ |")
    lines.append("|---|---|---|---|---|")
    for s, r in _sorted(results):
        d = r["determinism"]
        ing = r.get("ingest", {})
        iu = ing.get("setup_usage", {}) or {}
        lines.append(
            f"| {s} | {_fmt(d['exact_match'])} | {_fmt(d['mean_jaccard'])} | "
            f"{_fmt(ing.get('setup_seconds', 0.0), 1)} | {_fmt(iu.get('cost_usd', 0.0), 5)} |"
        )

    lines.append("\n## Quality by domain\n")
    domains = sorted({d for r in results.values() for d in r["by_domain"]})
    lines.append("| System | " + " | ".join(domains) + " |")
    lines.append("|---|" + "|".join("---" for _ in domains) + "|")
    for s, r in _sorted(results):
        cells = [_fmt(r["by_domain"].get(d, {}).get("primary")) for d in domains]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")

    lines.append("\n---\n_Primary quality = F1@k for answerable questions, correct "
                 "abstention for no-answer questions. Quality/cost/latency from the "
                 "first repeat; determinism across all repeats. Costs use the engine's "
                 "price book (see `manifest.json:pricing_fingerprint`)._\n")

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _speed_markdown(run_dir: Path, results: Dict[str, Any]) -> List[str]:
    out = ["\n## Speed: how it was measured, and what it shows\n",
           "Latency is the harness's wall clock around each system's retrieve call, the same "
           "boundary for every system, one question at a time. Each sample carries host evidence "
           "from every machine involved; a sample taken while the machine was busy (CPU steal or a "
           "foreign process) is rerun, and the figures above use clean samples only.\n",
           "| System | clean samples | flagged | reruns | under load p50 | q/min | concurrency |",
           "|---|---|---|---|---|---|---|"]
    for s, r in _sorted(results):
        sp = r.get("speed") or {}
        tp = sp.get("throughput") or {}
        ul = sp.get("under_load_ms") or {}
        out.append(f"| {s} | {sp.get('samples', '—')} | {sp.get('flagged', '—')} | {sp.get('reruns', '—')} | "
                   f"{_fmt(ul.get('p50'), 1) if ul else '—'} | {_fmt(tp.get('questions_per_minute'), 1) if tp else '—'} | "
                   f"{tp.get('concurrency', '—') if tp else '—'} |")
    splits = [(s, (r.get("speed") or {}).get("engine_split")) for s, r in _sorted(results)]
    splits = [(s, e) for s, e in splits if e]
    if splits:
        out += ["\nWhere a system reports its own split (median per question, ms):\n",
                "| System | total | engine's own work | waiting on Jev | waiting on answer model | db | storage |",
                "|---|---|---|---|---|---|---|"]
        for s, e in splits:
            out.append(f"| {s} | {e['total_ms']} | {e['engine_ms']} | {e['judge_wait_ms']} | "
                       f"{e['answer_model_wait_ms']} | {e['db_wait_ms']} | {e['storage_wait_ms']} |")
    try:
        man = json.loads((run_dir / "manifest.json").read_text())
        ex = man.get("extra") or {}
        net = ex.get("network") or {}
        if net:
            out += ["\nNetwork distance to each provider (median of fresh connections, ms):\n",
                    "| From | Target | TCP connect | TLS handshake |", "|---|---|---|---|"]
            for host, rows in net.items():
                for row in rows or []:
                    out.append(f"| {host} | {row.get('target')} | {row.get('tcp_connect_ms', '—')} | "
                               f"{row.get('tls_handshake_ms', '—')} |")
        mach = ex.get("machines") or {}
        if mach:
            out.append("\nMachines: " + "; ".join(
                f"{k}: {v.get('machine_type') or v.get('host') or '?'} ({v.get('cpu_count')} vCPU)"
                for k, v in mach.items() if isinstance(v, dict)) + "\n")
        ev = ex.get("engine_versions") or {}
        if ev:
            out.append("Engine build: " + "; ".join(f"{k}: {v.get('version') if isinstance(v, dict) else v}"
                                                   for k, v in ev.items()) + "\n")
        for w in man.get("warnings") or []:
            if "unattributable" in w or "stopped" in w:
                out.append(f"> **Warning:** {w}\n")
    except Exception:
        pass
    return out


def _sorted(results: Dict[str, Any]):
    return sorted(results.items(), key=lambda kv: kv[1]["quality"]["primary"], reverse=True)


# ---------------------------------------------------------------------------
# Self-contained HTML report — the artifact you actually open after a VM run.
# Inline CSS + an inline-SVG efficiency-frontier scatter, no JS, no deps.
# ---------------------------------------------------------------------------
_CSS = """
body{font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#0b0d12;color:#e6e9ef}
.wrap{max-width:1080px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:24px;margin:0 0 4px} h2{font-size:16px;margin:34px 0 10px;color:#aab3c5}
.sub{color:#8a93a6;font-size:13px;margin-bottom:8px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:7px 10px;text-align:right;border-bottom:1px solid #1c2230}
th:first-child,td:first-child{text-align:left}
th{color:#8a93a6;font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
tr:hover td{background:#11151f}
.sys{font-weight:600;color:#fff}
.best{color:#46d39a;font-weight:600}
.bad{color:#f0816b}
.na{color:#8a8f98;font-style:italic}
.card{background:#10141d;border:1px solid #1c2230;border-radius:12px;padding:18px 20px;margin-top:10px}
.note{font-size:12px;color:#727b8f;margin-top:24px;border-top:1px solid #1c2230;padding-top:12px}
svg{background:#0e121b;border:1px solid #1c2230;border-radius:12px}
.dot{fill:#5b8cff} .lbl{fill:#c6ccd8;font-size:11px}
"""


def _esc(s: object) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _frontier_svg(results: Dict[str, Any]) -> str:
    """Quality (y) vs $/query (x, log) scatter — the headline trade-off."""
    import math

    pts = [
        (s, r["cost"]["mean_cost_usd"], r["quality"]["primary"])
        for s, r in results.items()
        if r["n"] > 0
    ]
    if not pts:
        return "<p class='sub'>no successful runs to plot</p>"
    W, H, pad = 720, 320, 48
    costs = [max(c, 1e-7) for _, c, _ in pts]
    lo, hi = math.log10(min(costs)), math.log10(max(costs) * 1.3 or 1)
    if hi - lo < 0.5:
        lo, hi = lo - 0.5, hi + 0.5

    def x(c):
        return pad + (math.log10(max(c, 1e-7)) - lo) / (hi - lo) * (W - 2 * pad)

    def y(q):
        return H - pad - q * (H - 2 * pad)

    dots = []
    for s, c, q in pts:
        cx, cy = x(c), y(q)
        dots.append(f"<circle class='dot' cx='{cx:.0f}' cy='{cy:.0f}' r='6'/>")
        dots.append(f"<text class='lbl' x='{cx + 9:.0f}' y='{cy + 4:.0f}'>{_esc(s)}</text>")
    grid = "".join(
        f"<line x1='{pad}' y1='{y(g):.0f}' x2='{W - pad}' y2='{y(g):.0f}' stroke='#1c2230'/>"
        f"<text class='lbl' x='6' y='{y(g) + 4:.0f}'>{g:.1f}</text>"
        for g in (0, 0.25, 0.5, 0.75, 1.0)
    )
    return (
        f"<svg viewBox='0 0 {W} {H}' width='100%'>{grid}{''.join(dots)}"
        f"<text class='lbl' x='{W/2:.0f}' y='{H-12}'>$/query (log) → cheaper left</text>"
        f"<text class='lbl' x='10' y='20'>quality ↑</text></svg>"
    )


def _html_table(headers, rows) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(
            f"<td class='{cls}'>{_esc(v)}</td>" for v, cls in r
        ) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _write_html(run_dir: Path, results: Dict[str, Any], k: int) -> None:
    sd = _sorted(results)
    best_q = max((r["quality"]["primary"] for _, r in sd), default=0.0)

    frontier_rows = []
    for s, r in sd:
        q = r["quality"]["primary"]
        frontier_rows.append([
            (s, "sys"),
            (_fmt(q), "best" if q == best_q and q > 0 else ""),
            (_fmt(r["cost"]["mean_cost_usd"], 6), ""),
            (_fmt(r["latency_ms"]["p50"], 1), ""),
            (_fmt(r["latency_ms"]["p95"], 1), ""),
            (_fmt(r["cost"]["quality_per_1k_usd"], 1), ""),
            (_fmt(r["cost"]["cost_per_correct_usd"], 5), ""),
            (r["errors"], "bad" if r["errors"] else ""),
        ])
    qual_rows = [
        [(s, "sys")] + [(_fmt(r["quality"][m]), "") for m in
         (f"precision@{k}", f"recall@{k}", f"f1@{k}", f"ndcg@{k}", "mrr", f"hit@{k}")]
        for s, r in sd
    ]
    def _cite_row(s, r):
        c = r["citation"]
        cov = c.get("path_coverage") or 0.0
        pc, nm = _path_cells(c)
        measured_pc = c.get("path_correct_top1") is not None
        measured_nm = c.get("near_miss_rate") is not None
        path_cell = (pc, "" if measured_pc else "na")
        near_cell = (nm, ("bad" if c["near_miss_rate"] > 0.1 else "") if measured_nm else "na")
        return [(s, "sys"), (_fmt(c["span_in_top1"]), ""), path_cell, near_cell,
                (_fmt(cov), "bad" if 0 < cov < 0.5 else ""),
                (_fmt(r["abstention"]["abstained"]), "")]

    cite_rows = [_cite_row(s, r) for s, r in sd]
    det_rows = [
        [(s, "sys"),
         (_fmt(r["determinism"]["exact_match"]), ""),
         (_fmt(r["determinism"]["mean_jaccard"]), ""),
         (_fmt(r.get("ingest", {}).get("setup_seconds", 0.0), 1), ""),
         (_fmt((r.get("ingest", {}).get("setup_usage", {}) or {}).get("cost_usd", 0.0), 5), "")]
        for s, r in sd
    ]

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>vectorless-bench · {_esc(run_dir.name)}</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>vectorless-bench results</h1>
<div class="sub">run <code>{_esc(run_dir.name)}</code> · k={k} · primary quality = F1@k (answerable) / correct-abstention (no-answer)</div>
<h2>Efficiency frontier</h2>
<div class="card">{_frontier_svg(results)}</div>
{_html_table(["System","Quality","$/query","p50 ms","p95 ms","qual/$1k","$/correct","errors"], frontier_rows)}
<h2>Retrieval quality</h2>
{_html_table(["System",f"P@{k}",f"R@{k}",f"F1@{k}",f"nDCG@{k}","MRR",f"hit@{k}"], qual_rows)}
<h2>Citation exactness &amp; near-miss</h2>
<div class="sub">near-miss = retrieved a sibling of the right section (the vector failure mode). Lower is better; path metrics read n/a where they cannot be measured (no gold path, or a system that returns no paths).</div>
{_html_table(["System","span_in_top1","path_correct@1","near_miss","path_coverage","abstained(no-ans)"], cite_rows)}
<h2>Determinism &amp; ingest cost</h2>
{_html_table(["System","exact_match","mean_jaccard","ingest s","ingest $"], det_rows)}
<div class="note">Quality/cost/latency from the first repeat; determinism across all repeats. Costs use the engine's price book — see manifest.json.</div>
</div></body></html>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")
