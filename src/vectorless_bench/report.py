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
        ok = [r for r in first if not r.get("error")]
        errors = sum(1 for r in first if r.get("error"))

        qualities = [primary_quality(r["metrics"], k) for r in ok]
        costs = [float(r["usage"].get("cost_usd", 0.0)) for r in ok]
        toks = [int(r["usage"].get("total_tokens", 0)) for r in ok]
        calls = [int(r["usage"].get("llm_calls", 0)) for r in ok]
        latencies = [float(r["latency_ms"]) for r in rows if not r.get("error")]

        # determinism: group selected_ids by qid across repeats
        by_q: Dict[str, List[List[str]]] = defaultdict(list)
        for r in rows:
            if not r.get("error"):
                by_q[r["qid"]].append(r.get("selected_ids", []))
        det = [aggregate.determinism(v) for v in by_q.values() if len(v) > 1]

        def avg_metric(name: str) -> float:
            return _mean([r["metrics"].get(name, 0.0) for r in ok if name in r["metrics"]])

        results[system] = {
            "n": len(ok),
            "errors": errors,
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
            },
            "abstention": {
                "abstained": avg_metric("abstained"),
            },
            "cost": aggregate.cost_summary(costs, qualities, toks, calls),
            "latency_ms": aggregate.percentiles(latencies),
            "determinism": {
                "exact_match": _mean([d["exact_match"] for d in det]) if det else None,
                "mean_jaccard": _mean([d["mean_jaccard"] for d in det]) if det else None,
            },
            "ingest": setup.get(system, {}),
            "by_domain": _by_domain(ok, k),
        }
    return results


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
    lines.append("| System | span_in_top1 | path_correct_top1 | near_miss_rate | abstained(no-answer) |")
    lines.append("|---|---|---|---|---|")
    for s, r in _sorted(results):
        c = r["citation"]
        lines.append(
            f"| {s} | {_fmt(c['span_in_top1'])} | {_fmt(c['path_correct_top1'])} | "
            f"{_fmt(c['near_miss_rate'])} | {_fmt(r['abstention']['abstained'])} |"
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
    cite_rows = [
        [(s, "sys"),
         (_fmt(r["citation"]["span_in_top1"]), ""),
         (_fmt(r["citation"]["path_correct_top1"]), ""),
         (_fmt(r["citation"]["near_miss_rate"]), "bad" if r["citation"]["near_miss_rate"] > 0.1 else ""),
         (_fmt(r["abstention"]["abstained"]), "")]
        for s, r in sd
    ]
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
<div class="sub">near-miss = retrieved a sibling of the right section (the vector failure mode). Lower is better; chunk systems score 0 on path-correctness by construction.</div>
{_html_table(["System","span_in_top1","path_correct@1","near_miss","abstained(no-ans)"], cite_rows)}
<h2>Determinism &amp; ingest cost</h2>
{_html_table(["System","exact_match","mean_jaccard","ingest s","ingest $"], det_rows)}
<div class="note">Quality/cost/latency from the first repeat; determinism across all repeats. Costs use the engine's price book — see manifest.json.</div>
</div></body></html>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")
