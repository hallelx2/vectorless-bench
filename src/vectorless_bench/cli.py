"""Command line: `vlbench run | report | systems`.

Uses argparse (stdlib) so the CLI imports with zero third-party dependencies; a
config file needs pyyaml, but the flag-only path (used by the smoke test) does
not. `run` executes the suite then writes the reports in one shot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from .config import RunConfig, load_config
from .report import write_reports
from .runner import run as run_suite


def _split(csv: Optional[str]) -> Optional[List[str]]:
    return [s.strip() for s in csv.split(",") if s.strip()] if csv else None


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config) if args.config else RunConfig()
    # CLI flags override the config file
    if args.dataset:
        cfg.dataset = args.dataset
    if args.systems:
        cfg.systems = _split(args.systems)
    if args.k is not None:
        cfg.k = args.k
    if args.repeats is not None:
        cfg.repeats = args.repeats
    if args.limit is not None:
        cfg.limit = args.limit
    if args.model:
        cfg.model = args.model
    if args.out:
        cfg.out_dir = args.out
    if args.judge:
        cfg.judge = True
    if args.warm:
        cfg.cold = False

    run_dir = run_suite(cfg)
    results = write_reports(run_dir, cfg.k)
    print(f"\n[vlbench] report: {run_dir / 'report.md'}")
    _print_headline(results, cfg.k)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    if not (run_dir / "records.jsonl").exists():
        print(f"no records.jsonl in {run_dir}", file=sys.stderr)
        return 1
    results = write_reports(run_dir, args.k)
    _print_headline(results, args.k)
    print(f"[vlbench] wrote {run_dir / 'report.md'}")
    return 0


def cmd_systems(_: argparse.Namespace) -> int:
    from .retrievers import available

    print("systems:", ", ".join(available()))
    print("datasets: fixtures, financebench")
    return 0


def _print_headline(results, k: int) -> None:
    hdr = f"\n{'system':<14} {'quality':>8} {'$/query':>10} {'p50ms':>8} {'qual/$1k':>10} {'n':>4} {'err':>4}"
    print(hdr)
    rows = sorted(results.items(), key=lambda kv: kv[1]["quality"]["primary"], reverse=True)
    for s, r in rows:
        print(f"{s:<14} {r['quality']['primary']:>8.3f} "
              f"{r['cost']['mean_cost_usd']:>10.6f} {r['latency_ms']['p50']:>8.1f} "
              f"{r['cost']['quality_per_1k_usd']:>10.1f} {r['n']:>4} {r['errors']:>4}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vlbench", description="Vectorless benchmark suite")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run a benchmark and write reports")
    r.add_argument("--config", help="YAML config path (configs/*.yaml)")
    r.add_argument("--dataset", help="fixtures | financebench")
    r.add_argument("--systems", help="comma list, e.g. vectorless,vector_rag,bm25")
    r.add_argument("--k", type=int, help="top-k sections to retrieve")
    r.add_argument("--repeats", type=int, help="reruns per question (determinism)")
    r.add_argument("--limit", type=int, help="sample N questions (stratified)")
    r.add_argument("--model", help="retrieval model")
    r.add_argument("--out", help="output dir (default: runs/)")
    r.add_argument("--judge", action="store_true", help="enable LLM-judge axis")
    r.add_argument("--warm", action="store_true", help="allow warm caches (not cold)")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="(re)build reports from a run dir")
    rep.add_argument("run_dir")
    rep.add_argument("--k", type=int, default=5)
    rep.set_defaults(func=cmd_report)

    s = sub.add_parser("systems", help="list available systems and datasets")
    s.set_defaults(func=cmd_systems)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
