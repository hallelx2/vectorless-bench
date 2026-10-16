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
    if args.phase:
        cfg.phase = args.phase
    if args.concurrency is not None:
        cfg.concurrency = args.concurrency
    if args.max_spend is not None:
        cfg.max_spend_usd = args.max_spend
    if args.engine_probe:
        cfg.rig = {**cfg.rig, "probes": {**(cfg.rig.get("probes") or {}), "engine": args.engine_probe}}
    if args.no_rig:
        cfg.rig = {**cfg.rig, "enabled": False}

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
    from .datasets import available as datasets_available

    print("datasets:", ", ".join(datasets_available()))
    return 0


def _print_headline(results, k: int) -> None:
    hdr = f"\n{'system':<14} {'quality':>8} {'$/query':>10} {'p50ms':>8} {'qual/$1k':>10} {'n':>4} {'err':>4}"
    print(hdr)
    rows = sorted(results.items(), key=lambda kv: kv[1]["quality"]["primary"], reverse=True)
    for s, r in rows:
        print(f"{s:<14} {r['quality']['primary']:>8.3f} "
              f"{r['cost']['mean_cost_usd']:>10.6f} {r['latency_ms']['p50']:>8.1f} "
              f"{r['cost']['quality_per_1k_usd']:>10.1f} {r['n']:>4} {r['errors']:>4}")


def cmd_datasets(args: argparse.Namespace) -> int:
    import json as _json

    from .datasets import available as datasets_available
    from .datasets import build_dataset

    if args.action == "list":
        print("\n".join(datasets_available()))
        return 0
    names = datasets_available()[1:] if args.name == "all" else [args.name]
    params = _json.loads(args.params) if args.params else {}
    status = 0
    for name in names:
        d = build_dataset(name, **params)
        try:
            if args.action == "fetch":
                d.fetch()
            else:
                print(_json.dumps(d.check(), indent=1))
        except Exception as e:  # keep going across datasets
            status = 1
            print(f"[vlbench] {name}: {args.action} failed: {e}")
    return status


def cmd_embed(args: argparse.Namespace) -> int:
    """Build the per-document embedding cache for a dataset ahead of a run,
    with the same chunking the vector systems use. The seconds each document
    took are stored with its vectors and reported by the run that reads them."""
    import json as _json
    import os

    from .datasets import build_dataset
    from .retrievers.vector_rag import VectorRagRetriever

    os.environ.setdefault("VLBENCH_EMBED_PROGRESS", "1")
    names = [n for n in __import__("vectorless_bench.datasets", fromlist=["available"]).available() if n != "fixtures"] \
        if args.name == "all" else [args.name]
    for name in names:
        params = _json.loads(args.params) if args.params else {}
        if args.limit:
            # the runner's own selection, so the cache covers exactly the
            # documents a `run --limit N --seed S` will ask for
            from .config import RunConfig
            from .runner import _load_dataset

            _, corpus, _ = _load_dataset(RunConfig(dataset=name, dataset_params=params, limit=args.limit,
                                                   sample_seed=args.seed))
        else:
            corpus = build_dataset(name, **params).corpus()
        r = VectorRagRetriever(embedding_model=args.model, backend="memory",
                               chunk_tokens=args.chunk_tokens, overlap_tokens=args.overlap_tokens)
        r.setup(corpus)
        print(_json.dumps({"dataset": name, "model": args.model, "docs": len(corpus), "chunks": len(r._chunks),
                           "embedding_usd": round(r.setup_usage.cost_usd, 4), **r.setup_meta}), flush=True)
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    from .hostprobe import serve

    print(f"[vlbench] probe agent on {args.bind}:{args.port}", flush=True)
    serve(args.port, args.bind)
    return 0


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
    r.add_argument("--phase", choices=["latency", "throughput"], help="latency (default) or throughput")
    r.add_argument("--concurrency", type=int, help="questions in flight for --phase throughput")
    r.add_argument("--max-spend", type=float, help="stop once recorded spend reaches this many USD")
    r.add_argument("--engine-probe", help="engine VM probe agent URL, e.g. http://10.128.0.5:9101")
    r.add_argument("--no-rig", action="store_true", help="skip host evidence (local smoke runs only)")
    r.set_defaults(func=cmd_run)

    rep = sub.add_parser("report", help="(re)build reports from a run dir")
    rep.add_argument("run_dir")
    rep.add_argument("--k", type=int, default=5)
    rep.set_defaults(func=cmd_report)

    s = sub.add_parser("systems", help="list available systems and datasets")
    s.set_defaults(func=cmd_systems)

    ds = sub.add_parser("datasets", help="list, fetch or check the datasets")
    ds.add_argument("action", choices=["list", "fetch", "check"])
    ds.add_argument("name", nargs="?", default="all", help="a dataset name, or all")
    ds.add_argument("--params", help='dataset params as JSON, e.g. \'{"mini": true}\'')
    ds.set_defaults(func=cmd_datasets)

    em = sub.add_parser("embed", help="pre-build the embedding cache for a dataset")
    em.add_argument("name", help="a dataset name, or all")
    em.add_argument("--model", default="BAAI/bge-base-en-v1.5")
    em.add_argument("--chunk-tokens", type=int, default=512)
    em.add_argument("--overlap-tokens", type=int, default=64)
    em.add_argument("--params", help="dataset params as JSON")
    em.add_argument("--limit", type=int, help="only the documents of `run --limit N` (same sample)")
    em.add_argument("--seed", type=int, default=0, help="the run's sample seed")
    em.set_defaults(func=cmd_embed)

    pr = sub.add_parser("probe", help="host evidence agent (run on the engine VM)")
    pr.add_argument("--port", type=int, default=9101)
    pr.add_argument("--bind", default="0.0.0.0")
    pr.set_defaults(func=cmd_probe)
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
