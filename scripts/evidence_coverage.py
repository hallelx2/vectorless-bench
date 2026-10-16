"""Evidence-page coverage: does the TOC tree put each gold answer where it is?

    python scripts/evidence_coverage.py <trees-dir> [--json out.json]

FinanceBench gives a gold evidence page for every question. cmd/tocdump
gives the section tree the pipeline built for each filing, with page
ranges. This joins them and asks, per question: does the evidence page fall
inside a leaf section — and a leaf of sane size, not a catch-all that
covers half the document?

This is the accuracy number that gates any change to detection. "Jev
agreed with GLM" is not one; GLM's arm was silently empty on 13/21 docs.
Whether the tree puts the answer where it actually is, is.

Metrics:
  covered      gold page inside SOME leaf [start,end]
  covered_sane covered, and that leaf spans <= max_span pages
  median_span  median leaf width across the tree, in pages
Reported per document and overall.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def leaves(nodes, out):
    for n in nodes:
        kids = n.get("nodes") or []
        if kids:
            leaves(kids, out)
        else:
            s, e = n.get("start_page") or 0, n.get("end_page") or 0
            if s > 0:
                out.append((s, e if e >= s else s, n.get("title", "")))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trees")
    ap.add_argument("--max-span", type=int, default=25,
                    help="a leaf wider than this is a catch-all, not a section")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("PatronusAI/financebench", split="train")

    trees = {}
    for p in Path(args.trees).glob("*.json"):
        d = json.loads(p.read_text())
        trees[d["doc"]] = d

    rows, per_doc = [], {}
    for r in ds:
        doc = r["doc_name"]
        t = trees.get(doc)
        if not t:
            continue
        lv = leaves(t.get("nodes") or [], [])
        for ev in r.get("evidence") or []:
            page = ev.get("evidence_page_num")
            if page is None:
                continue
            page = int(page) + 1  # FinanceBench pages are 0-indexed; the tree is 1-indexed
            hits = [(s, e, ttl) for s, e, ttl in lv if s <= page <= e]
            covered = bool(hits)
            sane = any(e - s + 1 <= args.max_span for s, e, _ in hits)
            rows.append({"doc": doc, "qid": r["financebench_id"], "page": page,
                         "covered": covered, "covered_sane": sane,
                         "leaf": hits[0][2] if hits else None,
                         "leaf_span": (hits[0][1] - hits[0][0] + 1) if hits else None})
            pd = per_doc.setdefault(doc, {"n": 0, "cov": 0, "sane": 0, "leaves": len(lv),
                                          "median_span": statistics.median([e - s + 1 for s, e, _ in lv]) if lv else None,
                                          "err": t.get("err", "")})
            pd["n"] += 1; pd["cov"] += covered; pd["sane"] += sane

    if not rows:
        print("no questions matched any dumped tree — check doc names", file=sys.stderr)
        return 1

    print(f"{'document':<28} {'q':>3} {'covered':>8} {'sane':>6} {'leaves':>7} {'med span':>9}")
    print("-" * 66)
    for doc, pd in sorted(per_doc.items()):
        flag = "  !! " + pd["err"][:40] if pd["err"] else ""
        print(f"{doc:<28} {pd['n']:>3} {pd['cov']/pd['n']:>8.0%} {pd['sane']/pd['n']:>6.0%} "
              f"{pd['leaves']:>7} {str(pd['median_span']):>9}{flag}")

    n = len(rows); cov = sum(r["covered"] for r in rows); sane = sum(r["covered_sane"] for r in rows)
    print("-" * 66)
    print(f"{'ALL':<28} {n:>3} {cov/n:>8.1%} {sane/n:>6.1%}")
    print(f"\ncovered      = gold page inside some leaf section")
    print(f"sane         = ...and that leaf is <= {args.max_span} pages (not a catch-all)")

    if args.json:
        Path(args.json).write_text(json.dumps({"questions": n, "covered": cov, "covered_sane": sane,
                                               "per_doc": per_doc, "rows": rows}, indent=1))
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
