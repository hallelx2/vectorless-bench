"""Fetch + extract the FinanceBench source documents.

The FinanceBench QA set ships on HuggingFace but the underlying filings (large
PDFs) live in the Patronus AI GitHub repo. This script reads the doc_names the
QA set references, downloads each PDF, extracts plain text with pypdf, and writes
`<docs_dir>/<doc_name>.txt`. The FinanceBench dataset loader then reads those.

Idempotent: existing .txt files are skipped. Missing PDFs are reported and their
questions are simply skipped at run time (the loader degrades gracefully), so a
partial download still yields a valid, smaller benchmark.

Usage:
    pip install "vectorless-bench[data]"
    python scripts/download_financebench.py [--docs-dir data/financebench/docs] [--limit N]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

RAW_BASE = "https://raw.githubusercontent.com/patronus-ai/financebench/main/pdfs"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-dir", default="data/financebench/docs")
    ap.add_argument("--hf-name", default="PatronusAI/financebench")
    ap.add_argument("--split", default="train")
    ap.add_argument("--limit", type=int, default=None, help="cap distinct docs")
    args = ap.parse_args()

    try:
        import requests
        from datasets import load_dataset
        from pypdf import PdfReader
    except ImportError:
        print("install extras first: pip install 'vectorless-bench[data]'", file=sys.stderr)
        return 1

    docs_dir = Path(args.docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.hf_name, split=args.split)
    doc_names = sorted({str(r.get("doc_name", "")).strip() for r in ds if r.get("doc_name")})
    if args.limit:
        doc_names = doc_names[: args.limit]
    print(f"{len(doc_names)} distinct documents referenced by FinanceBench")

    import io

    ok = skipped = failed = 0
    for name in doc_names:
        out = docs_dir / f"{name}.txt"
        if out.exists():
            skipped += 1
            continue
        url = f"{RAW_BASE}/{name}.pdf"
        try:
            resp = requests.get(url, timeout=120)
            if resp.status_code != 200:
                print(f"  MISSING ({resp.status_code}): {name}")
                failed += 1
                continue
            # keep the original PDF (PageIndex parses it with its own pipeline)
            (docs_dir / f"{name}.pdf").write_bytes(resp.content)
            reader = PdfReader(io.BytesIO(resp.content))
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            out.write_text(text, encoding="utf-8")
            ok += 1
            print(f"  ok: {name} ({len(reader.pages)} pages)")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {name}: {e}")
            failed += 1

    print(f"\ndone: {ok} downloaded, {skipped} already present, {failed} missing/failed")
    print(f"docs in: {docs_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
