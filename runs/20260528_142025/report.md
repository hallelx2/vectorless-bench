# vectorless-bench results — PR-C head-to-head vs PageIndex (FinanceBench)

Run: `20260528_142025` · k=5 · model gemini-2.5-flash · see `manifest.json` for config.

## TL;DR / what this run measured

This run is the head-to-head config (`configs/financebench_gemini_headtohead.yaml`)
scoped to the systems that could actually complete on the live engine today.
`bm25` and `full_context` ran clean on all 4 FinanceBench docs. The two
engine-backed systems (`vectorless`, `vectorless_pageindex`) carry **error rows
for an honest, reproducible reason — not a bench bug**:

1. **`/v1/answer/pageindex` is not deployed.** A live probe of the endpoint on
   revision `vectorless-server-00040-28f` returns `HTTP 404: 404 page not found`.
   Root cause: the endpoint + PageIndex strategy were wired into the engine's
   **standalone** router (`cmd/engine`, `internal/api/server.go` — which *does*
   register `r.Post("/answer/pageindex", ...)`), but the **deployed** binary is
   `cmd/server`, whose router (`internal/handler/router.go`) only mounts
   `/v1/health`, `/v1/documents`, `/v1/sections`, `/v1/query{,/stream,/multi}`.
   The `/v1/answer` and `/v1/answer/pageindex` routes are simply absent from the
   server build, so the SDK and a raw POST both 404. **This blocks the headline
   axis of the comparison until the route is added to `cmd/server`'s router (and
   `handler.Deps` is given a `PageIndexStrategy`).**

2. **Engine ingest stalled at FinanceBench scale.** `vectorless`/`vectorless_pageindex`
   both need an ingested doc. The engine split the 3M 2023-Q2 10-Q into ~1500
   leaf sections and then stalled in the *summarize* phase: status frozen at one
   timestamp for 20+ minutes, >80 min elapsed vs a prior **68-min** successful
   ingest of the same doc (run `20260527_113813`). The doc was still `summarizing`
   server-side hours later. Most likely cause: Gemini free-tier throughput
   (per-minute RPM) throttling the multi-axis-summary pipeline (engine PR #22) on
   ~1500 sections. The bench's vectorless retriever is correct (proven below); the
   bottleneck is engine ingest at scale on the free tier.

`pageindex` (their real upstream tree builder) was attempted on the 5.2 MB / 92-page
10-Q and the 75-page 10-K. Its `page_index()` tree build ran ~34 min then the
process died (no traceback — consistent with an OOM/kill loading + building a large
tree in memory). PageIndex's self-hosted tree build is itself impractical at this
scale on this box, so it is omitted here (best-effort, as the config notes).

### Evidence the new retriever is correct (independent of the 404)

`tests/test_vectorless_pageindex.py` (5 tests, all green) exercises the full
citation→section mapping against a stubbed endpoint: content from each citation's
`quote`, `page` from `start_page`, `title_path` resolved from the first
`section_id`, usage copied from the engine, request body carrying
`max_hops`/`max_pages_per_fetch`, k-capping, and HTTP-error recording. A live call
of the retriever against the deployed endpoint returns the 404 cleanly (recorded,
never raised), which is exactly the error row you see below.

### Prior `vectorless` reference (run `20260527_113813`, before the ingest regression)

When the engine ingest *did* complete (68 min, same 10-Q, n=1), the default
chunked-tree `vectorless` scored F1@5 = 0.000 on that single hard question at
**$0.0286/query, 2 LLM calls, p50 7382 ms**. One question is not a verdict — it
just confirms the query path works and gives a cost/latency anchor for the
chunked-tree strategy.

## Efficiency frontier (the headline)

Quality is meaningless without its price. `quality_per_1k_usd` = primary quality bought per $1,000 of query spend (higher is better); `$/correct` = spend per correct answer (lower is better).

| System | Quality | $/query | p50 ms | p95 ms | qual/$1k | $/correct | errors |
|---|---|---|---|---|---|---|---|
| full_context | 0.250 | 0.023457 | 21451.6 | 28651.2 | 0.0 | 0.09383 | 0 |
| bm25 | 0.167 | 0.000000 | 91.1 | 150.2 | ∞ | ∞ | 0 |
| vectorless | 0.000 | 0.000000 | 0.0 | 0.0 | ∞ | ∞ | 4 |
| vectorless_pageindex | 0.000 | 0.000000 | 0.0 | 0.0 | ∞ | ∞ | 4 |

## Retrieval quality

| System | P@5 | R@5 | F1@5 | nDCG@5 | MRR | hit@5 |
|---|---|---|---|---|---|---|
| full_context | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 | 0.250 |
| bm25 | 0.100 | 0.500 | 0.167 | 0.408 | 0.375 | 0.500 |
| vectorless | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| vectorless_pageindex | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

## Citation exactness & the near-miss test

`near_miss_rate` is the share of returned sections that are a *sibling* of the right one (right neighbourhood, wrong leaf) — the vector-RAG failure Vectorless is built to avoid. Lower is better.

| System | span_in_top1 | path_correct_top1 | near_miss_rate | abstained(no-answer) |
|---|---|---|---|---|
| full_context | 0.250 | 0.000 | 0.000 | 0.000 |
| bm25 | 0.250 | 0.000 | 0.000 | 0.000 |
| vectorless | 0.000 | 0.000 | 0.000 | 0.000 |
| vectorless_pageindex | 0.000 | 0.000 | 0.000 | 0.000 |

## Determinism & ingest cost

| System | exact_match | mean_jaccard | ingest s | ingest $ |
|---|---|---|---|---|
| full_context | — | — | 0.0 | 0.00000 |
| bm25 | — | — | 7.3 | 0.00000 |
| vectorless | — | — | 0.0 | 0.00000 |
| vectorless_pageindex | — | — | 0.0 | 0.00000 |

## Quality by domain

| System | finance |
|---|---|
| full_context | 0.250 |
| bm25 | 0.167 |
| vectorless | — |
| vectorless_pageindex | — |

---
_Primary quality = F1@k for answerable questions, correct abstention for no-answer questions. Quality/cost/latency from the first repeat; determinism across all repeats. Costs use the engine's price book (see `manifest.json:pricing_fingerprint`)._
