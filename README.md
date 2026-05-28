# vectorless-bench

An advanced benchmarking suite for [Vectorless](https://vectorless.store) — the
reasoning-based ("vectorless") RAG engine where **the LLM is the retriever**, not
an embedding model.

It exists to turn the engine's claims (deterministic, citation-exact, accurate on
specialized domains, no vector DB) into **numbers you can defend**, measured head
to head against the systems Vectorless is positioned against.

```bash
# runs today, no API keys, no services — proves the harness end to end
pip install -e .
vlbench run --config configs/smoke.yaml
```

---

## Why benchmarking *this* engine is different

Standard RAG benchmarks assume retrieval is free and instant, so they only score
accuracy. Vectorless retrieves by **calling an LLM over a document map**, so every
query has a real **token cost** and **latency**. That single fact reframes the
whole exercise:

> The headline metric is not precision@k. It is **quality per dollar** and
> **quality per second** — the efficiency frontier.

A system that wins on F1 while costing 50× is not a win. `vlbench` puts quality,
cost, and latency in the **first table of every report** so that trade-off is
impossible to hide.

Three things in the engine's own code shape the methodology (and silently corrupt
naive benchmarks):

1. **Section IDs are random `sec_<uuid>`s**, regenerated on every ingest. Gold
   labels therefore can't be IDs — they are *stable anchors* (heading path /
   answer span / page) resolved to whatever each system returns. See
   [`anchors.py`](src/vectorless_bench/anchors.py).
2. **Caching zeroes cost.** Both the llmgate cache and the retrieval cache return
   `cost_usd=0` on a hit. Fair cost/latency requires a **cold cache** — run the
   server with `retrieval.cache.enabled=false`. The run manifest records the
   declared cache mode.
3. **Determinism is a claim, not a guarantee.** Temp=0 reduces but doesn't
   eliminate provider nondeterminism, so `vlbench` *measures* it (rerun the same
   query N times, report set-stability) instead of assuming it.

---

## What it measures — seven axes

| Axis | Metrics | What it tells you |
|---|---|---|
| **Retrieval quality** | precision/recall/F1@k, MRR, nDCG, hit@k | Did it fetch the right section? |
| **Citation exactness** | span-in-top1, **path-correct@1** | Can it point at the exact passage/heading? |
| **Near-miss** | sibling near-miss rate | Did it grab the *wrong fiscal year / wrong drug* (the vector failure mode)? |
| **Cost** | $/query, tokens/query, calls/query, **$/correct**, **quality per $1k** | The price of being right |
| **Latency** | p50 / p95 / p99, ingest time | Cold-cache, end to end |
| **Determinism** | exact-match + mean Jaccard across reruns | Is the published determinism claim real? |
| **Robustness** | abstention on no-answer, by-domain, by-answer-type | Does it over-retrieve when the answer isn't there? |

`path-correct@1` and `near-miss` are **structural** metrics: chunk systems (vector
RAG, BM25) score 0 on path-correctness by construction — that gap *is* the
differentiator the whitepaper argues for, made measurable.

---

## Systems compared

| System | What it is | Deps |
|---|---|---|
| `vectorless` | the engine under test (default chunked-tree strategy), via the Python SDK | `vectorless-sdk` + a running server |
| `vectorless_pageindex` | the engine's PageIndex-style page-based agentic strategy, via `POST /v1/answer/pageindex` (page navigation + answer + page-grounded citations in one round-trip) | a server exposing `/v1/answer/pageindex` |
| `vector_rag` | pgvector + OpenAI embeddings + cosine top-k (the ROADMAP baseline) | `[vector]` + Postgres/pgvector |
| `pageindex` | the real upstream [PageIndex](https://github.com/VectifyAI/PageIndex) — *their* tree builder (`page_index`/`md_to_tree`) + their reasoning retrieval, priced on our table | clone of PageIndex + `[llm]` |
| `full_context` | stuff the whole doc in the prompt — the quality **ceiling** + cost worst case | `[llm]` |
| `bm25` | lexical floor; free, no API, strong on exact-term lookups | `[bm25]` |
| `mock` | deterministic fake for harness CI — no services | none |

All LLM-using systems are priced from the **same table** the engine uses
([`pricing.py`](src/vectorless_bench/pricing.py), mirrored from `llmgate/pricing`),
so cost is apples-to-apples. Each baseline is a *fair* representative (standard
chunking, optional reranker hook), not a strawman.

---

## Datasets

- **`fixtures`** — a tiny in-repo curated set (finance + medicine) with stable
  anchors, a no-answer item, and a sibling near-miss trap. Seeds the
  "curated golden set" and powers the smoke test. Runs in seconds.
- **`financebench`** — the public 150-question [FinanceBench](https://github.com/patronus-ai/financebench)
  set over real 10-Ks. QA loads from HuggingFace; fetch the source PDFs with
  `python scripts/download_financebench.py`. Questions whose document text is
  missing are skipped (not failed), so a partial corpus still produces a valid run.

Add your own by subclassing `Dataset` (see
[`datasets/base.py`](src/vectorless_bench/datasets/base.py)) and emitting
`Question`s with `GoldAnchor`s. The only rule: **gold is stable anchors, never
engine IDs.**

---

## Running the real benchmark (FinanceBench)

```bash
pip install -e ".[all]"
cp .env.example .env            # fill in keys + DSN

# 1. fetch the source filings
python scripts/download_financebench.py

# 2. start a Vectorless server with caches OFF (fair cold-cache), then:
vlbench run --config configs/financebench.yaml

# 3. re-render the report from raw records any time
vlbench report runs/<stamp> --k 5
```

Each run writes a self-contained directory:

```
runs/<stamp>/
  records.jsonl    one scored (system, question, repeat) row each
  results.json     aggregated per-system summary
  report.md        the human report (frontier + per-axis tables)
  report.html      self-contained HTML report (frontier scatter + tables) — open this
  pareto.csv       quality vs cost vs latency, for plotting
  setup.json       per-system ingest time + cost
  manifest.json    repro: git sha, models, price fingerprint, cache mode, seed
```

---

## Running on a VM (bundle → run → view)

Real runs are long and need keys + Postgres, so the supported path is a Docker
bundle you run on a cloud VM, with results shipped to GCS for viewing. One command:

```bash
PROJECT=<gcp> BUCKET=gs://<bucket> ./deploy/gcp/run_on_gce.sh   # provision, run, upload, delete VM
BUCKET=gs://<bucket> RUN_ID=<name> ./deploy/gcp/fetch_results.sh # download + open report.html
```

Or locally with Docker (bundles pgvector + the real PageIndex repo):

```bash
docker compose build
docker compose run --rm --entrypoint python bench scripts/download_financebench.py
docker compose run --rm bench run --config configs/financebench.yaml --out /results --limit 10
```

Full details and prerequisites: [`deploy/README.md`](deploy/README.md).

---

## Validity controls (what makes the numbers credible)

- **Cost is never reported alone** — always beside quality, plus `$/correct`.
- **Cold-cache** enforced/declared and recorded in the manifest.
- **Gold defined independently** of any system's output (human/strong-model, then
  verified), and matched leniently on surface form, strictly on substance.
- **Retrieval quality scored separately from answer quality** — Vectorless is a
  retriever, so retrieval is the primary axis; the optional LLM-judge answer axis
  uses one judge model for all systems, blind to which system produced the answer.
- **Determinism uses real reruns**, not an assumption.
- **Bootstrap CIs** on primary quality so A-vs-B gaps come with uncertainty.
- **Reproducibility manifest** on every run, including a price-book fingerprint.

---

## Architecture

```
src/vectorless_bench/
  schema.py        dataclasses (Question, GoldAnchor, RetrievalResult, Usage, ...)
  pricing.py       engine-mirrored price book + token counting
  anchors.py       gold matching: the single definition of "right thing retrieved"
  metrics/         retrieval.py · citation.py · aggregate.py (+ score_question)
  retrievers/      base + registry + vectorless, vector_rag, pageindex,
                   full_context, bm25, mock
  datasets/        base + fixtures + financebench
  judge.py         optional LLM-as-judge answer axis
  runner.py        orchestrator -> records.jsonl + manifest
  report.py        records -> results.json + report.md + report.html + pareto.csv
  cli.py           `vlbench run | report | systems`
Dockerfile, docker-compose.yml   the runnable bundle (+ pgvector + real PageIndex)
deploy/gcp/      run_on_gce.sh · fetch_results.sh · startup-script.sh
```

Core (schema, pricing, anchors, metrics, runner, report, mock, fixtures) has
**zero third-party dependencies** and is fully unit-tested; everything that needs
a network or heavy library is an optional extra, imported lazily by the system
that uses it. `pytest` runs the whole harness, including a real end-to-end run,
with no keys.

### Extending

- **New retriever:** implement `setup(corpus)` + `retrieve(question, k, cold) ->
  RetrievalResult`, then `register("name", factory)` in `retrievers/registry.py`.
- **New dataset:** subclass `Dataset`, return a corpus + gold-anchored questions.
- **New metric:** add to `metrics/` and surface it in `report.py`.

---

## Roadmap

This closes the deferred item in the engine's own `ROADMAP.md`
("Benchmarks vs. traditional RAG … publish in benchmarks/README.md"):

- [x] **Phase 0** — harness, metrics, mock + fixtures, cold-cache control, reports (this repo)
- [x] **Phase 1** — vector RAG, BM25, full-context, real-PageIndex baselines; cost/latency frontier
- [x] **Phase 1.5** — Docker bundle + GCE run-and-view-results flow; HTML report
- [ ] **Phase 2** — full FinanceBench run on a VM with a live engine
- [ ] **Phase 3** — curated finance/law/medicine golden set; CUAD + multi-hop; CI regression gate; published leaderboard

## License

MIT.
