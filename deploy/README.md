# Running vectorless-bench on a VM

Real runs are long (LLM calls per query × systems × questions × repeats) and need
keys + Postgres, so the natural place is a cloud VM you fire and forget. The flow:
**bundle (Docker) → run on a GCE VM → results land in GCS → fetch + open the HTML
report.**

## 1. Bundle (Docker)

The `Dockerfile` builds one image with the suite, every baseline's deps, and the
real PageIndex repo (`/opt/PageIndex`). `docker-compose.yml` adds a pgvector
Postgres for the `vector_rag` baseline.

Run it locally first to sanity-check (swap in a small `--limit`):

```bash
cp .env.example .env            # fill in keys
docker compose build
docker compose run --rm --entrypoint python bench scripts/download_financebench.py
docker compose run --rm bench run --config configs/financebench.yaml --out /results --limit 10
# results in ./results/<stamp>/report.html
```

The `vectorless` system needs a reachable engine: set `VECTORLESS_BASE_URL` in
`.env` to your deployed server, or add an engine service to `docker-compose.yml`.

## 2. Run on a GCE VM (one command)

```bash
gcloud auth login && gcloud config set project <PROJECT>
# the VM's service account needs roles/storage.objectAdmin on your bucket

PROJECT=<PROJECT> BUCKET=gs://<your-bucket> ./deploy/gcp/run_on_gce.sh
```

`run_on_gce.sh` provisions an Ubuntu VM (Docker + Cloud CLI via `startup-script.sh`),
ships this repo + your `.env`, builds the image, downloads FinanceBench, runs the
benchmark, uploads `results/*` to `gs://<bucket>/<run-name>/`, and deletes the VM.

Knobs (env vars): `ZONE MACHINE DISK CONFIG ENV_FILE DOWNLOAD_DOCS DELETE_AFTER`.
Set `DELETE_AFTER=0` to keep the VM for debugging.

> Cost/safety: this creates a billable VM and (by default) deletes it when done.
> Use a small `--limit` (edit the config) for a first run.

## 3. View results

```bash
BUCKET=gs://<your-bucket> RUN_ID=<run-name printed above> ./deploy/gcp/fetch_results.sh
```

Downloads the run into `runs/<run-name>/` and opens `report.html` — the efficiency
frontier (quality vs cost/latency), per-axis tables, determinism, and per-domain
breakdown. `results.json` / `pareto.csv` / `records.jsonl` are there too for
deeper analysis, and `manifest.json` records exactly how the numbers were made.

## Files

- `../Dockerfile`, `../docker-compose.yml` — the bundle
- `gcp/startup-script.sh` — VM bootstrap (Docker + Cloud CLI)
- `gcp/run_on_gce.sh` — provision + run + upload + cleanup
- `gcp/fetch_results.sh` — download a run + open the report
