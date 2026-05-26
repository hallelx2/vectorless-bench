"""End-to-end smoke test: run the whole pipeline with the mock retriever over the
fixture corpus and assert the reports come out sane. No network, no API keys —
this is the "runs today" guarantee and the CI gate for the harness itself.
"""

import json

from vectorless_bench import RunConfig, run, write_reports


def test_end_to_end_mock_run(tmp_path):
    cfg = RunConfig(
        dataset="fixtures",
        systems=["mock"],
        k=5,
        repeats=3,            # exercises the determinism axis
        out_dir=str(tmp_path),
        system_params={"mock": {"accuracy": 1.0}},
    )
    run_dir = run(cfg, progress=False)
    results = write_reports(run_dir, cfg.k)

    assert (run_dir / "records.jsonl").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "results.json").exists()
    assert (run_dir / "pareto.csv").exists()
    assert (run_dir / "manifest.json").exists()
    assert "<svg" in (run_dir / "report.html").read_text(encoding="utf-8")

    mock = results["mock"]
    # a perfect oracle should score near the top on quality...
    assert mock["quality"]["primary"] >= 0.8
    # ...be perfectly deterministic (mock is seeded by qid only)...
    assert mock["determinism"]["exact_match"] == 1.0
    # ...and report a real, nonzero cost (priced from the shared table).
    assert mock["cost"]["mean_cost_usd"] > 0
    assert mock["errors"] == 0


def test_imperfect_mock_shows_near_miss(tmp_path):
    cfg = RunConfig(
        dataset="fixtures", systems=["mock"], k=5, repeats=1,
        out_dir=str(tmp_path), system_params={"mock": {"accuracy": 0.0}},
    )
    run_dir = run(cfg, progress=False)
    results = write_reports(run_dir, cfg.k)
    # accuracy=0 forces sibling near-misses on answerable questions
    assert results["mock"]["citation"]["near_miss_rate"] > 0


def test_manifest_records_pricing(tmp_path):
    cfg = RunConfig(dataset="fixtures", systems=["mock"], out_dir=str(tmp_path))
    run_dir = run(cfg, progress=False)
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["pricing_fingerprint"]
    assert manifest["dataset"] == "fixtures"
    assert "mock" in manifest["systems"]
