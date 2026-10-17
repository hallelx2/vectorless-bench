"""The scoreboard folds run directories into one board (HAL-3088)."""

import json
from pathlib import Path

from vectorless_bench.scoreboard import build, collect


def _run(root: Path, name: str, dataset: str, created: str, rows, k: int = 5) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"dataset": dataset, "k": k, "created_at": created,
                                                 "bench_git_sha": "abc123", "model": "gemini-3.5-flash-lite",
                                                 "extra": {"engine_versions": {"vectorless": {"version": "dev-1"}}}}))
    (d / "records.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    return d


def _rec(system, qid, hit, cost, ms, domain="comparison", native=None, phase="latency"):
    m = {"hit@5": hit, "recall@5": hit, "f1@5": hit, "mrr": hit, "ndcg@5": hit, "gold_reachable": 1.0}
    if native is not None:
        m["native_hits@5"] = native
    return {"qid": qid, "system": system, "repeat": 0, "domain": domain, "answer_type": "multihop",
            "difficulty": "medium", "latency_ms": ms, "wall_ms": ms, "phase": phase, "flagged": False,
            "attempts": 1, "usage": {"cost_usd": cost, "total_tokens": 10, "llm_calls": 1}, "metrics": m}


def test_scoreboard_keeps_the_latest_run_and_writes_every_output(tmp_path):
    old = _run(tmp_path, "r1", "multihop_rag", "2026-10-16T10:00:00",
               [_rec("vectorless", "q1", 0.0, 0.01, 900)])
    new = _run(tmp_path, "r2", "multihop_rag", "2026-10-16T12:00:00",
               [_rec("vectorless", "q1", 1.0, 0.02, 1500, native=0.5), _rec("bm25", "q1", 0.0, 0.0, 40, native=0.0),
                _rec("vectorless", "q2", 1.0, 0.02, 1700, domain="temporal", native=1.0)])
    board = collect([old, new])
    assert board["multihop_rag"]["vectorless"]["run"] == str(new)
    info = build([old, new], tmp_path / "out")
    assert info == {"datasets": 1, "rows": 2, "out": str(tmp_path / "out")}
    md = (tmp_path / "out" / "scoreboard.md").read_text()
    assert "## multihop_rag" in md and "Vectorless" in md and "hits@5" in md
    assert "By query type" in md and "temporal" in md
    assert "Judge" not in md and "jev" not in md.lower()
    rows = json.loads((tmp_path / "out" / "scoreboard.json").read_text())["rows"]
    v = next(r for r in rows if r["system"] == "vectorless")
    assert v["quality"] == 1.0 and v["native"] == 0.75 and v["usd_per_question"] == 0.02
    assert (tmp_path / "out" / "scoreboard.csv").read_text().startswith("dataset,system")
    assert "<circle" in (tmp_path / "out" / "quality_vs_cost.svg").read_text()


def test_an_unmeasured_cell_is_empty_not_zero(tmp_path):
    d = _run(tmp_path, "r", "financebench", "2026-10-16T10:00:00", [_rec("bm25", "q", 1.0, 0.0, 30)])
    build([d], tmp_path / "o")
    row = json.loads((tmp_path / "o" / "scoreboard.json").read_text())["rows"][0]
    assert row["native"] is None and row["answer_accuracy"] is None and row["questions_per_minute"] is None



def test_speed_comes_only_from_timed_runs_and_quality_from_the_latest(tmp_path):
    timed = _run(tmp_path, "rig", "qasper", "2026-10-16T10:00:00", [_rec("vectorless", "q1", 0.5, 0.01, 800)])
    acc = _run(tmp_path, "worker", "qasper", "2026-10-16T12:00:00",
               [_rec("vectorless", "q1", 1.0, 0.01, 9999, phase="accuracy"),
                _rec("vectorless", "q2", 1.0, 0.01, 9999, phase="accuracy")])
    build([timed, acc], tmp_path / "o")
    row = json.loads((tmp_path / "o" / "scoreboard.json").read_text())["rows"][0]
    assert row["quality"] == 1.0 and row["n"] == 2, "quality from the latest (accuracy) run"
    assert row["p50_ms"] == 800 and row["timing_run"] == str(timed), "speed only from the rig-timed run"

    only_acc = _run(tmp_path, "w2", "cuad", "2026-10-16T12:00:00", [_rec("bm25", "q", 1.0, 0.0, 9999, phase="accuracy")])
    build([only_acc], tmp_path / "o2")
    row2 = json.loads((tmp_path / "o2" / "scoreboard.json").read_text())["rows"][0]
    assert row2["p50_ms"] is None, "an accuracy pass alone reports no speed"


def test_an_accuracy_run_turns_the_rig_off():
    from vectorless_bench.config import RunConfig, validate

    validate(RunConfig(dataset="fixtures", phase="accuracy"))


def _setup_run(root, name, created, model, prompt):
    d = _run(root, name, "financebench", created, [_rec("vectorless", "q1", 1.0, 0.01, 900)])
    man = json.loads((d / "manifest.json").read_text())
    man["model"] = model
    man["extra"]["answer_setup"] = {"model": model, "gen_model": model, "answer_prompt": prompt, "judge_model": "jev-latest"}
    (d / "manifest.json").write_text(json.dumps(man))
    return d


def test_runs_answered_two_ways_are_never_folded_together(tmp_path):
    import pytest
    from vectorless_bench.scoreboard import MixedAnswerSetups

    a = _setup_run(tmp_path, "a", "2026-10-16T10:00:00", "gemini-3.5-flash-lite", "excerpts-calc-v3")
    b = _setup_run(tmp_path, "b", "2026-10-17T10:00:00", "gemini-2.5-flash-lite", "excerpts-calc-v3")
    with pytest.raises(MixedAnswerSetups, match="2.5-flash-lite.*3.5-flash-lite|3.5-flash-lite.*2.5-flash-lite"):
        collect([a, b])
    # a different prompt version is a different setup too
    c = _setup_run(tmp_path, "c", "2026-10-17T11:00:00", "gemini-2.5-flash-lite", "v1")
    with pytest.raises(MixedAnswerSetups):
        collect([b, c])
    # one setup folds; asked for explicitly, mixed runs fold with each row naming its setup
    assert collect([b])["financebench"]["vectorless"]["setup"].startswith("gemini-2.5-flash-lite")
    build([a, b], tmp_path / "out", allow_mixed=True)
    rows = json.loads((tmp_path / "out" / "scoreboard.json").read_text())["rows"]
    assert rows[0]["setup"].startswith("gemini-2.5-flash-lite")


def test_a_run_from_before_the_setup_was_recorded_reads_as_prompt_v1(tmp_path):
    from vectorless_bench.scoreboard import setup_of

    man = {"model": "gemini-3.5-flash-lite", "extra": {"config": {"model": "gemini-3.5-flash-lite", "judge": True,
                                                                  "gen_model": "gemini-3.5-flash-lite"}}}
    assert setup_of(man) == "gemini-3.5-flash-lite | answers gemini-3.5-flash-lite | prompt v1 | graded —"
    man["judge_model"] = "jev-latest"
    assert setup_of(man).endswith("| graded jev-latest")


def test_answer_model_sets_the_shared_model_and_the_answer_step():
    from vectorless_bench import cli

    args = cli.build_parser().parse_args(["run", "--config", "configs/pilot/financebench.yaml",
                                          "--answer-model", "gemini-2.5-flash-lite"])
    cfg = cli.config_from_args(args)
    assert cfg.model == cfg.gen_model == "gemini-2.5-flash-lite"
    assert cfg.params_for("pageindex")["retrieve_model"] == "gemini-2.5-flash-lite"
    # without the flag the pilot config's model stands
    plain = cli.config_from_args(cli.build_parser().parse_args(["run", "--config", "configs/pilot/financebench.yaml"]))
    assert plain.model == plain.gen_model == "gemini-3.5-flash-lite"


def test_two_graders_are_two_setups(tmp_path):
    import pytest
    from vectorless_bench.scoreboard import MixedAnswerSetups

    a = _setup_run(tmp_path, "a", "2026-10-17T10:00:00", "gemini-2.5-flash-lite", "excerpts-calc-v3")
    b = _setup_run(tmp_path, "b", "2026-10-17T11:00:00", "gemini-2.5-flash-lite", "excerpts-calc-v3")
    man = json.loads((b / "manifest.json").read_text())
    man["extra"]["answer_setup"]["judge_model"] = "jev-1.12.0"
    (b / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(MixedAnswerSetups):
        collect([a, b])


def test_mixed_board_never_pairs_one_setups_speed_with_anothers_quality(tmp_path):
    timed = _setup_run(tmp_path, "t", "2026-10-16T10:00:00", "gemini-3.5-flash-lite", "excerpts-calc-v3")
    acc = tmp_path / "acc"
    acc.mkdir()
    man = json.loads((timed / "manifest.json").read_text())
    man.update(created_at="2026-10-17T10:00:00", model="gemini-2.5-flash-lite")
    man["extra"]["answer_setup"] = {"model": "gemini-2.5-flash-lite", "gen_model": "gemini-2.5-flash-lite",
                                    "answer_prompt": "excerpts-calc-v3", "judge_model": "jev-latest"}
    (acc / "manifest.json").write_text(json.dumps(man))
    (acc / "records.jsonl").write_text(json.dumps(_rec("vectorless", "q1", 1.0, 0.01, 900, phase="accuracy")))
    build([timed], tmp_path / "alone")
    alone = json.loads((tmp_path / "alone" / "scoreboard.json").read_text())["rows"][0]
    assert alone["p50_ms"] is not None, "the timed run must carry latency for this test to mean anything"
    build([timed, acc], tmp_path / "out", allow_mixed=True)
    row = json.loads((tmp_path / "out" / "scoreboard.json").read_text())["rows"][0]
    assert row["setup"].startswith("gemini-2.5-flash-lite")
    assert row["p50_ms"] is None, "the 3.5 run's latency was shown beside the 2.5 run's quality"


def test_answer_model_overrides_a_systems_own_model_setting(tmp_path):
    from vectorless_bench import cli

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text("dataset: financebench\nsystems: [pageindex]\nmodel: gemini-3.5-flash-lite\n"
                        "system_params:\n  pageindex:\n    retrieve_model: gemini-3.5-flash-lite\n")
    cfg = cli.config_from_args(cli.build_parser().parse_args(["run", "--config", str(cfg_path),
                                                              "--answer-model", "gemini-2.5-flash-lite"]))
    assert cfg.params_for("pageindex")["retrieve_model"] == "gemini-2.5-flash-lite"
