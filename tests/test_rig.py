"""The speed rig: host evidence, flagging, reruns, throughput and the spend cap."""

import json
import subprocess
import sys
import time

import pytest

from vectorless_bench import hostprobe
from vectorless_bench.config import RunConfig
from vectorless_bench.rig import Rig, RigConfig
from vectorless_bench.runner import _Spend, SpendCapReached, run


STAT_A = "cpu  100 0 50 1000 10 0 0 0 0 0\ncpu0 100 0 50 1000 10 0 0 0 0 0\n"
STAT_B = "cpu  200 0 60 1800 10 0 0 40 0 0\ncpu0 200 0 60 1800 10 0 0 40 0 0\n"


def _snap(stat, procs, mono):
    return {"host": "h", "mono": mono, "cpu": hostprobe.cpu_ticks(stat), "procs": procs,
            "load1": 0.1, "mem_kb": {"MemAvailable": 1000}}


def test_evidence_steal_and_foreign_process():
    a = _snap(STAT_A, {"1": {"comm": "python", "ticks": 0}, "2": {"comm": "go", "ticks": 0}}, 0.0)
    b = _snap(STAT_B, {"1": {"comm": "python", "ticks": 50}, "2": {"comm": "go", "ticks": 90}}, 1.0)
    ev = hostprobe.evidence(a, b, allow=["python"])
    # steal 40 of total (100+10+800+0+40)=950 ticks
    assert abs(ev["steal_pct"] - 100 * 40 / 950) < 0.01
    names = [p["comm"] for p in ev["foreign"]]
    assert names == ["go"], "the allowlisted harness is not foreign; the stray go build is"


def test_rig_flags_steal_and_foreign(monkeypatch):
    seq = iter([
        _snap(STAT_A, {"2": {"comm": "stress-ng", "ticks": 0}}, 0.0),
        _snap(STAT_B, {"2": {"comm": "stress-ng", "ticks": 100}}, 1.0),
    ])
    monkeypatch.setattr(hostprobe, "snapshot", lambda: next(seq))
    rig = Rig(RigConfig(max_steal_pct=2.0, foreign_core_pct=10.0))
    v = rig.end(rig.begin())
    assert v["flagged"]
    assert any("steal" in r for r in v["reasons"])
    assert any("stress-ng" in r for r in v["reasons"])


def test_unreadable_remote_probe_is_flagged():
    rig = Rig(RigConfig(probes={"engine": "http://127.0.0.1:9"}))
    v = rig.end(rig.begin())
    assert v["flagged"] and "engine: probe unreadable" in v["reasons"]


def test_spend_cap():
    s = _Spend(1.0)
    s.add(0.6)
    with pytest.raises(SpendCapReached):
        s.add(0.5)


def test_real_snapshot_reads_this_machine():
    snap = hostprobe.snapshot()
    assert snap["ncpu"] >= 1 and snap["cpu"]["idle"] > 0 and snap["procs"]


def test_probe_agent_serves_snapshots_and_rtt(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "vectorless_bench.cli", "probe", "--port", "9137", "--bind", "127.0.0.1"])
    try:
        import urllib.request
        for _ in range(50):
            try:
                body = json.loads(urllib.request.urlopen("http://127.0.0.1:9137/snapshot", timeout=1).read())
                break
            except Exception:
                time.sleep(0.1)
        assert body["cpu"]["idle"] > 0
        rig = Rig(RigConfig(probes={"engine": "http://127.0.0.1:9137"}))
        v = rig.end(rig.begin())
        assert v["hosts"]["engine"] is not None
    finally:
        proc.terminate()


def test_latency_run_records_host_evidence_and_wall_clock(tmp_path):
    cfg = RunConfig(dataset="fixtures", systems=["mock"], out_dir=str(tmp_path), rig={"rtt_targets": []})
    run_dir = run(cfg, progress=False)
    rows = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines()]
    assert rows and all(r["phase"] == "latency" for r in rows)
    assert all(r["wall_ms"] is not None and r["latency_ms"] == r["wall_ms"] for r in rows)
    assert all("bench" in r["host"]["hosts"] for r in rows)
    man = json.loads((run_dir / "manifest.json").read_text())
    assert "machines" in man["extra"] and man["extra"]["spend_usd"] >= 0


def test_throughput_phase_reports_rate(tmp_path):
    cfg = RunConfig(dataset="fixtures", systems=["mock"], out_dir=str(tmp_path), phase="throughput",
                    concurrency=4, rig={"enabled": False})
    run_dir = run(cfg, progress=False)
    setup = json.loads((run_dir / "setup.json").read_text())
    tp = setup["mock"]["throughput"]
    assert tp["concurrency"] == 4 and tp["questions_per_minute"] > 0
    rows = [json.loads(l) for l in (run_dir / "records.jsonl").read_text().splitlines()]
    assert all(r["phase"] == "throughput" for r in rows)


def test_spend_cap_stops_a_run(tmp_path):
    cfg = RunConfig(dataset="fixtures", systems=["mock", "bm25"], out_dir=str(tmp_path),
                    rig={"enabled": False}, max_spend_usd=0.000001)
    run_dir = run(cfg, progress=False)
    man = json.loads((run_dir / "manifest.json").read_text())
    assert man["extra"]["stopped"] and "cap" in man["extra"]["stopped"]
    systems = {json.loads(l)["system"] for l in (run_dir / "records.jsonl").read_text().splitlines()}
    assert systems == {"mock"}, "the run stops at the cap instead of starting the next system"
