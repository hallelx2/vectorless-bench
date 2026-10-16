"""Host evidence for a timed sample: what else was this machine doing?

A latency number is publishable only if nothing else was competing for the
machine while it was measured (HAL-3032; HAL-1708 is the incident where a build
running alongside inflated one side of an A/B). This module reads, from /proc
alone, the things that would show competition:

  - CPU steal: time the hypervisor gave our vCPUs to someone else. On a
    dedicated-core machine it should be ~0; on shared-core types it is not.
  - iowait and overall busy share across all CPUs.
  - load average.
  - "foreign" processes: anything outside an allowlist that used a meaningful
    share of a core during the interval (a stray build, an indexer, a cron job).

It is stdlib-only so the same file runs on the engine VM as a tiny HTTP agent
(`python3 hostprobe.py serve --port 9101`) and inside the harness on the bench
VM, and it has no opinion about thresholds: rig.py decides what is flagged.

Snapshots are cumulative counters; evidence is the difference between two of
them, so a sample's evidence covers exactly the interval it was timed over.
"""

from __future__ import annotations

import json
import os
import socket
import ssl
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Dict, List, Optional
from urllib.parse import parse_qs, urlparse

CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100
PROC = os.environ.get("VLBENCH_PROC", "/proc")

# /proc/stat cpu line field order (man 5 proc).
_CPU_FIELDS = ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal", "guest", "guest_nice")


def _read(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def cpu_ticks(stat_text: Optional[str] = None) -> Dict[str, int]:
    """Aggregate CPU tick counters from the first line of /proc/stat."""
    text = stat_text if stat_text is not None else _read(f"{PROC}/stat")
    for line in text.splitlines():
        if line.startswith("cpu "):
            vals = [int(x) for x in line.split()[1:]]
            return {k: (vals[i] if i < len(vals) else 0) for i, k in enumerate(_CPU_FIELDS)}
    raise ValueError("no aggregate cpu line in /proc/stat")


def process_ticks() -> Dict[str, Dict[str, object]]:
    """Per-process CPU ticks (utime+stime) and command name, keyed by pid."""
    out: Dict[str, Dict[str, object]] = {}
    for pid in os.listdir(PROC):
        if not pid.isdigit():
            continue
        try:
            raw = _read(f"{PROC}/{pid}/stat")
        except OSError:
            continue  # the process exited between listdir and read
        # comm is in parentheses and may contain spaces; split after the last ')'.
        lp, rp = raw.find("("), raw.rfind(")")
        if lp < 0 or rp < 0:
            continue
        comm = raw[lp + 1 : rp]
        rest = raw[rp + 2 :].split()
        try:
            utime, stime = int(rest[11]), int(rest[12])
        except (IndexError, ValueError):
            continue
        out[pid] = {"comm": comm, "ticks": utime + stime}
    return out


_MACHINE: Optional[Dict[str, str]] = None


def machine() -> Dict[str, str]:
    """This VM's machine type, zone and CPU platform from the GCE metadata
    server, read once; empty off GCP."""
    global _MACHINE
    if _MACHINE is None:
        import urllib.request

        out: Dict[str, str] = {}
        base = "http://metadata.google.internal/computeMetadata/v1/instance/"
        for key, path in (("machine_type", "machine-type"), ("zone", "zone"), ("cpu_platform", "cpu-platform")):
            try:
                req = urllib.request.Request(base + path, headers={"Metadata-Flavor": "Google"})
                with urllib.request.urlopen(req, timeout=1.0) as r:  # noqa: S310 - metadata server
                    out[key] = r.read().decode().rsplit("/", 1)[-1]
            except Exception:
                break
        _MACHINE = out
    return _MACHINE


def snapshot() -> Dict[str, object]:
    """Cumulative counters at one instant."""
    load1, load5, _ = os.getloadavg()
    mem = {}
    try:
        for line in _read(f"{PROC}/meminfo").splitlines():
            k, _, v = line.partition(":")
            if k in ("MemTotal", "MemAvailable"):
                mem[k] = int(v.split()[0])  # kB
    except OSError:
        pass
    return {
        "t": time.time(),
        "mono": time.monotonic(),
        "host": socket.gethostname(),
        "machine": machine(),
        "ncpu": os.cpu_count() or 1,
        "cpu": cpu_ticks(),
        "procs": process_ticks(),
        "load1": load1,
        "load5": load5,
        "mem_kb": mem,
    }


def evidence(a: Dict[str, object], b: Dict[str, object], allow: List[str],
             foreign_min_pct: float = 5.0) -> Dict[str, object]:
    """What happened between snapshots a and b.

    Percentages of CPU are of the whole machine (all vCPUs) for steal, iowait
    and busy, and of ONE core for a foreign process, which is how a reader
    thinks about "a process pinned a core". Processes whose command name
    starts with an allowlisted prefix are ours and never count as foreign.
    """
    ca, cb = a["cpu"], b["cpu"]  # type: ignore[index]
    d = {k: cb[k] - ca[k] for k in _CPU_FIELDS}  # type: ignore[index]
    total = sum(d[k] for k in ("user", "nice", "system", "idle", "iowait", "irq", "softirq", "steal")) or 1
    secs = max(float(b["mono"]) - float(a["mono"]), 1e-6)  # type: ignore[arg-type]
    foreign = []
    pa, pb = a["procs"], b["procs"]  # type: ignore[index]
    for pid, info in pb.items():  # type: ignore[union-attr]
        comm = str(info["comm"])
        if any(comm.startswith(p) for p in allow):
            continue
        before = pa.get(pid, {}).get("ticks", 0) if pid in pa else 0  # type: ignore[union-attr]
        used = int(info["ticks"]) - int(before)
        pct = 100.0 * used / CLK_TCK / secs
        if pct >= foreign_min_pct:
            foreign.append({"pid": int(pid), "comm": comm, "core_pct": round(pct, 1)})
    foreign.sort(key=lambda x: -x["core_pct"])
    return {
        "host": b.get("host"),
        "seconds": round(secs, 4),
        "steal_pct": round(100.0 * d["steal"] / total, 3),
        "iowait_pct": round(100.0 * d["iowait"] / total, 3),
        "busy_pct": round(100.0 * (total - d["idle"] - d["iowait"]) / total, 3),
        "load1": b.get("load1"),
        "mem_available_kb": (b.get("mem_kb") or {}).get("MemAvailable"),  # type: ignore[union-attr]
        "foreign": foreign[:10],
    }


def rtt(url: str, timeout: float = 5.0, samples: int = 5) -> Dict[str, object]:
    """Network distance to a provider: median TCP connect and TLS handshake
    time over a few fresh connections. No request is sent, so nothing bills."""
    u = urlparse(url if "://" in url else f"https://{url}")
    host, port = u.hostname or "", u.port or (443 if u.scheme == "https" else 80)
    tcp, tls = [], []
    err = None
    for _ in range(samples):
        try:
            t0 = time.perf_counter()
            s = socket.create_connection((host, port), timeout=timeout)
            t1 = time.perf_counter()
            if u.scheme == "https":
                ctx = ssl.create_default_context()
                ss = ctx.wrap_socket(s, server_hostname=host)
                t2 = time.perf_counter()
                ss.close()
                tls.append((t2 - t1) * 1000)
            else:
                s.close()
            tcp.append((t1 - t0) * 1000)
        except OSError as e:
            err = str(e)
    med = lambda xs: round(sorted(xs)[len(xs) // 2], 2) if xs else None  # noqa: E731
    return {"target": f"{host}:{port}", "tcp_connect_ms": med(tcp), "tls_handshake_ms": med(tls),
            "samples": len(tcp), "error": err if not tcp else None}


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        if u.path == "/snapshot":
            body = snapshot()
        elif u.path == "/rtt":
            q = parse_qs(u.query)
            body = rtt(q.get("url", [""])[0])
        elif u.path == "/health":
            body = {"ok": True}
        else:
            self.send_response(404)
            self.end_headers()
            return
        raw = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def serve(port: int = 9101, bind: str = "0.0.0.0") -> None:
    """Run the probe agent. Read-only; exposes CPU counters and process
    names, so bind it to the VPC (the rig never opens it to the internet)."""
    ThreadingHTTPServer((bind, port), _Handler).serve_forever()


if __name__ == "__main__":  # pragma: no cover
    if len(sys.argv) >= 2 and sys.argv[1] == "serve":
        p = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 9101
        serve(p)
    else:
        print(json.dumps(snapshot(), indent=1)[:2000])
