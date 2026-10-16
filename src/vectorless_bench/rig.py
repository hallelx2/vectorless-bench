"""The speed rig: proof that nothing competed with the system being timed.

Every timed sample is bracketed by host snapshots from the bench VM (here, in
process) and from each remote host the system depends on (the engine VM, via
its hostprobe agent). The difference is the sample's host evidence, and a
sample is FLAGGED when it shows competition:

  - CPU steal above `max_steal_pct` on any host (the hypervisor took our CPU);
  - a process outside the allowlist using more than `foreign_core_pct` of a
    core (a stray build, an indexer, the dev shell);
  - a host whose probe could not be read (evidence missing is not evidence of
    a quiet machine).

A flagged sample is rerun, up to `rerun_flagged` times; the report computes
latency percentiles over unflagged samples only and says how many were
flagged. Nothing is averaged in that was measured on a busy machine.

The rig also records, once per run, the machines themselves (type, zone,
from the GCE metadata server) and the network distance from each host to
every provider the systems call, so provider distance is reported rather
than hidden inside latency.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from . import hostprobe

# Our own processes on either VM: never "foreign". Python is the harness and
# the in-process rivals; vls-server/postgres/docker are the engine stack.
DEFAULT_ALLOW = [
    "python", "vlbench", "uv", "vls-server", "postgres", "pgvector", "docker", "containerd",
    "kworker", "ksoftirqd", "rcu_", "migration", "systemd", "journald", "sshd", "google_",
    "chronyd", "irqbalance", "hostprobe", "snapd", "dockerd", "runc", "jbd2", "kswapd",
]

DEFAULT_RTT_TARGETS = [
    "https://aiplatform.googleapis.com",  # Vertex global endpoint
    "https://api.typesafe.ai",            # Jev
]


@dataclass
class RigConfig:
    # Remote hosts' probe agents, name -> base URL ("engine": "http://10.128.0.5:9101").
    probes: Dict[str, str] = field(default_factory=dict)
    max_steal_pct: float = 2.0
    foreign_core_pct: float = 10.0
    rerun_flagged: int = 2
    allow: List[str] = field(default_factory=lambda: list(DEFAULT_ALLOW))
    rtt_targets: List[str] = field(default_factory=lambda: list(DEFAULT_RTT_TARGETS))
    enabled: bool = True

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "RigConfig":
        if not d:
            return cls()
        known = cls().__dict__.keys()
        cfg = cls(**{k: v for k, v in d.items() if k in known})
        extra = d.get("allow_extra") or []
        cfg.allow = list(cfg.allow) + list(extra)
        return cfg


def _get_json(url: str, timeout: float = 3.0) -> Dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - VPC-internal URL from config
        return json.loads(r.read().decode())


class Rig:
    """Brackets samples with host snapshots and judges them."""

    def __init__(self, cfg: RigConfig) -> None:
        self.cfg = cfg

    def _snap_all(self) -> Dict[str, Optional[Dict[str, Any]]]:
        snaps: Dict[str, Optional[Dict[str, Any]]] = {"bench": hostprobe.snapshot()}
        for name, base in self.cfg.probes.items():
            try:
                snaps[name] = _get_json(base.rstrip("/") + "/snapshot")
            except Exception:
                snaps[name] = None
        return snaps

    def begin(self) -> Dict[str, Optional[Dict[str, Any]]]:
        if not self.cfg.enabled:
            return {}
        return self._snap_all()

    def end(self, start: Dict[str, Optional[Dict[str, Any]]]) -> Dict[str, Any]:
        """Evidence for the interval since `begin`, with a verdict."""
        if not self.cfg.enabled:
            return {"flagged": False, "reasons": [], "hosts": {}}
        stop = self._snap_all()
        hosts: Dict[str, Any] = {}
        reasons: List[str] = []
        for name in start:
            a, b = start.get(name), stop.get(name)
            if a is None or b is None:
                hosts[name] = None
                reasons.append(f"{name}: probe unreadable")
                continue
            ev = hostprobe.evidence(a, b, self.cfg.allow, foreign_min_pct=min(5.0, self.cfg.foreign_core_pct))
            hosts[name] = ev
            if ev["steal_pct"] > self.cfg.max_steal_pct:
                reasons.append(f"{name}: steal {ev['steal_pct']}%")
            for p in ev["foreign"]:
                if p["core_pct"] >= self.cfg.foreign_core_pct:
                    reasons.append(f"{name}: {p['comm']} used {p['core_pct']}% of a core")
        return {"flagged": bool(reasons), "reasons": reasons, "hosts": hosts}

    def network_baseline(self) -> Dict[str, Any]:
        """RTT from every host to every provider target."""
        out: Dict[str, Any] = {"bench": [hostprobe.rtt(t) for t in self.cfg.rtt_targets]}
        for name, base in self.cfg.probes.items():
            rows = []
            for t in self.cfg.rtt_targets:
                try:
                    rows.append(_get_json(f"{base.rstrip('/')}/rtt?url={urllib.request.quote(t, safe='')}", timeout=30))
                except Exception as e:
                    rows.append({"target": t, "error": str(e)})
            out[name] = rows
        return out


def machine_info() -> Dict[str, Any]:
    """This VM's identity from the GCE metadata server; empty off GCP."""
    base = "http://metadata.google.internal/computeMetadata/v1/instance/"
    out: Dict[str, Any] = {"cpu_count": os.cpu_count()}
    for key, path in (("name", "name"), ("machine_type", "machine-type"), ("zone", "zone"),
                      ("cpu_platform", "cpu-platform")):
        try:
            req = urllib.request.Request(base + path, headers={"Metadata-Flavor": "Google"})
            with urllib.request.urlopen(req, timeout=1.5) as r:  # noqa: S310 - metadata server
                out[key] = r.read().decode().rsplit("/", 1)[-1]
        except Exception:
            pass
    return out


def remote_machine_info(base: str) -> Dict[str, Any]:
    try:
        snap = _get_json(base.rstrip("/") + "/snapshot")
        return {"host": snap.get("host"), "cpu_count": snap.get("ncpu"), **(snap.get("machine") or {})}
    except Exception as e:
        return {"error": str(e)}
