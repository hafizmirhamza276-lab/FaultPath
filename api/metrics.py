#!/usr/bin/env python3
"""
metrics.py
Prometheus exposition. Hand-rolled: the counters are few, the format is stable,
and a dependency here would be more code than the thing it replaces.

Latency is kept as a sorted sample per node so p50/p95/p99 are exact over the
window rather than bucket approximations. The sample is capped; at that point
the oldest half is dropped, which is stated in the HELP text rather than left
for someone to discover from a suspicious percentile.
"""
from __future__ import annotations

import threading
from typing import Dict, List

SAMPLE_CAP = 4096


class Metrics:
    def __init__(self):
        self._lock = threading.RLock()
        self.counters: Dict[str, float] = {}
        self.latency: Dict[str, List[float]] = {}

    def inc(self, name: str, value: float = 1.0, **labels) -> None:
        key = _key(name, labels)
        with self._lock:
            self.counters[key] = self.counters.get(key, 0.0) + value

    def observe(self, node: str, ms: float) -> None:
        with self._lock:
            s = self.latency.setdefault(node, [])
            s.append(ms)
            if len(s) > SAMPLE_CAP:
                del s[:len(s) // 2]

    def reset(self) -> None:
        with self._lock:
            self.counters.clear()
            self.latency.clear()

    def render(self) -> str:
        with self._lock:
            counters = dict(self.counters)
            latency = {k: sorted(v) for k, v in self.latency.items()}

        out: List[str] = []
        seen_help = set()
        for key, val in sorted(counters.items()):
            name = key.split("{", 1)[0]
            if name not in seen_help:
                out.append(f"# HELP {name} {_HELP.get(name, name)}")
                out.append(f"# TYPE {name} counter")
                seen_help.add(name)
            out.append(f"{key} {val:g}")

        if latency:
            out.append("# HELP agent_node_latency_ms Per-node latency in "
                       "milliseconds. Exact percentiles over a capped sample; "
                       "when the cap is hit the oldest half is dropped.")
            out.append("# TYPE agent_node_latency_ms summary")
            for node, s in sorted(latency.items()):
                for q in (0.5, 0.95, 0.99):
                    out.append(
                        f'agent_node_latency_ms{{node="{node}",quantile="{q}"}} '
                        f"{_pct(s, q):g}")
                out.append(f'agent_node_latency_ms_count{{node="{node}"}} {len(s)}')
                out.append(f'agent_node_latency_ms_sum{{node="{node}"}} {sum(s):g}')
        return "\n".join(out) + "\n"


def _pct(sorted_vals: List[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    i = int(round(q * (len(sorted_vals) - 1)))
    return sorted_vals[max(0, min(i, len(sorted_vals) - 1))]


def _key(name: str, labels: dict) -> str:
    if not labels:
        return name
    inner = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
    return f"{name}{{{inner}}}"


_HELP = {
    "agent_sessions_total": "Sessions started.",
    "agent_turns_total": "Technician turns processed.",
    "agent_blocked_messages_total": "Outgoing messages blocked by the "
                                    "ungrounded-value guard.",
    "agent_gate_failures_total": "Boundary gate failures, by gate.",
    "agent_llm_calls_total": "LLM calls, by role.",
    "agent_llm_latency_ms_total": "Cumulative LLM latency in milliseconds.",
    "agent_errors_total": "Request errors, by class.",
    "agent_sessions_active": "Sessions currently held in memory.",
}

METRICS = Metrics()
