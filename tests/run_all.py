#!/usr/bin/env python3
"""
run_all.py
Every test level, in order, failing fast.

    python tests/run_all.py

Order is cheapest-first and dependency-first. A contract break should surface in
two seconds at the module level, not ninety seconds into an end-to-end run that
was doomed before it started.

  ground-truth   golden/ still matches the manual and the pinned counts
  harness        Tier-1 evaluation harness verifies itself
  agent          graph, replay determinism, good 7/7 vs bad 0/7
  module         API contracts, serialisation, guard, store, metrics
  pipeline+e2e   49 sessions in-process and over HTTP, plus overfitting guard

Fails fast: the first level to fail stops the run, because every later level
builds on it and their failures would be noise.
"""
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LEVELS = [
    ("ground-truth", "tests/test_extraction.py"),
    ("harness", "tests/test_eval_harness.py"),
    ("agent", "tests/test_agent_replay.py"),
    ("module", "tests/test_api_module.py"),
    ("pipeline+e2e", "tests/test_api_e2e.py"),
]


def main():
    only = sys.argv[1:] or None
    results, total = [], time.perf_counter()

    for name, path in LEVELS:
        if only and name not in only:
            continue
        print(f"\n{'=' * 66}\n{name.upper():<20} {path}\n{'=' * 66}")
        t0 = time.perf_counter()
        proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, path)],
                              cwd=REPO_ROOT,
                              env={**os.environ, "PYTHONIOENCODING": "utf-8"})
        dt = time.perf_counter() - t0
        results.append((name, proc.returncode, dt))
        if proc.returncode != 0:
            _summary(results, total, stopped_at=name)
            return proc.returncode

    _summary(results, total)
    return 0


def _summary(results, total, stopped_at=None):
    print(f"\n{'=' * 66}\nSUMMARY")
    for name, rc, dt in results:
        print(f"  [{'PASS' if rc == 0 else 'FAIL'}] {name:16} {dt:7.1f}s")
    print(f"  {'total':22} {time.perf_counter() - total:7.1f}s")
    if stopped_at:
        print(f"\nstopped at {stopped_at}; later levels build on it and their "
              "failures would be noise.")


if __name__ == "__main__":
    sys.exit(main())
