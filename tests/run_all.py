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

# EVERY ENTRY HERE MUST HAVE AN ORCHESTRATOR STAGE, and every test script run
# by a stage must appear here. tests/test_orchestrator.py derives both sides
# and fails the run on either kind of orphan -- see its parity check.
#
# That check exists because six of ten test files were orphans when it was
# written: four levels no stage ran, and two files in neither list that had
# only ever run when someone typed their names.
LEVELS = [
    ("ground-truth", "tests/test_extraction.py"),
    # Fidelity runs early: it asks whether golden/ says what the PDF says, and
    # every level after it assumes the answer is yes.
    ("fidelity", "tests/test_fidelity.py"),
    ("symptom-fixes", "tests/test_symptom_fixes.py"),
    ("symptom-map", "tests/test_symptom_map.py"),
    ("human-verify", "tests/test_human_verify.py"),
    ("harness", "tests/test_eval_harness.py"),
    # Expensive (~10 min: six full evaluations) and deliberately not optional.
    # The README table it checks claimed exactness for months after it stopped
    # reproducing, because nothing ran.
    ("readme-baselines", "tests/test_readme_baselines.py"),
    ("agent", "tests/test_agent_replay.py"),
    ("module", "tests/test_api_module.py"),
    ("pipeline+e2e", "tests/test_api_e2e.py"),
    ("orchestrator", "tests/test_orchestrator.py"),
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
