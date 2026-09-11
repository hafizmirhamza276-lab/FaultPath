#!/usr/bin/env python3
"""
test_orchestrator.py
MODULE, PIPELINE and E2E for the orchestrator, plus the overfitting guard.

Two faults matter most here. A stage that runs despite a failed upstream gate
produces a FALSE GREEN -- an end-to-end report built on a broken extraction.
And human_verified vanishing from the summary turns a known gap into an unknown
one, which is the failure the whole verification exercise was about.

Fake stages are used throughout the module level so the logic is tested without
a 6-minute pipeline behind it.
"""
import ast
import inspect
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from core import orchestrator as orch                   # noqa: E402

failures, timings = [], {}


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def fake(name, ok=True, skip=None, depends=None, marker=None):
    def _run(ctx):
        if marker is not None:
            marker.append(name)
        if skip:
            return {"skip": skip}
        return {"detail": {"ran": True},
                "gates": [orch._gate(f"{name}_ok", 1 if ok else 0, "==", 1, name)]}
    return orch.Stage(name, _run, inputs=["core/orchestrator.py"], outputs=[],
                      gates="synthetic", expected_s=0.0, depends=depends or [])


# ============================================================ MODULE
print("\nMODULE")
t0 = time.perf_counter()

check("stages are registered", len(orch.STAGES) == 11, str(len(orch.STAGES)))
names = [s.name for s in orch.ordered_stages()]
check("dependency order is topological",
      names.index("extract") < names.index("fidelity") < names.index("qa_set")
      < names.index("agent") < names.index("api")
      # the symptom audit gates the data fidelity is then measured over, so it
      # has to sit between extraction and fidelity, not after it
      and names.index("extract_symptoms") < names.index("audit_symptoms")
      < names.index("fidelity")
      and names.index("extract_symptoms") < names.index("symptom_map"),
      str(names))

# The rename that unblocked symptom_map. core/logging.py shadowed the standard
# library for any process whose script lives in core/, which is every
# orchestrator run. This asserts the shadowing file is gone rather than merely
# unimported: a compatibility shim would satisfy an import check and reinstate
# the defect, because the FILENAME is the defect.
_core_dir = os.path.join(REPO_ROOT, "core")
check("no core/logging.py shadowing the standard library",
      not os.path.exists(os.path.join(_core_dir, "logging.py")))
check("symptom_map runs in-process, not shelled out",
      "_script(\"core/symptom_match" not in inspect.getsource(orch.s_symptom_map)
      and "from core.symptom_match import" in inspect.getsource(orch.s_symptom_map))

# The shadow probe. This is the check that would have failed BEFORE the rename
# and passes after, which is the only thing that makes the rename meaningful --
# the whole suite passed with the bug present.
_shadow = subprocess.run(
    [sys.executable, "-c",
     "import sys, logging, pdfplumber; "
     "sys.stdout.write(logging.getLogger('probe').name + '|' + logging.__file__)"],
    cwd=_core_dir, capture_output=True, text=True,
    env={**os.environ, "PYTHONPATH": REPO_ROOT, "PYTHONIOENCODING": "utf-8"})
check("a process rooted in core/ imports pdfplumber and the real logging",
      _shadow.returncode == 0 and _shadow.stdout.startswith("probe|")
      and not _shadow.stdout.split("|")[1].startswith(REPO_ROOT),
      (_shadow.stdout + _shadow.stderr)[-300:])
check("every stage declares inputs, outputs, gates and a duration",
      all(s.inputs and s.gates and s.expected_s >= 0 for s in orch.STAGES))

git = orch.git_state()
s0 = orch.STAGES[0]
fp1 = orch.fingerprint(s0, git, orch.new_run_token())
fp2 = orch.fingerprint(s0, git, orch.new_run_token())
if git["dirty"]:
    check("a dirty tree invalidates every fingerprint across runs",
          fp1 != fp2 and fp1.startswith("dirty:"),
          "a resume that trusts a stale fingerprint reports green for code "
          "that is no longer there")
    tok = orch.new_run_token()
    check("a dirty fingerprint is stable within one run",
          orch.fingerprint(s0, git, tok) == orch.fingerprint(s0, git, tok))
else:
    check("fingerprints are stable on a clean tree", fp1 == fp2)
clean = {"sha": "a" * 40, "dirty": False}
check("fingerprint includes the git sha",
      orch.fingerprint(s0, clean) != orch.fingerprint(
          s0, {"sha": "b" * 40, "dirty": False}))

g = orch._gate("x", 0.9, ">=", 1.0, "s")
check("a gate below threshold fails", g["status"] == orch.FAIL)
check("a gate at threshold passes",
      orch._gate("x", 1.0, ">=", 1.0, "s")["status"] == orch.PASS)
check("a missing value fails rather than passing",
      orch._gate("x", None, ">=", 1.0, "s")["status"] == orch.FAIL)

ran = []
rec = orch.run(stages=[fake("a", marker=ran), fake("b", ok=False, marker=ran),
                       fake("c", marker=ran)], quiet=True)
check("a failing gate halts the run", rec["halted_by"] == "b")
check("no downstream stage runs after a failure", "c" not in ran,
      f"ran: {ran} -- a green report on a broken upstream is the worst output")
check("the skipped stage names its upstream cause",
      rec["stages"][2]["status"] == orch.SKIPPED
      and "b" in rec["stages"][2]["skip_reason"])

rec2 = orch.run(stages=[fake("a"), fake("b", skip="no PDF on this machine")],
                quiet=True)
check("a skip carries a reason",
      rec2["stages"][1]["status"] == orch.SKIPPED
      and rec2["stages"][1]["skip_reason"] == "no PDF on this machine")
check("a skipped stage is not counted as passed",
      rec2["stages"][1]["status"] != orch.PASS)

missing = orch.Stage("ghost", lambda c: {"gates": []},
                     inputs=["does/not/exist.py"], outputs=[])
rec3 = orch.run(stages=[missing], quiet=True)
check("a stage with missing inputs fails loudly, never skips",
      rec3["stages"][0]["status"] == orch.FAIL
      and rec3["stages"][0]["detail"]["missing_inputs"] == ["does/not/exist.py"])

check("the record carries known gaps", "known_gaps" in rec)
check("human_verified appears in the record",
      "human_verified" in rec["known_gaps"])
check("timings are separate from metrics", "timings" in rec
      and all("seconds" in s for s in rec["stages"]))
timings["module"] = time.perf_counter() - t0


# ========================================================== PIPELINE
print("\nPIPELINE")
t0 = time.perf_counter()

gaps = orch.known_gaps()
check("known_gaps reports human_verified as 0", gaps["human_verified"] == 0,
      str(gaps["human_verified"]))
check("known_gaps records the round as NOT PERFORMED",
      gaps["human_round"] == "NOT PERFORMED", gaps["human_round"])
check("known_gaps reports resolver_verified", gaps["resolver_verified"] == 3266,
      str(gaps["resolver_verified"]))
check("known_gaps reports the unresolved breakdown",
      gaps["unresolved_facts"].get("KNOWN_LIMITATION") == 504
      and gaps["unresolved_facts"].get("NEEDS_HUMAN_VERIFICATION") == 27
      # present AND zero. An absent key reads as "not measured".
      and gaps["unresolved_facts"].get("DEFECT") == 0,
      str(gaps["unresolved_facts"]))

counts = orch.regression_counts()
want = {"files": 174, "steps": 996, "measurements": 872, "branches": 1437,
        "pointer_only": 9, "column_split": 27, "crossrefs": 94,
        "format_split": "117 A / 57 B"}
check("all eight regression counts unchanged", counts == want,
      json.dumps({k: (want[k], counts.get(k)) for k in want
                  if want[k] != counts.get(k)}))

sp = orch.splits()
check("train/holdout split is recorded",
      sp.get("train", 0) + sp.get("holdout", 0) == 174, str(sp))

# The orchestrator must read the same values a stage produces standalone.
from pipeline import fidelity                            # noqa: E402
from eval.citations import PageText                      # noqa: E402
pages = PageText()
if pages.available():
    direct = fidelity.check_facts(pages=pages)
    pages.close()
    ctx = {}
    via = orch.s_fidelity(ctx)
    dg = {g["gate"]: g["value"] for g in fidelity.gates(direct)}
    og = {g["gate"]: g["value"] for g in via["gates"]}
    check("orchestrator and standalone agree on every fidelity gate",
          dg == og, f"standalone={dg} orchestrated={og}")
else:
    print("  SKIP  PDF unavailable; cross-check of fidelity values not run")
timings["pipeline"] = time.perf_counter() - t0


# =============================================================== E2E
print("\nE2E -- determinism of the record")
t0 = time.perf_counter()


def strip(rec):
    """Everything that must be reproducible. Timings excluded on purpose:
    wall time legitimately varies and would break the comparison for a reason
    that has nothing to do with determinism."""
    return {
        # seconds and fingerprint both excluded: wall time varies, and on a
        # dirty tree the fingerprint is deliberately run-unique so that no
        # cached stage can be reused. Neither is a metric.
        "stages": [{k: v for k, v in s.items()
                    if k not in ("seconds", "fingerprint")}
                   for s in rec["stages"]],
        "gates": rec["gates"], "known_gaps": rec["known_gaps"],
        "regression_counts": rec["regression_counts"], "splits": rec["splits"],
        "halted_by": rec["halted_by"],
    }


syn = [fake("a"), fake("b"), fake("c")]
r1 = orch.run(stages=syn, quiet=True)
r2 = orch.run(stages=syn, quiet=True)
check("two runs of the same inputs produce identical records",
      strip(r1) == strip(r2))
check("records differ in timings only",
      r1["timings"].keys() == r2["timings"].keys())

# resume: unchanged inputs skip, changed inputs rerun
ran_a, ran_b = [], []
st = [fake("a", marker=ran_a), fake("b", marker=ran_b)]
orch.run(stages=st, quiet=True)
before = len(ran_a)
orch.run(stages=st, resume=True, quiet=True)
if orch.git_state()["dirty"]:
    check("a dirty tree forces a rerun on --resume", len(ran_a) > before,
          "cannot know what changed, so nothing may be reused")
else:
    check("--resume skips a stage whose inputs are unchanged",
          len(ran_a) == before)
orch.run(stages=st, force=True, quiet=True)
check("--force reruns regardless", len(ran_a) > before)
timings["e2e"] = time.perf_counter() - t0


# ================================================ OVERFITTING GUARD
print("\nOVERFITTING GUARD")

tree = ast.parse(open(os.path.join(REPO_ROOT, "core", "orchestrator.py"),
                      encoding="utf-8").read())
imported = set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported |= {a.name.split(".")[0] for a in node.names}
    elif isinstance(node, ast.ImportFrom):
        imported.add((node.module or "").split(".")[0])
banned = {"openai", "anthropic", "langchain", "langchain_openai", "langgraph",
          "transformers", "cohere", "google", "ollama", "mistralai"}
check("the orchestrator imports no LLM client", not (imported & banned),
      f"imports: {sorted(imported & banned)}")
print(f"    imports: {sorted(imported - {'annotations'})}")

print("\n  MUTATION TABLE")
muts = []


def mut(name, caught):
    muts.append((name, caught))
    print(f"    {name:44} {'caught' if caught else '*** NOT CAUGHT ***':20} "
          f"{', '.join(caught) or '-'}")


ran = []
r = orch.run(stages=[fake("up", ok=False, marker=ran), fake("down", marker=ran)],
             quiet=True)
mut("stage_ran_despite_failed_upstream_gate",
    ["downstream skipped"] if "down" not in ran else [])

r = orch.run(stages=[fake("s", skip="no PDF")], quiet=True)
mut("skipped_stage_reported_as_passed",
    ["status is SKIPPED, not PASS"]
    if r["stages"][0]["status"] == orch.SKIPPED else [])

dirty = {"sha": "x" * 40, "dirty": True}
dirty_fp = orch.fingerprint(orch.STAGES[0], dirty, orch.new_run_token())
dirty_fp2 = orch.fingerprint(orch.STAGES[0], dirty, orch.new_run_token())
mut("stale_fingerprint_reused_after_tree_changed",
    ["dirty tree never matches a cached fingerprint"]
    if dirty_fp != dirty_fp2 else [])

import io                                                # noqa: E402
buf, old = io.StringIO(), sys.stdout
try:
    sys.stdout = buf
    orch.summary(r)
finally:
    sys.stdout = old
mut("human_verified_missing_from_summary",
    ["summary prints human_verified"] if "human_verified" in buf.getvalue() else [])

# a gate evaluated after the next stage started
order = []


def probe(name, ok=True):
    def _run(ctx):
        order.append(f"run:{name}")
        return {"gates": [orch._gate(f"{name}_ok", 1 if ok else 0, "==", 1, name)]}
    return orch.Stage(name, _run, inputs=["core/orchestrator.py"], outputs=[])


rr = orch.run(stages=[probe("p1", ok=False), probe("p2")], quiet=True)
mut("gate_evaluated_after_next_stage_started",
    ["p2 never ran; p1's gate was evaluated first"]
    if order == ["run:p1"] else [])

check("every planted fault is caught", all(c for _, c in muts),
      f"uncaught: {[n for n, c in muts if not c]}")

# negative coverage: every orchestrator gate kind has a failing case
check("gate evaluation has a failing case for each operator",
      orch._gate("a", 0, "==", 1, "s")["status"] == orch.FAIL
      and orch._gate("b", 0.5, ">=", 1.0, "s")["status"] == orch.FAIL
      and orch._gate("c", 2.0, "<=", 1.0, "s")["status"] == orch.FAIL)


print("\n" + "=" * 62)
print(f"module {timings.get('module', 0):.1f}s  "
      f"pipeline {timings.get('pipeline', 0):.1f}s  e2e {timings.get('e2e', 0):.1f}s")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: orchestrator")
sys.exit(0)
