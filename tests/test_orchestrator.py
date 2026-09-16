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
import collections
import glob
import inspect
import json
import os
import subprocess
import sys
import textwrap
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from core import orchestrator as orch                   # noqa: E402
from core import loader                                 # noqa: E402

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

check("stages are registered", len(orch.STAGES) == 14, str(len(orch.STAGES)))
names = [s.name for s in orch.ordered_stages()]
check("dependency order is topological",
      names.index("extract") < names.index("fidelity") < names.index("qa_set")
      < names.index("agent") < names.index("api")
      # human_verify reads the unresolved split fidelity writes, and self_test
      # asserts over the stage list every earlier stage is part of, so it runs
      # last of all.
      and names.index("fidelity") < names.index("human_verify")
      and names.index("api") < names.index("self_test")
      and names.index("self_test") == len(names) - 1
      # the symptom audit gates the data fidelity is then measured over, so it
      # has to sit between extraction and fidelity, not after it
      and names.index("extract_symptoms") < names.index("audit_symptoms")
      < names.index("fidelity")
      and names.index("extract_symptoms") < names.index("symptom_map"),
      str(names))

# ============================================ LEVEL / STAGE PARITY
#
# WHY THIS EXISTS. tests/test_human_verify.py was red for four commits while
# the orchestrator reported 11 stages green and 36/36 gates pass. Nothing was
# watching it: it was a level in run_all.py with no stage. Auditing that one
# level found six of ten test files orphaned -- four levels no stage ran, and
# two files (test_symptom_fixes.py, test_symptom_map.py) in NEITHER list, which
# had only ever run when someone typed their names.
#
# BOTH SIDES ARE DERIVED. run_all.py's LEVELS is read out of its parsed AST and
# each stage's test scripts out of its run function's source. Nothing here is a
# hand-written list of what ought to be present -- a hand-maintained list drifts
# exactly the way the expectation it is meant to protect just did.
#
# Pairing is by SCRIPT, not by name: the "module" level is run by the "api"
# stage, and matching on names would have to encode that by hand.
#
# This is orphan_detection, pointed at the orchestrator's own test surface.


def _levels_from_run_all() -> dict:
    """{level_name: script} parsed out of run_all.py. Never imported -- running
    it would run the whole suite."""
    tree = ast.parse(open(os.path.join(REPO_ROOT, "tests", "run_all.py"),
                          encoding="utf-8").read())
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign) and any(
                getattr(t, "id", "") == "LEVELS" for t in n.targets):
            return {e.elts[0].value: e.elts[1].value for e in n.value.elts}
    return {}


def _scripts_run_by(stage) -> set:
    """Test scripts a stage executes, read from its own source.

    Derived, so a stage that stops running a test is caught by the same check
    as a stage that never ran one.
    """
    body = inspect.getsource(stage.run)
    return {os.path.basename(c.value)
            for c in ast.walk(ast.parse(textwrap.dedent(body)))
            if isinstance(c, ast.Constant) and isinstance(c.value, str)
            and c.value.startswith("tests/") and c.value.endswith(".py")}


def parity(stages, levels, on_disk):
    """(levels_without_stage, stage_scripts_without_level, files_in_neither)."""
    run = {f for s in stages for f in _scripts_run_by(s)}
    lvl = {os.path.basename(p) for p in levels.values()}
    return sorted(lvl - run), sorted(run - lvl), sorted(on_disk - lvl - run)


_levels = _levels_from_run_all()
_on_disk = {os.path.basename(p)
            for p in glob.glob(os.path.join(REPO_ROOT, "tests", "test_*.py"))}
_no_stage, _no_level, _neither = parity(orch.STAGES, _levels, _on_disk)

check("run_all.py's LEVELS was parsed, not assumed", len(_levels) >= 8,
      str(_levels))
check("every level in run_all.py is run by a stage", not _no_stage,
      f"levels with no stage: {_no_stage}")
check("every test script a stage runs is a level in run_all.py", not _no_level,
      f"stage scripts with no level: {_no_level}")
check("no test file exists outside both lists", not _neither,
      f"in neither: {_neither}")

# SELF-TEST. A parity check that cannot fail is the same defect it exists to
# catch -- and this one reports zero on a healthy repo, which is precisely the
# shape audit checks E4 and H2 were in when they were read as clean.
_fake_level = dict(_levels, planted="tests/test_planted_orphan.py")
_p_no_stage, _, _ = parity(orch.STAGES, _fake_level, _on_disk)
check("parity self-test: a level with no stage fails the run",
      _p_no_stage == ["test_planted_orphan.py"], str(_p_no_stage))


def _planted_stage_run(ctx):
    _script_marker = "tests/test_unlisted_by_run_all.py"   # noqa: F841
    return {"gates": []}


_planted = orch.Stage("planted", _planted_stage_run,
                      inputs=["core/orchestrator.py"], outputs=[],
                      gates="synthetic", expected_s=0.0)
_, _p_no_level, _ = parity(list(orch.STAGES) + [_planted], _levels, _on_disk)
check("parity self-test: a stage running an unlisted test fails the run",
      _p_no_level == ["test_unlisted_by_run_all.py"], str(_p_no_level))

_, _, _p_neither = parity(orch.STAGES, _levels,
                          _on_disk | {"test_written_but_never_wired.py"})
check("parity self-test: a test file in neither list fails the run",
      _p_neither == ["test_written_but_never_wired.py"], str(_p_neither))

print(f"    {len(_levels)} levels / {len(orch.STAGES)} stages / "
      f"{len(_on_disk)} test files -- parity in both directions")

# ============================ SYMPTOM ENTRY METRICS: REGRESSION DETECTION
#
# The matcher's entry metrics are NOT registered as eval Tier-1 metrics, and
# that is a decision rather than an omission. Metric.compute(case, result)
# scores a system's response to a case over a retrieved corpus; the matcher
# takes a phrase and returns a routing decision, with no system, no retriever
# and no chunking. Registering it would mean either making the matcher
# masquerade as a Generator, or writing metrics that ignore their `result` and
# call the matcher themselves -- and the second produces a run record that
# lies: identical numbers under --system good and --system weak, while
# config.system names which system was supposedly under test.
#
# So they stay where they live, and regression detection comes to them. What
# was missing is the point of this block: a GATE CATCHES CROSSING A LINE, not
# 0.9890 sliding to 0.9500 while staying above 0.95.
from core.symptom_match import METRIC_DIRECTION      # noqa: E402
from eval.compare import compare as _compare         # noqa: E402

check("every symptom entry metric declares a diff direction",
      set(METRIC_DIRECTION) == {
          "symptom_entry_accuracy", "symptom_direct_entry_rate",
          "symptom_ask_rate", "symptom_wrong_tree_rate",
          "symptom_ask_without_answer_rate", "unmapped_rate"},
      str(sorted(METRIC_DIRECTION)))
check("ASK has NO better direction, so a diff can never judge it",
      METRIC_DIRECTION["symptom_ask_rate"] is None
      and METRIC_DIRECTION["symptom_direct_entry_rate"] is None,
      "a matcher that asks less is not better, it is more willing to guess")
check("wrong tree is lower-is-better and entry accuracy higher-is-better",
      METRIC_DIRECTION["symptom_wrong_tree_rate"] is False
      and METRIC_DIRECTION["symptom_entry_accuracy"] is True)


def _sym_record(overrides=None):
    """A minimal orchestrator record carrying the symptom metrics block."""
    from core.symptom_match import run_record_metrics
    scored = {
        "titles": {"n": 57, "symptom_entry_accuracy": 1.0,
                   "symptom_direct_entry_rate": 1.0, "symptom_ask_rate": 0.0,
                   "symptom_wrong_tree_rate": 0.0,
                   "symptom_ask_without_answer_rate": 0.0, "unmapped_rate": 0.0},
        "held_out": {"n": 91, "symptom_entry_accuracy": 0.989,
                     "symptom_direct_entry_rate": 0.0, "symptom_ask_rate": 1.0,
                     "symptom_wrong_tree_rate": 0.0,
                     "symptom_ask_without_answer_rate": 0.011,
                     "unmapped_rate": 0.0},
        "sealed": {"n": 46, "symptom_entry_accuracy": 1.0,
                   "symptom_direct_entry_rate": 0.0, "symptom_ask_rate": 1.0,
                   "symptom_wrong_tree_rate": 0.0,
                   "symptom_ask_without_answer_rate": 0.0, "unmapped_rate": 0.0},
    }
    m = run_record_metrics(scored)
    for k, v in (overrides or {}).items():
        m[k] = dict(m[k], value=v)
    return {"metrics": m}


_base = _sym_record()
check("the sealed/contaminated split survives into the record",
      _base["metrics"]["symptom_entry_accuracy[sealed]"]["contaminated"] is False
      and _base["metrics"]["symptom_entry_accuracy[held_out]"]["contaminated"] is True,
      "the contamination label has to travel with the number, or a "
      "contaminated figure gets quoted without it")
check("no blended number across sets is emitted",
      not [k for k in _base["metrics"] if "[" not in k],
      f"unscoped: {[k for k in _base['metrics'] if '[' not in k]}")

# THE CASE THIS EXISTS FOR. Above the gate both times; a gate sees nothing.
_slide = _compare(_base, _sym_record({"symptom_entry_accuracy[held_out]": 0.95}))
_row = next(r for r in _slide
            if r["metric"] == "symptom_entry_accuracy[held_out]")
check("a within-gate slide in entry accuracy is REGRESSED, not UNCHANGED",
      _row["status"] == "REGRESSED",
      f"0.9890 -> 0.9500 reported {_row['status']}; the gate is >= 0.95 and "
      f"would pass both runs")

_worse = _compare(_base, _sym_record({"symptom_wrong_tree_rate[sealed]": 0.02}))
check("a rise in wrong_tree_rate is REGRESSED",
      next(r for r in _worse
           if r["metric"] == "symptom_wrong_tree_rate[sealed]")["status"]
      == "REGRESSED")

# ASK must be diffable and unjudgeable, in BOTH directions. Asking more and
# asking less are both movement and neither is a regression.
for _key, _v, _lbl in (("symptom_ask_rate[titles]", 0.30, "up"),
                       ("symptom_ask_rate[held_out]", 0.40, "down"),
                       ("symptom_direct_entry_rate[titles]", 0.50, "down")):
    _r = next(r for r in _compare(_base, _sym_record({_key: _v}))
              if r["metric"] == _key)
    check(f"  {_key} moving {_lbl} is MOVED, never judged",
          _r["status"] == "MOVED",
          f"reported {_r['status']} ({_r['delta']:+.4f}) -- scoring ASK as a "
          f"regression is the pressure that turns a matcher into a guesser")

check("control: an unchanged record produces no regression",
      not [r for r in _compare(_base, _sym_record())
           if r["status"] == "REGRESSED"])

# Choice (b) means these are deliberately NOT in the eval registry, so the
# ceiling guard from 01e1845 does not and should not cover them. Asserted so
# nobody later assumes it does.
from eval.metrics import ceiling as _ceiling          # noqa: E402
check("symptom entry metrics are absent from the eval ceiling classification",
      not (set(METRIC_DIRECTION) & (set(_ceiling.CEILING)
                                    | set(_ceiling.NOT_CEILING))),
      "they are not Metric instances and are not scored over qa_set cases")


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
# PINNED AND DERIVED, both. The literal catches an unexpected change the way
# REGRESSION COUNTS does; the derivation catches the literal agreeing with a
# stale artefact, which is exactly how 3266 survived -- this pin and METRICS.md
# both read the same frozen file and their agreement read as corroboration.
check("known_gaps reports resolver_verified", gaps["resolver_verified"] == 6023,
      str(gaps["resolver_verified"]))
_ver = loader.load_verification()
check("the verification artefact's own counts agree with its facts",
      collections.Counter(_ver["facts"].values()) == collections.Counter(_ver["counts"])
      and _ver["total"] == len(_ver["facts"]),
      f"counts {_ver['counts']} vs {collections.Counter(_ver['facts'].values())}")
check("no fact is human_verified",
      _ver["counts"].get(loader.HUMAN_VERIFIED, 0) == 0,
      "claiming a human read pages nobody read is the self-agreement this "
      "exercise exists to avoid")
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
    # Every gate the MODULE defines must appear with the same value. Compared
    # as a subset, not as equality, because the stage also gates on its guard
    # script's exit code -- a gate the module has no opinion about.
    check("orchestrator and standalone agree on every fidelity gate",
          all(og.get(k) == v for k, v in dg.items()),
          f"standalone={dg} orchestrated={og}")
    # ...and the stage must not QUIETLY DROP one. Subset comparison alone would
    # pass an s_fidelity that forgot half the gates, which is the failure this
    # cross-check exists for.
    # The stage's own gates are NAMED, not counted. A fourth added later and
    # not listed here fails, which is the point -- an inequality that merely
    # allowed "more than the module's" would let a gate appear unnoticed.
    STAGE_ONLY_GATES = {"fidelity_test_exit_zero", "verification_not_stale",
                        "human_verified_still_zero"}
    check("the stage adds exactly its own gates and drops none of the module's",
          set(og) == set(dg) | STAGE_ONLY_GATES,
          f"extra={sorted(set(og) - set(dg) - STAGE_ONLY_GATES)} "
          f"missing={sorted(set(dg) - set(og))}")
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
