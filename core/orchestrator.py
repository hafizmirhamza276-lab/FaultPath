#!/usr/bin/env python3
"""
orchestrator.py
One controlled entry point for a pipeline that is already deterministic.

    python core/orchestrator.py                 # full run
    python core/orchestrator.py --resume        # only what changed, and downstream
    python core/orchestrator.py --from fidelity # from a named stage onward

ITS JOB IS SEQUENCE, GATING AND RECORDING. Never judgement. Every stage is
already a pure function; this adds order and a halt, nothing else.

WHAT IT MUST NOT DO, and does not:
  retry a failed stage, adjust a threshold, choose between alternatives, or
  interpret a result. Each of those would put a decision back into a system
  whose entire value is that it has none. If a stage fails it stops and reports.

A green end-to-end report built on a broken extraction is the worst output this
could produce, so no downstream stage runs after an upstream gate fails.

THREE OUTCOMES per stage: PASS, FAIL, SKIPPED. A skip always carries a reason
and always appears in the summary. Silence is never the safe default.

No LLM. This module imports no model client, asserted over its parsed AST by
tests/test_orchestrator.py.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Callable, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

RUNS_DIR = os.path.join(REPO_ROOT, "eval_out", "runs")
INDEX_PATH = os.path.join(RUNS_DIR, "index.jsonl")
STATE_PATH = os.path.join(RUNS_DIR, ".orchestrator_state.json")
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))

PASS, FAIL, SKIPPED = "PASS", "FAIL", "SKIPPED"


# ------------------------------------------------------------------ git

def git_state() -> dict:
    def run(*a):
        try:
            return subprocess.check_output(a, cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    sha = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {"sha": sha, "dirty": dirty}


# ---------------------------------------------------------- fingerprints

def _hash_paths(paths: List[str]) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        full = p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
        if os.path.isdir(full):
            for root, _, files in os.walk(full):
                for fn in sorted(files):
                    fp = os.path.join(root, fn)
                    h.update(os.path.relpath(fp, REPO_ROOT).encode())
                    try:
                        h.update(str(os.path.getsize(fp)).encode())
                        with open(fp, "rb") as f:
                            h.update(hashlib.sha256(f.read()).digest())
                    except OSError:
                        h.update(b"<unreadable>")
        elif os.path.isfile(full):
            h.update(os.path.relpath(full, REPO_ROOT).encode())
            with open(full, "rb") as f:
                h.update(hashlib.sha256(f.read()).digest())
        else:
            h.update(f"<missing:{p}>".encode())
    return h.hexdigest()[:16]


_RUN_TOKEN_SEQ = [0]


def new_run_token() -> str:
    """A value unique to this run, used only to poison dirty-tree caches.

    A counter rather than a timestamp: time.time_ns() returned the same value
    on two consecutive calls under Windows' clock resolution, so two stages in
    the same run could collide and a cached result would be reused on a dirty
    tree -- the exact failure the dirty rule exists to prevent.
    """
    _RUN_TOKEN_SEQ[0] += 1
    return f"{os.getpid()}-{_RUN_TOKEN_SEQ[0]}"


def fingerprint(stage: "Stage", git: dict, run_token: str = "") -> str:
    """Inputs plus the git state.

    A DIRTY TREE INVALIDATES EVERY CACHED STAGE. We cannot know what changed in
    an uncommitted working tree, and a resume that trusts a stale fingerprint
    produces a green report for code that is no longer there. The run token is
    stable across the stages of one run and different on the next, so a dirty
    run is internally consistent and never reusable.
    """
    if git.get("dirty"):
        return f"dirty:{run_token or new_run_token()}"
    return _hash_paths(list(stage.inputs)) + ":" + (git.get("sha") or "nosha")[:12]


# ---------------------------------------------------------------- stage

class Stage:
    def __init__(self, name: str, run: Callable[[dict], dict],
                 inputs: List[str], outputs: List[str],
                 gates: str = "", expected_s: float = 0.0,
                 needs_pdf: bool = False, depends: Optional[List[str]] = None):
        self.name = name
        self.run = run
        self.inputs = inputs
        self.outputs = outputs
        self.gates = gates
        self.expected_s = expected_s
        self.needs_pdf = needs_pdf
        self.depends = depends or []

    def missing_inputs(self) -> List[str]:
        out = []
        for p in self.inputs:
            full = p if os.path.isabs(p) else os.path.join(REPO_ROOT, p)
            if not os.path.exists(full):
                out.append(p)
        return out


def _gate(name: str, value, op: str, threshold, stage: str) -> dict:
    if value is None:
        ok = False
    elif op == ">=":
        ok = value >= threshold
    elif op == "<=":
        ok = value <= threshold
    elif op == "==":
        ok = value == threshold
    else:
        raise ValueError(op)
    return {"gate": name, "value": value, "op": op, "threshold": threshold,
            "status": PASS if ok else FAIL, "stage": stage}


def _script(path: str, args=()) -> dict:
    """Run an existing entry point. Its exit code is the only verdict taken."""
    t0 = time.perf_counter()
    proc = subprocess.run([sys.executable, os.path.join(REPO_ROOT, path), *args],
                          cwd=REPO_ROOT, capture_output=True, text=True,
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    return {"returncode": proc.returncode, "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-2000:],
            "seconds": round(time.perf_counter() - t0, 2)}


# --------------------------------------------------------------- stages

def s_extract(ctx) -> dict:
    r = _script("pipeline/extract_golden.py")
    # The regression guard runs again here, explicitly, even though
    # extract_golden.py already invokes it and exits non-zero on failure.
    # Redundant by 0.5s and worth it: run_all.py's "ground-truth" level is
    # tests/test_extraction.py, and the parity check pairs levels to stages by
    # the SCRIPT each runs. A guard reachable only from inside another script
    # is invisible to that pairing, which is how an orphan starts.
    g = _script("tests/test_extraction.py")
    return {"detail": {**r, "guard": g}, "gates": [
        _gate("extract_exit_zero", r["returncode"], "==", 0, "extract"),
        _gate("ground_truth_exit_zero", g["returncode"], "==", 0, "extract")]}


def s_audit(ctx) -> dict:
    r = _script("pipeline/audit_manual.py")
    findings = []
    p = os.path.join(REPO_ROOT, "reports", "audit_findings.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            findings = json.load(f)
    sev = {}
    for x in findings:
        sev[x["severity"]] = sev.get(x["severity"], 0) + 1
    return {"detail": {**r, "findings": len(findings), "by_severity": sev},
            "gates": [_gate("audit_exit_zero", r["returncode"], "==", 0, "audit")]}


def s_extract_symptoms(ctx) -> dict:
    r = _script("pipeline/extract_symptoms.py")
    n = len(glob.glob(os.path.join(REPO_ROOT, "golden", "symptoms", "*M*.json")))
    ctx["symptom_records"] = n
    # The guard for the S1/S2/C3/S4 parser fixes. It had no stage and no level
    # until the parity check found it: it had only ever run when someone typed
    # its name, which is not a guard, it is a habit.
    g = _script("tests/test_symptom_fixes.py")
    return {"detail": {**r, "records": n, "guard": g},
            "gates": [_gate("extract_symptoms_exit_zero", r["returncode"],
                            "==", 0, "extract_symptoms"),
                      _gate("symptom_fixes_guarded", g["returncode"], "==", 0,
                            "extract_symptoms")]}


def s_audit_symptoms(ctx) -> dict:
    """Audit of the H-Mode and S-Mode symptom trees.

    Positioned after extraction and before fidelity, so a symptom-audit failure
    halts the run like any other stage. Fidelity measures whether stored facts
    are on the pages they claim; this measures whether the facts are the right
    shape in the first place, and a wrong shape makes a clean fidelity figure
    meaningless -- S1's empty criteria resolved against every page in the
    document until they were refused explicitly.

    Gated on exit code AND on the HIGH count, which must be zero. The script's
    own 12 self-tests abort it before any check runs, so a non-zero exit covers
    a blind check as well as a failed one.
    """
    r = _script("pipeline/audit_symptoms.py")
    findings = []
    p = os.path.join(REPO_ROOT, "reports", "audit_symptoms.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            findings = json.load(f)
    sev = {}
    for x in findings:
        sev[x["severity"]] = sev.get(x["severity"], 0) + 1
    ctx["audit_symptoms"] = {"findings": len(findings), "by_severity": sev}
    return {"detail": {**r, "findings": len(findings), "by_severity": sev},
            "gates": [
                _gate("audit_symptoms_exit_zero", r["returncode"], "==", 0,
                      "audit_symptoms"),
                _gate("audit_symptoms_high", sev.get("HIGH", 0), "==", 0,
                      "audit_symptoms")]}


def s_symptom_map(ctx) -> dict:
    """Build the synonym map and score it on three sets.

    The map is validated against ground truth by its own builder -- a target id
    with no tree fails the build rather than shipping an entry that cites a
    page convincingly and wrongly.

    Gated on wrong_tree == 0 for EVERY set, including the sealed one. Entry
    accuracy is gated lower and separately, because asking is not a failure and
    a single number that mixes the two would let a guessing matcher score well.
    """
    from core.symptom_match import (SymptomMatcher, score, gates as sym_gates,
                                    run_record_metrics)
    build = _script("pipeline/build_symptom_map.py")
    if build["returncode"] != 0:
        return {"detail": build,
                "gates": [_gate("symptom_map_builds", build["returncode"], "==",
                                0, "symptom_map")]}
    held = _script("pipeline/build_symptom_heldout.py")

    m = SymptomMatcher()
    with open(os.path.join(REPO_ROOT, "knowledge", "symptom_synonyms.json"),
              encoding="utf-8") as f:
        smap = json.load(f)
    with open(os.path.join(REPO_ROOT, "knowledge", "symptom_heldout.json"),
              encoding="utf-8") as f:
        hold = json.load(f)

    sets = {
        "titles": [{"input": m.trees[s]["symptom"], "expected": [s]}
                   for s in m.trees],
        "built_from": [{"input": e["input"],
                        "expected": e["target_symptom_ids"]}
                       for e in smap["entries"]],
        "held_out": hold["cases"],
        "sealed": hold["sealed_cases"],
    }
    out_gates, detail = [
        _gate("symptom_map_builds", build["returncode"], "==", 0, "symptom_map"),
        _gate("symptom_heldout_builds", held["returncode"], "==", 0,
              "symptom_map"),
    ], {}
    scored = {}
    for name, cases in sets.items():
        sc = score(m, cases)
        scored[name] = sc
        detail[name] = {k: round(v, 4) for k, v in sc.items()
                        if isinstance(v, float)}
        detail[name]["n"] = sc["n"]
        for g in sym_gates(sc):
            out_gates.append(_gate(f"{g['gate']}[{name}]", g["value"], g["op"],
                                   g["threshold"], "symptom_map"))
    # ALL SIX metrics per set, in the shape compare.py already diffs.
    #
    # Only three of the six reach a gate, and a gate catches crossing a line --
    # not 0.9890 sliding to 0.9500 while staying above 0.95. The other three
    # (ask_rate, direct_entry_rate, unmapped_rate) lived in `detail`, which
    # report_orchestrator never diffed at all, so they had no regression
    # detection of any kind.
    ctx["symptom_metrics"] = run_record_metrics(scored)
    ctx["symptom_map"] = detail
    detail["entries"] = len(smap["entries"])
    detail["unmapped_recorded"] = len(smap["unmapped"])
    guard = _script("tests/test_symptom_map.py")
    detail["guard"] = guard
    out_gates.append(_gate("symptom_map_guarded", guard["returncode"], "==", 0,
                           "symptom_map"))
    return {"detail": detail, "gates": out_gates}


def s_fidelity(ctx) -> dict:
    from pipeline import fidelity
    from eval.citations import PageText
    pages = PageText()
    if not pages.available():
        return {"skip": f"source PDF not found at {pages.path}; fidelity is "
                        "measured against the document and cannot be inferred"}
    res = fidelity.check_facts(pages=pages)
    payload = fidelity.write_unresolved(res)
    pages.close()
    ctx["fidelity"] = res
    ctx["unresolved"] = payload
    gates = [_gate(g["gate"], g["value"], g["op"], g["threshold"], "fidelity")
             for g in fidelity.gates(res)]
    g = _script("tests/test_fidelity.py")
    gates.append(_gate("fidelity_test_exit_zero", g["returncode"], "==", 0,
                       "fidelity"))
    return {"detail": {"by_kind": {k: v["rate"] for k, v in res["by_kind"].items()},
                       "unresolved": payload["by_classification"],
                       "guard": g},
            "gates": gates}


def s_human_verify(ctx) -> dict:
    """The human-transcription round and the reclassification it rests on.

    This level had NO STAGE and was red for four commits while the orchestrator
    reported every gate green -- a false green, which is the one outcome this
    module exists to prevent. See reports/false_green_findings.md.
    """
    r = _script("tests/test_human_verify.py")
    return {"detail": r, "gates": [
        _gate("human_verify_exit_zero", r["returncode"], "==", 0,
              "human_verify")]}


def s_self_test(ctx) -> dict:
    """The orchestrator's own tests, including the level/stage parity check.

    Run as a subprocess, which is not a workaround here but the only correct
    shape: an orchestrator that imported its own test module in-process would
    be asserting over the objects it is currently executing.
    """
    r = _script("tests/test_orchestrator.py")
    return {"detail": r, "gates": [
        _gate("self_test_exit_zero", r["returncode"], "==", 0, "self_test")]}


def s_structural(ctx) -> dict:
    from pipeline import structural
    reader = structural.PageReader()
    if not reader.available():
        return {"skip": f"source PDF not found at {reader.path}"}
    res = structural.run(reader=reader)
    reader.close()
    gates = [_gate(g["gate"], g["value"], g["op"], g["threshold"], "structural")
             for g in structural.gates(res)]
    return {"detail": {"orphans": len(res["orphan_detection"]["orphans"]),
                       "branch_completeness": res["branch_completeness"]["rate"]},
            "gates": gates}


def s_qa_set(ctx) -> dict:
    """Rebuild the golden Q&A set and pin BOTH HALVES, not one total.

    This gate used to read `qa_set_cases == 1330`, which is the same
    under-specification that made tests/test_human_verify.py go red for four
    commits: a total pinned while a section was meant. When symptom cases
    landed, that total would have absorbed a second corpus, and bumping it to
    1865 alone would leave the next widening free to hide inside it.

    Two gates instead. The per-(section, type) split is asserted in
    tests/test_extraction.py, which the extract stage runs; these are the
    coarse backstop that fails if either half moves.
    """
    r = _script("eval/build_qa_set.py")
    n = n_s40 = 0
    p = os.path.join(GOLD, "qa_set.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            qa = json.load(f)
        n = len(qa)
        n_s40 = sum(1 for c in qa if c.get("section") == "section40")
    return {"detail": {**r, "cases": n, "section40_cases": n_s40,
                       "symptom_cases": n - n_s40},
            "gates": [_gate("qa_set_exit_zero", r["returncode"], "==", 0, "qa_set"),
                      _gate("qa_set_cases", n, "==", 1865, "qa_set"),
                      _gate("qa_set_section40_cases", n_s40, "==", 1330, "qa_set")]}


def s_harness(ctx) -> dict:
    r = _script("tests/test_eval_harness.py")
    return {"detail": r, "gates": [
        _gate("harness_exit_zero", r["returncode"], "==", 0, "harness")]}


def s_agent(ctx) -> dict:
    r = _script("tests/test_agent_replay.py")
    return {"detail": r, "gates": [
        _gate("agent_exit_zero", r["returncode"], "==", 0, "agent")]}


def s_api(ctx) -> dict:
    mod = _script("tests/test_api_module.py")
    e2e = _script("tests/test_api_e2e.py")
    return {"detail": {"module": mod, "e2e": e2e},
            "gates": [_gate("api_module_exit_zero", mod["returncode"], "==", 0, "api"),
                      _gate("api_e2e_exit_zero", e2e["returncode"], "==", 0, "api")]}


STAGES: List[Stage] = [
    Stage("extract", s_extract,
          inputs=["pipeline/extract_golden.py", "tests/test_extraction.py"],
          outputs=["golden/failure_codes", "golden/index.json"],
          gates="exit zero; the script's own regression guard enforces the "
                "eight counts", expected_s=260, needs_pdf=True),
    Stage("audit", s_audit, inputs=["pipeline/audit_manual.py"],
          outputs=["reports/audit_findings.json"], gates="exit zero",
          expected_s=200, needs_pdf=True, depends=["extract"]),
    # Symptom extraction was never a stage either. Auditing golden/symptoms
    # without regenerating it first would audit whatever happened to be on
    # disk, which is the shape of defect this project keeps removing.
    Stage("extract_symptoms", s_extract_symptoms,
          inputs=["pipeline/extract_symptoms.py"],
          outputs=["golden/symptoms"],
          gates="exit zero; the script's own self-tests run first",
          expected_s=40, needs_pdf=True, depends=["extract"]),
    Stage("audit_symptoms", s_audit_symptoms,
          inputs=["pipeline/audit_symptoms.py", "golden/symptoms"],
          outputs=["reports/audit_symptoms.json"],
          gates="exit zero; HIGH findings == 0",
          expected_s=5, needs_pdf=True, depends=["extract_symptoms"]),
    # Registered in-process, deliberately. It was held out of STAGES until
    # core/logging.py was renamed to core/run_log.py: that file shadowed the
    # standard library for any process whose script lives in core/, so this
    # stage's in-process pdfplumber import died on `logging.getLogger`. Running
    # it through _script() would have made the symptom disappear while leaving
    # the defect for the next in-process import. See core/run_log.py's header.
    Stage("symptom_map", s_symptom_map,
          inputs=["core/symptom_match.py", "pipeline/build_symptom_map.py",
                  "pipeline/build_symptom_heldout.py",
                  "knowledge/symptom_synonyms.source.tsv",
                  "knowledge/symptom_unmapped.tsv", "golden/symptoms"],
          outputs=["knowledge/symptom_synonyms.json",
                   "knowledge/symptom_heldout.json"],
          gates="map builds against ground truth; wrong_tree 0 on every set "
                "including sealed; entry accuracy >= 0.95; ask rate reported "
                "but never gated against wrong_tree",
          expected_s=15, needs_pdf=True, depends=["extract_symptoms"]),
    Stage("fidelity", s_fidelity,
          inputs=["pipeline/fidelity.py", "eval/citations.py",
                  "golden/failure_codes", "golden/symptoms"],
          outputs=["reports/unresolved_facts.json"],
          gates="measurement 1.0, branch 1.0 (per section and overall), "
                "page_containment 1.0, known_overhang_stable",
          expected_s=90, needs_pdf=True,
          depends=["extract", "extract_symptoms"]),
    Stage("structural", s_structural,
          inputs=["pipeline/structural.py", "golden/failure_codes"], outputs=[],
          gates="orphan_detection, step_count_agreement, step_contiguity",
          expected_s=20, needs_pdf=True, depends=["extract"]),
    Stage("qa_set", s_qa_set,
          inputs=["eval/build_qa_set.py", "golden/failure_codes"],
          outputs=["golden/qa_set.json"],
          gates="exit zero; 1,865 cases total and 1,330 of them Section 40",
          expected_s=5, depends=["extract", "fidelity"]),
    Stage("harness", s_harness,
          inputs=["tests/test_eval_harness.py", "eval", "golden/qa_set.json"],
          outputs=[], gates="good 7/7, weak 0/7, determinism",
          expected_s=290, depends=["qa_set"]),
    Stage("agent", s_agent,
          inputs=["tests/test_agent_replay.py", "agent", "golden/failure_codes"],
          outputs=[], gates="good 7/7, bad 0/7, replay determinism",
          expected_s=15, depends=["qa_set"]),
    Stage("api", s_api,
          inputs=["tests/test_api_module.py", "tests/test_api_e2e.py", "api", "agent"],
          outputs=[], gates="module + pipeline + e2e; HTTP matches in-process",
          expected_s=30, depends=["agent"]),
    # --- levels that had no stage until the parity check found them ---
    Stage("human_verify", s_human_verify,
          inputs=["tests/test_human_verify.py", "pipeline/human_verify.py",
                  "pipeline/human_select.py", "pipeline/human_kit.py",
                  "reports/unresolved_facts.json"],
          outputs=[],
          gates="unresolved split per section/kind/class; no unnamed bucket; "
                "DEFECT 0; transcription round recorded",
          expected_s=15, needs_pdf=True, depends=["fidelity"]),
    Stage("self_test", s_self_test,
          inputs=["tests/test_orchestrator.py", "core/orchestrator.py",
                  "tests/run_all.py"],
          outputs=[],
          gates="level/stage parity in both directions; no orphan test file; "
                "gate arithmetic; determinism of the record",
          expected_s=25, needs_pdf=True, depends=["api"]),
]

STAGE_BY_NAME = {s.name: s for s in STAGES}


def ordered_stages() -> List[Stage]:
    """Topological order. Declared dependencies must already be satisfied."""
    done, out = set(), []
    remaining = list(STAGES)
    while remaining:
        progressed = False
        for s in list(remaining):
            if all(d in done for d in s.depends):
                out.append(s)
                done.add(s.name)
                remaining.remove(s)
                progressed = True
        if not progressed:
            raise RuntimeError(f"cyclic stage dependencies: "
                               f"{[s.name for s in remaining]}")
    return out


# ----------------------------------------------------------- known gaps

def known_gaps() -> dict:
    """Surfaced in EVERY summary. A gap that stops being visible becomes an
    unknown gap, and human_verified: 0 is the one most easily forgotten."""
    gaps = {"human_verified": 0, "resolver_verified": 0, "unverified": 0,
            "unresolved_facts": {}, "human_round": "NOT PERFORMED"}
    try:
        from core import loader
        s = loader.summarise()
        gaps["human_verified"] = s[loader.HUMAN_VERIFIED]["n"]
        gaps["resolver_verified"] = s[loader.RESOLVER_VERIFIED]["n"]
        gaps["unverified"] = s[loader.UNVERIFIED]["n"]
    except Exception:
        pass
    p = os.path.join(REPO_ROOT, "reports", "unresolved_facts.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        gaps["unresolved_facts"] = d.get("by_classification", {})
        gaps["human_round"] = (d.get("human_verification") or {}).get(
            "status", "NOT PERFORMED")
    return gaps


def regression_counts() -> dict:
    import glob
    recs = {}
    for p in glob.glob(os.path.join(GOLD, "failure_codes", "*.json")):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["code"]] = r
    if not recs:
        return {}
    fmt = {}
    for r in recs.values():
        fmt[r["format"]] = fmt.get(r["format"], 0) + 1
    return {
        "files": len(recs),
        "steps": sum(len(r["steps"]) for r in recs.values()),
        "measurements": sum(len(r["standalone_measurements"]) for r in recs.values())
        + sum(len(s["measurements"]) for r in recs.values() for s in r["steps"]),
        "branches": sum(len(s["branches"]) for r in recs.values() for s in r["steps"]),
        "pointer_only": sum(1 for r in recs.values() if r["is_pointer_only"]),
        "column_split": sum(1 for r in recs.values() for s in r["steps"]
                            if s.get("extraction_warning") == "column_split_recovered"),
        "crossrefs": sum(len(r["refs_failure_codes"]) for r in recs.values()),
        "format_split": f"{fmt.get('A', 0)} A / {fmt.get('B', 0)} B",
    }


def splits() -> dict:
    try:
        from agent import holdout
        return {"train": len(holdout.train_codes()),
                "holdout": len(holdout.holdout_codes())}
    except Exception:
        return {}


# ------------------------------------------------------------------ run

def load_state() -> dict:
    if os.path.isfile(STATE_PATH):
        with open(STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    os.makedirs(os.path.dirname(STATE_PATH), exist_ok=True)
    with open(STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


class Term:
    def __init__(self, quiet=False, verbose=False):
        self.quiet, self.verbose = quiet, verbose

    def stage_start(self, s: Stage, i: int, n: int):
        if self.quiet:
            return
        print(f"\n[{i}/{n}] {s.name}"
              f"   (expects ~{s.expected_s:.0f}s)   gates: {s.gates or '-'}",
              flush=True)

    def stage_end(self, res: dict):
        if self.quiet:
            return
        st = res["status"]
        print(f"      {st}  {res['seconds']:.1f}s", end="")
        if st == SKIPPED:
            print(f"  -- {res['skip_reason']}")
        else:
            print()
        for g in res["gates"]:
            v = "n/a" if g["value"] is None else (
                f"{g['value']:.4f}" if isinstance(g["value"], float) else g["value"])
            print(f"        [{g['status']:4}] {g['gate']:34} {v} "
                  f"{g['op']} {g['threshold']}")
        if self.verbose and res.get("detail"):
            print(f"        detail: {json.dumps(res['detail'], default=str)[:400]}")


def run(only_from: Optional[str] = None, resume: bool = False,
        force: bool = False, quiet: bool = False, verbose: bool = False,
        stages: Optional[List[Stage]] = None) -> dict:
    term = Term(quiet, verbose)
    git = git_state()
    prev = load_state() if resume and not force else {}
    started = time.time()
    run_id = time.strftime("%Y%m%dT%H%M%S", time.gmtime(started))

    run_token = new_run_token()
    seq = stages if stages is not None else ordered_stages()
    if only_from:
        names = [s.name for s in seq]
        if only_from not in names:
            raise SystemExit(f"unknown stage {only_from!r}; have {names}")
        seq = seq[names.index(only_from):]

    ctx: dict = {}
    results: List[dict] = []
    halted_by: Optional[str] = None
    new_state = dict(prev)

    if not quiet:
        print("=" * 72)
        print(f"ORCHESTRATOR  run {run_id}  git "
              f"{(git['sha'] or '?')[:8]}{'  DIRTY' if git['dirty'] else ''}")
        if git["dirty"]:
            print("  tree is dirty: every cached fingerprint is invalidated, "
                  "because we cannot know what changed")
        print("=" * 72)

    for i, st in enumerate(seq, 1):
        fp = fingerprint(st, git, run_token)
        base = {"stage": st.name, "gates": [], "seconds": 0.0,
                "fingerprint": fp, "detail": None, "skip_reason": None}

        # No downstream stage runs after an upstream gate failed. A green
        # end-to-end report built on a broken extraction is the worst output
        # this could produce.
        if halted_by:
            base.update(status=SKIPPED,
                        skip_reason=f"upstream stage '{halted_by}' failed")
            results.append(base)
            term.stage_start(st, i, len(seq))
            term.stage_end(base)
            continue

        missing = st.missing_inputs()
        if missing:
            base.update(status=FAIL, skip_reason=None,
                        gates=[_gate(f"{st.name}_inputs_present", 0, "==", 1,
                                     st.name)],
                        detail={"missing_inputs": missing})
            term.stage_start(st, i, len(seq))
            term.stage_end(base)
            results.append(base)
            halted_by = st.name
            continue

        if resume and not force and prev.get(st.name, {}).get("fingerprint") == fp \
                and prev.get(st.name, {}).get("status") == PASS:
            base.update(status=SKIPPED,
                        skip_reason="inputs unchanged since the last passing run",
                        gates=prev[st.name].get("gates", []))
            term.stage_start(st, i, len(seq))
            term.stage_end(base)
            results.append(base)
            continue

        term.stage_start(st, i, len(seq))
        t0 = time.perf_counter()
        try:
            out = st.run(ctx)
        except Exception as exc:
            out = {"error": f"{type(exc).__name__}: {exc}",
                   "gates": [_gate(f"{st.name}_completed", 0, "==", 1, st.name)]}
        base["seconds"] = round(time.perf_counter() - t0, 2)

        if out.get("skip"):
            base.update(status=SKIPPED, skip_reason=out["skip"])
        else:
            base["gates"] = out.get("gates", [])
            base["detail"] = out.get("detail") or out.get("error")
            failed = [g for g in base["gates"] if g["status"] == FAIL]
            base["status"] = FAIL if failed else PASS
            if failed:
                halted_by = st.name

        term.stage_end(base)
        results.append(base)
        new_state[st.name] = {"fingerprint": fp, "status": base["status"],
                              "gates": base["gates"]}

    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y%m%dT%H%M%S", time.gmtime(started)),
        "git": git,
        "kind": "orchestrator",
        "stages": results,
        "gates": [g for r in results for g in r["gates"]],
        # Measurements that are NOT scored over qa_set cases and so have no
        # place in an eval run record -- today, the symptom matcher's entry
        # metrics. Same {value, n, higher_is_better} shape compare.py consumes,
        # so they diff with the existing machinery rather than new machinery.
        "metrics": ctx.get("symptom_metrics", {}),
        "regression_counts": regression_counts(),
        "known_gaps": known_gaps(),
        "splits": splits(),
        "halted_by": halted_by,
        # Timings live apart from everything else so wall-time variance can
        # never break a determinism comparison.
        "timings": {r["stage"]: r["seconds"] for r in results},
        "wall_seconds": round(time.time() - started, 2),
    }
    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{run_id}_orchestrator.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    _append_index(record, path)
    save_state(new_state)

    if not quiet:
        summary(record)
    return record


def _append_index(record: dict, path: str) -> None:
    row = {
        "run_id": record["run_id"], "timestamp": record["timestamp"],
        "kind": "orchestrator",
        "git_sha": record["git"]["sha"], "git_dirty": record["git"]["dirty"],
        "stages_pass": sum(1 for s in record["stages"] if s["status"] == PASS),
        "stages_fail": sum(1 for s in record["stages"] if s["status"] == FAIL),
        "stages_skipped": sum(1 for s in record["stages"] if s["status"] == SKIPPED),
        "gates_pass": sum(1 for g in record["gates"] if g["status"] == PASS),
        "gates_fail": sum(1 for g in record["gates"] if g["status"] == FAIL),
        "halted_by": record["halted_by"],
        "human_verified": record["known_gaps"]["human_verified"],
        "wall_seconds": record["wall_seconds"],
        "run_file": os.path.relpath(path, REPO_ROOT).replace("\\", "/"),
    }
    os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)
    with open(INDEX_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summary(rec: dict) -> None:
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'stage':18}{'status':9}{'secs':>8}  gates")
    for s in rec["stages"]:
        gp = sum(1 for g in s["gates"] if g["status"] == PASS)
        gf = sum(1 for g in s["gates"] if g["status"] == FAIL)
        note = f"  -- {s['skip_reason']}" if s["skip_reason"] else ""
        print(f"{s['stage']:18}{s['status']:9}{s['seconds']:>8.1f}  "
              f"{gp} pass / {gf} fail{note}")

    failed = [g for g in rec["gates"] if g["status"] == FAIL]
    if failed:
        print("\nFAILED GATES")
        for g in failed:
            print(f"  {g['stage']}.{g['gate']}: {g['value']} "
                  f"{g['op']} {g['threshold']}")

    rc = rec["regression_counts"]
    if rc:
        print(f"\nREGRESSION COUNTS  {rc['files']} / {rc['steps']} / "
              f"{rc['measurements']} / {rc['branches']} / {rc['pointer_only']} / "
              f"{rc['column_split']} / {rc['crossrefs']} / {rc['format_split']}")

    g = rec["known_gaps"]
    print("\nKNOWN GAPS  (always shown; a gap that stops being visible becomes "
          "an unknown gap)")
    print(f"  human_verified    : {g['human_verified']}"
          f"    <- human transcription round: {g['human_round']}")
    print(f"  resolver_verified : {g['resolver_verified']}")
    print(f"  unverified        : {g['unverified']}")
    print(f"  unresolved facts  : {g['unresolved_facts']}")
    sp = rec.get("splits") or {}
    if sp:
        print(f"  train / holdout   : {sp.get('train')} / {sp.get('holdout')}")

    n_fail = len(failed)
    print(f"\n{'FAILED' if n_fail or rec['halted_by'] else 'OK'}: "
          f"{len(rec['gates']) - n_fail}/{len(rec['gates'])} gates pass"
          + (f", halted at '{rec['halted_by']}'" if rec["halted_by"] else ""))


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Deterministic pipeline orchestrator")
    ap.add_argument("--resume", action="store_true",
                    help="rerun only stages whose inputs changed, plus downstream")
    ap.add_argument("--force", action="store_true", help="rerun every stage")
    ap.add_argument("--from", dest="only_from", default=None,
                    help="start at this stage")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--graph", action="store_true", help="print the stage graph")
    args = ap.parse_args()

    if args.graph:
        print(stage_graph())
        return 0

    rec = run(only_from=args.only_from, resume=args.resume, force=args.force,
              quiet=args.quiet, verbose=args.verbose)
    bad = sum(1 for g in rec["gates"] if g["status"] == FAIL)
    return 1 if (bad or rec["halted_by"]) else 0


def stage_graph() -> str:
    lines = ["STAGE GRAPH  (dependency order; a gate failure halts everything "
             "downstream)", ""]
    for s in ordered_stages():
        dep = " <- " + ", ".join(s.depends) if s.depends else ""
        lines.append(f"  {s.name}{dep}")
        lines.append(f"      inputs  : {', '.join(s.inputs)}")
        lines.append(f"      outputs : {', '.join(s.outputs) or '(none)'}")
        lines.append(f"      gates   : {s.gates or '(none)'}")
        lines.append(f"      ~{s.expected_s:.0f}s"
                     + ("   needs KOMATSU_PDF" if s.needs_pdf else ""))
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    sys.exit(main())
