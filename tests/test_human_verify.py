#!/usr/bin/env python3
"""
test_human_verify.py
MODULE, PIPELINE and E2E for the human-verified holdout, plus the overfitting
guard.

The fault that matters most here is a PRE-FILLED TEMPLATE. A form showing the
machine's answer produces agreement rather than verification, looks identical to
success, and would turn this entire exercise into a rubber stamp.
"""
import ast
import copy
import json
import os
import shutil
import sys
import tempfile
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from agent import holdout as metric_holdout           # noqa: E402
from agent import tools                               # noqa: E402
from core import loader                               # noqa: E402
from pipeline import fidelity, human_kit, human_select, human_verify  # noqa: E402

failures, timings = [], {}


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


# ============================================================ MODULE
print("\nMODULE")
t0 = time.perf_counter()

# --- reclassification
try:
    fidelity.self_test()
    check("fidelity self-tests pass (incl. overhang reclassification)", True)
except AssertionError as exc:
    check("fidelity self-tests pass", False, str(exc))

unres = json.load(open(os.path.join(REPO_ROOT, "reports",
                                    "unresolved_facts.json"), encoding="utf-8"))
check("unresolved_facts.json still holds 39", unres["total_unresolved"] == 39,
      str(unres["total_unresolved"]))
check("no fact is classified DEFECT",
      unres["by_classification"].get("DEFECT", 0) == 0,
      str(unres["by_classification"]))
check("12 KNOWN_LIMITATION and 27 NEEDS_HUMAN_VERIFICATION",
      unres["by_classification"].get("KNOWN_LIMITATION") == 12
      and unres["by_classification"].get("NEEDS_HUMAN_VERIFICATION") == 27,
      str(unres["by_classification"]))
oh = [i for i in unres["items"] if "overhang_pt" in i]
check("each overhang entry records the true text and the overhang",
      len(oh) == 9 and all(i.get("true_text") for i in oh),
      f"{len(oh)} entries carry an overhang measurement")
check("the reason names the precedent it follows",
      all("F@BBZL" in i["reason"] and "D8ARKR" in i["reason"] for i in oh))

# --- selection
try:
    human_select.self_test()
    check("selection is deterministic and excludes the metric holdout", True)
except AssertionError as exc:
    check("selection self-test", False, str(exc))

sel = human_select.select()
codes = {d["code"] for d in sel}
check("exactly 25 codes selected", len(sel) == 25, str(len(sel)))
check("selection repeats identically",
      [d["code"] for d in human_select.select()] == [d["code"] for d in sel])
check("disjoint from the 20% metric holdout",
      not (codes & metric_holdout.holdout_set()),
      f"overlap: {sorted(codes & metric_holdout.holdout_set())}")
check("covers pointer-only codes",
      sum(1 for d in sel if d["pointer_only"]) >= 2)
check("covers the 11-page extreme",
      max(len(d["pages"]) for d in sel) >= 11)
check("covers both stored formats", {d["format"] for d in sel} >= {"A", "B"})
check("includes every column-split code not in the metric holdout",
      {d["code"] for d in sel if d["column_split"]} ==
      {c for c, r in tools.records().items()
       if c not in metric_holdout.holdout_set()
       and any(s.get("extraction_warning") == "column_split_recovered"
               for s in r["steps"])})

# --- template generation
try:
    human_kit.self_test()
    check("assert_blank detects a pre-filled template", True)
except AssertionError as exc:
    check("assert_blank detects a pre-filled template", False, str(exc))

recs = tools.records()
tmpl = human_kit.blank_template("CA451", recs["CA451"])
flat = json.dumps(tmpl, ensure_ascii=False)
check("template carries no extracted value",
      "Defective wiring harness" not in flat and "Sensor output" not in flat)
check("template carries the right structure",
      len(tmpl["steps"]) == len(tools.real_steps(recs["CA451"])))
check("the worked example is not one of the 25",
      human_select.example_code() not in codes)

# --- comparator
try:
    human_verify.self_test()
    check("comparator self-tests pass", True)
except AssertionError as exc:
    check("comparator self-tests pass", False, str(exc))

check("a changed digit never normalises away",
      human_verify.compare("Min. 90kΩ", "Min. 100kΩ") == human_verify.DISAGREE)
check("whitespace and omega fold",
      human_verify.compare("max.  1 ohm", "Max. 1 Ω") == human_verify.NORMALISED)
check("line-break hyphenation folds",
      human_verify.compare("har- ness", "harness") == human_verify.NORMALISED)
check("a genuine hyphen survives",
      human_verify.compare("well-known", "wellknown") == human_verify.DISAGREE)
timings["module"] = time.perf_counter() - t0


# ========================================================== PIPELINE
print("\nPIPELINE")
t0 = time.perf_counter()
from eval.citations import PageText                   # noqa: E402
pages = PageText()
if pages.available():
    res = fidelity.check_facts(pages=pages)
    for g in fidelity.gates(res):
        check(f"gate {g['gate']} ({g['value']:.4f})", g["status"] == "PASS",
              g.get("detail", ""))
    print(f"  {'kind':14}{'rate':>9}")
    for k, v in res["by_kind"].items():
        print(f"  {k:14}{v['rate']:>9.4f}")
    # eight regression counts
    fmt = {}
    for r in recs.values():
        fmt[r["format"]] = fmt.get(r["format"], 0) + 1
    counts = {
        "files": len(recs),
        "steps": sum(len(r["steps"]) for r in recs.values()),
        "measurements": sum(len(r["standalone_measurements"]) for r in recs.values())
        + sum(len(s["measurements"]) for r in recs.values() for s in r["steps"]),
        "branches": sum(len(s["branches"]) for r in recs.values() for s in r["steps"]),
        "pointer": sum(1 for r in recs.values() if r["is_pointer_only"]),
        "colsplit": sum(1 for r in recs.values() for s in r["steps"]
                        if s.get("extraction_warning") == "column_split_recovered"),
        "xref": sum(len(r["refs_failure_codes"]) for r in recs.values()),
        "fmt": f"{fmt.get('A')} A / {fmt.get('B')} B"}
    want = {"files": 174, "steps": 996, "measurements": 872, "branches": 1437,
            "pointer": 9, "colsplit": 27, "xref": 94, "fmt": "117 A / 57 B"}
    check("all eight regression counts unchanged", counts == want,
          json.dumps({k: (want[k], counts[k]) for k in want if want[k] != counts[k]}))
else:
    print("  SKIP  PDF unavailable")
timings["pipeline"] = time.perf_counter() - t0


# =============================================================== E2E
print("\nE2E -- human_verified must reach the wire")
t0 = time.perf_counter()
from fastapi.testclient import TestClient             # noqa: E402
from api.app import create_app                        # noqa: E402
from api.store import SessionStore                    # noqa: E402

# Inject a human_verified fact so the path is exercised end to end. Nothing is
# written to golden/ -- verification is additive metadata held beside it.
base_ver = loader.load_verification()
tmp_ver = copy.deepcopy(base_ver)
tmp_ver["facts"]["CA451:1:step:0"] = loader.HUMAN_VERIFIED
ver_path = os.path.join(REPO_ROOT, "reports", "fact_verification.json")
backup = open(ver_path, encoding="utf-8").read()
try:
    with open(ver_path, "w", encoding="utf-8") as f:
        json.dump(tmp_ver, f)
    client = TestClient(create_app(store=SessionStore(max_sessions=50,
                                                      max_turns=50,
                                                      rate_limit=9999)))
    b = client.post("/sessions", json={"message": "CA451 aa raha hai"}).json()
    sid = b["session_id"]
    seen = []
    for i, m in enumerate(["PC200-10M0, serial 700123", "koi aur code nahi"],
                          start=1):
        r = client.post(f"/sessions/{sid}/messages",
                        json={"message": m, "turn_index": i}).json()
        seen += r.get("citations") or []
    statuses = {c["fact_id"]: c["verification"] for c in seen}
    check("a human_verified fact reaches the wire as human_verified",
          statuses.get("CA451:1:step:0") == "human_verified", str(statuses))
    check("the contract accepts all three statuses",
          set(statuses.values()) <= set(loader.STATUSES))
finally:
    with open(ver_path, "w", encoding="utf-8") as f:
        f.write(backup)
timings["e2e"] = time.perf_counter() - t0


# ================================================ OVERFITTING GUARD
print("\nOVERFITTING GUARD")

# a) independent derivation, over the AST -- structural.py's earlier version of
#    this check failed on its own docstring prose.
tree = ast.parse(open(os.path.join(REPO_ROOT, "pipeline", "human_verify.py"),
                      encoding="utf-8").read())
imported, names = set(), set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported |= {a.name.split(".")[0] for a in node.names}
    elif isinstance(node, ast.ImportFrom):
        imported.add((node.module or "").split(".")[0])
        imported |= {a.name for a in node.names}
    elif isinstance(node, ast.Name):
        names.add(node.id)
    elif isinstance(node, ast.Attribute):
        names.add(node.attr)
banned_mod = {"extract_golden", "pdfplumber", "citations"}
banned_name = {"parse_causes", "table_kind", "clean", "real_steps",
               "render_citations", "_fuse_split_decimals", "resolve"}
shared = sorted((imported & banned_mod) | (names & banned_name))
check("comparator shares no helper with the extractor or resolver", not shared,
      f"actually imports/calls: {shared}")
print(f"    comparator imports: {sorted(imported - {'annotations'})}")

# b) mutation check
print("\n  MUTATION TABLE")
rec = recs["CA451"]
steps = tools.real_steps(rec)


def transcribe_truth():
    """A perfect transcription -- used only as the base to plant faults into."""
    t = human_kit.blank_template("CA451", rec)
    t["transcriber"], t["date"] = "tester", "2026-09-11"
    for i, s in enumerate(steps):
        t["steps"][i]["cause"] = s.get("cause") or ""
        t["steps"][i]["procedure"] = s.get("procedure") or ""
        for j, m in enumerate(s.get("measurements") or []):
            for f in ("quantity", "point", "criteria"):
                t["steps"][i]["measurements"][j][f] = m.get(f) or ""
    return t


muts = []


def mut(name, fn):
    muts.append((name, fn))


mut("digit_differs", lambda t: _set_crit(t, lambda v: v.replace("0.2", "0.9")))
mut("whitespace_only", lambda t: _set_crit(t, lambda v: "  " + v.replace(" ", "  ")))
mut("prefilled_template", None)
mut("verified_without_transcription", None)
mut("disagreement_auto_resolved", None)


def _set_crit(t, f):
    for st in t["steps"]:
        for m in st["measurements"]:
            if m.get("criteria"):
                m["criteria"] = f(m["criteria"])
    return t


results = []
for name, fn in muts:
    caught = []
    if name == "prefilled_template":
        bad = human_kit.blank_template("CA451", rec)
        bad["steps"][0]["cause"] = steps[0].get("cause")
        try:
            human_kit.assert_blank(bad)
        except AssertionError:
            caught.append("assert_blank")
    elif name == "verified_without_transcription":
        # A template with no transcriber must not be loaded as a transcription,
        # so it can never yield a human_verified fact.
        d = tempfile.mkdtemp()
        try:
            os.makedirs(os.path.join(d, "CA451"))
            with open(os.path.join(d, "CA451", "CA451.json"), "w",
                      encoding="utf-8") as f:
                json.dump(human_kit.blank_template("CA451", rec), f)
            if not human_verify.load_transcriptions(d):
                caught.append("load_transcriptions")
        finally:
            shutil.rmtree(d, ignore_errors=True)
    elif name == "disagreement_auto_resolved":
        rows = [{"fact_id": "CA451:6:meas:0", "field": "criteria",
                 "kind": "measurement", "verdict": human_verify.DISAGREE,
                 "adjudication": None}]
        human_verify.apply_adjudications(rows, {})
        if rows[0]["adjudication"] == human_verify.UNADJUDICATED and \
                not human_verify.verified_fact_ids(rows):
            caught.append("unadjudicated_stays_unadjudicated")
    else:
        t = fn(transcribe_truth())
        rows = human_verify.diff_code(t, rec)
        dis = [r for r in rows if r["verdict"] == human_verify.DISAGREE]
        if name == "digit_differs" and dis:
            caught.append("comparator_disagreement")
        if name == "whitespace_only":
            # must NOT be a disagreement; must be absorbed as normalised
            if not dis and any(r["verdict"] == human_verify.NORMALISED
                               for r in rows):
                caught.append("normalised_not_flagged (correct behaviour)")
    results.append((name, caught))
    print(f"    {name:32} {'caught' if caught else '*** NOT CAUGHT ***':20} "
          f"{', '.join(caught) or '-'}")

uncaught = [n for n, c in results if not c]
check("every planted fault is caught", not uncaught, f"uncaught: {uncaught}")

# c) negative coverage for the new gate
bad_res = {"rows": [{"fact_id": f, "resolved": True}
                    for f in fidelity.CELL_OVERHANG_FACTS]}
check("known_overhang_stable has a failing case",
      fidelity.check_known_overhang(bad_res)["status"] == "FAIL")

# d) disjointness, asserted in code
check("transcription set and metric holdout are disjoint (asserted)",
      not (codes & metric_holdout.holdout_set()))

print("\n" + "=" * 62)
print(f"module {timings.get('module', 0):.1f}s  "
      f"pipeline {timings.get('pipeline', 0):.1f}s  e2e {timings.get('e2e', 0):.1f}s")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: reclassification + human-verified holdout")
sys.exit(0)
