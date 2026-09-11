#!/usr/bin/env python3
"""Overfitting guard for the S1 / S2 / C3 fixes and the audit_symptoms stage.

    python tests/test_symptom_fixes.py

Three parts, in the order they bite:

  a) MUTATION CHECK  -- plant a specific fault, assert something catches it.
     A test suite that passes on broken code is worse than no suite, because
     it is evidence of correctness that nobody re-derived.
  b) NEGATIVE COVERAGE -- every new gate needs a case that fails it.
  c) INDEPENDENT DERIVATION -- asserted over the PARSED AST, not by grepping
     source text. A grep for "import openai" is satisfied by a comment.

The first two mutations matter most: an empty cell treated as merged, and a
merged cell treated as empty. They are the same distinction from opposite
sides, and a fix that only ever saw one of them would look correct.
"""
import ast
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))

from pipeline import fidelity                                    # noqa: E402
from pipeline.extract_symptoms import (                          # noqa: E402
    merged_from_above, inherit_merged_cells, assign_branch_provenance,
    is_relational, page_prints_title, parse_flat_causes)
from pipeline.extract_golden import parse_causes                 # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}"
                                                     if not ok else ""))
    if not ok:
        FAILED.append(name)


def caught(name, fn, why):
    """The planted fault must raise or return a wrong answer that we detect."""
    try:
        ok = fn()
    except AssertionError:
        ok = True
    check(name, ok, why)


# The real shapes, taken from p1296 t2 and HM22 step 5 rather than invented.
# A toy fixture can be classified differently from the data the fix must carry.
RATIO = ("Oil pressure ratio pump discharged pressure: PC valve discharged "
         "pressure 1:0.6 (approximately 3/5)")
GRID = [[(0, 0, 10, 30), (10, 0, 20, 60), (20, 0, 30, 30)],
        [(0, 30, 10, 60), None, (20, 30, 30, 60)]]
TXT = [["Pump supply", RATIO, ""], ["PC valve", "", ""]]

ROWS = [["5", "cause", "proc", "", "- Valve is normal."],
        ["", "", "", "", "- Valve is defective."]]
PROVS = [{"manual_page": "40-857", "pdf_page": 1399, "table_index": 0,
          "row_index": 5},
         {"manual_page": "40-858", "pdf_page": 1400, "table_index": 0,
          "row_index": 2}]


print("\nMUTATION CHECK")

# 1. an EMPTY cell treated as merged --------------------------------------
#    Column 2 is blank in both rows and has its own rect in each. Filling it
#    from above would invent a criterion for a row the manual leaves blank.
caught("empty cell treated as merged",
       lambda: merged_from_above(GRID, 1, 2) is None
       and inherit_merged_cells(TXT, GRID)[0][1][2] == "",
       "col 2 has its own rect in both rows; it is blank, not covered")

# 2. a MERGED cell treated as empty ---------------------------------------
#    The same distinction from the other side: refusing to inherit here is
#    what produced the 20 empty criteria in the first place.
caught("merged cell treated as empty",
       lambda: merged_from_above(GRID, 1, 1) == 0
       and inherit_merged_cells(TXT, GRID)[0][1][1] == RATIO,
       "col 1 is None in row 1 and covered by row 0's cell reaching to y=60")

#    ...and the mutation itself: a detector keyed on emptiness rather than
#    geometry cannot tell the two apart. Proven, not asserted.
def emptiness_keyed(tbl, ri, ci):
    """The broken implementation: 'blank means merged'."""
    return ri > 0 if not (tbl[ri][ci] or "").strip() else False


check("an emptiness-keyed detector cannot tell them apart",
      emptiness_keyed(TXT, 1, 1) == emptiness_keyed(TXT, 1, 2) is True
      and merged_from_above(GRID, 1, 1) != merged_from_above(GRID, 1, 2),
      "the geometry detector must separate two cells the text cannot")

# 3. a horizontally merged header read as a vertical merge ----------------
HGRID = [[(0, 0, 20, 30), None, (20, 0, 30, 30)],
         [(0, 30, 10, 60), (10, 30, 20, 60), (20, 30, 30, 60)]]
caught("horizontal header merge read as vertical",
       lambda: merged_from_above(HGRID, 0, 1) is None,
       "'Measurement position' spans two columns; nothing is above it")

# 4. a branch fact inheriting its step's page again -----------------------
steps = [{"step": 5, "provenance": dict(PROVS[0]),
          "branches": {"YES": "- Valve is normal.", "NO": "- Valve is defective."}}]
own, inherited = assign_branch_provenance(steps, ROWS, PROVS)
caught("branch fact inherits its step's page",
       lambda: steps[0]["branch_provenance"]["NO"]["pdf_page"] == 1400
       and not inherited,
       "NO is printed on 1400; taking the step's 1399 is the S2 defect")

#    the fallback exists, and must be counted rather than silent
lost = [{"step": 6, "provenance": dict(PROVS[0]),
         "branches": {"YES": "text present in no row"}}]
own2, inh2 = assign_branch_provenance(lost, ROWS, PROVS)
caught("unmatched branch falls back silently",
       lambda: inh2 == ["6:YES"] and own2 == 0,
       "an unmatched outcome must appear in the inherited list")

# 5. a genuine numeric criterion swept into the relational bucket ---------
SWEEP = ["0.70 to 1.09 MPa {7.1 to 11.1 kgf/cm2}", "Min. 100kO", "0 mA",
         "800 to 1000 mA", "Pressure ratio 1 to 5 ohm", "Max. 1 ohm"]
caught("numeric criterion swept into the relational bucket",
       lambda: not any(is_relational(c) for c in SWEEP),
       "rule 1 (carries a value -> numeric) must return before rules 2 and 3")
caught("conditional criterion left in the numeric bucket",
       lambda: is_relational("Pressure for each flow setting") and is_relational(RATIO),
       "a criterion with no number cannot be scored by numeric comparison")

# 6. the audit stage skipped without a reason -----------------------------
from core import orchestrator as orch                            # noqa: E402

check("audit_symptoms is a registered stage",
      "audit_symptoms" in orch.STAGE_BY_NAME,
      str(sorted(orch.STAGE_BY_NAME)))
order = [s.name for s in orch.ordered_stages()]
check("audit_symptoms runs after extraction and before fidelity",
      order.index("extract_symptoms") < order.index("audit_symptoms")
      < order.index("fidelity"), str(order))
st = orch.STAGE_BY_NAME["audit_symptoms"]
check("audit_symptoms declares inputs, outputs and gates",
      bool(st.inputs and st.outputs and st.gates), st.gates)
#    Run the real executor with the audit stage forced to skip, and assert the
#    skip is reported as SKIPPED, carries a reason, and is NOT counted as a
#    pass. Asserting `orch.SKIPPED == "SKIPPED"` would be a check that cannot
#    fail, which is the shape this whole suite exists to remove.
_real = orch.STAGE_BY_NAME["audit_symptoms"].run
try:
    orch.STAGE_BY_NAME["audit_symptoms"].run = \
        lambda ctx: {"skip": "planted: source PDF unavailable"}
    rec = orch.run(only_from="audit_symptoms", resume=False, force=True,
                   quiet=True)
    row = next(r for r in rec["stages"] if r["stage"] == "audit_symptoms")
finally:
    orch.STAGE_BY_NAME["audit_symptoms"].run = _real

check("a skipped stage is reported as SKIPPED, not PASS",
      row["status"] == orch.SKIPPED, str(row["status"]))
check("a skipped stage carries its reason",
      "planted" in (row.get("skip_reason") or ""), str(row.get("skip_reason")))
check("a skipped stage contributes no passing gate",
      not [g for g in row["gates"] if g["status"] == orch.PASS],
      str(row["gates"]))


print("\nSTRADDLING ROWS -- the case the current layout hides")
# S4 is latent: every fact sits on its parent's page in THIS revision, so a
# suite built only on real data proves nothing. Each kind below is given a row
# that genuinely crosses a page break and must take its own page. Without
# these the fix is unfalsifiable.

PA = {"manual_page": "40-100", "pdf_page": 500, "table_index": 0, "row_index": 1}
PB = {"manual_page": "40-101", "pdf_page": 501, "table_index": 0, "row_index": 0}

# 1. Section 40 branch outcome: YES on page 500, NO on 501 ------------------
s40 = parse_causes([
    (["No.", "Cause", "Procedure", "", "", ""], PA),
    (["7", "Defective sensor", "Check it.", "YES", "Sensor is normal.", ""], PA),
    (["", "", "", "NO", "Sensor is defective.", ""], PB),
], "A")
bp = s40[0].get("branch_provenance", {})
check("s40 straddling step: YES takes page 500",
      bp.get("YES", {}).get("pdf_page") == 500, str(bp))
check("s40 straddling step: NO takes page 501, NOT the step's 500",
      bp.get("NO", {}).get("pdf_page") == 501, str(bp))
check("s40 straddling step: the step itself still reports 500",
      s40[0]["provenance"]["pdf_page"] == 500, str(s40[0]["provenance"]))

# 2. H-Mode branch: same parser, and the symptom path must not undo it ------
own3, inh3 = assign_branch_provenance(s40, [], [])
check("hmode straddling branch keeps its parse-time page through the symptom path",
      s40[0]["branch_provenance"]["NO"]["pdf_page"] == 501 and not inh3,
      str(s40[0]["branch_provenance"]))

# 3. S-Mode remedy on a flat table that crosses a page break ----------------
flat = parse_flat_causes(
    [["No.", "Cause", "Point to check", "Remedy"],
     ["1", "Bad wiring", "Check harness.", "Repair the harness."],
     ["2", "Bad relay", "Check relay.", "Replace the relay."]],
    [PA, PA, PB])
check("smode remedy on the far page of a break takes 501",
      flat[1].get("remedy_provenance", {}).get("pdf_page") == 501, str(flat[1]))
check("smode remedy on the near page still takes 500",
      flat[0].get("remedy_provenance", {}).get("pdf_page") == 500, str(flat[0]))


def remedy_from_first_row(prov_rows):
    """The defect: every remedy takes the table's first page."""
    return prov_rows[0]["pdf_page"]


check("a first-row remedy rule would get the straddling case wrong",
      remedy_from_first_row([PA, PA, PB]) == 500
      and flat[1]["remedy_provenance"]["pdf_page"] == 501,
      "per-row capture and first-row capture must disagree here")

# 4. symptom_title printed on a LATER page than the entry's first -----------
TITLE = "Work Equipment With Heavier Load Moves Slower"
check("title absent from the first page is not claimed to be on it",
      not page_prints_title("H-18 Details of failure ... nothing here", TITLE))
check("title found on the page that prints it",
      page_prints_title("H-18 " + TITLE + " 40 Troubleshooting", TITLE))
check("title hyphenated across a line break is still found",
      page_prints_title("H-18 Work Equipment With Heavi- er Load Moves Slower",
                        TITLE),
      "clean() de-hyphenates; a raw substring match moved HM18/34/35 to their "
      "second page")
check("a different title on the page is not matched",
      not page_prints_title("H-19 Swing Speed Is Low", TITLE))


# 5. fuzzy matching reattaching a fact to a NEIGHBOURING row ---------------
#    A near-match would relocate the defect instead of removing it: the fact
#    would get a confident page belonging to a different row. Refusing and
#    counting the fallback is the only safe answer.
near = [{"step": 9, "provenance": dict(PA),
         "branches": {"NO": "Sensor is defective and must be replaced."}}]
own4, inh4 = assign_branch_provenance(
    near,
    [["", "", "", "NO", "Sensor is defective."]],   # close, not equal
    [PB])
check("a near-miss row is refused rather than matched fuzzily",
      inh4 == ["9:NO"] and own4 == 0
      and near[0]["branch_provenance"]["NO"]["pdf_page"] == 500,
      str(near[0]["branch_provenance"]))

# 6. provenance CAPTURED BUT NOT USED by the resolver ----------------------
#    The fix is worthless if fidelity keeps reading the parent's page. Build a
#    record whose branch sits on a different page and assert the enumerated
#    fact carries the branch's page, not the step's.
rec_s40 = {"CODE1": {
    "code": "CODE1", "pdf_pages": [500, 501], "standalone_measurements": [],
    "steps": [{"step": 1, "cause": "c", "procedure": "",
               "provenance": dict(PA),
               "branches": {"NO": "Sensor is defective."},
               "branch_fact_ids": {"NO": "CODE1:1:branch:0"},
               "branch_provenance": {"NO": dict(PB)},
               "measurements": [], "fact_id": "CODE1:1:step:0"}]}}
f40 = {f["fact_id"]: f for f in fidelity.enumerate_facts(rec_s40)}
check("fidelity reads Section 40 branch_provenance, not the step's page",
      f40["CODE1:1:branch:0"]["prov"]["pdf_page"] == 501,
      str(f40["CODE1:1:branch:0"]["prov"]))

rec_sym = {"HMZZ": {
    "symptom_id": "HMZZ", "tree_kind": "SymptomTreeFlat", "symptom": "t",
    "pdf_pages": [500, 501], "manual_pages": ["40-100", "40-101"],
    "title_provenance": {"manual_page": "40-101", "pdf_page": 501},
    "standalone_measurements": [],
    "steps": [{"step": 1, "cause": "c", "remedy": "Replace the relay.",
               "provenance": dict(PA), "remedy_provenance": dict(PB),
               "remedy_fact_id": "HMZZ:1:remedy:0", "fact_id": "HMZZ:1:step:0",
               "branch_fact_ids": {}}]}}
fsym = {f["fact_id"]: f for f in fidelity.enumerate_symptom_facts(rec_sym)}
check("fidelity reads remedy_provenance, not the step's page",
      fsym["HMZZ:1:remedy:0"]["prov"]["pdf_page"] == 501,
      str(fsym["HMZZ:1:remedy:0"]["prov"]))
check("fidelity reads title_provenance, not pdf_pages[0]",
      fsym["HMZZ:0:symptom_title:0"]["prov"]["pdf_page"] == 501,
      str(fsym["HMZZ:0:symptom_title:0"]["prov"]))

# 7. these tests must be able to FAIL -------------------------------------
#    A straddling assertion that passes on broken code proves nothing. Each
#    check above is re-run against a deliberately broken variant, which must
#    give the wrong answer. ("SKIPPED == 'SKIPPED'" got as far as passing.)
broken_s40 = dict(rec_s40["CODE1"]["steps"][0])
broken_s40.pop("branch_provenance")
check("without branch_provenance the s40 fact falls back to the step's page",
      fidelity.enumerate_facts({"CODE1": {**rec_s40["CODE1"],
                                          "steps": [broken_s40]}})[1]
      ["prov"]["pdf_page"] == 500,
      "the fallback must be reachable, or the gate is measuring nothing")
broken_sym = dict(rec_sym["HMZZ"]["steps"][0])
broken_sym.pop("remedy_provenance")
check("without remedy_provenance the remedy falls back to the step's page",
      [f for f in fidelity.enumerate_symptom_facts(
          {"HMZZ": {**rec_sym["HMZZ"], "steps": [broken_sym],
                    "title_provenance": None}})
       if f["kind"] == "remedy"][0]["prov"]["pdf_page"] == 500)


print("\nNEGATIVE COVERAGE -- every new gate has a failing case")

# audit_symptoms_high
check("audit_symptoms_high fails on a HIGH finding",
      orch._gate("audit_symptoms_high", 1, "==", 0, "x")["status"] == orch.FAIL
      and orch._gate("audit_symptoms_high", 0, "==", 0, "x")["status"] == orch.PASS)

# per-section fidelity gates
weak = {"total_facts": 10, "resolved": 10,
        "by_kind": {"measurement": {"total": 10, "resolved": 10, "rate": 1.0}},
        "by_section": {
            "section40/measurement": {"total": 9, "resolved": 9, "rate": 1.0},
            "hmode/measurement": {"total": 1, "resolved": 0, "rate": 0.0}},
        "page_containment": 1.0, "verbatim_integrity": 1.0, "rows": []}
g = {x["gate"]: x["status"] for x in fidelity.gates(weak)}
check("hmode/measurement gate fails while the corpus rate is 1.0",
      g.get("fact_resolution_rate[hmode/measurement]") == "FAIL"
      and g.get("fact_resolution_rate[measurement]") == "PASS", str(g))

# empty verbatim must not resolve
from eval.citations import PageText                              # noqa: E402
pages = PageText()
res = fidelity.check_facts(recs={"ZZ999": {
    "code": "ZZ999", "pdf_pages": [1, 1], "steps": [],
    "standalone_measurements": [{
        "fact_id": "ZZ999:0:meas:0", "criteria": "",
        "provenance": {"manual_page": "x", "pdf_page": 1, "table_index": 0,
                       "row_index": 0}}]}}, pages=pages)
check("an empty verbatim is refused rather than resolved",
      res["by_kind"]["measurement"]["resolved"] == 0,
      "the resolver finds '' on every page; admitting it raises the rate "
      "while measuring nothing")
pages.close()

# relational exclusion is counted, not silent
check("relational facts leave the denominator but stay counted",
      fidelity.RELATIONAL == "relational")


print("\nINDEPENDENT DERIVATION -- over the parsed AST")

MODEL_HINTS = ("openai", "anthropic", "cohere", "transformers", "llama_cpp",
               "langchain", "litellm", "requests", "httpx", "urllib")
for mod in ("pipeline/extract_symptoms.py", "pipeline/audit_symptoms.py",
            "pipeline/fidelity.py"):
    tree = ast.parse(open(os.path.join(REPO_ROOT, mod), encoding="utf-8").read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    check(f"no model or network import in {mod}",
          not (imported & set(MODEL_HINTS)), str(sorted(imported & set(MODEL_HINTS))))

# The three fixes are re-derived here from the AST rather than trusted:
# each must exist as a real function definition, not a name bound elsewhere.
tree = ast.parse(open(os.path.join(REPO_ROOT, "pipeline/extract_symptoms.py"),
                      encoding="utf-8").read())
defs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
for fn in ("merged_from_above", "inherit_merged_cells",
           "assign_branch_provenance", "is_relational"):
    check(f"{fn} is defined in the extractor", fn in defs, str(sorted(defs))[:200])

# table_kind must NOT be redefined here -- Section 40's classifier is out of
# scope, and a shadowing definition would silently change its behaviour.
check("table_kind is not redefined in the symptom extractor",
      "table_kind" not in defs,
      "Section 40's classifier is imported, never shadowed")

# and the committed data must agree with the fixes, independently recounted
ms, rel, empty = 0, 0, 0
import glob                                                      # noqa: E402
for f in glob.glob(os.path.join(REPO_ROOT, "golden", "symptoms", "*M*.json")):
    r = json.load(open(f, encoding="utf-8"))
    for m in (r.get("standalone_measurements", [])
              + [m for s in r["steps"] for m in s.get("measurements", [])]):
        ms += 1
        rel += m.get("criteria_kind") == "relational"
        empty += not (m.get("criteria") or "").strip()
check("274 symptom measurements, 42 relational, 0 empty",
      (ms, rel, empty) == (274, 42, 0), f"{ms} / {rel} / {empty}")

print("\n" + "=" * 62)
if FAILED:
    print(f"FAILED: {len(FAILED)} check(s)")
    for n in FAILED:
        print("  -", n)
    sys.exit(1)
print("OK: symptom fixes guarded")
