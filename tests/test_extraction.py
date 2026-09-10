#!/usr/bin/env python3
"""
test_extraction.py
Regression guard for the golden dataset. Plain asserts, no framework.

    python tests/test_extraction.py        # exits 1 on any failure

Run this after every regeneration, before committing.

WHY THIS FILE EXISTS
--------------------
CA451 step 6 stores a common-rail pressure sensor range that pdfplumber splits
across two cells: 'Sensor output 0' + '.2 to 4.6V'. That split was repaired once.
The repair was verified once. Measurement detection was then rewritten to handle
the third cause-table layout, the repair was silently lost, and nothing
re-checked it -- the audit check that should have caught it (E4) required a digit
before the split point and so could never match a leading-dot fragment.

The result was a wrong voltage in the ground truth, propagated into the Q&A set
as a required verbatim answer, with a clean audit reported as proof of
correctness. README.md calls silent extraction loss "the real enemy" and says the
page-by-page check should be built into the pipeline rather than done as a
one-off review. This is that check.
"""
import json
import glob
import os
import re
import sys
import collections

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))

failures = []


def check(name, condition, detail=""):
    if condition:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


# ---------------------------------------------------------------- load corpus
paths = sorted(glob.glob(os.path.join(GOLD, "failure_codes", "*.json")))
if not paths:
    sys.exit(f"ERROR: no failure-code JSON found under {GOLD}/failure_codes/\n"
             "  Run pipeline/extract_golden.py first.")

recs = {}
for p in paths:
    with open(p, encoding="utf-8") as f:
        r = json.load(f)
    recs[r["code"]] = r


def all_measurements(rec):
    out = list(rec["standalone_measurements"])
    for st in rec["steps"]:
        out += st["measurements"]
    return out


# ============================================ 1. the record this file exists for
print("\nCA451 step 6 -- the split decimal")

step6 = [s for s in recs["CA451"]["steps"] if s["step"] == 6]
check("CA451 has a step 6", len(step6) == 1)

if step6:
    ms = step6[0]["measurements"]
    check("CA451 step 6 has exactly one measurement", len(ms) == 1,
          f"got {len(ms)}")
    if ms:
        m = ms[0]
        check("CA451 step 6 point is intact",
              m["point"] == "Between ECM (25) and (47)",
              f"got {m['point']!r} -- an orphaned integer part in the measuring "
              f"point means the criterion cell was chosen before the decimal "
              f"was rejoined")
        check("CA451 step 6 criteria is intact",
              m["criteria"] == "Sensor output 0.2 to 4.6V",
              f"got {m['criteria']!r} -- expected 'Sensor output 0.2 to 4.6V'")

# ================================================== 2. the eight regression counts
print("\nregression counts (CLAUDE.md)")

fmt_a = sum(1 for r in recs.values() if r["format"] == "A")
fmt_b = sum(1 for r in recs.values() if r["format"] == "B")

counts = {
    "failure-code JSON files": (len(recs), 174),
    "total steps": (sum(len(r["steps"]) for r in recs.values()), 996),
    "total measurements": (sum(len(all_measurements(r)) for r in recs.values()), 872),
    "YES/NO branch outcomes": (
        sum(len(s["branches"]) for r in recs.values() for s in r["steps"]), 1437),
    "is_pointer_only codes": (
        sum(1 for r in recs.values() if r["is_pointer_only"]), 9),
    "column_split_recovered steps": (
        sum(1 for r in recs.values() for s in r["steps"]
            if s.get("extraction_warning") == "column_split_recovered"), 27),
    "cross-reference edges": (
        sum(len(r["refs_failure_codes"]) for r in recs.values()), 94),
    "format A count": (fmt_a, 117),
    "format B count": (fmt_b, 57),
}
for name, (got, want) in counts.items():
    check(f"{name} == {want}", got == want, f"got {got}")

# ================================================ 3. no orphaned decimal fragments
print("\nsplit-decimal corruption across the whole corpus")

leading_dot = [
    f"{c} :: {m['criteria']!r}"
    for c, r in recs.items() for m in all_measurements(r)
    if (m.get("criteria") or "").startswith(".")
]
check("no criteria string begins with '.'", not leading_dot,
      "; ".join(leading_dot[:5]))

# The integer part stranded in the measuring point looks like a bare digit at the
# end of the string, preceded by whitespace: 'Sensor output 0'. A digit that is
# part of a token ('Between R01 and R02') is a connector name, not an orphan.
orphan_digit = [
    f"{c} :: {m['point']!r}"
    for c, r in recs.items() for m in all_measurements(r)
    if re.search(r"(?<=\s)\d+$", m.get("point") or "")
]
check("no measuring point ends in an orphaned bare digit", not orphan_digit,
      "; ".join(orphan_digit[:5]))

# Belt and braces: the shape E4 looks for, checked here too so the guard does not
# depend on the audit having been run.
split_dec = [
    f"{c} :: {m['criteria']!r}"
    for c, r in recs.items() for m in all_measurements(r)
    if re.search(r"\d\s+\.\d|\d\.\s+\d|^\s*\.\d|\d\s+\.\s*\d", m.get("criteria") or "")
]
check("no criteria string has a split decimal", not split_dec,
      "; ".join(split_dec[:5]))

# ---- and prove the detector above can actually fail, on the known-bad value.
# A check that cannot fail on its own motivating example is not a check.
_probe = re.compile(r"\d\s+\.\d|\d\.\s+\d|^\s*\.\d|\d\s+\.\s*\d")
check("detector self-test: matches the known-bad '.2 to 4.6V'",
      bool(_probe.search(".2 to 4.6V")))
check("detector self-test: accepts the corrected 'Sensor output 0.2 to 4.6V'",
      not _probe.search("Sensor output 0.2 to 4.6V"))

# ============================================ 4. qa_set.json agrees with truth
# qa_set.json went stale once: the ground truth was corrected and the test set
# kept requiring the old corrupt value as a verbatim answer, so a correct
# assistant would have been marked wrong on a sensor voltage. Nothing detected
# that, because nothing compared the two. This does.
QA_PATH = os.path.join(GOLD, "qa_set.json")
if not os.path.isfile(QA_PATH):
    print("\nqa_set.json -- SKIPPED (not present; run eval/build_qa_set.py)")
else:
    print("\nqa_set.json agrees with the ground truth")
    with open(QA_PATH, encoding="utf-8") as f:
        qa = json.load(f)

    check("qa_set.json holds 1,330 cases", len(qa) == 1330, f"got {len(qa)}")
    check("all case ids are unique",
          len({c["id"] for c in qa}) == len(qa))

    # Every pinned verbatim string must be a criterion that currently exists in
    # golden/ for the code the case names. This is the staleness check.
    truth = {}
    for code, rec in recs.items():
        truth[code] = {m["criteria"] for m in all_measurements(rec)}

    orphans = []
    for c in qa:
        for v in c.get("must_contain_verbatim", []):
            src = c.get("source_code")
            if src not in truth or v not in truth[src]:
                orphans.append(f"{c['id']} ({src}) pins {v!r}")
    check("every must_contain_verbatim string exists in golden/", not orphans,
          "; ".join(orphans[:5]))

    # The specific shape that went wrong.
    frags = [f"{c['id']}: {v!r}" for c in qa
             for v in c.get("must_contain_verbatim", []) if v.startswith(".")]
    check("no pinned verbatim string is a decimal fragment", not frags,
          "; ".join(frags[:5]))

    # Refusal cases must not smuggle in a correct answer.
    leaky = [c["id"] for c in qa if c.get("must_refuse")
             and (c.get("must_contain") or c.get("must_contain_verbatim")
                  or c.get("must_cite_page"))]
    check("refusal cases carry no answer content", not leaky,
          ", ".join(leaky[:5]))

    counts = collections.Counter(c["type"] for c in qa)
    for t, want in (("numeric_exactness", 846), ("direct_lookup", 173),
                    ("step_ordering", 164), ("branch_following", 116),
                    ("precondition", 10), ("cross_ref_hop", 9),
                    ("adversarial_unknown", 6), ("adversarial_model", 6)):
        check(f"qa {t} == {want}", counts.get(t, 0) == want, f"got {counts.get(t,0)}")

# ==================================================================== summary
print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    print("\nA changed count is a behaviour change in the parser, not a new\n"
          "baseline. Investigate before committing regenerated data.")
    sys.exit(1)

print(f"OK: all checks passed over {len(recs)} codes")
sys.exit(0)
