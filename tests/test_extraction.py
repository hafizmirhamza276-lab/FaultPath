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

# ================================================= 4. span-level provenance
# A code spans up to 11 pages, so a citation to the code's first page is not a
# citation to the measurement. 75% of measurements are not on their code's
# first page. Provenance is captured at parse time from the page the row was
# read from; these checks pin that it stayed attached and stayed complete.
print("\nspan-level provenance (schema v2)")

check("every record declares schema_version 2",
      all(r.get("schema_version") == 2 for r in recs.values()),
      f"missing/old: {sorted(c for c, r in recs.items() if r.get('schema_version') != 2)[:5]}")

PROV_KEYS = {"manual_page", "pdf_page", "table_index", "row_index"}


def prov_ok(p):
    return isinstance(p, dict) and PROV_KEYS <= set(p) and p.get("pdf_page")


missing_prov = [f"{c} step {s['step']}" for c, r in recs.items()
                for s in r["steps"] if not prov_ok(s.get("provenance"))]
check("every step carries complete provenance", not missing_prov,
      "; ".join(missing_prov[:5]))

missing_mprov = [f"{c} {m.get('fact_id')}" for c, r in recs.items()
                 for m in all_measurements(r) if not prov_ok(m.get("provenance"))]
check("every measurement carries complete provenance", not missing_mprov,
      "; ".join(missing_mprov[:5]))

# Provenance must point inside the code's own page range, or it is pointing at
# another code's page -- worse than no citation.
outside = []
for c, r in recs.items():
    lo, hi = r["pdf_pages"]
    for s in r["steps"]:
        for p in [s.get("provenance")] + [m.get("provenance") for m in s["measurements"]]:
            if p and not (lo <= p["pdf_page"] <= hi):
                outside.append(f"{c}: {p['pdf_page']} not in {lo}-{hi}")
check("no provenance points outside its code's page range", not outside,
      "; ".join(outside[:5]))

# Fact ids: unique, and shaped <code>:<step>:<kind>:<index>.
fact_ids = [s["fact_id"] for r in recs.values() for s in r["steps"]]
fact_ids += [m["fact_id"] for r in recs.values() for m in all_measurements(r)]
check("every fact id is unique", len(fact_ids) == len(set(fact_ids)),
      f"{len(fact_ids)} ids, {len(set(fact_ids))} unique")
FACT_RE = re.compile(r"^[A-Z0-9@#]{4,7}:\d+:(step|meas|branch):\d+$")
malformed = [f for f in fact_ids if not FACT_RE.match(f)]
check("every fact id is well formed", not malformed, "; ".join(malformed[:5]))


# EVERY FACT ID IS RE-DERIVED FROM POSITION AND COMPARED.
#
# Uniqueness and shape were the only checks, plus ONE hand-picked spot. An id
# that is unique, well formed and pointing at the WRONG FACT passed all three.
#
# That matters more here than anywhere else in the repo. eval/citations.py
# resolves citations BY FACT ID, and the whole provenance claim is that the
# model names an id and code renders the page -- "what the model does not
# generate it cannot get wrong". A mismatched id does not fail loudly; it
# renders a confident, correctly formatted, WRONG citation.
#
# The id is defined as a function of position and nothing else, so the stored
# copy is a derived value and this is the derivation it must match. See
# reports/stored_derived_inventory.md.
def derived_fact_ids(rec):
    """Every fact id this record's structure implies, in the schema's order."""
    code = rec["code"]
    out = []
    for step in rec.get("steps") or []:
        n = step["step"]
        out.append((f"{code}:{n}:step:0", step.get("fact_id"), f"step {n}"))
        for i, m in enumerate(step.get("measurements") or []):
            out.append((f"{code}:{n}:meas:{i}", m.get("fact_id"),
                        f"step {n} measurement {i}"))
        bf = step.get("branch_fact_ids") or {}
        for i, outcome in enumerate(sorted(bf)):
            out.append((f"{code}:{n}:branch:{i}", bf[outcome],
                        f"step {n} branch {outcome}"))
    for i, m in enumerate(rec.get("standalone_measurements") or []):
        out.append((f"{code}:0:meas:{i}", m.get("fact_id"),
                    f"standalone measurement {i}"))
    return out


def fact_id_mismatches(records):
    return [f"{r['code']} {what}: stored {got!r}, position implies {want!r}"
            for r in records.values()
            for want, got, what in derived_fact_ids(r) if got != want]


_pairs = [p for r in recs.values() for p in derived_fact_ids(r)]
_mismatch = fact_id_mismatches(recs)
check(f"all {len(_pairs)} fact ids match the position they are derived from",
      not _mismatch, "; ".join(_mismatch[:5]))

# SELF-TEST. This check reports zero on a healthy repo, which is the exact
# shape audit checks E4 and H2 were in when their zeros were read as clean.
_planted = json.loads(json.dumps(recs[sorted(recs)[0]]))
if _planted.get("steps"):
    _planted["steps"][0]["fact_id"] = f"{_planted['code']}:99:step:0"
    check("fact id self-test: a mismatched id fails the run",
          bool(fact_id_mismatches({"x": _planted})),
          "the check cannot detect an id that points at the wrong fact")

# The record this whole chain of work exists for.
_ca451 = [s for s in recs["CA451"]["steps"] if s["step"] == 6]
if _ca451 and _ca451[0]["measurements"]:
    _m = _ca451[0]["measurements"][0]
    check("CA451 step 6 measurement cites 40-242, not the code's first page",
          _m["provenance"]["manual_page"] == "40-242",
          f"got {_m['provenance']['manual_page']!r}")
    check("CA451 step 6 measurement has fact id CA451:6:meas:0",
          _m.get("fact_id") == "CA451:6:meas:0", f"got {_m.get('fact_id')!r}")


# ============================================ 5. qa_set.json agrees with truth
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

    # ---- NO EXPECTATION MAY END MID-WORD ----------------------------------
    #
    # eval/build_qa_set.py USED TO slice branch outcomes with `outcome[:60]`,
    # so 153 branch_following cases carried a must_contain that was a
    # 60-character PREFIX of the real outcome, 134 of them cut MID-WORD:
    #
    #   qa_set: '• A wiring harness or connector is defective. • Repair or re'
    #   golden: '• A wiring harness or connector is defective. • Repair or
    #            replace the defective wiring harness or connector. • Go to
    #            "Confirmation of repair".'
    #
    # golden/ IS INTACT -- this is the test set under-requiring, not the ground
    # truth being wrong -- so the effect is permissive rather than incorrect: an
    # answer that stops after 60 characters passes a check the full answer also
    # passes. It still means "the whole expected answer" is not what is being
    # required, and nothing was looking.
    #
    # DERIVED, not a word list: an expectation is truncated when it is a STRICT
    # PREFIX of a string in its own golden record and the characters either side
    # of the cut are both alphanumeric. No heuristic about what a word is.
    #
    # GATED AT ZERO. It was PINNED at 134 for exactly one commit, while the
    # defect existed and its fix was still pending. Leaving a pin in place after
    # the fix would permit the defect's return: a pin says "no worse", and what
    # is wanted now is "none".
    #
    # Covers BOTH sections. `recs` in this file is the 174 failure codes only,
    # which would leave the 18 symptom-side truncations unwatched -- the same
    # shape of blindness the check exists to remove.
    KNOWN_TRUNCATED = 0

    _all_recs = dict(recs)
    for _p in glob.glob(os.path.join(GOLD, "symptoms", "*.json")):
        if os.path.basename(_p) == "index.json":
            continue
        with open(_p, encoding="utf-8") as _f:
            _r = json.load(_f)
        _all_recs[_r.get("symptom_id") or _r.get("code")] = _r

    def _record_strings(rec):
        out = set()
        for st in rec.get("steps") or []:
            for v in (st.get("branches") or {}).values():
                out.add(v)
            for k in ("cause", "procedure", "remedy", "point_to_check"):
                if st.get(k):
                    out.add(st[k])
        return out

    def truncated_expectations(qa_cases, records):
        bad = []
        for c in qa_cases:
            rec = records.get(c.get("source_code"))
            if not rec:
                continue
            pool = _record_strings(rec)
            for field in ("must_contain", "must_contain_verbatim"):
                for want in (c.get(field) or []):
                    if not want or want in pool:
                        continue
                    for full in pool:
                        if not (full.startswith(want) and len(full) > len(want)):
                            continue
                        if want[-1].isalnum() and full[len(want)].isalnum():
                            bad.append((c["id"], c["type"], field, want))
                        break
        return bad

    _trunc = truncated_expectations(qa, _all_recs)
    check(f"no expectation ends mid-word ({KNOWN_TRUNCATED} permitted)",
          len(_trunc) == KNOWN_TRUNCATED,
          f"found {len(_trunc)} truncated expectations -- e.g. {_trunc[:2]}")
    check("  and none of them is a must_contain_verbatim",
          not [t for t in _trunc if t[2] == "must_contain_verbatim"],
          "a verbatim expectation must never be a fragment")

    # SELF-TEST. This reports a fixed number on a healthy repo, which is the
    # shape E4 and H2 were in when their zeros were read as clean. Plant one
    # more truncation and require it to be seen.
    # The cut position is DERIVED -- the first index where both neighbouring
    # characters are alphanumeric -- rather than a fixed offset that might land
    # on a space and make the probe vacuously pass.
    _probe_code = sorted(recs)[0]
    _probe_src = sorted(_record_strings(recs[_probe_code]), key=len, reverse=True)
    _cut = next((i for s in _probe_src[:1] for i in range(2, len(s))
                 if s[i - 1].isalnum() and s[i].isalnum()), None)
    check("truncation self-test: a planted mid-word expectation is detected",
          _cut is not None
          and len(truncated_expectations(
              [{"id": "planted", "type": "branch_following",
                "source_code": _probe_code,
                "must_contain": [_probe_src[0][:_cut]]}], recs)) == 1,
          "the check cannot see a fragment it was built to see")

    # EXPECTED PER (section, type), NOT AS A TOTAL.
    #
    # This replaced `len(qa) == 1330`. That assertion was not wrong -- it was
    # UNDER-SPECIFIED, in exactly the way the unresolved-facts expectation was
    # in a74b002: it pinned an unqualified total while meaning a Section 40
    # one. When symptom cases were added, the total it pinned absorbed a
    # second corpus and the assertion became false without any Section 40
    # count having changed. The Section 40 numbers below are the original
    # 1,330, unmoved and now named.
    QA_EXPECTED = {
        ("section40", "numeric_exactness"): 846,
        ("section40", "direct_lookup"): 173,
        ("section40", "step_ordering"): 164,
        ("section40", "branch_following"): 116,
        ("section40", "precondition"): 10,
        ("section40", "cross_ref_hop"): 9,
        ("section40", "adversarial_unknown"): 6,
        ("section40", "adversarial_model"): 6,
        ("symptoms", "numeric_exactness"): 232,
        ("symptoms", "step_ordering"): 57,
        ("symptoms", "branch_following"): 37,
        # 209 flat rows carry BOTH point_to_check and remedy.
        #
        # 207 of these cases derive THREE golden facts and 2 derive only two.
        # That split is the SOURCE DOCUMENT, not the builder: SM09 step 1 and
        # SM11 step 1 print "Unspecified fuel is used." in both the Cause and
        # the Point-to-check column of the manual's own table, so golden_facts
        # dedupes them to one string. Nothing is being dropped.
        ("symptoms", "symptom_remedy"): 209,
    }
    qa_built = collections.Counter((c.get("section"), c["type"]) for c in qa)
    qa_drift = [f"{k}: expected {QA_EXPECTED.get(k, 0)}, got {qa_built.get(k, 0)}"
                for k in sorted(set(qa_built) | set(QA_EXPECTED), key=str)
                if qa_built.get(k, 0) != QA_EXPECTED.get(k, 0)]
    check("qa_set.json matches the expected split by section and type",
          not qa_drift, "\n          ".join(qa_drift))
    # The two guards that a pinned total cannot give you: a bucket nobody named
    # (which is the event that broke the old assertion, and must fail by naming
    # the newcomer), and parts that do not sum to the whole.
    check("no qa_set bucket exists that the expectation does not name",
          set(qa_built) <= set(QA_EXPECTED),
          f"unnamed: {sorted(set(qa_built) - set(QA_EXPECTED), key=str)}")
    check("the qa_set parts account for the whole",
          sum(qa_built.values()) == len(qa),
          f"parts {sum(qa_built.values())} vs total {len(qa)}")
    check("every qa_set case declares a section",
          all(c.get("section") in ("section40", "symptoms") for c in qa))
    check("all case ids are unique",
          len({c["id"] for c in qa}) == len(qa))

    # Every pinned verbatim string must be a criterion -- or, for a flat
    # S-Mode row, a remedy -- that currently exists in golden/ for the record
    # the case names. This is the staleness check, and it now spans both
    # corpora because the cases do. Extending its source of truth is not
    # loosening it: every pinned string is still required to exist.
    truth = {}
    for code, rec in recs.items():
        truth[code] = {m["criteria"] for m in all_measurements(rec)}
    for p in sorted(glob.glob(os.path.join(GOLD, "symptoms", "*.json"))):
        if os.path.basename(p) == "index.json":
            continue
        with open(p, encoding="utf-8") as f:
            sr = json.load(f)
        vals = {m["criteria"] for m in (sr.get("standalone_measurements") or [])}
        for st in sr.get("steps", []):
            vals |= {m["criteria"] for m in (st.get("measurements") or [])}
            if st.get("remedy"):
                vals.add(st["remedy"])
        truth[sr["symptom_id"]] = vals

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

    # The per-type counts that used to live here were the SAME
    # under-specification one level finer: `numeric_exactness == 846` pinned a
    # type total while meaning a Section 40 one, so it broke the moment a
    # second corpus contributed to the same type. QA_EXPECTED above carries
    # both halves keyed on (section, type) and is checked with the two guards a
    # total cannot give you. Re-asserting the type totals here would only
    # restore the ambiguity.
    #
    # What is worth keeping separately is the SECTION 40 SUBTOTAL: the original
    # 1,330, which must not move because symptom cases were added beside it.
    s40 = [c for c in qa if c.get("section") == "section40"]
    check("the Section 40 half of qa_set is still exactly 1,330 cases",
          len(s40) == 1330, f"got {len(s40)}")

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
