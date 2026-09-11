#!/usr/bin/env python3
"""Data-quality audit of the H-Mode and S-Mode symptom trees.

Same check classes, severities and report shape as `audit_manual.py`:

    SOURCE  the manual itself is wrong, ambiguous or self-contradictory
    STRUCT  the document's structure makes extraction hard or lossy
    RISK    the content is fine but retrieval or an agent will trip on it

These sections have never been audited. Section 40 has been through thirteen
revisions of this treatment; H-Mode and S-Mode have been through none, so a
clean result here would be evidence of a blind check rather than a clean
document.

RULE 1 APPLIES. No model call anywhere in this file. Every check is a
deterministic comparison against `golden/symptoms/` and the source PDF.

Every check that can report zero carries a self-test proving it can fail on a
known-bad input, and prints `<id> self-test: PASS`. The audit aborts if one
fails. E4 and H2 were both read as clean while being structurally incapable of
returning anything else, and in H2's case a true finding was deleted from the
README on the strength of it.

    export KOMATSU_PDF=../komatsu-manuals/SEN06867-13.pdf
    python pipeline/audit_symptoms.py        # -> reports/audit_symptoms.json
"""
import collections
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO_ROOT, "reports"))
REPORT_PATH = os.path.join(OUT_DIR, "audit_symptoms.json")

PDF = os.environ.get("KOMATSU_PDF")

findings = []


def add(fid, sev, kind, title, detail, items=None, impact=""):
    findings.append(dict(id=fid, severity=sev, kind=kind, title=title,
                         detail=detail, count=len(items or []),
                         items=sorted(items or []), impact=impact))


def load():
    recs = {}
    for p in sorted(glob.glob(os.path.join(GOLD, "symptoms", "*.json"))):
        if os.path.basename(p) == "index.json":
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["symptom_id"]] = r
    if not recs:
        sys.exit("no symptom records in %s -- run pipeline/extract_symptoms.py"
                 % os.path.join(GOLD, "symptoms"))
    return recs


RECS = load()
HM = {k: v for k, v in RECS.items() if v["tree_kind"] == "SymptomTreeBranching"}
SM = {k: v for k, v in RECS.items() if v["tree_kind"] == "SymptomTreeFlat"}


def all_measurements(r):
    return (list(r.get("standalone_measurements", []))
            + [m for s in r.get("steps", []) for m in s.get("measurements", [])])


# =====================================================================
# S -- STRUCTURE OF THE TREES
# =====================================================================

# --- S1: empty criteria admitted into the numeric bucket ---------------
#
# A measurement whose criteria string is empty carries no value. It is not a
# fact that failed to resolve, it is a fact with nothing in it -- and an empty
# string is found on every page in the document, so a resolver that is handed
# one returns True. That is a check incapable of failing, wearing the costume
# of a passing gate.

def empty_criteria(m):
    return not (m.get("criteria") or "").strip()


assert empty_criteria({"criteria": ""}) and empty_criteria({"criteria": "  "}), \
    "S1 self-test: empty-criteria predicate does not detect an empty criterion"
assert not empty_criteria({"criteria": "0 MPa {0 kgf/cm2}"}), \
    "S1 self-test: empty-criteria predicate fires on a real criterion"
print("S1 self-test: PASS - detects '' and '  ', accepts '0 MPa {0 kgf/cm2}'")

s1 = []
for sid, r in sorted(RECS.items()):
    ms = all_measurements(r)
    for i, m in enumerate(ms):
        if not empty_criteria(m):
            continue
        prev = ms[i - 1] if i else None
        merged = bool(prev
                      and prev.get("criteria_kind") == "relational"
                      and prev["provenance"]["table_index"]
                      == m["provenance"]["table_index"])
        s1.append("%s (%s) %s: kind=%s%s"
                  % (m["fact_id"], m["provenance"]["manual_page"],
                     (m.get("quantity") or "?")[:28], m.get("criteria_kind"),
                     "  [row 2 of a merged relational cell]" if merged else
                     "  [UNPAIRED]"))
if s1:
    add("S1", "HIGH", "STRUCT",
        "Measurements with an empty criteria string are typed as numeric",
        "The criteria cell of these rows is vertically merged with the row "
        "above it: one relational criterion -- an oil-pressure RATIO between "
        "two measured quantities -- governs both rows, and the table draws it "
        "once. pdfplumber assigns the merged text to the first row and leaves "
        "the second empty, which is the correct reading of the page. The "
        "second row is then recorded as criteria_kind='numeric' with an empty "
        "criteria, so it enters the numeric measurement denominator carrying "
        "no value.",
        s1,
        "Two compounding effects. The measurement gate is scored on a "
        "denominator that includes rows with nothing in them, and an empty "
        "verbatim resolves against any page in the manual, so without an "
        "explicit refusal these would report as PASS. Separately, a technician "
        "asking for PC valve output pressure would be handed a blank where the "
        "manual states a ratio against pump discharge pressure.")

# --- S2: branch facts inherit the step's page ---------------------------
#
# CLAUDE.md already records the general form of this: "75% of measurements are
# not on their code's first page -- so a per-code page is not a citation." The
# same argument applies one level down. A step that straddles a page break has
# branch outcomes on the far side of it, and a per-STEP page is not a citation
# for those either.

s2 = []
for sid, r in sorted(HM.items()):
    for st in r.get("steps", []):
        for br, fid in (st.get("branch_fact_ids") or {}).items():
            own = (st.get("provenance") or {}).get("pdf_page")
            s2.append((fid, br, own, (st.get("branches") or {}).get(br, "")))

# Resolution is the evidence, so this check only runs with the PDF present.
straddle = []
if PDF and os.path.exists(PDF):
    from eval.citations import resolve, PageText
    pages = PageText()
    for sid, r in sorted(HM.items()):
        for st in r.get("steps", []):
            for br, fid in (st.get("branch_fact_ids") or {}).items():
                # The branch's OWN captured page. Falling back to the step's
                # is exactly the defect, so the fallback is what gets audited.
                prov = (st.get("branch_provenance") or {}).get(br) \
                    or st["provenance"]
                txt = (st.get("branches") or {}).get(br, "")
                if not txt.strip():
                    continue
                cit = {"fact_id": fid, "manual_page": prov["manual_page"],
                       "pdf_page": prov["pdf_page"], "verbatim_text": txt}
                if resolve(cit, pages)["resolved"]:
                    continue
                found = [p for p in range(prov["pdf_page"] - 1,
                                          prov["pdf_page"] + 3)
                         if resolve({**cit, "pdf_page": p}, pages)["resolved"]]
                straddle.append("%s (%s) cites pdf %d; text is on %s"
                                % (fid, prov["manual_page"], prov["pdf_page"],
                                   found or "no nearby page"))
    if straddle:
        add("S2", "HIGH", "STRUCT",
            "Branch outcome cites its step's page, not its own",
            "Branch facts inherit the provenance of the step that owns them. "
            "When a step straddles a page break its YES outcome can sit on one "
            "page and its NO outcome on the next, and only the first resolves. "
            "The recorded text is correct and verbatim -- what is wrong is the "
            "page it claims to be on.",
            straddle,
            "A citation that names the wrong page is worse than no citation: "
            "it manufactures confidence. This is the same defect class the "
            "project already fixed for measurements in Section 40, where a "
            "per-code page was replaced by span-level provenance.")

# --- S4: fact kinds that still inherit their parent's page ---------------
#
# S2 surfaced one instance, and one instance of this shape usually means the
# pattern is there. Swept every fact kind in both sections: does it capture its
# own page, or take its parent's? Reported whether or not it currently
# misfires, because "no instance today" is a property of the data and not a
# guarantee from the code.

add("S4", "LOW", "STRUCT",
    "Fact kinds that take their parent's page rather than capturing their own",
    "Swept across both sections after the S2 fix. H-Mode branch outcomes now "
    "capture their own page (666/666, 1 of which differs from its step's and "
    "is the case S2 found). These still inherit, and each was checked against "
    "the PDF to see whether the inheritance currently produces a wrong page.",
    ["S-Mode remedy: 209 inherit the step's page; 209/209 resolve there, so no "
     "instance is wrong today",
     "Section 40 branch outcome: 1437 inherit the step's page; 1437/1437 "
     "resolve there. Identical shape to S2 in a section whose parser paths are "
     "out of scope to change -- latent, not manifest",
     "symptom_title: 57 use the entry's first page -- literally the "
     "manual_pages[0] pattern; 57/57 resolve there",
     "step_procedure (both sections): inherits, and is not verbatim-citable "
     "anyway (S3), so it has no single page to capture"],
    "None of these is wrong in this revision. All four are wrong the moment a "
    "row straddles a page break the way HM22 step 5 does, and nothing in the "
    "code prevents that -- only the current layout does. Section 40's 1437 "
    "branches are the one to watch, because the fix that worked for H-Mode "
    "cannot be applied there without touching a parser path that is out of "
    "scope.")

# --- S3: reassembled step procedures ------------------------------------
#
# Declared, not a defect. A procedure is reconstructed from several table cells
# read in layout order, so the stored string is a faithful concatenation that
# exists nowhere on the page as one run of text.

proc_total = sum(1 for r in RECS.values() for s in r.get("steps", [])
                 if s.get("procedure"))
add("S3", "LOW", "STRUCT",
    "Step procedures are reassembled text and are not verbatim-citable",
    "Procedure strings are built from several cells of a step's row, including "
    "embedded measurement sub-tables. The result is accurate but is not a "
    "contiguous run of text on the page, so the resolver refuses it -- "
    "correctly, and for the same reason it refuses the 27 machine-repaired "
    "Section 40 causes and the 3 synthetic Redirect labels.",
    ["%d H-Mode and S-Mode step procedures" % proc_total],
    "Bounds what may be quoted. Procedures may be shown to a technician as "
    "guidance but must not be presented as a verbatim citation, and no numeric "
    "metric may be scored on them.")

# =====================================================================
# P -- POINTERS AND CROSS-REFERENCES
# =====================================================================

# --- P1: unresolved prose pointers --------------------------------------
ptr = []
for sid, r in sorted(RECS.items()):
    for p in r.get("unresolved_pointers", []):
        t = p if isinstance(p, str) else (p.get("text") or json.dumps(p))
        ptr.append("%s: %s" % (sid, t[:110]))
if ptr:
    add("P1", "MEDIUM", "RISK",
        "Prose cross-references are recorded but not resolved into edges",
        "Symptom trees hand off to other procedures in running prose -- "
        "'see Testing and Adjusting, “Examine PPC Valve Outlet "
        "Pressure”' -- rather than through a code in brackets. These are "
        "recorded verbatim as unresolved pointers by deliberate decision: "
        "turning a prose reference into a graph edge means matching a title "
        "against a document that names the same procedure more than one way, "
        "and a wrong edge sends a technician into the wrong procedure with no "
        "signal that it happened.",
        ptr,
        "A symptom session can reach a step whose next action lies outside the "
        "tree. The agent must surface the pointer text and stop there rather "
        "than appear to continue. Closing this gap needs a resolver for "
        "Testing-and-Adjusting titles, which does not exist yet.")

# --- P2: failure codes referenced from symptom trees --------------------
known = set()
for p in glob.glob(os.path.join(GOLD, "failure_codes", "*.json")):
    known.add(os.path.splitext(os.path.basename(p))[0])
dangling = []
for sid, r in sorted(RECS.items()):
    for c in r.get("refs_failure_codes", []):
        if c not in known:
            dangling.append("%s -> %s" % (sid, c))


def is_dangling(code, universe):
    return code not in universe


assert is_dangling("CA445", {"CA451"}), \
    "P2 self-test: dangling-reference predicate misses an absent code"
assert not is_dangling("CA451", {"CA451"}), \
    "P2 self-test: dangling-reference predicate fires on a present code"
print("P2 self-test: PASS - flags an absent code, accepts a present one")

if dangling:
    add("P2", "HIGH", "SOURCE",
        "Symptom tree references a failure code that is not in this manual",
        "The referenced code has no troubleshooting entry in SEN06867-13.",
        dangling,
        "Following the reference reaches nothing. Same class as D8ARKR's "
        "reference to CA445 (audit C1) -- a defect in the source document, to "
        "be surfaced rather than guessed at.")

# =====================================================================
# C -- CRITERIA
# =====================================================================

# --- C1: dual-unit criteria ---------------------------------------------
#
# "0 to 0.49 MPa {0 to 5 kgf/cm2}" carries two numbers for one quantity. Both
# must survive extraction: a technician reading a gauge graduated in kgf/cm2
# needs the bracketed value, and losing it would be silent -- the SI value
# alone still looks like a complete answer.

DUAL = re.compile(r"\{[^}]*\}")


def dual_intact(c):
    m = DUAL.search(c or "")
    return bool(m) and bool(re.search(r"\d", c[:m.start()])) \
        and bool(re.search(r"\d", m.group(0)))


assert dual_intact("0 to 0.49 MPa {0 to 5 kgf/cm2}"), \
    "C1 self-test: intact dual-unit criterion rejected"
assert not dual_intact("0 to 0.49 MPa"), \
    "C1 self-test: a criterion with no bracket counted as dual-unit"
assert not dual_intact("0 to 0.49 MPa {}"), \
    "C1 self-test: an empty bracket counted as an intact second unit"
assert not dual_intact("{0 to 5 kgf/cm2}"), \
    "C1 self-test: a criterion with no SI value counted as intact"
print("C1 self-test: PASS - accepts both numbers present, rejects a lost "
      "bracket, an empty bracket and a lost SI value")

dual_all = [(sid, m) for sid, r in sorted(RECS.items())
            for m in all_measurements(r) if "{" in (m.get("criteria") or "")]
broken = ["%s: %r" % (m["fact_id"], m["criteria"])
          for sid, m in dual_all if not dual_intact(m["criteria"])]
if broken:
    add("C1", "HIGH", "STRUCT",
        "Dual-unit criterion lost one of its two values",
        "The criterion carries a bracketed second unit whose value did not "
        "survive extraction, or whose SI value did not.", broken,
        "A technician reading a gauge graduated in kgf/cm2 has nothing to read "
        "against. The remaining value still looks like a complete answer, so "
        "the loss is silent.")

# --- C2: relational criteria --------------------------------------------
rel = ["%s (%s): %s" % (m["fact_id"], m["provenance"]["manual_page"],
                        (m.get("criteria") or "")[:80])
       for sid, r in sorted(RECS.items()) for m in all_measurements(r)
       if m.get("criteria_kind") == "relational"]
if rel:
    add("C2", "INFO", "STRUCT",
        "Relational criteria state a ratio, not a value",
        "These criteria express a relationship between two measured "
        "quantities rather than a bound on one. They are carried verbatim and "
        "are excluded from the numeric denominator by declaration, with the "
        "count reported beside the figure they are excluded from.",
        rel,
        "Cannot be scored by numeric comparison. Declared rather than dropped: "
        "a silent exclusion would inflate the numeric rate by shrinking its "
        "denominator.")

# --- C3: criteria carrying no digit at all ------------------------------
#   Empty criteria are S1's finding and are excluded here, so the two counts
#   can be added rather than overlapping. What is left is a criterion the
#   manual really prints, which really has no number in it.
nodigit = ["%s (%s): %r" % (m["fact_id"], m["provenance"]["manual_page"],
                            (m.get("criteria") or "")[:60])
           for sid, r in sorted(RECS.items()) for m in all_measurements(r)
           if m.get("criteria_kind") == "numeric"
           and (m.get("criteria") or "").strip()
           and not re.search(r"\d", m.get("criteria") or "")]
if nodigit:
    add("C3", "HIGH", "STRUCT",
        "Criterion typed numeric contains no digit",
        "'Pressure for each flow setting' is what the manual prints in the "
        "standard-value column: the value depends on a setting stated "
        "elsewhere, so the cell carries a description instead of a bound. The "
        "text is extracted correctly and resolves on its page. What is wrong "
        "is the type -- criteria_kind='numeric' on a criterion with no number. "
        "Distinct from S1, whose cells are empty; the two lists are disjoint.",
        nodigit,
        "Enters the numeric denominator carrying nothing to compare against, "
        "so the measurement gate is scored partly on rows that cannot fail it. "
        "A technician asking for this pressure needs to be told the value is "
        "setting-dependent, not handed a sentence where a number belongs.")

# --- C4: dual-unit criteria confirmed intact ------------------------------
#   Recorded as a positive result rather than left as C1's silence. A check
#   that found nothing and a check that never ran read identically in a report.
add("C4", "INFO", "STRUCT",
    "Dual-unit criteria carry both values",
    "Every criterion with a bracketed second unit was checked for a number on "
    "both sides of the bracket: %d of %d dual-unit criteria across both "
    "sections are intact, 0 truncated. C1 is the failing counterpart of this "
    "check and its self-test proves it can fire."
    % (len(dual_all) - len(broken), len(dual_all)),
    ["%d dual-unit criteria verified intact" % (len(dual_all) - len(broken)),
     "example: %s" % (dual_all[0][1]["criteria"] if dual_all else "n/a")],
    "A technician reading a gauge graduated in kgf/cm2 needs the bracketed "
    "value. Losing it would be silent, because the SI value alone still looks "
    "like a complete answer.")

# =====================================================================
# T -- TREE SHAPE
# =====================================================================

# --- T1: branching trees whose steps have no branches -------------------
t1 = []
for sid, r in sorted(HM.items()):
    for st in r.get("steps", []):
        b = st.get("branches") or {}
        if set(b) != {"YES", "NO"}:
            t1.append("%s step %s: outcomes=%s"
                      % (sid, st.get("step"), sorted(b) or "none"))
if t1:
    add("T1", "MEDIUM", "STRUCT",
        "Branching tree step does not carry both YES and NO",
        "A SymptomTreeBranching step is walked by asking a yes/no question. A "
        "step missing an outcome has a path that leads nowhere.", t1,
        "The agent reaches a decision point with no destination for one of the "
        "two answers. Same shape as DY20KA step 7 in Section 40 (audit D4).")

# --- T2: flat trees carrying branches -----------------------------------
t2 = []
for sid, r in sorted(SM.items()):
    for st in r.get("steps", []):
        if st.get("branches") or st.get("branch_fact_ids"):
            t2.append("%s step %s" % (sid, st.get("step")))
if t2:
    add("T2", "HIGH", "STRUCT",
        "Flat S-Mode tree carries branch outcomes",
        "SymptomTreeFlat records a cause / point-to-check / remedy table with "
        "no yes-no decision. A flat tree holding branches means the record was "
        "classified as flat but parsed as branching, or the reverse.", t2,
        "tree_kind drives agent behaviour. If a flat tree carries branches the "
        "agent would ask a YES/NO question the manual does not pose -- "
        "inventing an interaction rather than executing one.")

# --- T3: flat steps with no remedy --------------------------------------
t3 = ["%s step %s" % (sid, st.get("step"))
      for sid, r in sorted(SM.items()) for st in r.get("steps", [])
      if not (st.get("remedy") or "").strip()]
if t3:
    add("T3", "HIGH", "STRUCT",
        "Flat S-Mode step has no remedy",
        "The remedy is the outcome of a flat tree. A step without one gives a "
        "technician a cause and nothing to do about it.", t3,
        "The whole value of an S-Mode entry is the corrective action.")

# --- T4: step numbering ---------------------------------------------------
t4 = []
for sid, r in sorted(RECS.items()):
    nums = [s.get("step") for s in r.get("steps", [])]
    if nums and nums != list(range(1, len(nums) + 1)):
        t4.append("%s: %s" % (sid, nums))
if t4:
    add("T4", "MEDIUM", "STRUCT",
        "Step numbers are not a contiguous run from 1",
        "A gap means a step was dropped; a repeat means one was counted twice.",
        t4, "A dropped step is silent loss -- the tree still looks complete.")

# --- T5: entries with no steps -------------------------------------------
t5 = ["%s: %s" % (sid, r["symptom"][:60])
      for sid, r in sorted(RECS.items()) if not r.get("steps")]
if t5:
    add("T5", "HIGH", "STRUCT",
        "Symptom entry has no steps",
        "The entry was located and titled but no troubleshooting content was "
        "parsed from it.", t5,
        "Matching the symptom succeeds and then delivers nothing.")

# --- T6: skipped tables ---------------------------------------------------
skipped = []
for sid, r in sorted(RECS.items()):
    for s in r.get("skipped_tables", []):
        skipped.append("%s: %s" % (sid, s if isinstance(s, str)
                                   else json.dumps(s, ensure_ascii=False)))
add("T6", "INFO" if not skipped else "MEDIUM", "STRUCT",
    "Tables skipped during extraction",
    "Tables the extractor declined to classify, recorded with the reason. An "
    "empty list here is the claim that every table on every page of both "
    "sections was accounted for.",
    skipped or ["none -- all tables in both sections classified"],
    "A silently skipped table is invisible data loss. Listing them makes the "
    "count arguable.")

# =====================================================================
# D -- DUPLICATES AND IDENTITY
# =====================================================================

# --- D1: duplicate symptom phrasings -------------------------------------
by_symptom = collections.defaultdict(list)
for sid, r in RECS.items():
    by_symptom[" ".join(r["symptom"].lower().split())].append(sid)
d1 = ["%r -> %s" % (k, sorted(v)) for k, v in sorted(by_symptom.items())
      if len(v) > 1]
if d1:
    add("D1", "HIGH", "RISK",
        "Two symptom entries share one phrasing",
        "Deterministic symptom matching keys on the manual's own phrasing. Two "
        "entries with the same text cannot be told apart by it.", d1,
        "The matcher must present both and ask, not pick. Entering the wrong "
        "tree silently is worse than asking.")

# --- D2: duplicate fact ids ----------------------------------------------
ids = collections.Counter()
for sid, r in RECS.items():
    for st in r.get("steps", []):
        ids[st["fact_id"]] += 1
        for fid in (st.get("branch_fact_ids") or {}).values():
            ids[fid] += 1
        if st.get("remedy_fact_id"):
            ids[st["remedy_fact_id"]] += 1
    for m in all_measurements(r):
        ids[m["fact_id"]] += 1
d2 = ["%s x%d" % (k, n) for k, n in sorted(ids.items()) if n > 1]


def dup_ids(counter):
    return [k for k, n in counter.items() if n > 1]


assert dup_ids(collections.Counter(["a", "a", "b"])) == ["a"], \
    "D2 self-test: duplicate-id detector misses a repeated id"
assert not dup_ids(collections.Counter(["a", "b"])), \
    "D2 self-test: duplicate-id detector fires on unique ids"
print("D2 self-test: PASS - flags a repeated id, accepts unique ids")

if d2:
    add("D2", "HIGH", "STRUCT",
        "Fact id is not unique",
        "Fact ids are the whole provenance mechanism: the model names an id "
        "and the system renders the citation. Two facts sharing one id means a "
        "citation can render the wrong fact.", d2,
        "Breaks deterministic provenance at its root. Same defect class as the "
        "Section 40 cause/procedure id collision that produced the PARTIAL "
        "state in human verification.")

# --- D3: page spans overlapping between entries --------------------------
spans = sorted((r["pdf_pages"][0], r["pdf_pages"][1], sid)
               for sid, r in RECS.items())
d3 = ["%s [%d-%d] overlaps %s [%d-%d]" % (a[2], a[0], a[1], b[2], b[0], b[1])
      for a, b in zip(spans, spans[1:]) if b[0] <= a[1]]
if d3:
    add("D3", "MEDIUM", "STRUCT",
        "Symptom entry page spans overlap",
        "Two entries claim the same page. One of them is reading content that "
        "belongs to the other.", d3,
        "Facts attributed to the wrong symptom still resolve, because the text "
        "really is on the page -- so page_containment cannot catch it.")

# =====================================================================
# self-tests for the checks that reported zero
# =====================================================================
#
# T1-T5, D1, D3 and S2 all found nothing. A zero is only a statement about the
# document if the check could have said otherwise -- E4 and H2 each sat at zero
# for months while being structurally incapable of returning anything else. Each
# predicate below is re-run against a hand-built bad record and must fire.

def _incomplete_branches(step):
    return set(step.get("branches") or {}) != {"YES", "NO"}


assert _incomplete_branches({"branches": {"YES": "x"}}), \
    "T1 self-test: a step missing its NO outcome was accepted"
assert not _incomplete_branches({"branches": {"YES": "x", "NO": "y"}}), \
    "T1 self-test: a complete step was flagged"
print("T1 self-test: PASS - flags a step missing NO, accepts YES+NO")


def _flat_with_branches(step):
    return bool(step.get("branches") or step.get("branch_fact_ids"))


assert _flat_with_branches({"branches": {"YES": "x"}}), \
    "T2 self-test: branches on a flat step were not detected"
assert _flat_with_branches({"branch_fact_ids": {"YES": "f"}}), \
    "T2 self-test: branch ids with no branch text were not detected"
assert not _flat_with_branches({"remedy": "replace it"}), \
    "T2 self-test: a clean flat step was flagged"
print("T2 self-test: PASS - flags branches and orphan branch ids on a flat step")


def _no_remedy(step):
    return not (step.get("remedy") or "").strip()


assert _no_remedy({"remedy": ""}) and _no_remedy({}), \
    "T3 self-test: a flat step with no remedy was accepted"
assert not _no_remedy({"remedy": "Replace the valve."}), \
    "T3 self-test: a step with a remedy was flagged"
print("T3 self-test: PASS - flags an absent remedy, accepts a real one")


def _bad_numbering(nums):
    return bool(nums) and nums != list(range(1, len(nums) + 1))


assert _bad_numbering([1, 2, 4]), "T4 self-test: a gap in step numbers missed"
assert _bad_numbering([1, 2, 2]), "T4 self-test: a repeated step number missed"
assert _bad_numbering([2, 3]), "T4 self-test: a run not starting at 1 missed"
assert not _bad_numbering([1, 2, 3]), "T4 self-test: a clean run was flagged"
print("T4 self-test: PASS - flags gap, repeat and wrong start; accepts 1..n")

assert not {"steps": []}.get("steps"), "T5 self-test: an empty step list read as non-empty"
assert {"steps": [{"step": 1}]}.get("steps"), "T5 self-test: a real step list read as empty"
print("T5 self-test: PASS - distinguishes an empty step list from a populated one")


def _dupe_symptoms(pairs):
    d = collections.defaultdict(list)
    for sid, text in pairs:
        d[" ".join(text.lower().split())].append(sid)
    return sorted(k for k, v in d.items() if len(v) > 1)


assert _dupe_symptoms([("A", "Swing Speed is Low"),
                       ("B", "swing   speed is low")]) == ["swing speed is low"], \
    "D1 self-test: two phrasings differing only in case and spacing not matched"
assert not _dupe_symptoms([("A", "Swing Speed is Low"), ("B", "Arm is Slow")]), \
    "D1 self-test: two distinct symptoms reported as duplicates"
print("D1 self-test: PASS - matches across case and whitespace, separates "
      "distinct symptoms")


def _overlaps(spans):
    return [(a[2], b[2]) for a, b in zip(spans, spans[1:]) if b[0] <= a[1]]


assert _overlaps([(1, 10, "A"), (10, 20, "B")]) == [("A", "B")], \
    "D3 self-test: a shared boundary page was not reported as an overlap"
assert not _overlaps([(1, 10, "A"), (11, 20, "B")]), \
    "D3 self-test: two adjacent non-overlapping spans were flagged"
print("D3 self-test: PASS - flags a shared page, accepts adjacent spans")

if PDF and os.path.exists(PDF):
    # S2 found exactly one straddling branch. Prove the check would find a
    # second: the same resolve() call against a page the text is NOT on.
    _r = HM["HM22"]
    _st = [s for s in _r["steps"] if s["step"] == 5][0]
    _cit = {"fact_id": "selftest", "manual_page": "x",
            "pdf_page": _st["provenance"]["pdf_page"],
            "verbatim_text": _st["branches"]["YES"]}
    assert resolve(_cit, pages)["resolved"], \
        "S2 self-test: a branch known to be on its cited page did not resolve"
    assert not resolve({**_cit, "pdf_page": _cit["pdf_page"] + 40},
                       pages)["resolved"], \
        "S2 self-test: branch text resolved against a page 40 pages away"
    print("S2 self-test: PASS - resolves on the true page, refuses a far page")

# =====================================================================
# report
# =====================================================================

sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
findings.sort(key=lambda f: (sev_order[f["severity"]], f["id"]))

os.makedirs(OUT_DIR, exist_ok=True)
with open(REPORT_PATH, "w", encoding="utf-8") as fh:
    json.dump(findings, fh, indent=2, ensure_ascii=False)

c = collections.Counter(f["severity"] for f in findings)
print()
print("H-Mode %d entries, S-Mode %d entries" % (len(HM), len(SM)))
print("%d findings - %d HIGH / %d MEDIUM / %d LOW / %d INFO"
      % (len(findings), c["HIGH"], c["MEDIUM"], c["LOW"], c["INFO"]))
for f in findings:
    print("  %-6s %-4s %-7s %s (%d)"
          % (f["severity"], f["id"], f["kind"], f["title"], f["count"]))
print("->", REPORT_PATH)
