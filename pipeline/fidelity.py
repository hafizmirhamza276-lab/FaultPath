#!/usr/bin/env python3
"""
fidelity.py
Extraction fidelity: does golden/ actually say what the PDF says.

This closes the gap nothing measured. numeric_exactness asks whether the agent
reproduced a criterion faithfully; path_correctness asks whether it walked the
tree. Neither asks whether the tree was read off the page correctly in the first
place. 27 steps were silently dropped and no metric noticed -- they were caught
by a human reading pages.

The citation resolver already verifies facts against the PDF. It was reported.
Here it is enforced.

Deterministic. No LLM. Measures the parser; never alters it.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
OUT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO_ROOT, "reports"))
UNRESOLVED_PATH = os.path.join(OUT_DIR, "unresolved_facts.json")

FACT_KINDS = ("measurement", "branch", "cause", "step_procedure",
              "remedy", "symptom_title", "title", "action_level")

# Safety-critical kinds. Gated at 100% -- they already pass, so this pins
# current behaviour rather than asserting an aspiration.
GATED_KINDS = ("measurement", "branch")

# Every section is gated on its own. Section 40 is mature; a new section
# riding its denominator would be scored on someone else's work.
SECTIONS = ("section40", "hmode", "smode")

# ------------------------------------------------------- relational criteria
#
# A relational criterion states a RATIO between two measured quantities rather
# than a value for one -- "Oil pressure ratio pump discharged pressure : PC
# valve discharged pressure 1:0.6 (approximately 3/5)". There is no number to
# reproduce for the row on its own, so it cannot be scored by the numeric
# comparison that backs `numeric_exactness`.
#
# It is EXCLUDED FROM THE NUMERIC DENOMINATOR BY DECLARATION, and its count is
# reported next to the figure it is excluded from. Dropping it silently would
# inflate the numeric rate by shrinking the denominator; folding it in would
# fail a row for lacking a number it was never supposed to have. Both are
# wrong, and only one of them is visible.
RELATIONAL = "relational"

KNOWN_LIMITATION = "KNOWN_LIMITATION"
NEEDS_HUMAN = "NEEDS_HUMAN_VERIFICATION"
DEFECT = "DEFECT"

# ---------------------------------------------------- cell-border overhang
#
# NOT a parser defect. The document places a glyph outside its own ruled cell
# border, and both pdfplumber and PyMuPDF report what lies inside the rectangle
# -- correctly. Measured: 'f' of "of" starts at x=146.68 against a cell right
# edge of x=145.90, a 0.79pt overhang.
#
# This was recorded as DEFECT in e907cf4 on a wrong diagnosis: the stored cause
# was compared against get_text() over the WHOLE page, which matched a different
# occurrence of the same words in a neighbouring branch-outcome cell. Corrected
# here.
#
# Why no fix was applied:
#   - 8,108 cause-table cells were cross-checked pdfplumber vs PyMuPDF on the
#     same regions. ZERO cells show character loss by pdfplumber. There is no
#     parser error to repair.
#   - Reading the cell through PyMuPDF by bounding box -- the obvious fix --
#     returns the identical truncated string. It is a no-op.
#   - Padding the clip by 2.5pt does recover the glyphs, but on a 120-page
#     sample 178 of 2,520 cells (7.06%) gain text bled in from a neighbour.
#     That is roughly 570 corrupted cells corpus-wide to repair 9 cause
#     strings, and it would touch measurement and branch cells, which are
#     gated at 1.0000.
#   - Glyph-contiguity reconstruction would be narrower, but it is a heuristic
#     with unknown behaviour on any future manual, and it would be tuned on the
#     nine cases it is meant to fix.
#
# PRECEDENT. This project records source properties rather than patching around
# them: F@BBZL's action level conflicts between the summary table and its detail
# page (audit B1); D8ARKR references CA445, which does not exist in this manual
# (C1); DY20KA step 7 carries a NO branch with no YES (D4). None was silently
# corrected. A value the document itself renders ambiguously is surfaced, not
# guessed at -- a technician who sees the conflict can escalate, one handed a
# confidently repaired value cannot.
#
# No measurement criterion is affected: all 872 resolve exactly.
CELL_OVERHANG_FACTS = {
    "CA145:3:step:0": {
        "stored": "Short circuit in wiring ha ness",
        "true_text": "Short circuit in wiring harness",
        "overhang_pt": 2.03, "glyphs_outside": ["-"]},
    "DGH2KA:4:step:0": {
        "stored": "Hot short circu in wiring harness",
        "true_text": "Hot short circuit in wiring harness",
        "overhang_pt": 2.45, "glyphs_outside": ["i", "t"]},
    "DGH2KA:5:step:0": {
        "stored": "Reconfirmatio of inspection item",
        "true_text": "Reconfirmation of inspection item",
        "overhang_pt": 0.0, "glyphs_outside": [],
        "edge": "the lost glyph ends a wrapped line and is clipped at the "
                "cell's vertical bound rather than its right edge"},
    "DGH2KA:6:step:0": {
        "stored": "Confirmation o repair",
        "true_text": "Confirmation of repair",
        "overhang_pt": 0.79, "glyphs_outside": ["f"]},
    "DGH2KB:5:step:0": {
        "stored": "Reconfirmatio of inspection item",
        "true_text": "Reconfirmation of inspection item",
        "overhang_pt": 0.0, "glyphs_outside": [],
        "edge": "clipped at the cell's vertical bound, as DGH2KA:5"},
    "DGH2KB:6:step:0": {
        "stored": "Confirmation o repair",
        "true_text": "Confirmation of repair",
        "overhang_pt": 0.79, "glyphs_outside": ["f"]},
    "DW91KA:3:step:0": {
        "stored": "Open circuit i wiring harnes",
        "true_text": "Open circuit in wiring harness",
        "overhang_pt": 0.89, "glyphs_outside": ["s"]},
    "DWK8KA:3:step:0": {
        "stored": "Open circui wiring harn",
        "true_text": "Open circuit in wiring harness",
        "overhang_pt": 7.10, "glyphs_outside": ["i", "n", "s"]},
    "F313KA:3:step:0": {
        "stored": "Open circuit i wiring harness",
        "true_text": "Open circuit in wiring harness",
        "overhang_pt": 0.0, "glyphs_outside": [],
        "edge": "clipped at the cell's vertical bound"},
}

OVERHANG_REASON = (
    "cell-border overhang in the source document, not a parser error. The "
    "manual renders a glyph outside its own ruled cell boundary; pdfplumber and "
    "PyMuPDF agree on what lies inside the rectangle. 8,108 cells cross-checked "
    "with zero character loss by pdfplumber. No fix applied: reading the same "
    "bbox via PyMuPDF is a no-op, padding the clip corrupts ~7% of cells to "
    "repair 9 cause strings, and glyph-contiguity is an untested heuristic. "
    "Recorded as a source property, following the precedent of F@BBZL (B1), "
    "D8ARKR/CA445 (C1) and DY20KA step 7 (D4)."
)


def load_records(gold: Optional[str] = None) -> Dict[str, dict]:
    recs = {}
    for p in sorted(glob.glob(os.path.join(gold or GOLD, "failure_codes", "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["code"]] = r
    return recs


def load_symptom_records(gold: Optional[str] = None) -> Dict[str, dict]:
    recs = {}
    for p in sorted(glob.glob(os.path.join(gold or GOLD, "symptoms", "*.json"))):
        if os.path.basename(p) == "index.json":
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["symptom_id"]] = r
    return recs


def _fact(fid, kind, code, text, prov, section, warn=None, redirect=False,
          relational=False):
    return {"fact_id": fid, "kind": kind, "code": code, "text": text or "",
            "prov": prov, "warn": warn, "redirect": redirect,
            "section": section, "relational": relational}


def enumerate_symptom_facts(recs: Dict[str, dict]) -> List[dict]:
    """Facts from the H-Mode and S-Mode symptom trees.

    Sectioned by tree_kind rather than by id prefix: the section a fact belongs
    to is a property of the record, and reading it off a filename would break
    silently the first time an id scheme changes.
    """
    out = []
    for sid, r in recs.items():
        section = "smode" if r["tree_kind"] == "SymptomTreeFlat" else "hmode"
        prov0 = {"manual_page": r["manual_pages"][0], "pdf_page": r["pdf_pages"][0]}
        out.append(_fact(f"{sid}:0:symptom_title:0", "symptom_title", sid,
                         r["symptom"], prov0, section))

        def meas(m, warn=None):
            rel = m.get("criteria_kind") == RELATIONAL
            return _fact(m["fact_id"], "measurement", sid, m["criteria"],
                         m["provenance"], section, warn=warn, relational=rel)

        for m in r.get("standalone_measurements", []):
            out.append(meas(m))
        for st in r.get("steps", []):
            warn = st.get("extraction_warning")
            out.append(_fact(st["fact_id"], "cause", sid, st.get("cause"),
                             st["provenance"], section, warn=warn))
            if st.get("procedure"):
                out.append(_fact(f"{st['fact_id']}:proc", "step_procedure", sid,
                                 st["procedure"], st["provenance"], section,
                                 warn=warn))
            if st.get("remedy_fact_id"):
                out.append(_fact(st["remedy_fact_id"], "remedy", sid,
                                 st.get("remedy"), st["provenance"], section,
                                 warn=warn))
            for m in st.get("measurements", []):
                out.append(meas(m, warn))
            for br, fid in (st.get("branch_fact_ids") or {}).items():
                # The branch's OWN page, captured at parse time. Falls back to
                # the step's only where the extractor could not match the
                # outcome to a row -- and that fallback is counted in the
                # record, so it cannot pass unnoticed.
                bprov = (st.get("branch_provenance") or {}).get(br) \
                    or st["provenance"]
                out.append(_fact(fid, "branch", sid,
                                 (st.get("branches") or {}).get(br, ""),
                                 bprov, section, warn=warn))
    return out


def enumerate_facts(recs: Dict[str, dict]) -> List[dict]:
    """Every fact with a fact_id, its kind, and the page it claims to be on."""
    out = []
    for code, r in recs.items():
        for m in r.get("standalone_measurements", []):
            out.append({"fact_id": m["fact_id"], "kind": "measurement",
                        "code": code, "text": m["criteria"],
                        "prov": m["provenance"], "warn": None, "redirect": False})
        for st in r.get("steps", []):
            out.append({"fact_id": st["fact_id"], "kind": "cause", "code": code,
                        "text": st.get("cause") or "", "prov": st["provenance"],
                        "warn": st.get("extraction_warning"),
                        "redirect": bool(st.get("redirect"))})
            for m in st.get("measurements", []):
                out.append({"fact_id": m["fact_id"], "kind": "measurement",
                            "code": code, "text": m["criteria"],
                            "prov": m["provenance"], "warn": st.get("extraction_warning"),
                            "redirect": False})
            for br, fid in (st.get("branch_fact_ids") or {}).items():
                # A branch fact id whose branch text is gone is itself the
                # finding: the id promises a fact the record no longer holds.
                # Defaulting to "" makes it unresolvable rather than crashing.
                out.append({"fact_id": fid, "kind": "branch", "code": code,
                            "text": (st.get("branches") or {}).get(br, ""),
                            "prov": st["provenance"],
                            "warn": st.get("extraction_warning"), "redirect": False})
            if st.get("procedure"):
                out.append({"fact_id": f"{st['fact_id']}:proc",
                            "kind": "step_procedure", "code": code,
                            "text": st["procedure"], "prov": st["provenance"],
                            "warn": st.get("extraction_warning"),
                            "redirect": bool(st.get("redirect"))})
    for f in out:
        f.setdefault("section", "section40")
        f.setdefault("relational", False)
    return out


# ------------------------------------------------------------- the checks

def check_facts(recs=None, pages=None) -> dict:
    """Resolve every fact against the PDF and report by kind."""
    from eval.citations import resolve, PageText
    if recs is None:
        recs = dict(load_records())
        recs.update(load_symptom_records())
    pages = pages or PageText()
    facts = [f for f in enumerate_facts(
                 {k: v for k, v in recs.items() if "code" in v})]
    facts += enumerate_symptom_facts(
        {k: v for k, v in recs.items() if "symptom_id" in v})

    rows, by_kind, by_sec = [], {}, {}
    for f in facts:
        entry = {"fact_id": f["fact_id"], "kind": f["kind"], "code": f["code"],
                 "section": f["section"], "relational": f["relational"],
                 "manual_page": f["prov"].get("manual_page"),
                 "pdf_page": f["prov"].get("pdf_page"),
                 "warn": f["warn"], "redirect": f["redirect"],
                 "resolved": None, "reason": "", "verbatim": f["text"]}
        if not (f["text"] or "").strip():
            # An empty verbatim is not a fact that resolves, it is a fact with
            # nothing in it. The resolver would find "" on any page in the
            # document and return True, so admitting it would push the rate UP
            # while measuring nothing -- the same shape as an audit check that
            # cannot fail. Refused explicitly.
            entry["resolved"] = False
            entry["reason"] = "empty verbatim text"
        elif pages.available():
            # The citation is built from the RECORDS PASSED IN, not re-rendered
            # from disk. An earlier version called citations.render(), which
            # reloads golden/ -- so a corrupted digit or a shifted page in the
            # supplied records was invisible, and the mutation harness could
            # not fail by construction. Measure what you were handed.
            cit = {"fact_id": f["fact_id"],
                   "manual_page": f["prov"].get("manual_page"),
                   "pdf_page": f["prov"].get("pdf_page"),
                   "verbatim_text": f["text"]}
            res = resolve(cit, pages)
            entry["resolved"] = res["resolved"]
            entry["reason"] = res["reason"]
        rows.append(entry)
        for bucket, key in ((by_kind, f["kind"]),
                            (by_sec, (f["section"], f["kind"]))):
            k = bucket.setdefault(key, {"total": 0, "resolved": 0,
                                        "relational": 0})
            if f["relational"]:
                # counted, reported, and kept out of the scored denominator
                k["relational"] += 1
                continue
            k["total"] += 1
            if entry["resolved"]:
                k["resolved"] += 1

    # page_containment: a fact must sit inside its own code's page span.
    # A fact resolving on another code's page is worse than not resolving.
    outside = []
    for f in facts:
        lo, hi = recs[f["code"]]["pdf_pages"]
        p = f["prov"].get("pdf_page")
        if p is None or not (lo <= p <= hi):
            outside.append(f["fact_id"])

    scored = [r for r in rows if not r["relational"]]

    # verbatim_integrity: resolved text matches after the DOCUMENTED
    # normalisation only. Digits are never normalised.
    #
    # Counted over `scored`, NOT over `rows`. Its denominator is `resolved`,
    # which excludes relational facts; counting the numerator over every row
    # let a resolving relational fact into the top and not the bottom, and the
    # ratio came out at 1.0028. A rate above 1.0 is the arithmetic announcing
    # that two different populations were divided.
    integrity = sum(1 for r in scored
                    if r["resolved"] and "whitespace/hyphenation" not in r["reason"])

    total = len(scored)
    resolved = sum(1 for r in scored if r["resolved"])

    def rated(bucket):
        return {k: {**v, "rate": (v["resolved"] / v["total"]) if v["total"] else 0.0}
                for k, v in sorted(bucket.items())}

    return {
        "available": pages.available(),
        "total_facts": total,
        "resolved": resolved,
        "relational_excluded": sum(1 for r in rows if r["relational"]),
        "fact_resolution_rate": (resolved / total) if total else 0.0,
        "by_kind": rated(by_kind),
        "by_section": {f"{s}/{k}": v for (s, k), v in rated(by_sec).items()},
        "page_containment": 1.0 - (len(outside) / total if total else 0.0),
        "page_containment_failures": outside,
        "verbatim_integrity": (integrity / resolved) if resolved else 0.0,
        "rows": rows,
    }


# ------------------------------------------------ classify the unresolved

def classify(row: dict) -> dict:
    """KNOWN_LIMITATION / NEEDS_HUMAN_VERIFICATION / DEFECT.

    Anything that does not match a known, explained shape is a DEFECT. Silence
    must never be the safe option: an unclassifiable fact is precisely the case
    where nobody has looked, and defaulting it to "probably fine" is how the
    27 dropped steps survived.
    """
    fid, reason = row["fact_id"], row.get("reason", "")

    if fid in CELL_OVERHANG_FACTS:
        info = CELL_OVERHANG_FACTS[fid]
        return {"classification": KNOWN_LIMITATION, "reason": OVERHANG_REASON,
                "true_text": info["true_text"],
                "overhang_pt": info["overhang_pt"],
                "glyphs_outside_cell": info["glyphs_outside"],
                "edge_note": info.get("edge")}

    if row["redirect"]:
        return {"classification": KNOWN_LIMITATION,
                "reason": "synthetic 'Redirect' label. The extractor emits this "
                          "for a pointer-only code whose cause table is a single "
                          "unnumbered row; it is not text printed on the page and "
                          "must not be citable as if it were."}

    if row.get("kind") == "step_procedure":
        return {"classification": KNOWN_LIMITATION,
                "reason": "reassembled step procedure. Built from several cells "
                          "of a step's row, including embedded measurement "
                          "sub-tables, so the stored string is accurate but is "
                          "not a contiguous run on the page. Audit S3. Ungated, "
                          "not verbatim-citable, and no numeric metric is "
                          "scored on it."}

    if row.get("kind") == "measurement" and reason == "empty verbatim text":
        return {"classification": DEFECT,
                "reason": "empty criteria string. The criteria cell is "
                          "vertically merged with the row above, whose "
                          "relational criterion governs both rows; this row "
                          "inherits nothing and is typed numeric anyway. "
                          "Audit S1 -- needs a parser change, not a scorer "
                          "tolerance."}

    if row["warn"] == "column_split_recovered":
        return {"classification": NEEDS_HUMAN,
                "reason": "machine-repaired column split. The step number was "
                          "typeset inside the cause text and the extractor "
                          "reassembled it, so the stored string is not a "
                          "contiguous run on the page. README and METRICS both "
                          "say these 27 steps need eye verification; none has "
                          "had it."}

    if reason.startswith("verbatim text NOT"):
        return {"classification": DEFECT,
                "reason": "characters missing from the stored text. pdfplumber's "
                          "extract_tables() dropped glyphs from this cell; the "
                          "PDF text layer contains the full word. Not a "
                          "hyphenation difference -- letters are absent."}

    return {"classification": DEFECT,
            "reason": f"unclassified resolver failure: {reason or 'no reason given'}"}


def write_unresolved(result: dict, out_path: Optional[str] = None) -> dict:
    out_path = out_path or UNRESOLVED_PATH
    unresolved = [r for r in result["rows"] if r["resolved"] is False]
    items = []
    for r in unresolved:
        c = classify(r)
        entry = {
            "fact_id": r["fact_id"], "kind": r["kind"], "code": r["code"],
            "classification": c["classification"], "reason": c["reason"],
            "should_be_on_page": r["manual_page"], "pdf_page": r["pdf_page"],
            "stored_text": r["verbatim"], "resolver_reason": r["reason"],
        }
        for extra in ("true_text", "overhang_pt", "glyphs_outside_cell",
                      "edge_note"):
            if c.get(extra) is not None:
                entry[extra] = c[extra]
        items.append(entry)
    counts = {}
    for i in items:
        counts[i["classification"]] = counts.get(i["classification"], 0) + 1
    payload = {
        "total_unresolved": len(items),
        "by_classification": counts,
        # RECORDED, NOT QUIETLY DROPPED. The human round was designed, the kit
        # was built (27 steps, 54 crop boxes, ~1-2 hours of work) and it was
        # never executed. A gap that stops being visible becomes an unknown gap.
        "human_verification": {
            "status": "NOT PERFORMED",
            "human_verified_facts": 0,
            "detail": (
                "The 27 machine-repaired column-split steps have no human "
                "verification. The transcription round was built and not run; "
                "human_verified is 0 across all 3,305 facts."),
            "what_is_still_covered": (
                "3,266 of 3,305 facts are resolver-verified against the PDF. "
                "All 872 measurement criteria and all 1,437 branch outcomes "
                "resolve exactly, so every safety-critical value has external "
                "confirmation from the source document."),
            "what_remains_unverified": (
                "Reassembled cause text on 27 steps: the extractor recovered a "
                "step number typeset inside the cause column and kept the "
                "mangled fragments. No person has read those pages."),
            "supporting_evidence": (
                "The crop work produced independent support for the extractor's "
                "behaviour on these tables: glyphs overflow their declared "
                "column by up to 36pt (measured on page 784, cause text "
                "x0=70.4..164.4 against a declared cell of 106.3..181.3), and "
                "pdfplumber and PyMuPDF agree on what lies inside the "
                "rectangle. That is consistent with the extractor reading the "
                "tables correctly. It is NOT a substitute for a human reading, "
                "because both libraries share the same notion of a cell."),
            "to_close_it": (
                "reports/transcription/crops/index.html -- 54 boxes, "
                "0.9-1.8 hours. pipeline/human_verify.py compares the result."),
        },
        "items": sorted(items, key=lambda x: (x["classification"], x["fact_id"]))}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


# --------------------------------------------------------------- gating

def check_known_overhang(result: dict) -> dict:
    """The 9 cell-border-overhang facts must STAY unresolvable.

    A known limitation that quietly starts passing is not good news -- it means
    the extractor's cell-text behaviour changed and nobody decided to change it.
    Either the text is now being recovered (in which case these entries are
    stale and the reclassification needs revisiting) or something else moved.
    Both need a person, so this fails rather than celebrating.
    """
    by_id = {r["fact_id"]: r for r in result.get("rows", [])}
    unexpectedly_resolving, missing = [], []
    for fid in CELL_OVERHANG_FACTS:
        row = by_id.get(fid)
        if row is None:
            missing.append(fid)
        elif row.get("resolved") is True:
            unexpectedly_resolving.append(fid)
    ok = not unexpectedly_resolving and not missing
    return {"gate": "known_overhang_stable", "status": "PASS" if ok else "FAIL",
            "expected_unresolvable": len(CELL_OVERHANG_FACTS),
            "now_resolving": unexpectedly_resolving,
            "missing_from_corpus": missing,
            "detail": ("a recorded known limitation started resolving; the "
                       "extractor changed and the classification must be "
                       "re-examined" if unexpectedly_resolving else
                       "fact ids absent from the corpus" if missing else "")}


def gates(result: dict) -> List[dict]:
    out = []
    for kind in GATED_KINDS:
        k = result["by_kind"].get(kind, {"rate": 0.0, "total": 0})
        out.append({"gate": f"fact_resolution_rate[{kind}]", "value": k["rate"],
                    "op": ">=", "threshold": 1.0,
                    "status": "PASS" if k["rate"] >= 1.0 else "FAIL",
                    "n": k["total"]})
        # ...and again per section. A corpus-wide rate lets a mature section
        # carry a weak new one: 274 H-Mode measurements against Section 40's
        # 872 move the combined figure by a quarter of what they should.
        for sec in SECTIONS:
            k = result.get("by_section", {}).get(f"{sec}/{kind}")
            if not k or not k["total"]:
                continue
            out.append({"gate": f"fact_resolution_rate[{sec}/{kind}]",
                        "value": k["rate"], "op": ">=", "threshold": 1.0,
                        "status": "PASS" if k["rate"] >= 1.0 else "FAIL",
                        "n": k["total"]})
    out.append({"gate": "page_containment", "value": result["page_containment"],
                "op": ">=", "threshold": 1.0,
                "status": "PASS" if result["page_containment"] >= 1.0 else "FAIL",
                "n": result["total_facts"]})
    if result.get("rows"):
        oh = check_known_overhang(result)
        out.append({"gate": oh["gate"], "value": 1.0 if oh["status"] == "PASS" else 0.0,
                    "op": ">=", "threshold": 1.0, "status": oh["status"],
                    "n": oh["expected_unresolvable"], "detail": oh["detail"]})
    return out


def self_test() -> None:
    """Each fidelity check must be able to fail on known-bad input.

    E4, H2 and D3 all sat at zero while being incapable of returning anything
    else. A fidelity gate with no failing case would be the same mistake in a
    new place.
    """
    bad = {"total_facts": 2, "resolved": 1,
           "by_kind": {"measurement": {"total": 2, "resolved": 1, "rate": 0.5}},
           "page_containment": 0.5, "verbatim_integrity": 0.5, "rows": []}
    g = gates(bad)
    assert any(x["status"] == "FAIL" for x in g), \
        "fidelity gates cannot fail on a half-resolved corpus"
    good = {"total_facts": 2, "resolved": 2,
            "by_kind": {"measurement": {"total": 2, "resolved": 2, "rate": 1.0},
                        "branch": {"total": 1, "resolved": 1, "rate": 1.0}},
            "page_containment": 1.0, "verbatim_integrity": 1.0, "rows": []}
    assert all(x["status"] == "PASS" for x in gates(good)), \
        "fidelity gates cannot pass on a clean corpus"

    # A per-section gate must fail on a weak section even when the corpus-wide
    # rate is carried to 1.0 by a larger, mature one. This is the whole reason
    # the section split exists: 254 H-Mode measurements against Section 40's
    # 872 move a combined figure by a quarter of what they should.
    masked = {
        "total_facts": 1000, "resolved": 1000,
        "by_kind": {"measurement": {"total": 1000, "resolved": 1000, "rate": 1.0}},
        "by_section": {
            "section40/measurement": {"total": 990, "resolved": 990, "rate": 1.0},
            "hmode/measurement": {"total": 10, "resolved": 5, "rate": 0.5}},
        "page_containment": 1.0, "verbatim_integrity": 1.0, "rows": []}
    mg = gates(masked)
    assert any(x["gate"] == "fact_resolution_rate[hmode/measurement]"
               and x["status"] == "FAIL" for x in mg), \
        "a weak section passes when a mature section carries the combined rate"
    assert any(x["gate"] == "fact_resolution_rate[section40/measurement]"
               and x["status"] == "PASS" for x in mg), \
        "the section split fails a section that is genuinely clean"

    # An empty verbatim must never count as resolved. The resolver finds "" on
    # every page, so admitting one would raise the rate while measuring
    # nothing -- a gate that cannot fail, wearing the costume of one that passed.
    from eval.citations import PageText
    empty = check_facts(recs={"ZZ999": {
        "code": "ZZ999", "pdf_pages": [1, 1],
        "standalone_measurements": [{
            "fact_id": "ZZ999:0:meas:0", "criteria": "",
            "provenance": {"manual_page": "x", "pdf_page": 1,
                           "table_index": 0, "row_index": 0}}],
        "steps": []}}, pages=PageText())
    assert empty["by_kind"]["measurement"]["resolved"] == 0, \
        "an empty criteria string was counted as a resolved fact"

    # Relational facts leave the scored denominator but stay counted.
    rel = check_facts(recs={"HMZZ": {
        "symptom_id": "HMZZ", "tree_kind": "SymptomTreeBranching",
        "symptom": "test", "pdf_pages": [1, 1], "manual_pages": ["x"],
        "steps": [], "standalone_measurements": [{
            "fact_id": "HMZZ:0:meas:0", "criteria": "ratio 1:0.6",
            "criteria_kind": RELATIONAL,
            "provenance": {"manual_page": "x", "pdf_page": 1,
                           "table_index": 0, "row_index": 0}}]}},
        pages=PageText())
    assert rel["by_kind"]["measurement"]["total"] == 0, \
        "a relational criterion entered the numeric denominator"
    assert rel["relational_excluded"] == 1, \
        "a relational criterion was excluded without being counted"
    # classification must never fall through to silence
    assert classify({"fact_id": "X:1:step:0", "redirect": False, "warn": None,
                     "reason": ""})["classification"] == DEFECT, \
        "an unclassifiable fact must default to DEFECT, not to silence"


if __name__ == "__main__":
    self_test()
    from eval.citations import PageText
    pages = PageText()
    if not pages.available():
        sys.exit(f"ERROR: source PDF not found at {pages.path}. Fidelity is "
                 "measured against the document; it cannot be inferred.")
    res = check_facts(pages=pages)
    print(f"facts {res['resolved']}/{res['total_facts']} "
          f"= {res['fact_resolution_rate']:.4f}\n")
    print(f"{'kind':16}{'resolved':>10}{'total':>8}{'rate':>10}")
    for k, v in res["by_kind"].items():
        print(f"{k:16}{v['resolved']:>10}{v['total']:>8}{v['rate']:>10.4f}")
    # By section as well as by kind. A combined rate lets Section 40's
    # maturity mask a weak new section, which is the failure this split exists
    # to prevent.
    print(f"\n{'kind':16}{'section40':>24}{'hmode':>24}{'smode':>24}")
    for k in ("measurement", "branch", "cause", "step_procedure", "remedy",
              "symptom_title"):
        cells = []
        for s in SECTIONS:
            v = res["by_section"].get(f"{s}/{k}")
            if not v or not (v["total"] + v["relational"]):
                cells.append("n/a")
            else:
                extra = f" +{v['relational']}R" if v["relational"] else ""
                cells.append(f"{v['rate']:.4f} {v['resolved']}/{v['total']}{extra}")
        print(f"{k:16}" + "".join(f"{c:>24}" for c in cells))
    print(f"\n  +NR = relational criteria, carried and counted but excluded "
          f"from the scored denominator by declaration ({res['relational_excluded']} total)")
    print(f"\npage_containment    {res['page_containment']:.4f}")
    print(f"verbatim_integrity  {res['verbatim_integrity']:.4f}")
    payload = write_unresolved(res)
    print(f"\nunresolved: {payload['total_unresolved']} -> "
          f"{payload['by_classification']}")
    print(f"written to {os.path.relpath(UNRESOLVED_PATH, REPO_ROOT)}")
    print("\nGATES")
    for g in gates(res):
        print(f"  [{g['status']}] {g['gate']:34} {g['value']:.4f} "
              f"{g['op']} {g['threshold']}  n={g['n']}")
    pages.close()
    sys.exit(0 if all(g["status"] == "PASS" for g in gates(res)) else 1)
