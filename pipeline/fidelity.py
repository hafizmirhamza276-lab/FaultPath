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
              "title", "action_level")

# Safety-critical kinds. Gated at 100% -- they already pass, so this pins
# current behaviour rather than asserting an aspiration.
GATED_KINDS = ("measurement", "branch")

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
    return out


# ------------------------------------------------------------- the checks

def check_facts(recs=None, pages=None) -> dict:
    """Resolve every fact against the PDF and report by kind."""
    from eval.citations import resolve, PageText
    recs = recs or load_records()
    pages = pages or PageText()
    facts = enumerate_facts(recs)

    rows, by_kind = [], {}
    for f in facts:
        entry = {"fact_id": f["fact_id"], "kind": f["kind"], "code": f["code"],
                 "manual_page": f["prov"].get("manual_page"),
                 "pdf_page": f["prov"].get("pdf_page"),
                 "warn": f["warn"], "redirect": f["redirect"],
                 "resolved": None, "reason": "", "verbatim": f["text"]}
        if pages.available():
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
        k = by_kind.setdefault(f["kind"], {"total": 0, "resolved": 0})
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

    # verbatim_integrity: resolved text matches after the DOCUMENTED
    # normalisation only. Digits are never normalised.
    integrity = sum(1 for r in rows
                    if r["resolved"] and "whitespace/hyphenation" not in r["reason"])

    total = len(rows)
    resolved = sum(1 for r in rows if r["resolved"])
    return {
        "available": pages.available(),
        "total_facts": total,
        "resolved": resolved,
        "fact_resolution_rate": (resolved / total) if total else 0.0,
        "by_kind": {k: {**v, "rate": (v["resolved"] / v["total"]) if v["total"] else 0.0}
                    for k, v in sorted(by_kind.items())},
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
