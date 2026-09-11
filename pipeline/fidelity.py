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
        items.append({
            "fact_id": r["fact_id"], "kind": r["kind"], "code": r["code"],
            "classification": c["classification"], "reason": c["reason"],
            "should_be_on_page": r["manual_page"], "pdf_page": r["pdf_page"],
            "stored_text": r["verbatim"], "resolver_reason": r["reason"],
        })
    counts = {}
    for i in items:
        counts[i["classification"]] = counts.get(i["classification"], 0) + 1
    payload = {"total_unresolved": len(items), "by_classification": counts,
               "items": sorted(items, key=lambda x: (x["classification"], x["fact_id"]))}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


# --------------------------------------------------------------- gating

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
