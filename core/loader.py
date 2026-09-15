#!/usr/bin/env python3
"""
loader.py
The two roles of golden/, made explicit.

golden/ is both the agent's knowledge base and the evaluation ground truth. That
is circular for some metrics and not others, and the fix is not to pretend
otherwise:

  NOT circular  -- path_correctness, gate_enforcement, the protocol rules,
     numeric_exactness, fabricated_values, content_recall. These ask whether the
     agent walked the tree and quoted it faithfully. The tree is the right
     reference for that.
  WAS circular  -- citation_accuracy, before span-level provenance. Fixed.
  NOT MEASURED  -- whether extraction read the PDF correctly. Closed by
     pipeline/fidelity.py and pipeline/structural.py.

The files are NOT split. What changes is that the two roles are now
distinguishable in code, and every ground-truth record carries how it was
verified:

  resolver_verified  the citation resolver confirmed it against the PDF
  human_verified     a person read the page and confirmed it
  unverified         neither

A metric computed over unverified facts is worth less than the same number over
verified ones. Until now that distinction was invisible; every metric can now
report split by it.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
REPORTS = os.environ.get("OUT_DIR", os.path.join(REPO_ROOT, "reports"))
VERIFICATION_PATH = os.path.join(REPORTS, "fact_verification.json")

RESOLVER_VERIFIED = "resolver_verified"
HUMAN_VERIFIED = "human_verified"
UNVERIFIED = "unverified"
STATUSES = (HUMAN_VERIFIED, RESOLVER_VERIFIED, UNVERIFIED)


def _read(gold: Optional[str] = None) -> Dict[str, dict]:
    recs = {}
    for p in sorted(glob.glob(os.path.join(gold or GOLD, "failure_codes", "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["code"]] = r
    return recs


def load_knowledge_base(gold: Optional[str] = None) -> Dict[str, dict]:
    """What the agent reads at runtime.

    Deliberately WITHOUT verification annotations. The agent must behave the
    same whether or not a fact has been human-checked -- it executes the manual
    either way. Letting verification status reach the agent would make its
    behaviour depend on how well we had audited ourselves, which is a different
    system on every audit pass.
    """
    return _read(gold)


def load_ground_truth(gold: Optional[str] = None,
                      verification: Optional[dict] = None) -> Dict[str, dict]:
    """What evaluation scores against, annotated with verification status.

    Same records. The difference is that these carry provenance about how much
    each fact is trusted, so a metric can say what it is standing on.
    """
    recs = _read(gold)
    ver = verification if verification is not None else load_verification()
    by_fact = ver.get("facts", {})
    for r in recs.values():
        for m in r.get("standalone_measurements", []):
            m["verification"] = by_fact.get(m.get("fact_id"), UNVERIFIED)
        for st in r.get("steps", []):
            st["verification"] = by_fact.get(st.get("fact_id"), UNVERIFIED)
            for m in st.get("measurements", []):
                m["verification"] = by_fact.get(m.get("fact_id"), UNVERIFIED)
            st["branch_verification"] = {
                br: by_fact.get(fid, UNVERIFIED)
                for br, fid in (st.get("branch_fact_ids") or {}).items()}
    return recs


def load_verification(path: Optional[str] = None) -> dict:
    p = path or VERIFICATION_PATH
    if not os.path.isfile(p):
        return {"facts": {}, "counts": {}, "note": "not yet generated"}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def build_verification(fidelity_result: dict,
                       human_verified_ids: Optional[List[str]] = None,
                       path: Optional[str] = None) -> dict:
    """Populate verification status from a fidelity run.

    human_verified is empty for now -- no one has read the pages yet, and
    claiming otherwise would be the same kind of self-agreement this whole
    exercise is about. The human holdout task fills it.
    """
    human = set(human_verified_ids or [])
    facts: Dict[str, str] = {}
    for row in fidelity_result.get("rows", []):
        fid = row["fact_id"]
        if fid in human:
            facts[fid] = HUMAN_VERIFIED
        elif row.get("resolved") is True:
            facts[fid] = RESOLVER_VERIFIED
        else:
            facts[fid] = UNVERIFIED
    counts: Dict[str, int] = {}
    for v in facts.values():
        counts[v] = counts.get(v, 0) + 1
    payload = {"facts": facts, "counts": counts,
               "total": len(facts),
               "note": ("resolver_verified means the citation resolver found the "
                        "text on the cited PDF page. human_verified means a "
                        "person read the page. No fact is human_verified yet.")}
    out = path or VERIFICATION_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return payload


def verification_drift(fidelity_result: dict,
                       verification: Optional[dict] = None) -> dict:
    """What is on disk vs what the current fidelity run says. BOTH DERIVED.

    Regenerating the artefact once does not stop it going stale again, and the
    way it went stale was invisible: load_ground_truth falls through to
    UNVERIFIED for a missing fact, so a fact absent from the file and a fact
    genuinely unverified are the same value to every reader. 2,757 facts were
    reported unverified while the resolver verified them -- see
    reports/frozen_verification_findings.md.

    So both sides are derived and compared. `missing` is the half that caused
    the understatement; `ghost` is the other direction, a fact the artefact
    claims that the run no longer produces; `disagree` is a fact both know
    about and label differently.
    """
    on_disk = (verification if verification is not None
               else load_verification()).get("facts", {})
    live = {}
    for row in fidelity_result.get("rows", []):
        live[row["fact_id"]] = (RESOLVER_VERIFIED if row.get("resolved") is True
                                else UNVERIFIED)
    missing = sorted(set(live) - set(on_disk))
    ghost = sorted(set(on_disk) - set(live))
    disagree = sorted(f for f in set(live) & set(on_disk)
                      # A human_verified stamp is not drift: it is a claim the
                      # resolver cannot make and must not be overwritten by a
                      # rerun. Nothing is human_verified yet, so this is a
                      # guard against a future round, not present behaviour.
                      if on_disk[f] != live[f] and on_disk[f] != HUMAN_VERIFIED)
    return {"missing": missing, "ghost": ghost, "disagree": disagree,
            "n_missing": len(missing), "n_ghost": len(ghost),
            "n_disagree": len(disagree),
            "stale": bool(missing or ghost or disagree),
            "on_disk_total": len(on_disk), "live_total": len(live)}


def verification_of(fact_id: str, verification: Optional[dict] = None) -> str:
    ver = verification if verification is not None else load_verification()
    return ver.get("facts", {}).get(fact_id, UNVERIFIED)


def split_by_verification(fact_ids, verification: Optional[dict] = None) -> dict:
    """Group fact ids by verification status, so a metric can report split."""
    ver = verification if verification is not None else load_verification()
    facts = ver.get("facts", {})
    out = {s: [] for s in STATUSES}
    for fid in fact_ids:
        out[facts.get(fid, UNVERIFIED)].append(fid)
    return out


def summarise(verification: Optional[dict] = None) -> dict:
    ver = verification if verification is not None else load_verification()
    counts = ver.get("counts", {})
    total = ver.get("total", 0) or 1
    return {s: {"n": counts.get(s, 0), "share": counts.get(s, 0) / total}
            for s in STATUSES}
