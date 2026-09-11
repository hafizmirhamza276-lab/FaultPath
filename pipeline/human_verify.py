#!/usr/bin/env python3
"""
human_verify.py
Diff completed human transcriptions against golden/, and adjudicate.

INDEPENDENT BY CONSTRUCTION. This module imports no extractor helper -- not
clean(), not parse_causes, not table_kind, not the citation resolver's
normaliser. It carries its own copy of the three documented normalisations, so
a bug in the extractor's version cannot make a disagreement disappear. If both
sides shared the normaliser they would share its blind spots, which is the whole
reason the resolver alone is not enough: the resolver and the extractor already
share theirs.

DIGITS ARE NEVER NORMALISED. Whitespace, line-break hyphenation and ohm variants
are folded. Nothing else. "Min. 100kOhm" and "Min. 90kOhm" must stay different.

ADJUDICATION IS EXPLICIT. A disagreement is never auto-resolved:
  extractor_wrong  a real defect -- reported, not patched here
  human_wrong      transcription error -- logged and corrected
  contested        both defensible; the source is ambiguous
The default when unclear is extractor_wrong. The burden sits on the machine: it
is the side that cannot be asked what it meant.

human_verified is set ONLY on agreement. There is no other path to it.

No LLM.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
import unicodedata
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
REPORTS = os.environ.get("OUT_DIR", os.path.join(REPO_ROOT, "reports"))
KIT_DIR = os.path.join(REPORTS, "transcription")
RESULT_PATH = os.path.join(REPORTS, "human_verification.json")

EXACT = "exact"
NORMALISED = "normalised"
DISAGREE = "DISAGREEMENT"

EXTRACTOR_WRONG = "extractor_wrong"
HUMAN_WRONG = "human_wrong"
CONTESTED = "contested"
UNADJUDICATED = "unadjudicated"

_OHM = ("Ω", "Ω", "ω", "ohms", "ohm")


def normalise(text: str) -> str:
    """The three documented normalisations, and only those.

    Own implementation on purpose -- see the module docstring. Digits are
    untouched; that is the whole point of the metric this feeds.
    """
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    s = s.replace("\n", " ")
    s = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", s)        # line-break hyphen
    for form in _OHM:
        s = re.sub(re.escape(form), " ohm ", s, flags=re.I)
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def compare(human: str, machine: str) -> str:
    if human is None or str(human).strip() == "":
        return None                       # not transcribed; not a disagreement
    if str(human) == str(machine):
        return EXACT
    if normalise(human) == normalise(machine):
        return NORMALISED
    return DISAGREE


def load_records() -> Dict[str, dict]:
    recs = {}
    for p in sorted(glob.glob(os.path.join(GOLD, "failure_codes", "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["code"]] = r
    return recs


def _real_steps(rec: dict) -> List[dict]:
    """Local copy. Importing the agent's helper would couple the comparator to
    the thing it audits."""
    return [s for s in rec.get("steps", []) if not s.get("redirect")]


def load_transcriptions(kit_dir: Optional[str] = None) -> Dict[str, dict]:
    """Completed transcriptions only. A template with no transcriber name has
    not been done, and counting it as agreement would be a rubber stamp."""
    out = {}
    for p in sorted(glob.glob(os.path.join(kit_dir or KIT_DIR, "*", "*.json"))):
        if os.path.basename(os.path.dirname(p)) == "example":
            continue
        with open(p, encoding="utf-8") as f:
            t = json.load(f)
        if not str(t.get("transcriber") or "").strip():
            continue
        out[t["code"]] = t
    return out


def diff_code(transcription: dict, rec: dict) -> List[dict]:
    """Field-by-field. Produces rows; decides nothing."""
    rows = []

    def add(fact_id, kind, field, human, machine):
        verdict = compare(human, machine)
        if verdict is None:
            return
        rows.append({"code": rec["code"], "fact_id": fact_id, "kind": kind,
                     "field": field, "human": human, "machine": machine,
                     "verdict": verdict,
                     "adjudication": UNADJUDICATED if verdict == DISAGREE else None})

    steps = _real_steps(rec)
    for ht in transcription.get("steps", []):
        i = ht.get("step", 0) - 1
        if not (0 <= i < len(steps)):
            rows.append({"code": rec["code"], "fact_id": f"{rec['code']}:{ht.get('step')}:step:0",
                         "kind": "structure", "field": "step_exists",
                         "human": f"step {ht.get('step')} transcribed",
                         "machine": "no such step in golden/",
                         "verdict": DISAGREE, "adjudication": UNADJUDICATED})
            continue
        ms = steps[i]
        add(ms.get("fact_id"), "cause", "cause", ht.get("cause"), ms.get("cause"))
        add(ms.get("fact_id"), "step_procedure", "procedure",
            ht.get("procedure"), ms.get("procedure"))
        br = ms.get("branches") or {}
        bf = ms.get("branch_fact_ids") or {}
        add(bf.get("YES"), "branch", "branch_yes", ht.get("branch_yes"), br.get("YES"))
        add(bf.get("NO"), "branch", "branch_no", ht.get("branch_no"), br.get("NO"))
        for j, hm in enumerate(ht.get("measurements") or []):
            if j >= len(ms.get("measurements") or []):
                rows.append({"code": rec["code"], "fact_id": f"{ms.get('fact_id')}#m{j}",
                             "kind": "structure", "field": "measurement_exists",
                             "human": json.dumps(hm, ensure_ascii=False)[:80],
                             "machine": "no such measurement in golden/",
                             "verdict": DISAGREE, "adjudication": UNADJUDICATED})
                continue
            mm = ms["measurements"][j]
            for f in ("quantity", "point", "criteria"):
                add(mm.get("fact_id"), "measurement", f, hm.get(f), mm.get(f))

    for j, hm in enumerate(transcription.get("standalone_measurements") or []):
        sm = rec.get("standalone_measurements") or []
        if j >= len(sm):
            continue
        for f in ("quantity", "point", "criteria"):
            add(sm[j].get("fact_id"), "measurement", f, hm.get(f), sm[j].get(f))

    if transcription.get("page_disagrees_with_template"):
        rows.append({"code": rec["code"], "fact_id": None, "kind": "structure",
                     "field": "step_count",
                     "human": transcription.get("notes") or "page disagrees",
                     "machine": f"{len(steps)} steps extracted",
                     "verdict": DISAGREE, "adjudication": UNADJUDICATED})
    return rows


def load_adjudications(path: Optional[str] = None) -> dict:
    p = path or os.path.join(REPORTS, "adjudications.json")
    if not os.path.isfile(p):
        return {}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def apply_adjudications(rows: List[dict], adjudications: dict) -> List[dict]:
    """Attach a human decision to each disagreement.

    An unadjudicated disagreement stays unadjudicated and counts AGAINST the
    extractor. Nothing here decides on its own -- the default exists so that
    silence is never scored as agreement.
    """
    for r in rows:
        if r["verdict"] != DISAGREE:
            continue
        key = f"{r['fact_id']}|{r['field']}"
        decision = adjudications.get(key) or adjudications.get(r["fact_id"] or "")
        if isinstance(decision, dict):
            r["adjudication"] = decision.get("decision", UNADJUDICATED)
            r["adjudication_note"] = decision.get("note", "")
        elif isinstance(decision, str):
            r["adjudication"] = decision
        else:
            r["adjudication"] = UNADJUDICATED
            r["adjudication_note"] = (
                "not adjudicated; counted against the extractor by default -- "
                "the machine cannot be asked what it meant")
    return rows


def verified_fact_ids(rows: List[dict]) -> List[str]:
    """Facts a human confirmed. Agreement on EVERY transcribed field of that
    fact, and no disagreement anywhere on it."""
    by_fact: Dict[str, List[dict]] = {}
    for r in rows:
        if r["fact_id"]:
            by_fact.setdefault(r["fact_id"], []).append(r)
    out = []
    for fid, rs in by_fact.items():
        if all(r["verdict"] in (EXACT, NORMALISED) for r in rs):
            out.append(fid)
    return sorted(out)


def summarise(rows: List[dict]) -> dict:
    by_kind: Dict[str, dict] = {}
    for r in rows:
        k = by_kind.setdefault(r["kind"], {"total": 0, "agree": 0, "disagree": 0})
        k["total"] += 1
        if r["verdict"] in (EXACT, NORMALISED):
            k["agree"] += 1
        else:
            k["disagree"] += 1
    for v in by_kind.values():
        v["rate"] = v["agree"] / v["total"] if v["total"] else 0.0

    disagreements = [r for r in rows if r["verdict"] == DISAGREE]
    adj: Dict[str, int] = {}
    for r in disagreements:
        adj[r.get("adjudication") or UNADJUDICATED] = \
            adj.get(r.get("adjudication") or UNADJUDICATED, 0) + 1

    total = len(rows) or 1
    return {
        "fields_compared": len(rows),
        "human_agreement_rate": sum(1 for r in rows
                                    if r["verdict"] in (EXACT, NORMALISED)) / total,
        "by_kind": by_kind,
        "disagreements": len(disagreements),
        "adjudication": adj,
        # THE NUMBER THIS EXERCISE EXISTS TO PRODUCE: facts the resolver passed
        # that a human disputes. The resolver and the extractor share
        # normalisation rules, so a shared misreading is invisible to it.
        "resolver_blind_spots": [],
    }


def resolver_blind_spots(rows: List[dict], verification: dict) -> List[dict]:
    facts = (verification or {}).get("facts", {})
    out = []
    for r in rows:
        if r["verdict"] != DISAGREE or not r["fact_id"]:
            continue
        if facts.get(r["fact_id"]) == "resolver_verified":
            out.append({"fact_id": r["fact_id"], "field": r["field"],
                        "human": r["human"], "machine": r["machine"],
                        "adjudication": r.get("adjudication")})
    return out


def column_split_accuracy(rows: List[dict], recs: Dict[str, dict]) -> dict:
    """Agreement over the 27 machine-repaired steps specifically.

    The highest-risk facts in the corpus and the reason the human round exists.
    """
    repaired = {s.get("fact_id") for r in recs.values() for s in r.get("steps", [])
                if s.get("extraction_warning") == "column_split_recovered"}
    rs = [r for r in rows if r["fact_id"] in repaired]
    agree = sum(1 for r in rs if r["verdict"] in (EXACT, NORMALISED))
    return {"fields_compared": len(rs), "agree": agree,
            "rate": (agree / len(rs)) if rs else None,
            "note": "no machine-repaired step transcribed yet" if not rs else ""}


def run(kit_dir: Optional[str] = None, adjudications: Optional[dict] = None,
        verification: Optional[dict] = None) -> dict:
    recs = load_records()
    trs = load_transcriptions(kit_dir)
    rows: List[dict] = []
    for code, t in trs.items():
        if code in recs:
            rows += diff_code(t, recs[code])
    rows = apply_adjudications(rows, adjudications or load_adjudications())

    from core import loader
    ver = verification if verification is not None else loader.load_verification()
    out = summarise(rows)
    out["resolver_blind_spots"] = resolver_blind_spots(rows, ver)
    out["column_split_accuracy"] = column_split_accuracy(rows, recs)
    out["codes_transcribed"] = sorted(trs)
    out["human_verified_fact_ids"] = verified_fact_ids(rows)
    out["rows"] = rows
    return out


def write(result: dict, path: Optional[str] = None) -> str:
    p = path or RESULT_PATH
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return p


def self_test() -> None:
    """Every comparison rule proves it can fail before it is trusted."""
    assert compare("Max. 1 Ω", "Max. 1 Ω") == EXACT
    assert compare("max.  1 ohm", "Max. 1 Ω") == NORMALISED, \
        "documented normalisations must fold"
    assert compare("Min. 90kΩ", "Min. 100kΩ") == DISAGREE, \
        "a changed digit must NEVER normalise away"
    assert compare("har- ness", "harness") == NORMALISED, \
        "line-break hyphenation must fold"
    assert compare("well-known", "wellknown") == DISAGREE, \
        "a genuine hyphen must not be folded away"
    assert compare("", "anything") is None, \
        "an untranscribed field is not a disagreement"

    # an unadjudicated disagreement must not drift to agreement
    rows = [{"fact_id": "X:1:meas:0", "field": "criteria", "kind": "measurement",
             "verdict": DISAGREE, "adjudication": None}]
    apply_adjudications(rows, {})
    assert rows[0]["adjudication"] == UNADJUDICATED, \
        "an unadjudicated disagreement must stay unadjudicated"
    assert not verified_fact_ids(rows), \
        "a disputed fact must never be marked human_verified"

    ok = [{"fact_id": "X:1:meas:0", "field": "criteria", "kind": "measurement",
           "verdict": EXACT, "adjudication": None}]
    assert verified_fact_ids(ok) == ["X:1:meas:0"]


if __name__ == "__main__":
    self_test()
    res = run()
    if not res["codes_transcribed"]:
        print("no completed transcriptions yet.")
        print(f"kit is at {os.path.relpath(KIT_DIR, REPO_ROOT)}; a transcription "
              "counts as complete once its 'transcriber' field is filled.")
        sys.exit(0)
    print(f"codes transcribed: {len(res['codes_transcribed'])}")
    print(f"fields compared  : {res['fields_compared']}")
    print(f"agreement        : {res['human_agreement_rate']:.4f}")
    for k, v in sorted(res["by_kind"].items()):
        print(f"  {k:16}{v['agree']:>6}/{v['total']:<6}{v['rate']:.4f}")
    print(f"disagreements    : {res['disagreements']} {res['adjudication']}")
    print(f"resolver blind spots: {len(res['resolver_blind_spots'])}")
    print(f"column-split accuracy: {res['column_split_accuracy']}")
    print(f"written to {os.path.relpath(write(res), REPO_ROOT)}")
