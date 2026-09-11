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


def load_manifest(kit_dir: Optional[str] = None) -> dict:
    p = os.path.join(kit_dir or KIT_DIR, "manifest.json")
    if not os.path.isfile(p):
        return {"codes": []}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def round_of(code: str, manifest: Optional[dict] = None) -> Optional[int]:
    """READ from the manifest, never inferred.

    Deriving the round from a record's own properties would put the rule in two
    places, and two places deciding the same thing drift apart silently.
    """
    m = manifest if manifest is not None else load_manifest()
    for c in m.get("codes", []):
        if c.get("code") == code:
            return c.get("round")
    return None


def load_transcriptions(kit_dir: Optional[str] = None) -> Dict[str, dict]:
    """Completed transcriptions only. A template with no transcriber name has
    not been done, and counting it as agreement would be a rubber stamp."""
    out = {}
    root = kit_dir or KIT_DIR
    for p in sorted(glob.glob(os.path.join(root, "*", "*.json"))
                    + glob.glob(os.path.join(root, "round2", "*", "*.json"))):
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


VERIFIED = "verified"
PARTIAL = "PARTIAL"
NOT_TRANSCRIBED = "NOT_TRANSCRIBED"
DISPUTED = "DISPUTED"


def expected_fields(recs: Dict[str, dict]) -> Dict[str, set]:
    """Which fields EXIST for each fact, from golden/.

    A fact is only verified when every field it actually has was transcribed.
    Without this the step fact_id is shared by `cause` and `procedure`, so a
    correct cause with a blank procedure marked the whole step human_verified
    -- crediting the record with work nobody did. Same class as the audit
    checks that sat at zero because they could not fail.
    """
    out: Dict[str, set] = {}
    for rec in recs.values():
        for m in rec.get("standalone_measurements", []):
            if m.get("fact_id"):
                out[m["fact_id"]] = {f for f in ("quantity", "point", "criteria")
                                     if str(m.get(f) or "").strip()}
        for st in rec.get("steps", []):
            if st.get("fact_id"):
                out[st["fact_id"]] = {f for f in ("cause", "procedure")
                                      if str(st.get(f) or "").strip()}
            for m in st.get("measurements", []):
                if m.get("fact_id"):
                    out[m["fact_id"]] = {f for f in ("quantity", "point", "criteria")
                                         if str(m.get(f) or "").strip()}
            for br, fid in (st.get("branch_fact_ids") or {}).items():
                out[fid] = {f"branch_{br.lower()}"}
    return out


def fact_states(rows: List[dict], expected: Dict[str, set]) -> Dict[str, str]:
    """verified / PARTIAL / DISPUTED / NOT_TRANSCRIBED, per fact.

    PARTIAL exists so an incomplete step reports as incomplete instead of
    quietly counting as agreement. It is never treated as verified anywhere.
    """
    seen: Dict[str, set] = {}
    bad: Dict[str, bool] = {}
    for r in rows:
        fid = r.get("fact_id")
        if not fid:
            continue
        seen.setdefault(fid, set()).add(r["field"])
        if r["verdict"] == DISAGREE:
            bad[fid] = True
    states = {}
    for fid, want in expected.items():
        got = seen.get(fid, set())
        if bad.get(fid):
            states[fid] = DISPUTED
        elif not got:
            states[fid] = NOT_TRANSCRIBED
        elif want and not want <= got:
            states[fid] = PARTIAL
        else:
            states[fid] = VERIFIED
    return states


def verified_fact_ids(rows: List[dict],
                      expected: Optional[Dict[str, set]] = None) -> List[str]:
    """Facts a human confirmed: every field that exists, all agreeing.

    `expected` is required for the completeness rule. Without it the old
    behaviour returns -- any agreeing field verifies the fact -- so callers
    that cannot supply it get the conservative reading instead: a fact with
    fewer rows than fields is NOT verified.
    """
    if expected is None:
        from agent import tools as _t
        try:
            expected = expected_fields(_t.records())
        except Exception:
            expected = {}
    states = fact_states(rows, expected)
    return sorted(f for f, s in states.items() if s == VERIFIED)


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


def coverage(trs: Dict[str, dict], recs: Dict[str, dict],
             manifest: dict) -> dict:
    """The denominator, made impossible to lose.

    A partial result presented as a whole-corpus number is the quiet
    overstatement this project keeps removing. Every report carries this.
    """
    total_facts = 0
    for r in recs.values():
        total_facts += len(r.get("standalone_measurements", []))
        for st in r.get("steps", []):
            total_facts += 1 + len(st.get("measurements", []))                 + len(st.get("branch_fact_ids") or {})
    done_facts = 0
    for code in trs:
        r = recs.get(code)
        if not r:
            continue
        done_facts += len(r.get("standalone_measurements", []))
        for st in r.get("steps", []):
            done_facts += 1 + len(st.get("measurements", []))                 + len(st.get("branch_fact_ids") or {})
    selected = [c["code"] for c in manifest.get("codes", [])]
    r1 = [c["code"] for c in manifest.get("codes", []) if c.get("round") == 1]
    return {"codes_transcribed": len(trs), "codes_selected": len(selected),
            "round1_codes": len(r1),
            "round1_transcribed": len([c for c in trs if c in set(r1)]),
            "facts_covered": done_facts, "facts_total": total_facts,
            "statement": (f"{len(trs)} of {len(selected)} codes, "
                          f"{done_facts} of {total_facts} facts"),
            "is_partial": done_facts < total_facts}


def column_split_per_step(rows: List[dict], recs: Dict[str, dict],
                          trs: Dict[str, dict],
                          states: Optional[Dict[str, str]] = None) -> List[dict]:
    """Every machine-repaired step, one row each. NEVER aggregated.

    One wrong step among 27 matters; 96.3% hides it. The reader gets the
    evidence -- what the extractor recovered, what the human typed, the verdict
    -- whether or not anything disagrees.
    """
    by_fact = {}
    for r in rows:
        if r["field"] == "cause" and r["fact_id"]:
            by_fact[r["fact_id"]] = r
    out = []
    for code, rec in sorted(recs.items()):
        for st in rec.get("steps", []):
            if st.get("extraction_warning") != "column_split_recovered":
                continue
            fid = st.get("fact_id")
            row = by_fact.get(fid)
            state = states.get(fid, NOT_TRANSCRIBED) if states else None
            verdict = (row["verdict"] if row else NOT_TRANSCRIBED)
            # PARTIAL outranks a per-field verdict: a step with an agreeing
            # cause and a blank procedure must not read as a match.
            if state == PARTIAL:
                verdict = PARTIAL
            out.append({
                "fact_id": fid, "code": code, "step": st.get("step"),
                "manual_page": (st.get("provenance") or {}).get("manual_page"),
                "extractor": st.get("cause"),
                "human": row["human"] if row else None,
                "verdict": verdict, "state": state,
                "transcribed": code in trs})
    return out


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
    manifest = load_manifest(kit_dir)
    out = summarise(rows)
    out["coverage"] = coverage(trs, recs, manifest)
    out["resolver_blind_spots"] = resolver_blind_spots(rows, ver)
    out["column_split_accuracy"] = column_split_accuracy(rows, recs)
    expected = expected_fields(recs)
    states = fact_states(rows, expected)
    out["column_split_steps"] = column_split_per_step(rows, recs, trs, states)
    out["fact_states"] = {s_: sum(1 for v in states.values() if v == s_)
                          for s_ in (VERIFIED, PARTIAL, DISPUTED, NOT_TRANSCRIBED)}
    out["rounds"] = {c: round_of(c, manifest) for c in sorted(trs)}
    out["codes_transcribed"] = sorted(trs)
    out["human_verified_fact_ids"] = verified_fact_ids(rows, expected)
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
    exp = {"X:1:meas:0": {"quantity", "point", "criteria"}}
    assert not verified_fact_ids(rows, exp), \
        "a disputed fact must never be marked human_verified"

    # COMPLETENESS. The previous version of this assertion encoded the bug it
    # was meant to guard -- one agreeing field verified the whole fact, so a
    # step with a correct cause and a blank procedure read as verified. A fact
    # is verified only when every field it actually has was transcribed.
    one_of_three = [{"fact_id": "X:1:meas:0", "field": "criteria",
                     "kind": "measurement", "verdict": EXACT,
                     "adjudication": None}]
    assert fact_states(one_of_three, exp)["X:1:meas:0"] == PARTIAL, \
        "a partially transcribed fact must report PARTIAL"
    assert not verified_fact_ids(one_of_three, exp), \
        "PARTIAL must never count as verified"

    all_three = [{"fact_id": "X:1:meas:0", "field": f, "kind": "measurement",
                  "verdict": EXACT, "adjudication": None}
                 for f in ("quantity", "point", "criteria")]
    assert fact_states(all_three, exp)["X:1:meas:0"] == VERIFIED
    assert verified_fact_ids(all_three, exp) == ["X:1:meas:0"]
    assert fact_states([], exp)["X:1:meas:0"] == NOT_TRANSCRIBED


# ---------------------------------------------------------------- progress

def progress(kit_dir: Optional[str] = None, round_no: Optional[int] = 1) -> dict:
    """Which codes are done, which are pending, and the running rate.

    Transcription accuracy degrades well before hour eight, so the work should
    be split across sessions. Making it easy to stop and resume is the point --
    a transcriber who pushes on because picking the thread back up is awkward
    produces exactly the agreement-by-fatigue this design is built against.
    """
    manifest = load_manifest(kit_dir)
    trs = load_transcriptions(kit_dir)
    recs = load_records()
    wanted = [c for c in manifest.get("codes", [])
              if round_no is None or c.get("round") == round_no]
    rows: List[dict] = []
    for c in trs:
        if c in recs:
            rows += diff_code(trs[c], recs[c])
    rows = apply_adjudications(rows, load_adjudications())
    done = [c for c in wanted if c["code"] in trs]
    pending = [c for c in wanted if c["code"] not in trs]
    expected = expected_fields(recs)
    states = fact_states(rows, expected)
    n_partial = sum(1 for v in states.values() if v == PARTIAL)
    agree = sum(1 for r in rows if r["verdict"] in (EXACT, NORMALISED))
    mins = [trs[c["code"]].get("minutes_taken") for c in done
            if trs[c["code"]].get("minutes_taken")]
    return {
        "round": round_no,
        "done": [c["code"] for c in done],
        "pending": [c["code"] for c in pending],
        "done_count": len(done), "total": len(wanted),
        "fields_compared": len(rows),
        "running_agreement": (agree / len(rows)) if rows else None,
        "disagreements": sum(1 for r in rows if r["verdict"] == DISAGREE),
        "partial_facts": n_partial,
        "verified_facts": sum(1 for v in states.values() if v == VERIFIED),
        "minutes_spent": sum(mins) if mins else 0,
        "minutes_remaining_estimate": (
            round(sum(mins) / len(done) * len(pending)) if mins and done else None),
    }


def print_progress(kit_dir: Optional[str] = None, round_no: int = 1) -> dict:
    p = progress(kit_dir, round_no)
    print(f"ROUND {p['round']}: {p['done_count']} of {p['total']} codes transcribed")
    if p["done"]:
        print(f"  done    : {', '.join(p['done'])}")
    print(f"  pending : {', '.join(p['pending']) if p['pending'] else '-- none --'}")
    if p["fields_compared"]:
        print(f"  fields compared   : {p['fields_compared']}")
        print(f"  running agreement : {p['running_agreement']:.4f}")
        print(f"  disagreements     : {p['disagreements']}")
        print(f"  facts verified    : {p['verified_facts']}")
        print(f"  facts PARTIAL     : {p['partial_facts']}"
              + ("   <- incomplete, NOT counted as verified"
                 if p["partial_facts"] else ""))
    else:
        print("  nothing compared yet")
    if p["minutes_spent"]:
        print(f"  time spent        : {p['minutes_spent']} min")
        if p["minutes_remaining_estimate"]:
            print(f"  remaining (est)   : {p['minutes_remaining_estimate']} min")
    return p


def print_column_split_table(result: dict) -> None:
    """The 27 machine-repaired steps, one row each. Never a single rate."""
    rows = result.get("column_split_steps") or []
    cov = result.get("coverage", {})
    print(f"\nCOLUMN-SPLIT STEPS -- {len(rows)} machine-repaired steps, "
          f"reported individually")
    print(f"coverage: {cov.get('statement', 'unknown')}")
    print(f"{'fact_id':22}{'page':9}{'verdict':17}extractor / human")
    for r in rows:
        print(f"{r['fact_id']:22}{str(r['manual_page']):9}{r['verdict']:17}"
              f"{(r['extractor'] or '')[:44]!r}")
        if r["human"] is not None:
            print(f"{'':48}{(r['human'] or '')[:44]!r}")
    done = [r for r in rows if r["verdict"] != "NOT_TRANSCRIBED"]
    dis = [r for r in done if r["verdict"] == DISAGREE]
    print(f"\n{len(done)} of {len(rows)} transcribed; {len(dis)} disagree")
    if len(rows) != len(done):
        print(f"{len(rows) - len(done)} not yet transcribed -- a rate over these "
              "would be an average of the ones we happened to do")


if __name__ == "__main__":
    self_test()
    if "--progress" in sys.argv:
        rn = 1
        if "--round" in sys.argv:
            rn = int(sys.argv[sys.argv.index("--round") + 1])
        print_progress(round_no=rn)
        sys.exit(0)
    res = run()
    print_column_split_table(res)
    if not res["codes_transcribed"]:
        print("\nno completed transcriptions yet.")
        print(f"kit is at {os.path.relpath(KIT_DIR, REPO_ROOT)}; a transcription "
              "counts as complete once its 'transcriber' field is filled.")
        sys.exit(0)
    print(f"\nCOVERAGE: {res['coverage']['statement']}"
          + ("  (PARTIAL)" if res["coverage"]["is_partial"] else ""))
    print(f"rounds transcribed: {res['rounds']}")
    print(f"fields compared  : {res['fields_compared']}")
    print(f"agreement        : {res['human_agreement_rate']:.4f}")
    for k, v in sorted(res["by_kind"].items()):
        print(f"  {k:16}{v['agree']:>6}/{v['total']:<6}{v['rate']:.4f}")
    print(f"disagreements    : {res['disagreements']} {res['adjudication']}")
    print(f"resolver blind spots: {len(res['resolver_blind_spots'])}")
    print(f"column-split accuracy: {res['column_split_accuracy']}")
    print(f"written to {os.path.relpath(write(res), REPO_ROOT)}")
