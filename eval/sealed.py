#!/usr/bin/env python3
"""
sealed.py
The sealed evaluation holdout: qa_set cases a system has never been tuned on.

    python eval/sealed.py            # describe the split

WHY A THIRD HOLDOUT, AND HOW IT DIFFERS FROM THE OTHER TWO
----------------------------------------------------------
Three exist now. They measure different things and must not be blended:

  agent/holdout.py              32 of 174 FAILURE CODES. Excluded from scripted
                                agent fixtures by construction, so
                                path_correctness and diagnosis_correct can say
                                whether the graph generalises past its own
                                fixtures. Unit: a code.
  knowledge/symptom_heldout.json  PHRASINGS for the symptom matcher, sealed 46
                                and contaminated 91. Measures entry matching,
                                not answer quality. Unit: a phrase.
  THIS FILE                     qa_set CASES. Measures a system that answers
                                questions -- which is what a real model will
                                be. Unit: a case, grouped by its source record.

This one exists because neither of the others gives a clean unseen set for a
generative model. The only clean set before it was sealed 46, which measures a
matcher.

THE SELECTION RULE READS NO EVAL RESULT. EVER.
----------------------------------------------
Selection depends on the DATA ONLY -- section, case type, record identity, a
content hash. It never reads which cases a system passes or fails.

That rule is not hygiene, it is the specific way the 91 held-out symptom
phrasings became contaminated: retrieval scoring was revised after looking at
which of them failed, so their score is partly in-sample and can no longer be
quoted as held out. A set chosen or adjusted with any knowledge of outcomes is
spent at the moment of choosing. Written down here so the next person has the
reason and not just the rule.

THE UNIT IS THE RECORD, NOT THE CASE
------------------------------------
Cases from one record share a page, a tree and a page image. Splitting them
across train and sealed leaks: a model tuned on CA451's step_ordering has seen
the page that CA451's numeric_exactness case asks about. So whole records move
together, and the cost is that per-type case counts deviate from an exact 20% --
records carry between 1 and 40 cases each.

SIZE: 20% of records, decided before selecting and for reasons independent of
what it selects.
  - matches HOLDOUT_FRACTION in agent/holdout.py; one fraction across the repo
  - ~370 cases, enough for the four large buckets to carry a rate
  - the small buckets are corpus-bound, not fraction-bound: cross_ref_hop has 9
    cases in the whole corpus and precondition 10. At any fraction they are
    single digit. Doubling the fraction would buy 4 instead of 2 and cost twice
    the corpus.
  - a sealed set is spent once anything is tuned against it, so it is sized to
    be sufficient rather than large

COVERAGE, not just proportion. Every (section, type) present in the corpus must
appear in the sealed set. A holdout that is 90% numeric_exactness measures one
thing and reports it as a general number; one that silently drops cross_ref_hop
reports a general number that is blind to a behaviour.

No LLM, no randomness, no stored list of ids. Deterministic and derived, so it
cannot drift from the data the way a committed list would.
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
from typing import Dict, List, Set, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
QA_PATH = os.path.join(GOLD, "qa_set.json")

SEALED_FRACTION = 0.20

# A different salt from agent/holdout.py's on purpose. Sharing one would make
# the two splits correlate -- the same records drawn for both -- and a code
# held out of agent fixtures AND of model evaluation is held out twice while
# another is held out never.
SPLIT_SALT = "komatsu-sealed-cases-v1"

# Cases with no source record: adversarial_unknown names a FABRICATED code, so
# there is no record to leak from and no record to group by. They are their own
# stratum, keyed on the case id, and are sampled at the same fraction -- a
# sealed set with no refusal test cannot tell you whether a model refuses.
NO_RECORD = "<no-record>"


def _rank(key: str) -> float:
    """Stable pseudo-random rank in [0,1). Same order on every machine."""
    h = hashlib.sha256(f"{SPLIT_SALT}:{key}".encode()).hexdigest()
    return int(h[:12], 16) / 0xFFFFFFFFFFFF


def load_cases(path: str = None) -> List[dict]:
    p = path or QA_PATH
    if not os.path.isfile(p):
        raise SystemExit(f"ERROR: no qa_set at {p}\n"
                         "  Run eval/build_qa_set.py first.")
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def record_of(case: dict) -> str:
    return case.get("source_code") or NO_RECORD


def type_signature(cases: List[dict]) -> Tuple:
    """The (section, type) pairs a record produces, as a stratum key.

    Stratifying on WHAT A RECORD GENERATES is what keeps every case type in the
    sealed set. Stratifying on the record's own properties -- format, page span
    -- would balance the records and leave case types to chance, which is how a
    type ends up with zero and the headline number goes quietly blind to it.
    """
    return tuple(sorted({(c["section"], c["type"]) for c in cases}))


def split(cases: List[dict] = None) -> Tuple[List[str], List[str]]:
    """(train_case_ids, sealed_case_ids). Deterministic; reads no eval result."""
    cases = cases if cases is not None else load_cases()
    by_record: Dict[str, List[dict]] = collections.defaultdict(list)
    for c in cases:
        by_record[record_of(c)].append(c)

    # The no-record group is split per CASE, not as one block: it has no record
    # to leak through, and holding it whole would put all six refusal tests on
    # one side.
    loose = by_record.pop(NO_RECORD, [])

    by_stratum: Dict[Tuple, List[str]] = collections.defaultdict(list)
    for rec, rcases in by_record.items():
        by_stratum[type_signature(rcases)].append(rec)

    sealed_records: List[str] = []
    remainders = []
    for key in sorted(by_stratum, key=str):
        members = sorted(by_stratum[key], key=_rank)
        exact = len(members) * SEALED_FRACTION
        take = int(exact)
        sealed_records += members[:take]
        if take < len(members):
            remainders.append((exact - take, _rank(members[take]), members[take]))

    target = int(round(len(by_record) * SEALED_FRACTION))
    for _, _, rec in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if len(sealed_records) >= target:
            break
        sealed_records.append(rec)

    sealed = set(sealed_records)

    # COVERAGE GUARANTEE. Every (section, type) in the corpus must be in the
    # sealed set, and never at the cost of emptying it from train -- a bucket a
    # system can no longer see is worse than one it cannot be scored on.
    present = {(c["section"], c["type"])
               for r in sealed for c in by_record[r]}
    for want in sorted({(c["section"], c["type"]) for c in cases
                        if record_of(c) != NO_RECORD}):
        if want in present:
            continue
        cands = sorted((r for r, rc in by_record.items()
                        if r not in sealed
                        and want in {(c["section"], c["type"]) for c in rc}),
                       key=_rank)
        for r in cands:
            others = [x for x, rc in by_record.items()
                      if x != r and x not in sealed
                      and want in {(c["section"], c["type"]) for c in rc}]
            if others:
                sealed.add(r)
                present |= {(c["section"], c["type"]) for c in by_record[r]}
                break

    sealed_ids = [c["id"] for r in sealed for c in by_record[r]]
    loose_sealed = [c["id"] for c in sorted(loose, key=lambda c: _rank(c["id"]))
                    [:int(round(len(loose) * SEALED_FRACTION))]]
    sealed_ids += loose_sealed
    sid = set(sealed_ids)
    train_ids = [c["id"] for c in cases if c["id"] not in sid]
    return sorted(train_ids), sorted(sid)


_cache = None


def _split_cached():
    global _cache
    if _cache is None:
        _cache = split()
    return _cache


def sealed_records() -> Set[str]:
    """The RECORDS on the sealed side.

    Generated cases -- injection, conversation -- are built at run time and are
    not in qa_set, so they have no case id to look up. They do carry a
    source_code, so they split on the same record rule, which keeps the leak
    guarantee (a record never straddles the boundary) without dropping them.
    """
    cases = load_cases()
    sealed = sealed_ids()
    return {c.get("source_code") for c in cases
            if c["id"] in sealed and c.get("source_code")}


def sealed_ids() -> Set[str]:
    return set(_split_cached()[1])


def train_ids() -> Set[str]:
    return set(_split_cached()[0])


def reset_cache() -> None:
    global _cache
    _cache = None


def describe(cases: List[dict] = None) -> dict:
    """Everything needed to audit the split without re-deriving it."""
    cases = cases if cases is not None else load_cases()
    train, sealed = split(cases)
    sset = set(sealed)
    by = collections.defaultdict(lambda: {"corpus": 0, "sealed": 0})
    for c in cases:
        k = (c["section"], c["type"])
        by[k]["corpus"] += 1
        if c["id"] in sset:
            by[k]["sealed"] += 1
    cov = {f"{k[0]}/{k[1]}": dict(v, share=v["sealed"] / v["corpus"])
           for k, v in sorted(by.items())}
    recs = {c.get("source_code") for c in cases if c["id"] in sset}
    return {"n_cases": len(cases), "n_train": len(train), "n_sealed": len(sealed),
            "fraction": len(sealed) / len(cases),
            "sealed_records": len([r for r in recs if r]),
            "coverage": cov}


def self_test() -> None:
    cases = load_cases()
    a, b = split(cases), split(cases)
    assert a == b, "sealed selection is not deterministic"
    train, sealed = a
    assert not (set(train) & set(sealed)), "train and sealed overlap"
    assert set(train) | set(sealed) == {c["id"] for c in cases}, \
        "split does not cover qa_set"

    # No record straddles the boundary. This is the leak the record unit exists
    # to prevent, so it is asserted rather than trusted to the grouping code.
    sset = set(sealed)
    by_rec = collections.defaultdict(set)
    for c in cases:
        if c.get("source_code"):
            by_rec[c["source_code"]].add(c["id"] in sset)
    straddle = sorted(r for r, sides in by_rec.items() if len(sides) > 1)
    assert not straddle, f"records split across train and sealed: {straddle[:5]}"

    d = describe(cases)
    for k, v in d["coverage"].items():
        assert v["sealed"] > 0, f"{k} has no sealed cases -- the set is blind to it"
        assert v["corpus"] - v["sealed"] > 0, f"{k} has no train cases left"
    assert 0.15 <= d["fraction"] <= 0.25, \
        f"sealed fraction drifted to {d['fraction']:.3f}"


if __name__ == "__main__":
    d = describe()
    print(f"qa_set {d['n_cases']} cases -> train {d['n_train']} / "
          f"sealed {d['n_sealed']}  ({d['fraction']:.1%}), "
          f"{d['sealed_records']} records\n")
    print(f"{'section/type':36}{'corpus':>8}{'sealed':>8}{'share':>8}")
    for k, v in d["coverage"].items():
        print(f"{k:36}{v['corpus']:>8}{v['sealed']:>8}{v['share']:>8.1%}")
    self_test()
    print("\nself-test passed")
