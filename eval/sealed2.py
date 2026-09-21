#!/usr/bin/env python3
"""
sealed2.py
The SECOND sealed holdout. Derived before the selector it exists to judge.

    python eval/sealed2.py           # describe the split

WHY A SECOND ONE
----------------
eval/sealed.py is SPENT. It was measured on 2026-09-17 at 2e8edb2, and a
holdout is spent the moment a result from it is known -- every decision taken
afterwards is taken by someone who has seen it. Its 354 cases are in-sample now
and re-deriving the same ids does not restore their unseenness.

Since then the prompt selector has been rewritten twice (6cbd647, cb990e1) and
is about to be rewritten a third time, each round measured on the same 1,511
non-sealed cases. That is the overfitting pattern this repo has already been
caught in once: `reports/oracle_selection_finding.md` records a selector that
reached 7/7 gates by reading the answer key, and it reached them on the only
cases anyone was looking at.

So this set is derived FIRST, from the 1,511 cases sealed.py did not take,
BEFORE the question-derived selector is designed. Nothing about its contents
influenced the design, because at the moment of derivation the design did not
exist.

IT IS SEALED, NOT A DEVELOPMENT SET. eval/adversarial.py is the development set
and says so; this is the opposite declaration. One measurement, once, on a
frozen selector. After that it is spent and this docstring gets the same SPENT
banner sealed.py carries.

THE RULE IS sealed.derive_split, NOT A COPY OF IT
-------------------------------------------------
Same algorithm, different salt and a different input set. The record unit, the
(section, type) stratification, the coverage repair and the loose-case handling
are all the original's -- imported, not reimplemented, because a second copy
drifts from the first and this repo has already paid for one judgement
implemented twice (eval/refusal.py).

What differs, and only this:

    input     the 1,511 cases sealed.py left behind, never the full 1,865
    salt      a new one, so the draw is independent of the first split rather
              than a re-run of it on a smaller pool

DISJOINTNESS IS BY CONSTRUCTION AND ASSERTED ANYWAY. sealed2 can contain no
sealed1 case because sealed1's cases are not in the input. The self-test checks
it regardless, because "by construction" is the phrase that precedes most of
the findings in reports/.

THE RECORD UNIT, AGAIN
----------------------
Cases from one record share a page, a tree and a page image, so whole records
move together. sealed1 already took whole records, which means every record
represented in the input is represented WHOLLY -- the boundary this split draws
cannot cut one that the first split left intact.

WHAT THIS SET CANNOT MEASURE, stated before it is spent
-------------------------------------------------------
`injection_resistance` -- 0 of 15. Injection cases are generated at run time by
`safety.injection_cases`, which wraps every seventh code of `sorted(records)`,
and none of those 15 codes landed on the sealed2 side. The coverage guarantee
in derive_split works on the (section, type) pairs of qa_set cases and cannot
see generated ones; teaching it to would mean importing eval.metrics into a
split rule, which is the one import class this module is forbidden (a split
that can see a metric can be adjusted to flatter it).

This is not silent: run_eval lists `injection_resistance` under `not_measured`
on every sealed2 run, and the metric is measured on dev with n=15. But a
sealed2 number is a number about answering, not about resisting, and must not
be quoted as though the injection gate had been re-tested.

`clean_refusal` and `refusal_correctness` are thin for the same structural
reason -- the adversarial cases are loose (no record) and split per case, so
sealed2 holds 2 of them.

READS NO EVAL RESULT. EVER.
---------------------------
Section, case type, record identity and a content hash. It does not read which
cases any selector passes, and it was derived before any of them had a score.

No LLM, no randomness, no stored list of ids.
"""
from __future__ import annotations

import collections
import os
from typing import Dict, List, Set, Tuple

from eval.sealed import (NO_RECORD, derive_split, load_cases, record_of,
                         sealed_ids as sealed1_ids)

SEALED2_FRACTION = 0.20

# Distinct from SPLIT_SALT in sealed.py and from agent/holdout.py's. Sharing a
# salt would make the draws correlate -- the same records ranked the same way --
# so the second holdout would be the first one's leftovers in the same order
# rather than an independent sample of them.
SPLIT_SALT2 = "komatsu-sealed2-cases-v1"


def input_cases(cases: List[dict] = None) -> List[dict]:
    """The pool: everything sealed.py did not take.

    Scoring code may read eval results; THIS reads only the first split's ids,
    which are themselves derived from data. No run record is opened here or
    anywhere upstream of here.
    """
    cases = cases if cases is not None else load_cases()
    spent = sealed1_ids()
    return [c for c in cases if c["id"] not in spent]


def split(cases: List[dict] = None) -> Tuple[List[str], List[str]]:
    """(dev_case_ids, sealed2_case_ids). Deterministic."""
    return derive_split(input_cases(cases), SEALED2_FRACTION, SPLIT_SALT2)


_cache = None


def _split_cached():
    global _cache
    if _cache is None:
        _cache = split()
    return _cache


def sealed2_ids() -> Set[str]:
    return set(_split_cached()[1])


def dev_ids() -> Set[str]:
    """The cases a selector may be tuned on. 1,511 minus sealed2."""
    return set(_split_cached()[0])


def sealed2_records() -> Set[str]:
    """The RECORDS on the sealed2 side.

    Generated cases -- injection, conversation -- are built at run time and
    carry a source_code but no qa_set id, so they split on the record rule.
    """
    held = sealed2_ids()
    return {c.get("source_code") for c in load_cases()
            if c["id"] in held and c.get("source_code")}


def reset_cache() -> None:
    global _cache
    _cache = None


def describe(cases: List[dict] = None) -> dict:
    """Everything needed to audit the split without re-deriving it."""
    pool = input_cases(cases)
    dev, held = split(cases)
    hset = set(held)
    by = collections.defaultdict(lambda: {"pool": 0, "sealed2": 0})
    for c in pool:
        k = (c["section"], c["type"])
        by[k]["pool"] += 1
        if c["id"] in hset:
            by[k]["sealed2"] += 1
    cov = {f"{k[0]}/{k[1]}": dict(v, share=v["sealed2"] / v["pool"])
           for k, v in sorted(by.items())}
    recs = {c.get("source_code") for c in pool if c["id"] in hset}
    return {"n_pool": len(pool), "n_dev": len(dev), "n_sealed2": len(held),
            "fraction": len(held) / len(pool),
            "sealed2_records": len([r for r in recs if r]),
            "coverage": cov}


def self_test() -> None:
    cases = load_cases()
    a, b = split(cases), split(cases)
    assert a == b, "sealed2 selection is not deterministic"
    dev, held = a
    assert not (set(dev) & set(held)), "dev and sealed2 overlap"
    pool = {c["id"] for c in input_cases(cases)}
    assert set(dev) | set(held) == pool, "split does not cover the pool"

    # DISJOINT FROM THE SPENT SET. True by construction; asserted because the
    # cost of it silently becoming false is a "clean" number that is not one.
    assert not (set(held) & sealed1_ids()), \
        "sealed2 contains cases from the spent sealed set"

    # No record straddles dev / sealed2.
    hset = set(held)
    by_rec = collections.defaultdict(set)
    for c in cases:
        if c["id"] in pool and c.get("source_code"):
            by_rec[c["source_code"]].add(c["id"] in hset)
    straddle = sorted(r for r, sides in by_rec.items() if len(sides) > 1)
    assert not straddle, f"records split across dev and sealed2: {straddle[:5]}"

    d = describe(cases)
    for k, v in d["coverage"].items():
        assert v["sealed2"] > 0, f"{k} has no sealed2 cases -- the set is blind to it"
        assert v["pool"] - v["sealed2"] > 0, f"{k} has no dev cases left"
    assert 0.15 <= d["fraction"] <= 0.25, \
        f"sealed2 fraction drifted to {d['fraction']:.3f}"

    # The salt must actually differ from the first split's, or this is sealed1
    # re-run on its own leftovers.
    from eval.sealed import SPLIT_SALT
    assert SPLIT_SALT2 != SPLIT_SALT, "sealed2 shares sealed1's salt"


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    d = describe()
    print(f"pool {d['n_pool']} non-sealed cases -> dev {d['n_dev']} / "
          f"sealed2 {d['n_sealed2']}  ({d['fraction']:.1%}), "
          f"{d['sealed2_records']} records\n")
    print(f"{'section/type':36}{'pool':>8}{'sealed2':>9}{'share':>8}")
    for k, v in d["coverage"].items():
        print(f"{k:36}{v['pool']:>8}{v['sealed2']:>9}{v['share']:>8.1%}")
    self_test()
    print("\nself-test passed")
