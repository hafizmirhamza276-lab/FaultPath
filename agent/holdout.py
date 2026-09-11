#!/usr/bin/env python3
"""
holdout.py
The held-out 20% of failure codes, stratified.

Lives here rather than under tests/ so that fixture builders exclude it BY
CONSTRUCTION instead of by remembering to. A holdout a fixture can reach by
accident is not a holdout, and the accident is silent -- the metric still
reports a number, it just stops meaning generalisation.

THE COLUMN-SPLIT EXCLUSION
--------------------------
No code carrying extraction_warning == "column_split_recovered" may be in the
holdout. Those 12 codes have steps the parser RECONSTRUCTED from a mangled
table. Asking "did we tune to our fixtures?" of a code whose content the parser
partly invented is circular in its own right: a disagreement there could be
overfitting or could be the reconstruction, and the holdout cannot tell you
which. They belong in the human-transcription pool, where an independent reader
settles it.

This is a deliberate trade, and it costs something -- see BIAS below.

SELECTION RULE, in code rather than in a comment:
  1. Eligible = all codes minus the column-split codes.
  2. Stratify on (format, action_level, page-span band, pointer-only).
  3. Take floor(0.2 * n) from each stratum by stable rank, so no stratum is
     over- or under-drawn relative to its size.
  4. Top up to the global 20% target from the highest-ranked remaining codes,
     so the total is exact.
Deterministic and content-addressed: the same split on every machine, forever.
A random split would make train-vs-holdout comparisons noise.

BIAS, stated rather than papered over
-------------------------------------
The transcription set is defined by a property of the data (column_split) and
the holdout is drawn from what remains. That ordering IS a bias and it has a
direction: the holdout now systematically excludes the pages the parser found
hardest, so it measures generalisation on cleaner-than-average material and
cannot detect tuning that only shows up on difficult layouts.

That limitation is real and is not fixable by reordering -- the two constraints
genuinely conflict. What closes it is the other half: all 12 column-split codes
go to human transcription, where a person reads the page directly. The holdout
covers "did we tune to our fixtures on cleanly-extracted codes"; the human round
covers "did the parser read the hard pages correctly". Neither answers both.
Quoting either as if it covered the whole corpus would be wrong.
"""
from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Set, Tuple

HOLDOUT_FRACTION = 0.20
SPLIT_SALT = "komatsu-holdout-v2-stratified"

COLUMN_SPLIT_WARNING = "column_split_recovered"


def _rank(code: str) -> float:
    """Stable pseudo-random rank in [0,1). Same order on every machine."""
    h = hashlib.sha256(f"{SPLIT_SALT}:{code}".encode()).hexdigest()
    return int(h[:12], 16) / 0xFFFFFFFFFFFF


def has_column_split(rec: dict) -> bool:
    return any(s.get("extraction_warning") == COLUMN_SPLIT_WARNING
               for s in rec.get("steps", []))


def page_span_band(rec: dict) -> str:
    n = len(set(rec.get("manual_pages") or []))
    return "1" if n <= 1 else "2-3" if n <= 3 else "4-6" if n <= 6 else "7+"


def stratum(rec: dict) -> Tuple[str, str, str, str]:
    """The four axes the split must stay balanced across."""
    return (rec.get("format") or "?",
            str(rec.get("action_level") or "-"),
            page_span_band(rec),
            "ptr" if rec.get("is_pointer_only") else "std")


def split(records: Dict[str, dict]) -> Tuple[List[str], List[str]]:
    """(train, holdout). Stratified, deterministic, column-split-free."""
    eligible = {c: r for c, r in records.items() if not has_column_split(r)}
    excluded = sorted(set(records) - set(eligible))

    by_stratum: Dict[tuple, List[str]] = {}
    for c, r in eligible.items():
        by_stratum.setdefault(stratum(r), []).append(c)

    target = int(round(len(eligible) * HOLDOUT_FRACTION))

    # Largest-remainder allocation. A plain floor sends every stratum smaller
    # than 1/FRACTION to zero, and with a four-axis key most strata are small --
    # action level L04 (7 codes) fragments across five buckets and would take
    # none of them. Largest remainder gives the fractional entitlements to the
    # strata that earned most of one.
    holdout: List[str] = []
    remainders: List[tuple] = []
    for key in sorted(by_stratum):
        members = sorted(by_stratum[key], key=_rank)
        exact = len(members) * HOLDOUT_FRACTION
        take = int(exact)
        holdout += members[:take]
        if take < len(members):
            remainders.append((exact - take, _rank(members[take]), key,
                               members[take]))
    for _, _, _, code in sorted(remainders, key=lambda x: (-x[0], x[1])):
        if len(holdout) >= target:
            break
        holdout.append(code)

    # Axis-coverage guarantee. A band with enough members to be worth measuring
    # must appear in the holdout, or the split silently stops covering it.
    # Never taken from a stratum whose last train member it would be: a value
    # the agent can no longer see is worse than one it cannot be scored on.
    hset = set(holdout)
    for axis_fn in (lambda r: str(r.get("action_level") or "-"), page_span_band,
                    lambda r: r.get("format") or "?"):
        counts: Dict[str, int] = {}
        held: Dict[str, int] = {}
        for c, r in eligible.items():
            k = axis_fn(r)
            counts[k] = counts.get(k, 0) + 1
            if c in hset:
                held[k] = held.get(k, 0) + 1
        for k, n in sorted(counts.items()):
            if n >= 5 and held.get(k, 0) == 0:
                cands = sorted((c for c, r in eligible.items()
                                if axis_fn(r) == k and c not in hset), key=_rank)
                for c in cands:
                    others = [x for x in by_stratum[stratum(eligible[c])]
                              if x != c and x not in hset]
                    if others:                       # not the last of its stratum
                        holdout.append(c)
                        hset.add(c)
                        break

    hset = set(holdout)
    train = sorted([c for c in eligible if c not in hset] + excluded)
    return train, sorted(hset)


_cache = None


def _split_cached():
    global _cache
    if _cache is None:
        from agent import tools
        _cache = split(tools.records())
    return _cache


def train_codes() -> List[str]:
    return _split_cached()[0]


def holdout_codes() -> List[str]:
    return _split_cached()[1]


def holdout_set() -> Set[str]:
    return set(_split_cached()[1])


def reset_cache() -> None:
    """For tests that need to re-derive the split under mutated input."""
    global _cache
    _cache = None


def describe(records: Dict[str, dict] = None) -> dict:
    """Everything needed to audit the split without re-deriving it."""
    if records is None:
        from agent import tools
        records = tools.records()
    train, hold = split(records)
    hs = set(hold)
    cov = {}
    for axis, fn in (("format", lambda r: r.get("format")),
                     ("action_level", lambda r: str(r.get("action_level") or "-")),
                     ("page_span", page_span_band),
                     ("pointer", lambda r: "ptr" if r.get("is_pointer_only") else "std")):
        allc: Dict[str, int] = {}
        hc: Dict[str, int] = {}
        for c, r in records.items():
            k = fn(r)
            allc[k] = allc.get(k, 0) + 1
            if c in hs:
                hc[k] = hc.get(k, 0) + 1
        cov[axis] = {k: {"corpus": allc[k], "holdout": hc.get(k, 0),
                         "share": (hc.get(k, 0) / allc[k]) if allc[k] else 0.0}
                     for k in sorted(allc)}
    return {"train": len(train), "holdout": len(hold),
            "fraction": len(hold) / len(records),
            "excluded_column_split": sorted(
                c for c, r in records.items() if has_column_split(r)),
            "strata_total": len({stratum(r) for r in records.values()}),
            "strata_eligible": len({stratum(r) for c, r in records.items()
                                    if not has_column_split(r)}),
            "coverage": cov, "holdout_codes": hold}


def self_test() -> None:
    from agent import tools
    recs = tools.records()
    a = split(recs)
    b = split(recs)
    assert a == b, "holdout selection is not deterministic"
    train, hold = a
    assert set(train) & set(hold) == set(), "train and holdout overlap"
    assert set(train) | set(hold) == set(recs), "split does not cover the corpus"

    cs = {c for c, r in recs.items() if has_column_split(r)}
    assert not (set(hold) & cs), \
        f"column-split codes in the holdout: {sorted(set(hold) & cs)}"
    assert cs <= set(train), "column-split codes must all be available to train"

    frac = len(hold) / len(recs)
    assert 0.15 <= frac <= 0.25, f"holdout fraction drifted to {frac:.3f}"

    # every axis value present in the corpus must still be represented in train
    d = describe(recs)
    for axis, vals in d["coverage"].items():
        for k, v in vals.items():
            assert v["corpus"] - v["holdout"] > 0, \
                f"stratum {axis}={k} has no train members left"
