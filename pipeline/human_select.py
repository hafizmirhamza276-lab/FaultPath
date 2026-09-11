#!/usr/bin/env python3
"""
human_select.py
Choose the 25 codes a person will transcribe by hand.

The resolver verifies 3,266 facts, but the resolver and the extractor share
normalisation rules. If both misread the same thing the same way, nothing
catches it -- and "nothing catches it" is precisely how 27 steps went missing and
how citation_accuracy read 1.0000 over 635 wrong pages. Independent human
transcription is the only external truth available.

Selection is deterministic, seeded, and justified per code. Convenience would
bias the sample toward pages that are easy to read, which are exactly the pages
least likely to be wrong.

DISJOINT from the 20% metric holdout, by construction and by test. If the two
overlapped, the human check and the generalisation check would be measuring the
same codes and neither would mean what it says.

No LLM. Nothing here reads an extracted value to decide anything.
"""
from __future__ import annotations

import hashlib
import os
import sys
from typing import Dict, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import holdout as metric_holdout          # noqa: E402
from agent import tools                              # noqa: E402

TARGET = 25
SEED = "komatsu-human-verify-v1"


def _rank(code: str) -> float:
    """Stable pseudo-random rank. Same order on every machine, forever."""
    h = hashlib.sha256(f"{SEED}:{code}".encode()).hexdigest()
    return int(h[:12], 16) / 0xFFFFFFFFFFFF


def select(records: Dict[str, dict] = None) -> List[dict]:
    recs = records or tools.records()
    excluded = metric_holdout.holdout_set()
    pool = {c: r for c, r in recs.items() if c not in excluded}

    chosen: Dict[str, str] = {}

    def take(code, reason):
        if code in pool and code not in chosen:
            chosen[code] = reason

    # 1. Every code carrying a column-split recovery. Highest risk in the
    #    corpus: the extractor reassembled the text, no one has read the page,
    #    and both README and METRICS say these need eye verification.
    split = sorted(c for c, r in pool.items()
                   if any(s.get("extraction_warning") == "column_split_recovered"
                          for s in r.get("steps", [])))
    for c in split:
        take(c, "column-split recovery: machine-repaired, never eye-verified")

    # 2. All three cause-table formats represented. Format C is the one that
    #    silently collapsed to a flat list before it was handled.
    for fmt in ("A", "B"):
        for c in sorted((c for c, r in pool.items() if r["format"] == fmt),
                        key=_rank):
            if sum(1 for x in chosen if pool[x]["format"] == fmt) >= 3:
                break
            take(c, f"format {fmt} coverage")
    # The KOMTRAX/communication layout (7 columns, YES/NO in column 6) is
    # stored as format A but is structurally distinct; sample it explicitly.
    for c in sorted((c for c in pool if c.startswith("F3")), key=_rank)[:2]:
        take(c, "KOMTRAX/communication cause-table layout (the third format)")

    # 3. Multi-page extremes, including the 11-page code. Provenance spanning
    #    many pages is where a per-code citation goes most wrong.
    by_pages = sorted(pool, key=lambda c: (-len(set(pool[c]["manual_pages"])),
                                           _rank(c)))
    for c in by_pages[:3]:
        take(c, f"multi-page extreme: spans "
                f"{len(set(pool[c]['manual_pages']))} manual pages")

    # 4. At least two pointer-only codes -- their whole content is a redirect
    #    and the synthetic label is not page text.
    for c in sorted((c for c, r in pool.items() if r["is_pointer_only"]),
                    key=_rank)[:2]:
        take(c, "pointer-only redirect")

    # 5. Fill the remainder by stratified sample across action levels, so the
    #    sample is not concentrated in one severity band.
    by_level: Dict[str, List[str]] = {}
    for c, r in pool.items():
        by_level.setdefault(str(r.get("action_level") or "-"), []).append(c)
    levels = sorted(by_level)
    i = 0
    while len(chosen) < TARGET and levels:
        lvl = levels[i % len(levels)]
        i += 1
        cands = [c for c in sorted(by_level[lvl], key=_rank) if c not in chosen]
        if not cands:
            levels.remove(lvl)
            i = 0
            continue
        take(cands[0], f"stratified fill: action level {lvl}")

    out = [{"code": c, "reason": chosen[c],
            # ROUND ASSIGNMENT, recorded here and written to the manifest --
            # never inferred at runtime. Round 1 is every column-split code:
            # machine-repaired text no person has ever read, flagged
            # NEEDS_HUMAN_VERIFICATION since the beginning. ~3 hours instead of
            # 13, and it covers the highest-risk facts in the corpus.
            "round": 1 if any(
                s.get("extraction_warning") == "column_split_recovered"
                for s in pool[c]["steps"]) else 2,
            "format": pool[c]["format"],
            "pages": sorted(set(pool[c]["manual_pages"])),
            "pdf_pages": pool[c]["pdf_pages"],
            "n_steps": len(tools.real_steps(pool[c])),
            "n_measurements": len(pool[c]["standalone_measurements"])
            + sum(len(s["measurements"]) for s in pool[c]["steps"]),
            "column_split": any(
                s.get("extraction_warning") == "column_split_recovered"
                for s in pool[c]["steps"]),
            "pointer_only": pool[c]["is_pointer_only"]}
           for c in sorted(chosen, key=_rank)]
    return out[:TARGET]


def round1_codes(records=None):
    """The Round 1 set. Derived once, from the manifest's own rule."""
    return [d for d in select(records) if d["round"] == 1]


def round2_codes(records=None):
    return [d for d in select(records) if d["round"] == 2]


def repaired_step_coverage(records=None) -> dict:
    """How many of the 27 machine-repaired steps Round 1 can actually reach.

    NOT 27. Four column-split codes fall inside the 20% metric holdout and
    cannot be transcribed without destroying its independence, so 10 repaired
    steps are out of reach by design. Stating the real denominator matters more
    than reaching a round number.
    """
    recs = records or tools.records()
    def steps_of(codes):
        return sum(1 for c in codes for s in recs[c]["steps"]
                   if s.get("extraction_warning") == "column_split_recovered")
    all_cs = {c for c, r in recs.items()
              if any(s.get("extraction_warning") == "column_split_recovered"
                     for s in r["steps"])}
    r1 = {d["code"] for d in round1_codes(recs)}
    blocked = all_cs & metric_holdout.holdout_set()
    return {"total_repaired_steps": steps_of(all_cs),
            "round1_codes": sorted(r1),
            "round1_repaired_steps": steps_of(r1),
            "blocked_codes": sorted(blocked),
            "blocked_repaired_steps": steps_of(blocked),
            "reason_blocked": "inside the 20% metric holdout; transcribing them "
                              "would make the human check and the generalisation "
                              "check measure the same codes"}


def example_code(records: Dict[str, dict] = None) -> str:
    """A worked example, deliberately NOT one of the 25.

    The example shows a completed transcription. If it were one of the 25 the
    person would have seen the answer before transcribing it, which is a review
    rather than an independent read.
    """
    recs = records or tools.records()
    picked = {d["code"] for d in select(recs)}
    excluded = metric_holdout.holdout_set() | picked
    for c in sorted(recs, key=_rank):
        r = recs[c]
        if c in excluded or r["is_pointer_only"]:
            continue
        if 2 <= len(tools.real_steps(r)) <= 4 and len(set(r["manual_pages"])) == 1:
            return c
    return sorted(set(recs) - excluded)[0]


def self_test() -> None:
    """Selection must be deterministic and must not touch the metric holdout."""
    a = [d["code"] for d in select()]
    b = [d["code"] for d in select()]
    assert a == b, "selection is not deterministic"
    assert len(a) == len(set(a)) == TARGET, f"expected {TARGET} unique codes, got {len(a)}"
    overlap = set(a) & metric_holdout.holdout_set()
    assert not overlap, \
        f"transcription set intersects the metric holdout: {sorted(overlap)}"
    assert example_code() not in set(a), \
        "the worked example must not be one of the 25"


if __name__ == "__main__":
    self_test()
    sel = select()
    print(f"{len(sel)} codes selected for human transcription\n")
    print(f"{'code':9}{'fmt':5}{'steps':>6}{'meas':>6}{'pages':>7}  reason")
    for d in sel:
        print(f"{d['code']:9}{d['format']:5}{d['n_steps']:>6}{d['n_measurements']:>6}"
              f"{len(d['pages']):>7}  {d['reason']}")
    tot_steps = sum(d["n_steps"] for d in sel)
    tot_meas = sum(d["n_measurements"] for d in sel)
    print(f"\ntotals: {tot_steps} steps, {tot_meas} measurements, "
          f"{sum(len(d['pages']) for d in sel)} manual pages")
    print(f"worked example (not in the 25): {example_code()}")
    print(f"disjoint from metric holdout: "
          f"{not (set(d['code'] for d in sel) & metric_holdout.holdout_set())}")
