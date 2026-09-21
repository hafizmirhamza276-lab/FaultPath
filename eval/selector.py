#!/usr/bin/env python3
"""
selector.py
Which rows of a retrieved record the model is shown. Derived from the QUESTION.

WHAT THIS REPLACES, AND WHY
---------------------------
The previous selector read case["expected"] -- the answer key. It promoted the
row carrying `expected.point`, and chose its row order from
`bool(expected.criteria)`. `expected` does not exist at inference time, so the
harness was locating the answer row and then scoring the model on whether it
read the row it had been handed. Measured: 7/7 gates with the oracle, 5/7
without. See reports/oracle_selection_finding.md.

Nothing in this module may read `expected`, `gold`, or any key of either.
tests/test_eval_harness.py asserts that over the parsed AST of every function
reachable from build_prompt, with the forbidden key set DERIVED from qa_set
rather than listed here, and with a planted read proving the check fires.

THE RULE: MATCH THE QUESTION AGAINST TEXT WE OWN
------------------------------------------------
Rule 4 in CLAUDE.md. The question is the technician's; the record's rows are
ours, extracted deterministically from the manual. So every judgement here is
"does this question token appear in this row", and never "does this question
look like one of the following phrasings".

There is no phrase list in this file, in either language. There is no stop-word
list either, and that is not an oversight: the questions are romanised Hinglish
over English technical content ("CA122 ke liye Between ECM (female) (37) and
(44) par resistance ki standard value kya honi chahiye?"), and the scaffold
words -- ke, liye, par, ki, kya -- appear in no row of any record, so they
score zero without anyone having to enumerate them. A stop list would be a
phrase list wearing a different hat, and would need a new entry per language.

DISCRIMINATIVENESS IS IDF OVER THE RECORD'S OWN ROWS
----------------------------------------------------
Not over the corpus, and not a weighting anyone chose. A token that appears in
every row of a record identifies nothing inside it: `resistance` in a record
that is forty resistance measurements is worth ~0. A token in one row carries
the whole match: a pin number, a connector designation, a one-off component
name.

This is also where abbreviation robustness comes from for free. A technician
writing "res" instead of "resistance" loses a token that was worth almost
nothing anyway, because that record's rows all said Resistance. No synonym
table, no prefix matching, no per-abbreviation rule.

THE THREE PRIORITIES
--------------------
  0. HEADER rows, plus the FIRST row of each other kind -- the record's entry
     points. Structural and case-independent: a record's opening step is part
     of what makes the extract readable as a procedure, and "which check comes
     first" is answered by the record's own order, which no question token
     points at. Costs ~4 rows and is what takes dev reachability from 99.66% to
     100%.
  1. Rows the question MATCHES, best first. Ranked by score / sqrt(row length),
     because the budget is spent in characters: a long row must earn its cost.
     Raw score leaves 7 dev answers outside the budget, density leaves 4.
  2. Everything else, by KIND, in an order the question also decides -- kinds
     ranked by the total match they attracted, ties broken by TAIL. When a
     question matches nothing specific ("sab se pehle kya check karna
     chahiye?") every row scores ~0, no kind outranks another, and TAIL puts
     step prose first. That is the correct default for a question with no
     measurement signal in it, and it is derived from the absence of evidence
     rather than from a flag.

Rows are then emitted in RECORD order, so the model reads something shaped like
the manual rather than a ranked list.

WHAT THIS MISSES, stated rather than discovered later
-----------------------------------------------------
  - A question naming the measuring point in words the manual does not use at
    all -- a different name for the same connector. There is no synonym source
    in this repo that is not a model, so there is nothing to derive one from.
  - A question whose only discriminating token is numeric and appears in many
    rows: "(64)" in a record where forty rows mention pin 64. IDF drives it to
    near zero and the tail order decides, correctly but weakly.
  - Perturbation below the token: a one-character typo removes that token
    entirely. Measured in tests/test_eval_harness.py against a deterministic
    English perturbation set (eval/perturb.py) rather than assumed.
  - ON THIS CORPUS THE QUESTION CONTAINS THE MEASURING POINT VERBATIM IN 100%
    OF CASES (701 of 701 dev cases with an expected point). build_qa_set.py
    templates it in. So question-anchoring is being measured under conditions
    far friendlier than a technician typing freely, and the perturbation
    numbers, not the clean ones, are the honest estimate.

No LLM. Deterministic.
"""
from __future__ import annotations

import collections
import math
import re

# Words and numbers. `p68` and `cm02` stay whole; `(37)` yields `37`; `&`, `-`
# and punctuation vanish, so "pins 37 and 44", "(37) and (44)" and "37-44" all
# reduce to the same two numeric tokens.
TOKEN = re.compile(r"[a-z]+[0-9]*|[0-9]+")

# The fallback order for kinds the question says nothing about. Step prose
# first: a question with no measurement signal is a question about procedure.
TAIL = ("step", "remedy", "branch", "measurement")

# The one order used when the selector is told nothing about the question --
# see model_system.selector_blind(). Measurements first, because the head slice
# this whole line of work began with kept header and prose and dropped the
# measurement table.
BLIND_ORDER = ("header", "measurement", "remedy", "branch", "step")


def tokens(text: str) -> list:
    return TOKEN.findall((text or "").lower())


def row_kind(line: str) -> str:
    """BOTH renderers. Section 40 emits "Step n"; a FLAT symptom tree emits
    "Check n" and "Check n remedy", because the two tree kinds invert and
    eval/chunkers.py keeps that distinction in the text deliberately. Treating
    a Check row as header -- which a Step-only rule does -- puts every flat
    symptom row in priority 0 and spends the budget before the measurements.
    """
    if line.startswith("Measurement.") or " measurement." in line[:26]:
        return "measurement"
    if line.startswith("Step ") and (" YES:" in line or " NO:" in line):
        return "branch"
    if " remedy:" in line[:24]:
        return "remedy"
    if line.startswith(("Step ", "Check ")):
        return "step"
    if line.startswith("Refers elsewhere:"):
        return "step"
    return "header"


def score_rows(lines, question):
    """Per-row match of `question` against THESE rows. -> (scores, idf).

    IDF is computed over this record only, so "discriminating" means
    "discriminating INSIDE the record the model is looking at" -- which is the
    judgement actually being made. A corpus-wide IDF would rate `resistance`
    highly in a record where every row is a resistance measurement.
    """
    q = set(tokens(question))
    if not q or not lines:
        return [0.0] * len(lines), {}
    row_toks = [set(tokens(ln)) for ln in lines]
    n = len(lines)
    df = collections.Counter()
    for rt in row_toks:
        for t in rt & q:
            df[t] += 1
    idf = {t: max(0.0, math.log(n / (1 + d))) for t, d in df.items()}
    return [sum(idf.get(t, 0.0) for t in (rt & q)) for rt in row_toks], idf


def kind_order(kinds, scores):
    """Which kinds the question is about, by the match they attracted."""
    totals = collections.Counter()
    for k, s in zip(kinds, scores):
        if k != "header":
            totals[k] += s
    return sorted(TAIL, key=lambda k: (-totals[k], TAIL.index(k)))


def entry_points(kinds):
    """The index of the first row of each non-header kind.

    A record's opening step, first measurement and first branch are what make
    an extract read as a procedure rather than as fragments, and they are the
    answer to "what do I check first" -- a question that names no row and so
    can never be anchored. Structural, case-independent, ~4 rows.
    """
    first = {}
    for i, k in enumerate(kinds):
        if k != "header":
            first.setdefault(k, i)
    return set(first.values())


def select(text: str, question: str, budget: int, blind: bool = False) -> str:
    """The rows of one record worth showing, within `budget` characters.

    Takes the QUESTION, not the case: a function that never receives the case
    object cannot read its answer key, which is a stronger guarantee than a
    convention about which fields to touch.
    """
    lines = (text or "").split("\n")
    if not lines:
        return ""

    if blind:
        ranked = [(BLIND_ORDER.index(row_kind(ln)), 0.0, i, ln)
                  for i, ln in enumerate(lines)]
    else:
        scores, _ = score_rows(lines, question)
        kinds = [row_kind(ln) for ln in lines]
        order = kind_order(kinds, scores)
        entry = entry_points(kinds)
        ranked = []
        for i, ln in enumerate(lines):
            if kinds[i] == "header" or i in entry:
                ranked.append((0, 0.0, i, ln))
            elif scores[i] > 0:
                ranked.append((1, -scores[i] / math.sqrt(len(ln) + 1), i, ln))
            else:
                ranked.append((2 + order.index(kinds[i]), 0.0, i, ln))
    ranked.sort(key=lambda r: (r[0], r[1], r[2]))

    kept, used = [], 0
    for _, _, i, ln in ranked:
        if used + len(ln) + 1 > budget:
            continue              # a later, shorter row may still fit
        kept.append((i, ln))
        used += len(ln) + 1
    # RECORD ORDER, not priority order.
    return "\n".join(ln for _, ln in sorted(kept))


def wants_value(text: str, question: str) -> bool:
    """Does this question look like it is after a measured value.

    DERIVED FROM THE QUESTION AND THE RECORD, never from expected.criteria. It
    is the same signal the kind order already uses -- whether the match
    concentrates in measurement rows -- exposed so its accuracy can be
    REPORTED. Scoring may read the answer key; selection may not, and this is
    selection, so it is measured against expected.criteria and never fed by it.
    """
    lines = (text or "").split("\n")
    scores, _ = score_rows(lines, question)
    return kind_order([row_kind(ln) for ln in lines], scores)[0] == "measurement"


def self_test() -> None:
    """A selector that cannot fail to find a row is not selecting."""
    # A record shaped like the real ones: many near-identical measurement
    # rows, the asked-about one late. This is the shape the head slice and the
    # blind order both fail on.
    rows = ["Failure code CA122 Controller: engine"]
    rows += [f"Step {i}. Cause: Circuit {i}. Procedure: check circuit {i}."
             for i in range(1, 10)]
    rows += [f"Step {i} measurement. Resistance. Measuring point: Between "
             f"ECM (female) ({i}) and ({i + 30}). Standard value: Max. 1 ohm"
             for i in range(1, 10)]
    rec = "\n".join(rows)
    q = "CA122 ke liye Between ECM (female) (9) and (39) par resistance ki value?"
    budget = 600
    out = select(rec, q, budget)
    assert "(9) and (39)" in out, "the anchored row was not selected"
    # The fixture must SEPARATE. Blind fills with the earlier measurements and
    # never reaches row 9; if it did, nothing below would be evidence.
    assert "(9) and (39)" not in select(rec, q, budget, blind=True), \
        "the fixture does not separate question-derived from blind"
    # A procedure question anchors nothing and must still open with step 1.
    assert "Step 1. Cause" in select(rec, "CA122 sab se pehla check kya hai?",
                                     budget=300)
    assert wants_value(rec, q) and not wants_value(
        rec, "CA122 sab se pehla check kya hai?")


if __name__ == "__main__":
    self_test()
    print("selector self-test passed")
