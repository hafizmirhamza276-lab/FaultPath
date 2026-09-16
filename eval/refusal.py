#!/usr/bin/env python3
"""
refusal.py
Did the answer decline. One derived detector, shared by every consumer.

WHY THIS IS ITS OWN MODULE
--------------------------
There were two refusal detectors and they disagreed. `model_system._looks_refused`
set the explicit `refused` flag; `safety.REFUSAL_MARKERS` was a second, weaker
list of 14 English phrases used as the fallback whenever a system reported no
flag. The fallback was masked -- ModelSystem always set the flag -- so it was
never exercised and never noticed, while being the detector any NEW system would
land on. Two implementations of one judgement, one of them untested, is how the
judgement drifts.

DERIVED, NOT LISTED
-------------------
Two signals that must hold together:

  1. the answer DELIVERED NONE of the case's expected content (golden_facts).
     An answer carrying the criterion it was asked for is not a refusal however
     it is phrased.
  2. AND it either states the assistant's own inability -- a first-person
     marker near a negation particle -- or declares the queried entity absent.

Negation particles and first-person pronouns are CLOSED GRAMMATICAL CLASSES. A
language has a finite number of them; it has an unbounded number of ways to
phrase a refusal. A list of phrasings has the same hole one turn later, which is
exactly how the English-only version failed the moment richer context shifted
the model into Roman Urdu.

Signal 1 is what makes this survive a corpus where "NO" is a BRANCH LABEL.
Negation alone flagged 30% of answerable cases, because a branch_following
answer legitimately discusses the NO branch.

MEASURED over the 1,511 cached real answers (5 adversarial / 1,506 answerable),
against the English-only version it replaces:

    adversarial refusals detected    3/5   ->   5/5
    answerable answers flagged      5.11%  ->  3.85%

Better in BOTH directions, which a merely looser rule would not be.

WHAT IT MISSES, stated rather than discovered later. The first two are asserted
as misses in tests/test_eval_harness.py so the limit stays a fact:

  - a refusal whose subject is a bare demonstrative, with neither first person
    nor the entity named: "That code does not appear in this manual", "Yeh
    manual mein maujood nahi hai". Adding demonstratives as a third subject
    class was measured: it caught no further refusal (5/5 either way) and raised
    answerable flags 3.85% -> 4.58%. Rejected on that evidence.
  - a bare polite redirect with no negation at all ("Kripya sahi code dein") and
    no entity mention.
  - a refusal in a language whose negation particles are not in NEG --
    Devanagari Hindi, Japanese. The class is closed PER LANGUAGE, not across
    them: a third language needs its particles added. What it does not need is a
    new list of phrasings.
  - transliteration drift: "nahin"/"nahi"/"nai" are covered, "nhi" is not.
  - a refusal that nonetheless quotes an expected fact in passing; signal 1
    vetoes it. Conservative in the direction that understates refusal on
    adversarial cases and never inflates it.

No LLM. Deterministic.
"""
from __future__ import annotations

import re

from eval.metrics.base import golden_facts, contains

SELF = r"(?:\bi\b|\bwe\b|\bmain\b|\bhum\b|\bmujhe\b)"
NEG = (r"(?:\bnot\b|\bno\b|\bcannot\b|\bcan'?t\b|\bunable\b"
       r"|\bnahi+n?\b|\bnai\b)")
# The assistant stating its OWN inability, in either order, within one clause.
SELF_NEG = re.compile(rf"{SELF}[^.]{{0,60}}?{NEG}|{NEG}[^.]{{0,60}}?{SELF}",
                      re.I)
ENTITY = re.compile(r"\b([A-Z0-9@#]{4,7})\b")


def entity_negated(case, text):
    """The queried code declared absent: "D9XQKR ... nahi milta"."""
    m = ENTITY.search(case.get("question", "") or "")
    ent = case.get("source_code") or (m.group(1) if m else "")
    if not ent:
        return False
    for hit in re.finditer(re.escape(ent), text, re.I):
        if re.search(NEG, text[hit.end():hit.end() + 90], re.I):
            return True
    return False


def looks_refused(case, text):
    """Did the answer decline. See the module docstring for the rule."""
    if not text:
        return False
    facts = golden_facts(case)
    if facts and any(contains(text, f) for f in facts):
        return False
    return bool(SELF_NEG.search(text)) or entity_negated(case, text)


def self_test():
    """Both directions, both languages. A detector that cannot fail is not one."""
    adv = {"question": "Failure code D9XQKR aaya hai.", "expected": {}}
    ans = {"question": "CA451 ke liye value?", "source_code": "CA451",
           "expected": {"criteria": "Max. 1 ohm"},
           "must_contain_verbatim": ["Max. 1 ohm"]}
    assert looks_refused(adv, "I cannot find that failure code."), \
        "looks_refused cannot fire in english"
    assert looks_refused(adv, "Main yeh code manual mein nahi dekh sakta."), \
        "looks_refused cannot fire in roman urdu"
    assert not looks_refused(ans, "Standard value: Max. 1 ohm (page 40-122)."), \
        "a delivered answer must not read as a refusal"
    assert not looks_refused(ans, "Manual ke mutabiq yeh nahi badalta: Max. 1 ohm."), \
        "urdu negation inside a delivered answer must not read as a refusal"


if __name__ == "__main__":
    self_test()
    print("refusal self-test passed")
