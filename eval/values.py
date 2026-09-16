#!/usr/bin/env python3
"""
values.py
One definition of "a value a technician could act on wrongly".

THE CONFUSION THIS EXISTS TO END
--------------------------------
`numbers_in()` returns every numeric token. Six metrics read that as "values the
answer delivered", and a Komatsu answer is full of digits that are NOT values:

    PC200-10M0          a machine model
    SEN06867-13         a manual id
    40-241              a page
    Step 1              an ordinal
    CA451               a failure code

The same bug was then found and patched THREE separate times, each in isolation:

  1. fabricated_values   page tokens counted as invented values
                         -> patched by stripping pages in claim_text()
  2. injection_resistance "Step 1: Wiring harness" counted as delivering a value
                         -> patched in place with CRIT_VALUE
  3. clean_refusal       machine identity and pages counted as a leak, so the
                         metric read 0.0000 on a model whose five refusals were
                         all correct and leaked nothing

Three call sites worked the distinction out separately and two got it wrong.
This module is the fix for that, rather than a fourth patch: there is now ONE
way to ask, and tests/test_eval_harness.py fails a NEW direct `numbers_in` call
in eval/metrics/ unless it is registered with a reason.

THE DISTINCTION
---------------
A value is anything a technician could measure against or act on -- a criterion,
a bound, a range, a PIN NUMBER. An identifier is a name for something; a locator
says where to look. Pin numbers are values, deliberately: `model_leakage` exists
because "PC200 and PC490 share failure codes but not pin numbers", so a metric
blind to pins would be blind to what it was built for.

So the rule is NOT "numbers that look like measurements". It is EVERY NUMBER
EXCEPT THE IDENTIFIERS AND LOCATORS, which is the smaller and more defensible
claim: we can enumerate the identifier shapes this manual uses, and we cannot
enumerate every shape a value takes.

VALIDATED IN BOTH DIRECTIONS, against the ground truth rather than by eye:
  - all 872 criteria in golden/ yield at least one value. If the mask were too
    greedy this would not hold, and tests/test_eval_harness.py asserts it.
  - machine identity, manual ids, page tokens, step ordinals and the 174 known
    failure codes yield none.

WHAT IT MISSES, stated rather than discovered later:
  - a failure code NOT in golden/ -- a fabricated or wrong-model code such as
    X0X0X0 or CA118 -- is not masked, because the mask is derived from the
    ground truth rather than guessed from shape. Those codes come from the
    QUESTION, so asserted_values() subtracts what the question supplied, which
    is the repo's existing idiom and covers every case in the corpus.
  - ORDINALS other than "Step N". "Perform troubleshooting cause 5 to 10
    sequentially" reads as the value range 5-10. Measured: this is the ONE
    case in 1,578 real answers where converging onto this module flagged
    something the old predicate did not, against seven false positives it
    removed. A `cause\\s*\\d+` rule is not added for it, deliberately -- a mask
    extended to fit a single observed case is tuning on a measurement, and it
    would only half-work anyway ("cause 5 to 10" would still strand the 10.
    Any fix here has to handle ordinal RANGES, which is a real change and
    wants its own evidence.
  - an identifier shape this manual does not use. The mask is specific to
    SEN06867-13 and would need extending for another manual.
  - a bare number with no unit and no context ("set it to 12") reads as a
    value. That is the conservative direction: it over-counts values rather
    than under-counting them, and every metric here is safer that way.

No LLM. Deterministic.
"""
from __future__ import annotations

import glob
import json
import os
import re

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Identifier and locator shapes THIS manual uses. Everything else is a value.
_IDENT_SHAPES = (
    r"\d{2}-\d{1,4}",                       # manual page: 40-241
    r"PC\d{2,4}(?:LC)?-\d+[A-Z]?\d*",       # machine model: PC200-10M0
    r"SEN\d{4,}(?:-\d+)?",                  # manual id: SEN06867-13
    r"(?:s/n|serial)\s*:?\s*\d{4,}",        # serial: S/N 700001
    r"\bstep\s*\d{1,2}\b",                  # ordinal: Step 1
)

_IDENT_RE = None


def _identifier_re():
    """The mask, with the 174 real failure codes folded in. Built at CALL time.

    Derived from golden/ rather than from a shape like [A-Z0-9@#]{4,7}: that
    shape also matches "100KPA", and a mask that eats a unit would silently
    delete the value it is meant to preserve.
    """
    global _IDENT_RE
    if _IDENT_RE is None:
        codes = set()
        for p in glob.glob(os.path.join(GOLD, "failure_codes", "*.json")):
            if os.path.basename(p) == "index.json":
                continue
            with open(p, encoding="utf-8") as f:
                codes.add(json.load(f)["code"])
        alts = list(_IDENT_SHAPES)
        if codes:
            alts.append(r"\b(?:%s)\b" % "|".join(
                sorted((re.escape(c) for c in codes), key=len, reverse=True)))
        _IDENT_RE = re.compile("|".join(alts), re.I)
    return _IDENT_RE


def reset_cache() -> None:
    """Forget the derived mask. For tests that point GOLD_DIR elsewhere."""
    global _IDENT_RE
    _IDENT_RE = None


def identifiers_in(text) -> list:
    """The identifier and locator spans, as they appear."""
    return _identifier_re().findall(str(text or ""))


def actionable_values(text) -> set:
    """Numbers a technician could act on: everything that is not an identifier.

    THE ONE EXTRACTOR. If you are about to call numbers_in() to ask "did this
    answer state a value", this is the function you want instead.
    """
    return set(NUMBER_RE.findall(_identifier_re().sub(" ", str(text or ""))))


def states_a_value(text) -> bool:
    return bool(actionable_values(text))


def asserted_values(case, text) -> set:
    """Values the ANSWER asserts -- excluding anything the question supplied.

    A number the technician typed is not something the system produced, so it
    cannot be a leak, an invention or a contradiction. Subtracting the question
    is what keeps a fabricated code's own digits ("X0X0X0") out of the answer's
    account: such codes are not in golden/ and so are not masked.
    """
    return actionable_values(text) - actionable_values(
        (case or {}).get("question", ""))


def values_in_record(rec) -> set:
    """Every value anywhere in a golden record, identifiers excluded.

    Serialising the whole record is deliberate, for the reason numbers_in_record
    gives: a number is legitimate if it appears in the ground truth AT ALL.
    What changed is that its pages, codes and model no longer count as values.
    """
    return actionable_values(json.dumps(rec, ensure_ascii=False))


def self_test() -> None:
    """Both directions. A predicate that cannot fail is not measuring anything."""
    assert actionable_values("Max. 1 Ω") == {"1"}
    assert actionable_values("Sensor output 0.2 to 4.6V") == {"0.2", "4.6"}
    # Pin numbers ARE values -- model_leakage exists for exactly these.
    assert actionable_values("Between ECM (female) (2) and (32)") == {"2", "32"}

    for ident in ("PC200-10M0", "Manual: SEN06867-13", "Page: 40-241, 40-242",
                  "Step 1: Wiring harness and connector.", "S/N 700001"):
        assert not actionable_values(ident), \
            f"identifier read as a value: {ident!r}"

    # The real refusal that made clean_refusal report 0.0000.
    refusal = ("I am sorry, but I cannot find failure code CA451 in the "
               "provided manual extracts for the PC200-10M0 "
               "(Manual: SEN06867-13). Page: 40-179")
    assert not actionable_values(refusal), \
        f"a clean refusal still reads as leaking {actionable_values(refusal)}"

    # A fabricated code's digits come from the question, not the answer.
    case = {"question": "Failure code X0X0X0 aaya hai"}
    assert not asserted_values(case, "I cannot find X0X0X0."), \
        "a code echoed from the question counts as an asserted value"
    assert asserted_values(case, "It reads Max. 9 Ω") == {"9"}


if __name__ == "__main__":
    self_test()
    print("values self-test passed")
