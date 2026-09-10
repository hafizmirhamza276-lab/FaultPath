#!/usr/bin/env python3
"""
generation.py
Grounding, exactness and citation. Deterministic.

GROUNDEDNESS AND FAITHFULNESS ARE NOT THE SAME NUMBER and this file never
collapses them:

  groundedness      claims supported by the GROUND TRUTH
  faithfulness_det  claims supported by the RETRIEVED CONTEXT

A system that retrieves the wrong code's page and then quotes it accurately
scores high on faithfulness and low on groundedness. That is a retrieval bug
wearing a generation bug's clothes, and averaging the two hides it completely.
The reverse -- grounded but unfaithful -- means the model knew the answer
without reading the context, which is its own problem.

"Claims" are decomposed deterministically: a sentence carries checkable atoms
(numbers, code references, criteria strings) or it does not. Sentences with no
atoms are not scored either way rather than counted as free passes.
"""
import re
import json

from .base import (Metric, normalise, contains, numbers_in, numbers_in_record,
                   sentences, golden_facts, is_adversarial)

CODE_RE = re.compile(r"\b([A-Z0-9@#]{4,7})\b")

NON_ADVERSARIAL = ("numeric_exactness", "direct_lookup", "step_ordering",
                   "branch_following", "precondition", "cross_ref_hop")


class _GenMetric(Metric):
    APPLIES_TO = NON_ADVERSARIAL

    def applies(self, case):
        return case["type"] in self.APPLIES_TO


def _atoms(text):
    """Checkable tokens: numbers and code-shaped identifiers.

    Used on BOTH sides of every support check. Extracting codes from the answer
    but only numbers from the source would fail any sentence naming the model or
    a cross-referenced code -- the support set would not contain a token the
    extractor was capable of producing.
    """
    return set(numbers_in(text)) | {c for c in CODE_RE.findall(str(text or ""))
                                    if any(ch.isdigit() for ch in c)}


# ------------------------------------------------------------------ exactness

class NumericExactness(_GenMetric):
    """The criterion reproduced verbatim.

    Whitespace and ohm spelling are tolerated. Digits are not: `Min. 100kΩ` and
    `Min. 90kΩ` must remain different strings, or this metric measures nothing.
    The expected value is the WHOLE criterion string as golden/ stores it --
    `Sensor output 0.2 to 4.6V`, not a trimmed `0.2 to 4.6V`.
    """
    name = "numeric_exactness"
    APPLIES_TO = ("numeric_exactness",)

    def compute(self, case, result):
        if not self.applies(case):
            return None
        want = case.get("must_contain_verbatim") or []
        if not want:
            return None
        ans = result.get("answer", "")
        return sum(1.0 for v in want if contains(ans, v)) / len(want)

    def self_test(self):
        m = NumericExactness()
        case = {"type": "numeric_exactness",
                "must_contain_verbatim": ["Sensor output 0.2 to 4.6V"]}
        assert m.compute(case, {"answer": "Sensor output 0.2 to 4.6V"}) == 1.0
        # ohm folding and spacing tolerated
        ohm = {"type": "numeric_exactness", "must_contain_verbatim": ["Max. 1 Ω"]}
        assert m.compute(ohm, {"answer": "max.  1 ohm"}) == 1.0, \
            "numeric_exactness must tolerate ohm spelling and spacing"
        # digits are NOT normalised -- this is the failure that matters
        assert m.compute(case, {"answer": "Sensor output 0.3 to 4.6V"}) == 0.0, \
            "numeric_exactness cannot fail on a changed digit"
        assert m.compute(case, {"answer": "0.2 to 4.6V"}) == 0.0, \
            "numeric_exactness must require the whole criterion string"


class FabricatedValues(_GenMetric):
    """THE HEADLINE HALLUCINATION FIGURE. Lower is better.

    Every number in the answer is checked against the numbers appearing anywhere
    in that code's golden record, plus numbers echoed from the question and
    legitimate step indices. Whatever is left was not read from the manual.

    It is a count, not an impression, and every flagged value is kept in the
    per-case rows so any individual claim traces back.
    """
    name = "fabricated_values"
    HIGHER_IS_BETTER = False
    APPLIES_TO = None  # adversarial answers must not invent numbers either

    def __init__(self, records):
        self.records = records

    def _allowed(self, case):
        allowed = set(numbers_in(case.get("question", "")))
        rec = self.records.get(case.get("source_code"))
        if rec:
            allowed |= numbers_in_record(rec)
            # step indices are legitimate even when phrased as "step 3 of 7"
            allowed |= {str(s["step"]) for s in rec.get("steps", [])}
            allowed |= {str(len(rec.get("steps", [])))}
        return allowed

    def compute(self, case, result):
        allowed = self._allowed(case)
        found = numbers_in(result.get("answer", ""))
        if not found:
            return 0.0
        bad = [n for n in found if n not in allowed]
        return len(bad) / len(found)

    def flagged(self, case, result):
        """The actual invented values, for the per-case row."""
        allowed = self._allowed(case)
        return sorted({n for n in numbers_in(result.get("answer", ""))
                       if n not in allowed})

    def self_test(self):
        m = FabricatedValues({"X": {"steps": [], "criteria": "Max. 1"}})
        case = {"type": "numeric_exactness", "question": "X ke liye?",
                "source_code": "X"}
        assert m.compute(case, {"answer": "Max. 1"}) == 0.0
        assert m.compute(case, {"answer": "Max. 7 and 9"}) == 1.0, \
            "fabricated_values cannot fail on invented numbers"


# ------------------------------------------------------- grounding vs context

class Groundedness(_GenMetric):
    """Atom-bearing claims supported by the GROUND TRUTH."""
    name = "groundedness"
    APPLIES_TO = None

    def __init__(self, records):
        self.records = records

    def _support(self, case):
        rec = self.records.get(case.get("source_code"))
        support = _atoms(json.dumps(rec, ensure_ascii=False)) if rec else set()
        # Tokens the technician supplied are not the system's inventions.
        return support | _atoms(case.get("question", ""))

    def compute(self, case, result):
        support = self._support(case)
        scored, ok = 0, 0
        for s in sentences(result.get("answer", "")):
            a = _atoms(s)
            if not a:
                continue
            scored += 1
            ok += int(a <= support)
        return ok / scored if scored else None

    def self_test(self):
        m = Groundedness({"X": {"a": "1 2 3"}})
        case = {"type": "direct_lookup", "source_code": "X", "question": ""}
        assert m.compute(case, {"answer": "Value is 1."}) == 1.0
        assert m.compute(case, {"answer": "Value is 99."}) == 0.0, \
            "groundedness cannot fail"
        # A code named in the question is supported, not fabricated.
        c2 = {"type": "direct_lookup", "source_code": "X",
              "question": "PC200-10M0 par CA451?"}
        assert m.compute(c2, {"answer": "On PC200-10M0, code CA451 applies."}) == 1.0, \
            "groundedness must extract codes from the source, not only numbers"


class FaithfulnessDeterministic(_GenMetric):
    """Atom-bearing claims supported by the RETRIEVED CONTEXT.

    Deliberately separate from groundedness. High here plus low there means the
    system faithfully quoted the wrong chunk.
    """
    name = "faithfulness_det"
    APPLIES_TO = None

    def compute(self, case, result):
        ctx = " ".join(c.get("text", "") for c in result.get("retrieved", []))
        if not ctx:
            return None
        support = _atoms(ctx) | _atoms(case.get("question", ""))
        scored, ok = 0, 0
        for s in sentences(result.get("answer", "")):
            a = _atoms(s)
            if not a:
                continue
            scored += 1
            ok += int(a <= support)
        return ok / scored if scored else None

    def self_test(self):
        m = FaithfulnessDeterministic()
        case = {"type": "direct_lookup"}
        r = {"answer": "It is 42.", "retrieved": [{"text": "the value 7"}]}
        assert m.compute(case, r) == 0.0, "faithfulness_det cannot fail"


# ------------------------------------------------------------------ coverage

class ContentRecall(_GenMetric):
    """Required facts present in the answer."""
    name = "content_recall"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        facts = golden_facts(case)
        if not facts:
            return None
        ans = result.get("answer", "")
        return sum(1.0 for f in facts if contains(ans, f)) / len(facts)


class Completeness(_GenMetric):
    """All steps of a procedure delivered, none skipped.

    Applies only where a full procedure is the expected shape of the answer --
    a code with more than one real step. Asking this of a single-value lookup
    would penalise a correct, appropriately short answer.
    """
    name = "completeness"
    APPLIES_TO = ("step_ordering", "cross_ref_hop", "branch_following")

    def __init__(self, records):
        self.records = records

    def _steps(self, case):
        rec = self.records.get(case.get("source_code"))
        if not rec:
            return []
        return [s for s in rec.get("steps", []) if not s.get("redirect")
                and (s.get("cause") or "").strip()]

    def applies(self, case):
        return case["type"] in self.APPLIES_TO and len(self._steps(case)) > 1

    def compute(self, case, result):
        if not self.applies(case):
            return None
        steps = self._steps(case)
        ans = result.get("answer", "")
        return sum(1.0 for s in steps if contains(ans, s["cause"])) / len(steps)


class Contradiction(_GenMetric):
    """The answer states a different value than the manual for the same point.

    Distinct from a missing answer. Silence is a gap; a confident wrong number
    is a machine going back out broken. Lower is better.
    """
    name = "contradiction"
    HIGHER_IS_BETTER = False
    APPLIES_TO = ("numeric_exactness",)

    def compute(self, case, result):
        if not self.applies(case):
            return None
        want = (case.get("expected") or {}).get("criteria")
        if not want:
            return None
        ans = result.get("answer", "")
        if contains(ans, want):
            return 0.0
        # The correct string is absent. If the answer names the measuring point
        # and still emits numbers, it is asserting a different value.
        point = (case.get("expected") or {}).get("point") or ""
        if point and contains(ans, point) and numbers_in(ans):
            return 1.0
        return 0.0

    def self_test(self):
        m = Contradiction()
        case = {"type": "numeric_exactness",
                "expected": {"criteria": "Max. 1 Ω", "point": "Between A and B"}}
        bad = {"answer": "Between A and B should read Max. 9 Ω"}
        assert m.compute(case, bad) == 1.0, "contradiction cannot fire"
        good = {"answer": "Between A and B should read Max. 1 Ω"}
        assert m.compute(case, good) == 0.0


# ------------------------------------------------------------------ citation

class CitationPresence(_GenMetric):
    """Did the answer cite anything at all.

    Reported separately from accuracy on purpose. A system that always cites but
    cites the wrong page is worse than one that admits uncertainty, because it
    manufactures confidence. Collapsing the two hides exactly that.
    """
    name = "citation_presence"

    def applies(self, case):
        return case.get("must_cite_page") is not None

    def compute(self, case, result):
        if not self.applies(case):
            return None
        return float(bool(result.get("citations")))


class CitationAccuracy(_GenMetric):
    """Is the cited page the page the fact actually came from.

    Scored only where a citation was made, so it does not double-count the
    absence already measured by citation_presence.
    """
    name = "citation_accuracy"

    def applies(self, case):
        return case.get("must_cite_page") is not None

    def compute(self, case, result):
        if not self.applies(case):
            return None
        cites = result.get("citations") or []
        if not cites:
            return None
        want = normalise(case["must_cite_page"])
        return float(any(normalise(c) == want for c in cites))

    def self_test(self):
        m = CitationAccuracy()
        case = {"type": "direct_lookup", "must_cite_page": "40-181"}
        assert m.compute(case, {"citations": ["40-181"]}) == 1.0
        assert m.compute(case, {"citations": ["40-999"]}) == 0.0, \
            "citation_accuracy cannot fail"


def build(records):
    return [
        NumericExactness(),
        FabricatedValues(records),
        Groundedness(records),
        FaithfulnessDeterministic(),
        ContentRecall(),
        Completeness(records),
        Contradiction(),
        CitationPresence(),
        CitationAccuracy(),
    ]
