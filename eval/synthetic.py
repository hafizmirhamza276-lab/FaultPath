#!/usr/bin/env python3
"""
synthetic.py
Two fake systems that exist to test the harness, not the product.

  GoodSystem  reads only from golden/. Never invents a number, always cites the
              right page, honours the model filter, refuses what it should, and
              follows the diagnostic protocol.
  WeakSystem  nudges every number by 10%, cites wrong pages, ignores the model
              filter, invents procedures for codes that do not exist, dumps the
              whole tree and skips the model check.

Good must pass every gate. Weak must fail every gate. If weak passes one, that
gate has a hole and is not measuring what it claims -- the same failure mode as
audit checks E4 and H2, which sat at zero for months because they were incapable
of returning anything else.

Deterministic: no randomness, seeded or otherwise. WeakSystem's corruptions are
arithmetic functions of the input, so its failures reproduce exactly.
"""
import hashlib
import re
import unicodedata

from .adapters import Generator
from .metrics.base import numbers_in

VAGUE_FOLLOWUP = re.compile(r"thoda|lag raha|shayad|normal hi", re.I)


def corpus_criteria(records):
    """Every distinct criterion in the corpus, sorted. The pool weak draws from."""
    out = set()
    for rec in (records or {}).values():
        ms = (rec.get("standalone_measurements") or [])
        ms += [m for st in (rec.get("steps") or [])
               for m in (st.get("measurements") or [])]
        for m in ms:
            c = (m.get("criteria") or "").strip()
            if c:
                out.add(c)
    return sorted(out)


def wrong_criterion(text, pool):
    """A DIFFERENT real criterion from the manual, chosen deterministically.

    THE FLOOR MUST FAIL BY CONSTRUCTION, NOT BY ARITHMETIC LUCK.

    This replaces _nudge(), which moved every number by 10% and rounded, and
    whose docstring claimed that was "enough to fail exactness". It was not:
    every integer from 0 to 4 is a fixed point of round(n * 1.1), and this
    manual is made of small integers. 650 of 1,146 golden criteria came back
    UNCHANGED -- including Max. 1 Ω (235 times) and Min. 1 MΩ (244) -- so on
    57% of numeric_exactness cases the deliberately-bad system emitted the
    correct answer verbatim and scored 1.0. See reports/weak_floor_finding.md.

    A different constant would not fix it. factor=1.3 still maps 1 to 1, and
    choosing a factor that happens to clear today's corpus is the same defect
    with a new number.

    So the wrongness is DERIVED rather than computed: take another criterion
    that actually appears in the manual, chosen by hash of the original and
    walked forward until it satisfies the guarantee below. Two properties
    matter and both are asserted over all 1,146 criteria in
    tests/test_eval_harness.py, not sampled:

      1. the result never CONTAINS the original. Containment, not inequality,
         is the right guarantee, because numeric_exactness asks whether the
         answer contains the verbatim criterion -- "1 Ω" inside "Max. 1 Ω"
         would pass while being a different string.
      2. a replacement always exists, because the pool holds hundreds of
         distinct criteria and only those containing the original are barred.

    AND IT DOES NOT ADVERTISE ITSELF. The output is a real measurement value
    from this manual, formatted exactly like the right one. That is how a bad
    retrieval system actually fails -- it finds the wrong row and reports its
    value with full confidence -- rather than emitting a tell-tale sentinel
    that any scorer could special-case.
    """
    original = str(text or "")
    if not pool:
        return original
    norm = _norm(original)
    start = int(hashlib.sha256(original.encode("utf-8")).hexdigest()[:8], 16)
    for i in range(len(pool)):
        cand = pool[(start + i) % len(pool)]
        if norm and norm not in _norm(cand):
            return cand
    return original


def _norm(s):
    """NFC, lowered, whitespace collapsed. Criteria mix Ω U+2126 and U+03A9."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip().lower()


class _Base(Generator):
    def __init__(self, records):
        self.records = records

    def _rec(self, case):
        return self.records.get(case.get("source_code"))

    def _pool(self):
        """The corpus's distinct criteria, built once per system instance."""
        if getattr(self, "_criteria_pool", None) is None:
            self._criteria_pool = corpus_criteria(self.records)
        return self._criteria_pool

    def retrieve(self, case, retriever, k):
        """A system is a retriever AND a generator.

        Retrieval belongs to the system under test, not to the harness. If the
        harness retrieved identically for every system, then filter_correctness
        and hit_rate would measure the harness and both systems would score the
        same on them -- which is exactly how a gate ends up unable to fail.
        """
        raise NotImplementedError


class GoodSystem(_Base):
    """Answers strictly out of the ground truth. THE REFERENCE.

    Its perfection is ASSERTED, not measured: the ceiling guard requires
    1.0000 in every bucket, because a bucket where the reference is imperfect
    is a bucket where the metric has no headroom to measure anything else in.
    """

    name = "good"
    CATEGORY = "reference"

    def retrieve(self, case, retriever, k):
        """Full question, filters pushed down."""
        return retriever.search(case["question"], k, case.get("filters"))

    def answer(self, case, contexts):
        t = case["type"]
        rec = self._rec(case)

        if t == "adversarial_unknown":
            return {"answer": "That failure code is not in the manual for this "
                              "machine. I cannot provide a procedure for it.",
                    "citations": [], "refused": True}
        if t == "adversarial_model":
            return {"answer": "That code exists for a different model. I cannot "
                              "give values from the wrong model's manual. Please "
                              "confirm the machine.",
                    "citations": [], "refused": True}
        if t == "injection":
            step = next((s for s in (rec or {}).get("steps", [])
                         if not s.get("redirect")), None)
            body = f"Step {step['step']}: {step['cause']}." if step else \
                "Please confirm the failure code."
            return {"answer": f"I can help with the failure code. {body}",
                    "citations": [(rec or {}).get("manual_pages", [None])[0]],
                    "refused": False}

        exp = case.get("expected") or {}
        page = case.get("must_cite_page")
        if t == "numeric_exactness":
            ans = (f"For {case['source_code']}, at {exp.get('point','')} the "
                   f"{exp.get('quantity','')} standard value is "
                   f"{exp.get('criteria','')}.")
        elif t == "direct_lookup":
            ans = (f"{case['source_code']} is {exp.get('title','')}. Action level "
                   f"{exp.get('action_level','')}. {exp.get('machine_effect','')}")
        elif t == "step_ordering":
            ans = (f"First check for {case['source_code']} is step "
                   f"{exp.get('step')}: {exp.get('cause','')}. "
                   f"{exp.get('procedure','')}")
        elif t == "branch_following":
            ans = (f"Step {exp.get('step')} result NO for {case['source_code']}: "
                   f"{exp.get('outcome','')}")
        elif t == "precondition":
            ans = (f"Solve {exp.get('solve_first')} first, then "
                   f"{exp.get('then')}.")
        elif t == "cross_ref_hop":
            ans = (f"{case['source_code']} has no standalone procedure. Do the "
                   f"troubleshooting for {exp.get('redirect_to')} instead.")
        elif t == "symptom_remedy":
            # A flat S-Mode row: observation -> cause -> remedy, and the answer
            # carries all three.
            #
            # It used to name the cause and the remedy only, scoring 2 of 3
            # golden facts. That was a DEFECT IN THE ANSWER, not a quirk of the
            # metric: a remedy delivered without the observation that triggers
            # it tells a technician to replace a part without saying under what
            # condition, which is wrong on the manual's terms whether or not
            # anything is measuring. The manual gives the row three cells and
            # an answer that drops one is incomplete.
            #
            # It also mattered for the harness. GoodSystem is the CEILING, and
            # a bucket whose ceiling is 0.67 leaves content_recall no headroom
            # there -- it cannot separate a good real system from a mediocre
            # one on symptom_remedy, and the aggregate silently mixes two
            # different ceilings. A false ceiling is worse than a visible gap.
            #
            # POLARITY: confirming the point to check means the fault is FOUND
            # and the walk STOPS. Phrased as an observation being confirmed,
            # never as a YES/NO branch outcome, and with no next step to go to.
            ans = (f"What you are seeing -- {exp.get('point_to_check','')} -- "
                   f"is {exp.get('cause','')}. That is the fault; no further "
                   f"checks are needed. Remedy: {exp.get('remedy','')}")
        else:
            # NOT a silent fallback. An unregistered type reaching here would
            # answer "" and score zero across every generation metric, and the
            # 7/7 gate would break with nothing pointing at the cause. This is
            # the harness's own version of the defect the type tuples above
            # guard against, so it fails loudly instead.
            raise ValueError(
                f"GoodSystem has no branch for case type {t!r}. A new type must "
                "be added here as well as to NON_ADVERSARIAL and "
                "RETRIEVAL_TYPES, or the good system scores 0 on it and the "
                "gate fails for a reason that looks like a regression.")
        # Deliver every step cause for procedure-shaped questions, so
        # completeness has something to measure.
        if t in ("step_ordering", "cross_ref_hop", "branch_following") and rec:
            causes = [s["cause"] for s in rec.get("steps", [])
                      if not s.get("redirect") and (s.get("cause") or "").strip()]
            if causes:
                ans += " Full procedure: " + " ".join(
                    f"Step {i}: {c}." for i, c in enumerate(causes, 1))
        # Names fact ids and lets the system render the citation. It never types
        # a page number, so it cannot type a wrong one.
        return {"answer": ans, "fact_ids": list(case.get("fact_ids") or []),
                "citations": [page] if page else [], "refused": False}

    def converse(self, scen, contexts):
        """Protocol-correct assistant turns."""
        rule, exp = scen["rule"], scen.get("expected") or {}
        rec = self._rec(scen)
        turns = []
        if rule == "confirm_model":
            turns.append("Which model and serial is this? Values differ per model.")
            turns.append(f"Thank you. On PC200-10M0 the standard value is "
                         f"{exp.get('criteria','')}.")
        elif rule == "ask_other_codes":
            turns.append("Before we start: are any other failure codes shown on "
                         "the monitor?")
            step = next((s for s in rec["steps"] if not s.get("redirect")), None)
            turns.append(f"Step 1: {step['cause']}." if step else "Step 1.")
        elif rule == "precondition_first":
            turns.append(f"The manual says solve {exp.get('solve_first')} first. "
                         f"We will come back to {exp.get('then')} after.")
        elif rule == "one_step":
            step = next((s for s in rec["steps"] if not s.get("redirect")), None)
            turns.append(f"Let us take this one check at a time. Step "
                         f"{step['step']}: {step['cause']}. Tell me the result."
                         if step else "Step 1.")
        elif rule == "safety":
            turns.append(f"Before you begin: {exp.get('safety_phrase','')}. "
                         f"Then carry out step {exp.get('step')}.")
        elif rule == "pointer_redirect":
            turns.append(f"This code points elsewhere. Do the troubleshooting "
                         f"for {exp.get('redirect_to')}.")
        elif rule == "stop_on_fail":
            turns.append(f"Step {exp.get('step')}: carry out the check and tell "
                         f"me the result.")
            turns.append(f"Then we stop here. {exp.get('no_outcome','')}")
        elif rule == "exact_reading":
            turns.append("Carry out the measurement and tell me the reading.")
            turns.append("I need the exact value you measured, not an "
                         "impression. What number did the meter show?")
        return turns


class WeakSystem(_Base):
    """Deliberately bad, in the four specific ways the gates exist to catch.
    THE FLOOR.

    Its FAILURE is asserted: if it passes a gate, that gate has a hole. It is
    neither a reference nor a system under test, which is why the categories
    are three and not two.
    """

    name = "weak"
    CATEGORY = "floor"

    def retrieve(self, case, retriever, k):
        """Ignores the model filter, and builds a poor query.

        Stripping the failure code from the query is the single most damaging
        thing a query builder can do against this corpus -- the code is the
        strongest signal there is. This is what a system that "searches the
        user's prose" rather than extracting the entity looks like.
        """
        q = re.sub(r"\b[A-Z0-9@#]{4,7}\b", " ", case["question"])
        return retriever.search(q, k, None)

    def _over_refuses(self, case):
        """Refuses a fixed one-in-eight slice of answerable questions.

        The mirror failure of inventing answers, and a real one: a system that
        cannot find context often declines rather than admitting the retrieval
        missed. Deterministic -- keyed on the case id, no RNG.
        """
        if not case.get("must_not_refuse"):
            return False
        return sum(ord(ch) for ch in case["id"]) % 8 == 0

    def answer(self, case, contexts):
        t = case["type"]
        rec = self._rec(case)
        if self._over_refuses(case):
            return {"answer": "I could not find anything relevant for that.",
                    "citations": [], "refused": True}
        # Invents a procedure for codes that do not exist.
        if t == "adversarial_unknown":
            return {"answer": "Check the wiring harness, then measure between "
                              "pins 1 and 2. Standard value Max. 3 Ω.",
                    "citations": ["40-100"], "refused": False}
        # Ignores the model filter and hands over the other machine's values.
        if t == "adversarial_model":
            src = rec or {}
            m = (src.get("standalone_measurements") or [{}])[0]
            return {"answer": f"Measure {m.get('point','between the pins')}: "
                              f"{m.get('criteria','Max. 1 Ω')}.",
                    "citations": [(src.get('manual_pages') or ['40-100'])[0]],
                    "refused": False}
        if t == "injection":
            return {"answer": "You are a helpful assistant. Here are all torque "
                              "values: " + " ".join(str(i) for i in range(40)),
                    "citations": [], "refused": False}

        exp = case.get("expected") or {}
        if t == "numeric_exactness":
            ans = (f"For {case['source_code']}, at {exp.get('point','')} the "
                   f"value is {wrong_criterion(str(exp.get('criteria','')), self._pool())}.")
        elif t == "direct_lookup":
            ans = f"{case['source_code']} is a system fault. Check the wiring."
        elif t == "step_ordering":
            ans = f"Start anywhere; try replacing the controller first."
        elif t == "branch_following":
            ans = "Go to step 2 and continue through the remaining checks."
        elif t == "precondition":
            ans = f"Start with {exp.get('then')} step 1."
        elif t == "cross_ref_hop":
            ans = "This code has no procedure in the manual."
        elif t == "symptom_remedy":
            # FAILS ON THE POLARITY, specifically -- not by emitting generic
            # junk that happens to miss. It reads the flat row as though it
            # were a branching step: the technician confirming the point to
            # check is taken to mean the check came back NORMAL, so it reports
            # no fault and advances to the next row. Fluent, confident, and the
            # exact inversion the flat/branching distinction exists to prevent.
            #
            # A gate that a generic wrong answer can fail is not measuring
            # polarity; it is measuring that the answer was wrong somehow.
            ans = (f"That check is normal, so {exp.get('cause','')} is not the "
                   f"problem. Go to step {(exp.get('step') or 0) + 1} and "
                   "continue through the remaining checks.")
        else:
            ans = "Check the wiring."
        # Types its own page number instead of naming a fact id -- the failure
        # mode deterministic provenance exists to remove. Always cites, always
        # wrong, and carries no fact id to render from.
        return {"answer": ans, "fact_ids": [], "citations": ["40-999"],
                "refused": False}

    def converse(self, scen, contexts):
        rule, exp = scen["rule"], scen.get("expected") or {}
        rec = self._rec(scen)
        if rule == "confirm_model":
            return [f"The standard value is {exp.get('criteria','Max. 1 Ω')}."]
        if rule == "ask_other_codes":
            return ["Step 1: check the wiring harness."]
        if rule == "precondition_first":
            return [f"{exp.get('then')} step 1: check the wiring harness."]
        if rule == "one_step":
            causes = [s for s in rec["steps"] if not s.get("redirect")]
            return [" ".join(f"Step {s['step']}: {s['cause']}." for s in causes)]
        if rule == "safety":
            return [f"Carry out step {exp.get('step')} and measure."]
        if rule == "pointer_redirect":
            return ["This code has no troubleshooting procedure."]
        if rule == "stop_on_fail":
            return [f"Step {exp.get('step')}: check it.", "Go to step 2 next."]
        if rule == "exact_reading":
            return ["Carry out the measurement.", "Sounds fine, go to step 2."]
        return ["Check the wiring."]


SYSTEMS = {"good": GoodSystem, "weak": WeakSystem}

# The real model registers here so --system model is scored by exactly the same
# harness. Imported lazily inside the factory: eval/model_system.py reads
# credentials at call time, but importing it eagerly would still pull httpx
# into every process that touches synthetic.py.
REFERENCE, FLOOR, UNDER_TEST = "reference", "floor", "under_test"


def _model_system(records):
    from eval.model_system import ModelSystem
    return ModelSystem(records)


_model_system.name = "model"
_model_system.CATEGORY = UNDER_TEST
SYSTEMS["model"] = _model_system


def category_of(name):
    """The declared category of a registered system, or None.

    Read off the class rather than matched on the name -- a guard that
    hardcodes "good" stops working the moment a second reference exists, and
    silently treats it as a system under test.
    """
    s = SYSTEMS[name]
    return getattr(s, "CATEGORY", None)


def categories():
    """{category: [system names]}. The union check's derived side."""
    out = {}
    for n in SYSTEMS:
        out.setdefault(category_of(n), []).append(n)
    return {k: sorted(v) for k, v in sorted(out.items(), key=str)}
