#!/usr/bin/env python3
"""
conversation.py
The Komatsu diagnostic protocol. Rule-based, deterministic, no industry
equivalent -- these encode how a service engineer is supposed to run a session,
not how a chatbot is supposed to sound.

Each rule is scored SEPARATELY and never averaged into a single "protocol
score". They fail for different reasons and need different fixes: dumping the
whole tree is a prompt problem, missing a precondition is a retrieval problem,
and skipping the model check is a policy problem. One blended number would tell
you something is wrong and nothing about what.

Scenarios are multi-turn. The technician side is scripted deterministically from
golden/; the system under test supplies the assistant turns.
"""
import re

from .base import Metric, normalise, contains, numbers_in

# A step that must be surfaced before the technician touches the machine.
SAFETY_RE = re.compile(
    r"turn the starting switch to the off position|stop the engine|"
    r"lower the work equipment|disconnect the (?:battery|connector)|"
    r"wait (?:for )?\d+|apply the (?:parking )?brake|before removing",
    re.I)

VALUE_RE = re.compile(
    r"(?:max|min|approx)\.?\s*\d|\d+(?:\.\d+)?\s*(?:to|-)\s*\d|"
    r"\d+\s*(?:k|m)?(?:ohm|Ω|Ω|v|a|kpa|mpa|rpm|hz)\b", re.I)

ASK_MODEL_RE = re.compile(
    r"which model|what model|confirm (?:the )?(?:model|machine)|model number|"
    r"serial|kaunsa model|model bataye|machine model", re.I)

ASK_OTHER_CODES_RE = re.compile(
    r"other (?:failure )?codes?|any other code|aur koi code|"
    r"koi aur (?:failure )?code|additional codes|other codes shown", re.I)

ASK_EXACT_RE = re.compile(
    r"exact (?:reading|value|number)|what (?:value|number) (?:did|do) you|"
    r"exact kitna|kitna aaya|please (?:give|provide) the (?:exact|measured)|"
    r"measured value|specific reading", re.I)


def assistant_turns(result):
    return [t for t in result.get("transcript", []) if t.get("role") == "assistant"]


def first_value_turn(result):
    """Index of the first assistant turn that states a measurement value."""
    for i, t in enumerate(assistant_turns(result)):
        if VALUE_RE.search(t.get("text", "")):
            return i
    return None


class _ConvMetric(Metric):
    """Applies only to conversation scenarios exercising this rule."""
    RULE = None

    def applies(self, case):
        return case.get("type") == "conversation" and case.get("rule") == self.RULE


class ModelConfirmedFirst(_ConvMetric):
    """No technical value before the model is confirmed.

    PC200 and PC490 share failure codes but not pin numbers. A value handed over
    before the machine is pinned down is a guess wearing a number's clothes.
    """
    name = "protocol_model_confirmed_first"
    RULE = "confirm_model"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        turns = assistant_turns(result)
        v = first_value_turn(result)
        if v is None:
            return 1.0  # no value given; nothing to be premature about
        return float(any(ASK_MODEL_RE.search(t.get("text", ""))
                         for t in turns[:v + 1]))

    def self_test(self):
        m = ModelConfirmedFirst()
        case = {"type": "conversation", "rule": "confirm_model"}
        bad = {"transcript": [{"role": "assistant", "text": "It is Max. 1 Ω"}]}
        assert m.compute(case, bad) == 0.0, "model-confirmed rule cannot fail"
        good = {"transcript": [
            {"role": "assistant", "text": "Which model is it?"},
            {"role": "assistant", "text": "Then it is Max. 1 Ω"}]}
        assert m.compute(case, good) == 1.0


class OtherCodesAsked(_ConvMetric):
    """Ask what else is on the monitor before starting a procedure.

    Codes arrive in clusters and the manual routinely says to solve one first.
    Starting down the wrong tree wastes the whole session.
    """
    name = "protocol_other_codes_asked"
    RULE = "ask_other_codes"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        turns = assistant_turns(result)
        cut = len(turns)
        for i, t in enumerate(turns):
            if re.search(r"\bstep 1\b|first check|pehla check", t.get("text", ""), re.I):
                cut = i + 1
                break
        return float(any(ASK_OTHER_CODES_RE.search(t.get("text", ""))
                         for t in turns[:cut]))

    def self_test(self):
        m = OtherCodesAsked()
        case = {"type": "conversation", "rule": "ask_other_codes"}
        bad = {"transcript": [{"role": "assistant", "text": "Step 1: check wiring"}]}
        assert m.compute(case, bad) == 0.0, "other-codes rule cannot fail"


class PreconditionSolvedFirst(_ConvMetric):
    """When the manual says solve X first, the session must go to X."""
    name = "protocol_precondition_first"
    RULE = "precondition_first"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        want = case["expected"]["solve_first"]
        other = case["expected"]["then"]
        for t in assistant_turns(result):
            txt = t.get("text", "")
            if contains(txt, want):
                return 1.0
            if contains(txt, other) and re.search(r"\bstep 1\b|first check",
                                                  txt, re.I):
                return 0.0  # started the wrong tree
        return 0.0

    def self_test(self):
        m = PreconditionSolvedFirst()
        case = {"type": "conversation", "rule": "precondition_first",
                "expected": {"solve_first": "CA227", "then": "CA122"}}
        bad = {"transcript": [{"role": "assistant", "text": "CA122 step 1: check wiring"}]}
        assert m.compute(case, bad) == 0.0, "precondition rule cannot fail"
        good = {"transcript": [{"role": "assistant", "text": "Solve CA227 first."}]}
        assert m.compute(case, good) == 1.0


class OneStepAtATime(_ConvMetric):
    """Deliver one check, wait for the result. Do not dump the tree.

    A technician working through a dumped procedure has no branch guidance and
    will run checks the manual would have skipped.
    """
    name = "protocol_one_step_at_a_time"
    RULE = "one_step"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        turns = assistant_turns(result)
        if not turns:
            return 0.0
        steps = set(re.findall(r"\bstep\s*(\d+)", turns[0].get("text", ""), re.I))
        return float(len(steps) <= 1)

    def self_test(self):
        m = OneStepAtATime()
        case = {"type": "conversation", "rule": "one_step"}
        dump = {"transcript": [{"role": "assistant",
                                "text": "Step 1 ... Step 2 ... Step 3 ..."}]}
        assert m.compute(case, dump) == 0.0, "one-step rule cannot fail"
        one = {"transcript": [{"role": "assistant", "text": "Step 1: check wiring."}]}
        assert m.compute(case, one) == 1.0


class SafetySurfaced(_ConvMetric):
    """When the step carries a safety precondition, say it with the step."""
    name = "protocol_safety_surfaced"
    RULE = "safety"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        want = case["expected"]["safety_phrase"]
        return float(any(contains(t.get("text", ""), want)
                         for t in assistant_turns(result)))

    def self_test(self):
        m = SafetySurfaced()
        case = {"type": "conversation", "rule": "safety",
                "expected": {"safety_phrase": "Turn the starting switch to the OFF position"}}
        bad = {"transcript": [{"role": "assistant", "text": "Measure between (1) and (2)."}]}
        assert m.compute(case, bad) == 0.0, "safety rule cannot fail"


class PointerRedirected(_ConvMetric):
    """Pointer-only codes must hop, not dead-end.

    The retrieved chunk is correct, relevant and useless. This is the one place
    where a technically perfect retrieval result helps nobody.
    """
    name = "protocol_pointer_redirected"
    RULE = "pointer_redirect"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        target = case["expected"]["redirect_to"]
        return float(any(contains(t.get("text", ""), target)
                         for t in assistant_turns(result)))

    def self_test(self):
        m = PointerRedirected()
        case = {"type": "conversation", "rule": "pointer_redirect",
                "expected": {"redirect_to": "CA441"}}
        bad = {"transcript": [{"role": "assistant",
                               "text": "This code has no procedure."}]}
        assert m.compute(case, bad) == 0.0, "pointer rule cannot fail"


class StopsAtFirstFailure(_ConvMetric):
    """When a check fails, act on it. Do not carry on down the list.

    The manual's NO branch is the answer at that point; continuing to step 2
    means the branch was ignored.
    """
    name = "protocol_stops_at_first_failure"
    RULE = "stop_on_fail"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        turns = assistant_turns(result)
        if len(turns) < 2:
            return 0.0
        after = turns[-1].get("text", "")
        outcome = case["expected"]["no_outcome"]
        if re.search(r"\bstep\s*2\b|next check|agla check", after, re.I) \
                and not contains(after, outcome[:40]):
            return 0.0
        return float(contains(after, outcome[:40]))

    def self_test(self):
        m = StopsAtFirstFailure()
        case = {"type": "conversation", "rule": "stop_on_fail",
                "expected": {"no_outcome": "A wiring harness or connector is defective."}}
        bad = {"transcript": [{"role": "assistant", "text": "Step 1?"},
                              {"role": "assistant", "text": "Go to step 2."}]}
        assert m.compute(case, bad) == 0.0, "stop-on-fail rule cannot fail"


class AsksForExactReading(_ConvMetric):
    """A vague answer is not a measurement. Ask for the number."""
    name = "protocol_asks_exact_reading"
    RULE = "exact_reading"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        turns = assistant_turns(result)
        return float(any(ASK_EXACT_RE.search(t.get("text", "")) for t in turns))

    def self_test(self):
        m = AsksForExactReading()
        case = {"type": "conversation", "rule": "exact_reading"}
        bad = {"transcript": [{"role": "assistant", "text": "OK, go to step 2."}]}
        assert m.compute(case, bad) == 0.0, "exact-reading rule cannot fail"
        good = {"transcript": [{"role": "assistant",
                                "text": "What exact value did you measure?"}]}
        assert m.compute(case, good) == 1.0


# ---------------------------------------------------------------- scenarios

def _first_real_step(rec):
    for s in rec.get("steps", []):
        if not s.get("redirect") and (s.get("cause") or "").strip():
            return s
    return None


def _is_failure_code(rec):
    """A Section 40 failure-code record, judged by what it carries.

    Positive on the fields these scenarios actually read, not negative on
    "has no symptom_id": a new record shape that happened to lack a symptom_id
    would pass a negative test and then fail on the first field access. This
    asks for what is needed.
    """
    return ("code" in rec and "is_pointer_only" in rec and "format" in rec
            and "refs_failure_codes" in rec)


def build_scenarios(records, qa_cases):
    """Deterministic multi-turn scenarios, one family per protocol rule.

    Built from sorted golden data with no randomness, so the scenario set is
    identical on every machine and two runs are comparable case by case.

    SECTION 40 ONLY, SELECTED ON THE RECORD'S OWN SHAPE. `records` may now hold
    symptom trees as well, and every scenario family below reads a failure-code
    field -- is_pointer_only, format, refs_failure_codes -- that a symptom tree
    does not have.

    Filtered with an explicit predicate rather than with .get() defaults. A
    default would let a symptom tree into a code scenario and build a scenario
    with an empty expectation: a case that scores, passes, and measures nothing.
    A KeyError would at least have been loud. This is neither -- the tree is
    simply not eligible, and saying so once here is clearer than defending
    against it in eight places.

    Symptom-entry scenarios are a separate family and are not invented here by
    reusing code-shaped ones.
    """
    codes = sorted(c for c, r in records.items() if _is_failure_code(r))
    scen = []

    def add(rule, code, turns, expected, note):
        scen.append({
            "id": f"conv_{rule}_{code}",
            "type": "conversation",
            "rule": rule,
            "difficulty": "hard",
            "question": turns[0],
            "turns": turns,
            "filters": {"model": records[code]["model"],
                        "manual_id": records[code]["manual_id"]},
            "expected": expected,
            "source_code": code,
            "trap": note,
        })

    # 1. model confirmed before any value
    for code in [c for c in codes if records[c]["standalone_measurements"]][:6]:
        m = records[code]["standalone_measurements"][0]
        add("confirm_model", code,
            [f"{code} aaya hai. {m['point']} par kya value honi chahiye?"],
            {"criteria": m["criteria"]},
            "The technician never said which machine. Values differ per model.")

    # 2. other codes asked before starting
    for code in [c for c in codes if _first_real_step(records[c])][:6]:
        add("ask_other_codes", code,
            [f"{code} aa raha hai PC200-10M0 par. Troubleshooting shuru karein."],
            {}, "Codes arrive in clusters; the manual often says to solve one first.")

    # 3. precondition solved first
    for c in [c for c in qa_cases if c["type"] == "precondition"][:6]:
        add("precondition_first", c["source_code"],
            [c["question"]], dict(c["expected"]),
            "The manual names a code that must be solved first.")

    # 4. one step at a time
    for code in [c for c in codes
                 if len([s for s in records[c]["steps"] if not s.get("redirect")]) >= 3][:6]:
        add("one_step", code,
            [f"{code} ka pura procedure batao PC200-10M0 par."],
            {}, "Dumping the tree removes the branch guidance that makes it a tree.")

    # 5. safety precondition surfaced
    for code in codes:
        st = next((s for s in records[code]["steps"]
                   if SAFETY_RE.search(s.get("procedure") or "")), None)
        if st:
            phrase = SAFETY_RE.search(st["procedure"]).group(0)
            add("safety", code,
                [f"{code} par step {st['step']} kaise karun? PC200-10M0."],
                {"safety_phrase": phrase, "step": st["step"]},
                "The step carries a precondition the technician must act on first.")
        if len([s for s in scen if s["rule"] == "safety"]) >= 6:
            break

    # 6. pointer-only redirect
    for code in [c for c in codes if records[c]["is_pointer_only"]][:6]:
        refs = records[code]["refs_failure_codes"]
        if refs:
            add("pointer_redirect", code,
                [f"{code} aa raha hai PC200-10M0 par, kya karun?"],
                {"redirect_to": refs[0]},
                "The code's own page is correct, relevant and useless on its own.")

    # 7. stop at the first failing check
    for code in [c for c in codes if records[c]["format"] == "A"][:6]:
        st = next((s for s in records[code]["steps"] if "NO" in s.get("branches", {})), None)
        if st:
            add("stop_on_fail", code,
                [f"{code} PC200-10M0 par. Step {st['step']} start karo.",
                 f"Step {st['step']} ka result NO aaya."],
                {"no_outcome": st["branches"]["NO"], "step": st["step"]},
                "The NO branch is the answer at that point; step 2 is not.")

    # 8. vague reading must be pinned down
    for code in [c for c in codes if records[c]["standalone_measurements"]][:6]:
        m = records[code]["standalone_measurements"][0]
        add("exact_reading", code,
            [f"{code} PC200-10M0 par. {m['point']} check karna hai.",
             "Thoda kam lag raha hai, normal hi hoga shayad."],
            {"criteria": m["criteria"]},
            "A vague impression is not a measurement.")

    return scen


def build():
    return [
        ModelConfirmedFirst(), OtherCodesAsked(), PreconditionSolvedFirst(),
        OneStepAtATime(), SafetySurfaced(), PointerRedirected(),
        StopsAtFirstFailure(), AsksForExactReading(),
    ]
