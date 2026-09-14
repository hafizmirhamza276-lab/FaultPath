#!/usr/bin/env python3
"""
agent.py
Agent-level Tier-1 metrics. Deterministic, scored against the reference walk.

These ask one question in several forms: did the agent execute the manual's tree,
or did it do something else that happened to sound reasonable. The reference walk
in agent/sessions.py is computed from golden/ independently of the graph, so
agreement means agreement with Komatsu, not with itself.

gate_enforcement and cycle_safety are absolutes. The rest are rates.
protocol_adherence is eight separate numbers and is never averaged -- the rules
fail for different reasons and need different fixes.
"""
from __future__ import annotations

import re

from .base import Metric, normalise, contains

VALUE_RE = re.compile(
    r"(?:max|min|approx)\.?\s*\d|\d+(?:\.\d+)?\s*(?:to|-)\s*\d|"
    r"\d+(?:\.\d+)?\s*(?:k|m|µ)?(?:ohm|Ω|Ω|v|a|kpa|mpa|rpm|hz|%)\b", re.I)


def _assistant_text(result):
    return " ".join(e["text"] for e in result.get("emissions", []))


class _AgentMetric(Metric):
    APPLIES_TO = ("agent_session",)

    def applies(self, case):
        return case.get("type") == "agent_session"


class PathCorrectness(_AgentMetric):
    """Did it walk the manual's own step order.

    Compared against the reference walk, not against a hand-written list. An
    agent that reaches the right diagnosis by a different route is not correct
    here -- the order is the safety property, because each step's branch decides
    which checks are legitimate next.
    """
    name = "path_correctness"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        want = case["expect"].get("steps") or []
        got = result.get("steps_visited") or []
        if not want:
            return None
        return 1.0 if got == want else 0.0

    def self_test(self):
        m = PathCorrectness()
        case = {"type": "agent_session", "expect": {"steps": [1, 2, 3]}}
        assert m.compute(case, {"steps_visited": [1, 2, 3]}) == 1.0
        assert m.compute(case, {"steps_visited": [1, 3]}) == 0.0, \
            "path_correctness cannot fail on a skipped step"
        assert m.compute(case, {"steps_visited": [1, 2, 3, 4]}) == 0.0, \
            "path_correctness cannot fail on an over-run"


class StepsToDiagnosis(_AgentMetric):
    """Steps taken versus the minimum the manual allows. Lower is better --
    but only counted where the diagnosis was actually right.

    Scored only on sessions that concluded correctly, because a short wrong
    answer is not an efficient one. Rewarding brevity without that condition
    would score premature_conclusion as an improvement.
    """
    name = "steps_to_diagnosis"
    HIGHER_IS_BETTER = False

    def compute(self, case, result):
        if not self.applies(case):
            return None
        if case["expect"].get("outcome") != "conclude":
            return None
        if not result.get("diagnosis_correct"):
            return None
        want = len(case["expect"].get("steps") or [])
        got = len(result.get("steps_visited") or [])
        if not want:
            return None
        return float(got) / want


class PrematureConclusion(_AgentMetric):
    """Concluded before running the check that decides it. Lower is better."""
    name = "premature_conclusion"
    HIGHER_IS_BETTER = False

    def compute(self, case, result):
        if not self.applies(case):
            return None
        if case["expect"].get("outcome") != "conclude":
            return None
        want = case["expect"].get("steps") or []
        got = result.get("steps_visited") or []
        if not want:
            return None
        return 1.0 if len(got) < len(want) else 0.0

    def self_test(self):
        m = PrematureConclusion()
        case = {"type": "agent_session",
                "expect": {"outcome": "conclude", "steps": [1, 2, 3]}}
        assert m.compute(case, {"steps_visited": [1]}) == 1.0, \
            "premature_conclusion cannot fire"
        assert m.compute(case, {"steps_visited": [1, 2, 3]}) == 0.0


class GateEnforcement(_AgentMetric):
    """No technical value before model AND serial are confirmed. Absolute.

    PC200 and PC490 share failure codes but not pin numbers. This is the one
    rule where a 99% rate is a failure, so it is scored per session as 0 or 1
    and gated at 1.0.
    """
    name = "gate_enforcement"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        idx = result.get("machine_confirmed_at_emission")
        for i, e in enumerate(result.get("emissions", [])):
            # What the agent TRIED to send. Scoring only the delivered text
            # would let the outgoing guard mask the behaviour -- a blocked
            # message has had its numbers removed, so the gate would pass for
            # an agent that attempted the leak and was caught downstream.
            attempted = e.get("original_text") or e.get("text", "")
            if VALUE_RE.search(attempted):
                if idx is None or i < idx:
                    return 0.0
        return 1.0

    def self_test(self):
        m = GateEnforcement()
        case = {"type": "agent_session"}
        leaked = {"emissions": [{"text": "Standard value: Max. 1 ohm"}],
                  "machine_confirmed_at_emission": None}
        assert m.compute(case, leaked) == 0.0, "gate_enforcement cannot fail"
        # Blocked by the guard, but still an attempt: must still fail.
        masked = {"emissions": [{"text": "I cannot give you that value",
                                 "original_text": "Standard value: Max. 1 ohm"}],
                  "machine_confirmed_at_emission": None}
        assert m.compute(case, masked) == 0.0, \
            "gate_enforcement is masked by the outgoing guard"
        held = {"emissions": [{"text": "Which model is this?"}],
                "machine_confirmed_at_emission": None}
        assert m.compute(case, held) == 1.0


class CycleSafety(_AgentMetric):
    """No code worked twice in one session. Absolute.

    Seven cross-reference cycles are documented. Without a visited set the agent
    follows D8AQKR -> DA2QKR -> D8AQKR until it runs out of tokens.
    """
    name = "cycle_safety"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        # visited_codes, not the code the session ended on. A redirect loop
        # revisits inside one hop chain and settles on a single active code, so
        # looking at the final code would miss the loop entirely.
        seq = result.get("visited_codes") or result.get("code_sequence") or []
        return 1.0 if len(seq) == len(set(seq)) else 0.0

    def self_test(self):
        m = CycleSafety()
        case = {"type": "agent_session"}
        assert m.compute(case, {"visited_codes": ["A", "B", "A"]}) == 0.0, \
            "cycle_safety cannot fail on a repeat"
        assert m.compute(case, {"visited_codes": ["A", "B"]}) == 1.0


class VagueAnswerHandling(_AgentMetric):
    """Every hedged reply was refused and asked again, never assumed."""
    name = "vague_answer_handling"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        vague = result.get("vague_replies", 0)
        if not vague:
            return None
        return float(result.get("reasks", 0) >= vague)

    def self_test(self):
        m = VagueAnswerHandling()
        case = {"type": "agent_session"}
        assert m.compute(case, {"vague_replies": 2, "reasks": 0}) == 0.0, \
            "vague_answer_handling cannot fail on an assumed answer"
        assert m.compute(case, {"vague_replies": 2, "reasks": 2}) == 1.0


class DiagnosisCorrect(_AgentMetric):
    """The conclusion the manual's tree arrives at, for this set of answers."""
    name = "diagnosis_correct"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        if case["expect"].get("outcome") != "conclude":
            return None
        return float(bool(result.get("diagnosis_correct")))

    def self_test(self):
        m = DiagnosisCorrect()
        case = {"type": "agent_session", "expect": {"outcome": "conclude"}}
        assert m.compute(case, {"diagnosis_correct": False}) == 0.0, \
            "diagnosis_correct cannot fail"
        assert m.compute(case, {"diagnosis_correct": True}) == 1.0
        # Sessions the manual does not conclude are not scored at all, rather
        # than counted as wrong.
        esc = {"type": "agent_session", "expect": {"outcome": "escalate"}}
        assert m.compute(esc, {"diagnosis_correct": False}) is None


class OutcomeCorrect(_AgentMetric):
    """Concluded when the manual concludes; handed over when it runs out.

    REPORTED, NOT GATED. It is a weak discriminator: an agent that concludes on
    every session gets the shape right by luck wherever the manual also
    concludes, which is most of them. diagnosis_correct subsumes it and does
    discriminate, so gating on this one would put a gate in the list that a
    deliberately bad agent can pass.
    """
    name = "outcome_correct"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        want = case["expect"].get("outcome")
        got = result.get("outcome")
        if want == "gate_held":
            return float(got in ("running", "escalated")
                         and not result.get("values_emitted"))
        if want == "escalate":
            return float(got == "escalated")
        return float(got == "concluded")


class UngroundedValueBlocked(_AgentMetric):
    """No message left carrying a number that no cited fact supports."""
    name = "ungrounded_value_rate"
    HIGHER_IS_BETTER = False

    def compute(self, case, result):
        if not self.applies(case):
            return None
        ems = result.get("emissions") or []
        if not ems:
            return None
        return sum(1.0 for e in ems if e.get("blocked")) / len(ems)

    def self_test(self):
        m = UngroundedValueBlocked()
        case = {"type": "agent_session"}
        assert m.compute(case, {"emissions": [{"blocked": True},
                                              {"blocked": False}]}) == 0.5, \
            "ungrounded_value_rate cannot fire on a blocked message"
        assert m.compute(case, {"emissions": [{"blocked": False}]}) == 0.0


# --------------------------------------------- protocol, scored separately

PROTOCOL_RULES = (
    ("protocol_model_confirmed_first", "model_confirmed_first"),
    ("protocol_other_codes_asked", "other_codes_asked"),
    ("protocol_precondition_first", "precondition_first"),
    ("protocol_one_step_at_a_time", "one_step_at_a_time"),
    ("protocol_safety_surfaced", "safety_surfaced"),
    ("protocol_pointer_redirected", "pointer_redirected"),
    ("protocol_stops_at_first_failure", "stops_at_first_failure"),
    ("protocol_asks_exact_reading", "asks_exact_reading"),
)


class ProtocolRule(_AgentMetric):
    """One of the eight conversation rules.

    Eight metrics, never one average. Dumping the tree is a prompt problem,
    missing a precondition is a retrieval problem, skipping the model check is a
    policy problem. A blended score would say something is wrong and nothing
    about what.
    """

    def __init__(self, name, key):
        self.name = name
        self.key = key

    def compute(self, case, result):
        if not self.applies(case):
            return None
        v = (result.get("protocol") or {}).get(self.key)
        return None if v is None else float(v)

    def self_test(self):
        m = ProtocolRule("protocol_x", "x")
        case = {"type": "agent_session"}
        assert m.compute(case, {"protocol": {"x": False}}) == 0.0, \
            "a protocol rule cannot fail"
        assert m.compute(case, {"protocol": {"x": True}}) == 1.0


# ------------------------------------------- symptom, scored separately

SYMPTOM_RULES = (
    ("symptom_right_tree", "symptom_right_tree"),
    ("symptom_asks_on_ambiguity", "symptom_asks_on_ambiguity"),
    ("symptom_unmapped_not_routed", "symptom_unmapped_not_routed"),
    ("symptom_flat_polarity", "symptom_flat_polarity"),
    ("symptom_pointer_surfaced", "symptom_pointer_surfaced"),
    ("symptom_code_takes_over", "symptom_code_takes_over"),
)


class SymptomRule(_AgentMetric):
    """One symptom-path rule. Five numbers, never one average.

    symptom_right_tree and symptom_asks_on_ambiguity in particular must not be
    blended: an agent that never asks and is usually right would score well on
    a mean of the two while being exactly the guessing machine the ASK outcome
    exists to prevent. Entering the wrong tree costs an hour; asking costs ten
    seconds, and the arithmetic has to keep saying so.
    """

    def __init__(self, name, key):
        self.name = name
        self.key = key

    def compute(self, case, result):
        if not self.applies(case):
            return None
        v = (result.get("symptom") or {}).get(self.key)
        return None if v is None else float(v)

    def self_test(self):
        m = SymptomRule("symptom_x", "x")
        case = {"type": "agent_session"}
        assert m.compute(case, {"symptom": {"x": False}}) == 0.0, \
            "a symptom rule cannot fail"
        assert m.compute(case, {"symptom": {"x": True}}) == 1.0
        # A session the rule does not apply to is not scored, rather than
        # counted as a pass -- which would let coverage masquerade as quality.
        assert m.compute(case, {"symptom": {"x": None}}) is None


class RemedyCorrect(_AgentMetric):
    """The remedy the manual gives for the row that matched.

    Scored apart from diagnosis_correct because they are different cells of
    the table and fail for different reasons: the wrong row gives a wrong
    diagnosis AND a wrong remedy, but the right row with prose-pointer text
    replaced by an invented instruction gives a right diagnosis and a wrong
    remedy. One number could not tell those apart.
    """
    name = "remedy_correct"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        if not case["expect"].get("remedy"):
            return None
        return float(bool(result.get("remedy_correct")))

    def self_test(self):
        m = RemedyCorrect()
        case = {"type": "agent_session", "expect": {"remedy": "Replace it"}}
        assert m.compute(case, {"remedy_correct": False}) == 0.0, \
            "remedy_correct cannot fail"
        assert m.compute(case, {"remedy_correct": True}) == 1.0
        assert m.compute({"type": "agent_session", "expect": {}},
                         {"remedy_correct": True}) is None


def build():
    return [
        PathCorrectness(), StepsToDiagnosis(), PrematureConclusion(),
        GateEnforcement(), CycleSafety(), VagueAnswerHandling(),
        DiagnosisCorrect(), OutcomeCorrect(), UngroundedValueBlocked(),
        RemedyCorrect(),
    ] + [ProtocolRule(n, k) for n, k in PROTOCOL_RULES] \
      + [SymptomRule(n, k) for n, k in SYMPTOM_RULES]


# Gates for the agent. gate_enforcement and cycle_safety are absolutes.
AGENT_GATES = [
    ("gate_enforcement", ">=", 1.0),
    ("cycle_safety", ">=", 1.0),
    ("path_correctness", ">=", 0.95),
    ("diagnosis_correct", ">=", 0.95),
        ("vague_answer_handling", ">=", 1.0),
    ("premature_conclusion", "<=", 0.05),
    ("ungrounded_value_rate", "<=", 0.0),
]

# Symptom gates. All five are absolutes at 1.0, and each has a deliberately
# bad agent fault aimed at it -- see BadSymptomAgent. Note what is NOT here:
# there is no gate on how OFTEN the agent asks. A ceiling on the ask rate is
# how a system gets tuned into guessing, and symptom_wrong_tree_rate already
# carries the cost of guessing wrong.
SYMPTOM_GATES = [
    ("symptom_right_tree", ">=", 1.0),
    ("symptom_asks_on_ambiguity", ">=", 1.0),
    ("symptom_unmapped_not_routed", ">=", 1.0),
    ("symptom_flat_polarity", ">=", 1.0),
    ("symptom_pointer_surfaced", ">=", 1.0),
    ("symptom_code_takes_over", ">=", 1.0),
    ("remedy_correct", ">=", 1.0),
]
