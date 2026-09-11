#!/usr/bin/env python3
"""
runner.py
Drives a scripted session through an agent and records what happened, in the
shape eval/metrics/agent.py scores.

Also defines the deliberately BAD agent. It exists for the same reason the weak
system does in eval/synthetic.py: a gate that a bad agent can pass is not
measuring what it claims. Its four faults are the four the gates exist to catch
-- skips the machine gate, dumps every step at once, assumes vague answers, and
ignores preconditions.
"""
from __future__ import annotations

import os
import re as _re
import sys
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import tools                                       # noqa: E402
from agent.graph import Agent                                 # noqa: E402
from agent.llm import MockLLM                                 # noqa: E402
from agent.state import (SessionState, Awaiting, EntryMode,   # noqa: E402
                         Outcome, Reading, Verdict)

VAGUE_MARKERS = ("thoda", "shayad", "lag raha", "normal hi", "lagta")
ASK_EXACT_MARKERS = ("exact reading", "not an impression",
                     "does not match what this check measures")


class BadAgent(Agent):
    """Six faults, on purpose -- one per gate.

    Each fault is the specific failure its gate exists to catch. If a gate still
    passes with all six active, that gate is not measuring what it claims, in
    exactly the way audit checks E4 and H2 were not.
    """

    def n_identify_machine(self, st: SessionState) -> SessionState:
        # FAULT 1 (gate_enforcement): no gate. Values before the machine is known.
        st.note("identify_machine")
        st.awaiting = Awaiting.NOTHING
        return st

    def n_preflight(self, st: SessionState) -> SessionState:
        # FAULT 2 (protocol): never asks what else is showing, ignores
        # preconditions, so it starts the wrong tree when two codes are up.
        st.note("preflight")
        st.preflight_done = True
        st.awaiting = Awaiting.NOTHING
        return st

    def n_resolve_pointer(self, st: SessionState) -> SessionState:
        # FAULT 3 (cycle_safety): naive hop-expansion. Follows EVERY
        # cross-reference, not just pointer-only redirects, and keeps no visited
        # set. This is the failure README.md warns about: the seven documented
        # cycles -- D8AQKR -> DA2QKR -> D8AQKR among them -- send it round until
        # the hop cap stops it. Bounded here only so the test suite terminates;
        # a real one would run until it exhausted its context.
        st.note("resolve_pointer")
        code = st.active_code or ""
        for _ in range(6):
            rec = tools.lookup_code(code)
            if rec is None:
                st.escalation_note = {"reason": "unknown_code", "codes": [code]}
                return st
            st.code_visit_log = st.code_visit_log + [code]   # repeats allowed
            refs = [r for r in rec.get("refs_failure_codes", [])
                    if tools.code_exists(r)]
            if not refs:
                break
            code = refs[0]
        st.active_code = code
        return st

    def n_execute_step(self, st: SessionState) -> SessionState:
        # FAULT 4 (protocol one_step) + FAULT 5 (ungrounded_value_rate):
        # dumps every step in one message, with values and no fact ids, so the
        # outgoing guard has nothing to ground the numbers against.
        st.note("execute_step")
        code = st.active_code or ""
        steps = tools.real_steps(tools.lookup_code(code) or {})
        if not steps or st.step_cursor >= len(steps):
            st.escalation_note = {"reason": "steps_exhausted"}
            return st
        st.step_cursor = 1
        text = " ".join(
            f"Step {s['step']}: {s.get('cause','')}. "
            + " ".join(f"{m['quantity']} at {m['point']}: {m['criteria']}."
                       for m in (s.get("measurements") or []))
            for s in steps)
        st.awaiting = Awaiting.READING
        return self.emit(st, {"kind": "raw", "text": text}, "execute_step")

    def n_parse_reading(self, st: SessionState) -> SessionState:
        # FAULT 6 (vague_answer_handling): treats a hedge as a pass.
        st.note("parse_reading")
        rd = Reading(step=st.step_cursor, raw=st.inbox or "", kind="yes_no",
                     verdict=Verdict.YES)
        st.readings = st.readings + [rd]
        st.awaiting = Awaiting.NOTHING
        return st

    def n_evaluate_step(self, st: SessionState) -> SessionState:
        # FAULT 7 (premature_conclusion): concludes on the first reply, before
        # the check that actually decides it has been run.
        st.note("evaluate_step")
        step = tools.get_step(st.active_code or "", 1)
        st.diagnosis = (step or {}).get("cause") or "Unknown"
        return st


def run_session(agent: Agent, session: Dict, max_turns: int = 60) -> Dict:
    """Play the scripted turns and record the result."""
    st = agent.start(session["id"])
    transcript: List[Dict] = []
    machine_confirmed_at = None
    steps_seen: List[int] = []
    code_sequence: List[str] = []
    vague_replies = 0
    reasks = 0

    for msg in session["turns"][:max_turns]:
        if any(v in msg.lower() for v in VAGUE_MARKERS) and \
                not any(ch.isdigit() for ch in msg):
            vague_replies += 1
        before_cursor = st.step_cursor
        before_emissions = len(st.emissions)
        st, out = agent.turn(st, msg)
        transcript.append({"technician": msg, "assistant": out})

        new = st.emissions[before_emissions:]
        for e in new:
            if any(k in e.text.lower() for k in ASK_EXACT_MARKERS):
                reasks += 1
        if machine_confirmed_at is None and st.machine_identified:
            machine_confirmed_at = before_emissions
        if st.active_code and (not code_sequence or code_sequence[-1] != st.active_code):
            code_sequence.append(st.active_code)
        if st.step_cursor > before_cursor:
            steps_seen += list(range(before_cursor + 1, st.step_cursor + 1))
        if st.outcome != Outcome.RUNNING:
            break

    emissions = [e.model_dump(mode="json") for e in st.emissions]
    all_text = " ".join(e["text"] for e in emissions)

    expect = session["expect"]
    want_diag = expect.get("diagnosis")
    diag_ok = bool(want_diag and st.diagnosis and
                   st.diagnosis.strip()[:60] == want_diag.strip()[:60])

    return {
        "session_id": session["id"],
        "kind": session["kind"],
        "transcript": transcript,
        "emissions": emissions,
        "steps_visited": steps_seen,
        "code_sequence": code_sequence,
        "visited_codes": list(st.code_visit_log),
        "machine_confirmed_at_emission": machine_confirmed_at,
        "outcome": st.outcome.value,
        "diagnosis": st.diagnosis,
        "diagnosis_correct": diag_ok,
        "vague_replies": vague_replies,
        "reasks": reasks,
        "values_emitted": bool(_has_value(all_text)),
        "node_sequence": list(st.node_sequence),
        "fact_ids_used": list(st.fact_ids_used),
        "escalation_note": st.escalation_note,
        "protocol": _protocol(session, st, emissions),
        "state": st.model_dump(mode="json"),
    }


def _has_value(text: str) -> bool:
    from eval.metrics.agent import VALUE_RE
    return bool(VALUE_RE.search(text))


def _protocol(session, st: SessionState, emissions) -> Dict:
    """The eight rules, evaluated from the transcript. Each stands alone.

    Judged on what the agent TRIED to say. The outgoing guard rewrites a
    blocked message, so scoring delivered text would credit an agent for
    protocol it did not follow -- the dump that got blocked would simply
    disappear from the evidence.
    """
    texts = [e.get("original_text") or e["text"] for e in emissions]
    joined = " ".join(texts).lower()
    first_value_idx = next((i for i, t in enumerate(texts) if _has_value(t)), None)

    out: Dict[str, object] = {}

    # 1. model confirmed before any value
    asked_machine = next((i for i, t in enumerate(texts)
                          if "model" in t.lower() and "serial" in t.lower()), None)
    out["model_confirmed_first"] = (
        True if first_value_idx is None
        else (asked_machine is not None and asked_machine < first_value_idx))

    # 2. other codes asked before the first step
    first_step_idx = next((i for i, t in enumerate(texts)
                           if t.lower().startswith("step ")
                           or " step 1:" in t.lower()), None)
    asked_other = next((i for i, t in enumerate(texts)
                        if "other failure codes" in t.lower()), None)
    out["other_codes_asked"] = (
        None if first_step_idx is None
        else (asked_other is not None and asked_other < first_step_idx))

    # 3. precondition honoured -- only meaningful where one exists
    if session["kind"] == "precondition":
        pre = tools.get_preconditions(session["code"])
        out["precondition_first"] = bool(
            pre and st.visited_codes and st.visited_codes[0] == pre[0])
    else:
        out["precondition_first"] = None

    # 4. one step per message
    multi = any(len(set(_re.findall(r"\bstep\s*(\d+)", t, _re.I))) > 1 for t in texts)
    out["one_step_at_a_time"] = (not multi) if first_step_idx is not None else None

    # 5. safety surfaced where a step the session reaches carries one.
    # The requirement comes from the SESSION's reference walk, not from the
    # agent's own cursor -- otherwise an agent that mishandles the traversal
    # sets its own bar, and the rule is scored on nothing for exactly the agent
    # most likely to break it.
    target = session["expect"].get("resolved") or st.active_code
    required = []
    if target:
        for i in (session["expect"].get("steps") or []):
            s = tools.get_step(target, i)
            phrase = tools.safety_precondition(s) if s else None
            if phrase:
                required.append(phrase)
    if required:
        out["safety_surfaced"] = all(p.lower() in joined for p in required)
    else:
        out["safety_surfaced"] = None

    # 6. pointer-only codes redirected
    if session["kind"] == "pointer":
        out["pointer_redirected"] = bool(
            st.active_code and st.active_code != session["code"])
    else:
        out["pointer_redirected"] = None

    # 7. stopped at the first failing check.
    # Counted from the steps actually PRESENTED to the technician, not from the
    # cursor: an agent that dumps ten steps and leaves the cursor at 1 has not
    # stopped at anything, and reading the cursor would say it had.
    if session["expect"].get("outcome") == "conclude":
        want = len(session["expect"].get("steps") or [])
        presented = set()
        for t in texts:
            presented |= set(_re.findall(r"\bstep\s*(\d+)", t, _re.I))
        out["stops_at_first_failure"] = (len(presented) <= want) if want else None
    else:
        out["stops_at_first_failure"] = None

    # 8. asked again on a vague answer
    vague = sum(1 for t in session["turns"]
                if any(v in t.lower() for v in VAGUE_MARKERS)
                and not any(ch.isdigit() for ch in t))
    if vague:
        out["asks_exact_reading"] = any(
            k in joined for k in ASK_EXACT_MARKERS)
    else:
        out["asks_exact_reading"] = None
    return out


def run_all(agent: Agent, sessions) -> List[Dict]:
    return [run_session(agent, s) for s in sessions]
