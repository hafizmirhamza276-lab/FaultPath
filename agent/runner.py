#!/usr/bin/env python3
"""
runner.py
Drives a scripted session through an agent and records what happened, in the
shape eval/metrics/agent.py scores.

Also defines the two deliberately BAD agents. They exist for the same reason
the weak system does in eval/synthetic.py: a gate that a bad agent can pass is
not measuring what it claims.

BadAgent breaks the code path -- skips the machine gate, dumps every step at
once, assumes vague answers, ignores preconditions, loses the visited set.

BadSymptomAgent breaks the symptom path, and every one of its five faults
produces PLAUSIBLE OUTPUT AND NO ERROR: it picks a tree when it should ask,
forces an unmapped input to the nearest tree, walks a flat tree with branching
polarity, answers a prose pointer as though it held the section it points at,
and carries on down a symptom tree after the technician reads out a failure
code. None of those raises; only the tree walked can tell.
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
                         Outcome, Reading, Verdict, TreeKind)

STEP_N_RE = r"\b(?:step|check)\s*(\d+)"

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


class BadSymptomAgent(Agent):
    """Five faults on the symptom path, one per symptom gate.

    Every one of them produces PLAUSIBLE OUTPUT AND NO ERROR, which is why they
    need a deliberately bad agent rather than an exception handler. A technician
    reading this agent's transcript cannot tell it is wrong; only the tree it
    walked can.
    """

    def n_resolve_entry(self, st: SessionState) -> SessionState:
        st.note("resolve_entry")
        if st.escalation_note and st.escalation_note.get("reason") == "unknown_code":
            return st
        if st.entry_resolved:
            return st
        if st.entry_mode != EntryMode.SYMPTOM:
            st.awaiting = Awaiting.ENTRY
            return self.emit(st, {"kind": "ask_entry"}, "resolve_entry")

        res = tools.search_symptoms(st.symptom_text or "")
        # FAULT 1 (symptom_never_guesses): picks the first candidate on an ASK
        # instead of presenting them. Confident, fast, and wrong half the time.
        if res["symptom_ids"]:
            sid = sorted(res["symptom_ids"])[0]
            st.enter_symptom(sid, tools.tree_kind(sid), "guessed")
            return st
        # FAULT 2 (symptom_unmapped_not_routed): forces an unmapped input to
        # the nearest tree by token overlap, with no floor at all.
        best, score = None, -1.0
        want = set((st.symptom_text or "").lower().split())
        for cand in tools.symptoms():
            have = set(tools.lookup_symptom(cand)["symptom"].lower().split())
            s = len(want & have) / (len(want) or 1)
            if s > score:
                best, score = cand, s
        if best:
            st.enter_symptom(best, tools.tree_kind(best), "nearest")
        return st

    def n_preflight(self, st: SessionState) -> SessionState:
        # FAULT 5 (symptom_code_takes_over): notes the failure code the
        # technician just read off the monitor and carries on down the symptom
        # tree anyway, because that is what was originally asked about. The
        # manual says the opposite -- clear the code first -- and the session
        # spends twenty minutes on the wrong tree while a named code sits
        # unworked.
        st.note("preflight")
        if st.awaiting == Awaiting.OTHER_CODES and st.inbox is not None:
            parsed = self.llm.parse_intake(st.inbox)
            st.pending_codes = list(dict.fromkeys(
                st.pending_codes + [c for c in parsed.codes
                                    if tools.code_exists(c)]))
            st.preflight_done = True
            st.awaiting = Awaiting.NOTHING
            return st
        if not st.preflight_done:
            st.awaiting = Awaiting.OTHER_CODES
            return self.emit(st, {"kind": "ask_other_codes"}, "preflight")
        return st

    def n_execute_step(self, st: SessionState) -> SessionState:
        # FAULT 3 (flat_tree_polarity): walks a FLAT tree as though it branched
        # -- asks "is it normal, yes or no?" on rows that have no branches, and
        # treats a confirmation as a reason to advance.
        if st.tree_kind == TreeKind.FLAT:
            st.note("execute_step")
            n = st.step_cursor + 1
            step = tools.get_symptom_step(st.active_symptom or "", n)
            if step is None:
                st.escalation_note = {"reason": "rows_exhausted"}
                return st
            st.step_cursor = n
            st.awaiting = Awaiting.READING          # not OBSERVATION
            return self.emit(st, {"kind": "step", "step": step["step"],
                                  "cause": step.get("cause") or "",
                                  "procedure": (step.get("point_to_check") or "")[:400],
                                  "safety": None, "expects": "yes_no"},
                             "execute_step", [step.get("fact_id")],
                             [step.get("cause") or "",
                              (step.get("point_to_check") or "")[:400]])
        return super().n_execute_step(st)

    def n_evaluate_step(self, st: SessionState) -> SessionState:
        if st.tree_kind == TreeKind.FLAT:
            st.note("evaluate_step")
            step = tools.get_symptom_step(st.active_symptom or "", st.step_cursor)
            rd = st.readings[-1]
            # Branching polarity on a flat tree: YES advances, NO concludes.
            # Exactly backwards, and it produces a fluent wrong answer.
            if rd.verdict in (Verdict.YES, Verdict.OBSERVED):
                return st
            st.diagnosis = (step or {}).get("cause") or ""
            # FAULT 4 (prose_pointer_not_followed): reads the pointer and
            # answers as though it had the section it points at.
            ptr = tools.step_prose_pointer(st.active_symptom or "", step or {})
            if ptr:
                st.remedy = ("Follow the E mode procedure for the engine not "
                             "cranking and replace the starting motor.")
            else:
                st.remedy = (step or {}).get("remedy") or ""
            return st
        return super().n_evaluate_step(st)


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
    want_rem = expect.get("remedy")
    remedy_ok = (None if not want_rem else
                 bool(st.remedy and st.remedy.strip()[:60] == want_rem.strip()[:60]))

    return {
        "session_id": session["id"],
        "kind": session["kind"],
        "symptom_id": st.active_symptom,
        "tree_kind": getattr(st.tree_kind, "value", None),
        "symptom_expected": session.get("symptom_id"),
        "remedy": st.remedy,
        "remedy_correct": remedy_ok,
        "prose_pointer": st.prose_pointer,
        "entry_switch_log": list(st.entry_switch_log),
        "symptom_match_layer": st.symptom_match_layer,
        "symptom": _symptom_protocol(session, st, emissions),
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


# Phrases that only a YES/NO question produces, and only an observation
# question produces. Kept apart because the whole flat-tree argument is that
# these are different questions, not two wordings of one.
YES_NO_MARKERS = ("is it normal? answer yes or no",)
OBSERVATION_MARKERS = ("is that what you are seeing",)
ASK_CHOICE_MARKERS = ("which one matches", "more than one troubleshooting tree")
UNMAPPED_MARKERS = ("no troubleshooting tree for that",
                    "closest-looking one")


def _symptom_protocol(session, st: SessionState, emissions) -> Dict:
    """Five symptom rules, each standing alone, none averaged with another.

    Scored on what the agent TRIED to say, like the other eight: the outgoing
    guard rewrites a blocked message, and an agent that attempted to ask
    yes/no on a flat tree and got blocked for an unrelated reason must still
    be recorded as having attempted it.
    """
    texts = [e.get("original_text") or e["text"] for e in emissions]
    joined = " ".join(texts).lower()
    exp = session.get("expect") or {}
    out: Dict[str, object] = {}

    # 1. entered the tree the walk says, or did not enter one at all where the
    #    manual has none. Applies to every symptom session.
    want_sid = exp.get("symptom_id") if "symptom_id" in exp else None
    if session["kind"].startswith("symptom") and session["kind"] != "symptom_to_code":
        out["symptom_right_tree"] = (st.active_symptom == want_sid)
    else:
        out["symptom_right_tree"] = None

    # 2. ambiguity was presented, never resolved silently.
    if session["kind"] == "symptom_ask":
        asked = any(k in joined for k in ASK_CHOICE_MARKERS)
        # Presenting ONE candidate is not presenting ambiguity.
        listed = max((t.count("(page ") for t in texts), default=0)
        out["symptom_asks_on_ambiguity"] = bool(asked and listed >= 2)
    else:
        out["symptom_asks_on_ambiguity"] = None

    # 3. an unmapped input was refused, not routed to the nearest tree.
    if session["kind"] == "symptom_unmapped":
        out["symptom_unmapped_not_routed"] = bool(
            any(k in joined for k in UNMAPPED_MARKERS)
            and st.active_symptom is None)
    else:
        out["symptom_unmapped_not_routed"] = None

    # 4. FLAT TREES ARE NOT ASKED YES/NO. The rule the polarity argument rests
    #    on: a flat tree has no branch outcomes to read back, and confirming a
    #    row means the fault is found rather than that the check was normal.
    if exp.get("tree_kind") == "SymptomTreeFlat":
        asked_yes_no = any(k in joined for k in YES_NO_MARKERS)
        asked_observation = any(k in joined for k in OBSERVATION_MARKERS)
        stopped_right = st.step_cursor == len(exp.get("steps") or [])
        out["symptom_flat_polarity"] = bool(
            asked_observation and not asked_yes_no and stopped_right)
    else:
        out["symptom_flat_polarity"] = None

    # 5. a real failure code produced mid-symptom takes over the session. The
    #    manual's own pre-troubleshooting instruction, so following it is
    #    reading the manual rather than preferring a code.
    if session["kind"] == "symptom_to_code":
        out["symptom_code_takes_over"] = bool(
            st.active_code == session.get("code")
            and st.active_symptom is None
            and st.entry_switch_log)
    else:
        out["symptom_code_takes_over"] = None

    # 6. a prose pointer is surfaced with its page and NOT followed. The only
    #    real instance in the corpus is SM01's remedy, which is stated in the
    #    fixture so one passing test is not mistaken for broad coverage.
    if exp.get("pointer"):
        ptr = st.prose_pointer or {}
        surfaced = bool(ptr.get("text") == exp["pointer"] and ptr.get("manual_page"))
        # "Follow the ... procedure and replace ..." is the agent answering as
        # though it held the section it was pointed at.
        invented = bool(st.remedy and st.remedy.strip() != (exp.get("remedy") or "").strip())
        out["symptom_pointer_surfaced"] = bool(surfaced and not invented)
    else:
        out["symptom_pointer_surfaced"] = None
    return out


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
    # "Check N:" as well as "Step N:" -- a flat S-Mode row is presented as a
    # check, and a rule that only counted steps would score every flat session
    # on nothing while still reporting a number for it.
    first_step_idx = next((i for i, t in enumerate(texts)
                           if t.lower().startswith(("step ", "check "))
                           or " step 1:" in t.lower()
                           or " check 1:" in t.lower()), None)
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
    multi = any(len(set(_re.findall(STEP_N_RE, t, _re.I))) > 1 for t in texts)
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
            presented |= set(_re.findall(STEP_N_RE, t, _re.I))
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
