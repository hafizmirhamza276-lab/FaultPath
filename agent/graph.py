#!/usr/bin/env python3
"""
graph.py
The LangGraph diagnostic agent.

  START -> receive
  receive        --> intake | identify_machine | preflight | parse_reading
  intake         --> identify_machine
  identify_machine --> pause(ask) | resolve_entry
  resolve_entry  --> pause(ask) | preflight | escalate
  preflight      --> pause(ask) | resolve_pointer
  resolve_pointer--> execute_step | escalate
  execute_step   --> pause(await reading)
  parse_reading  --> pause(re-ask) | evaluate_step
  evaluate_step  --> execute_step | conclude | escalate
  conclude / escalate / pause --> END

Every edge is a pure function of state. The LLM appears in exactly three places
-- intake, parse_reading, and phrasing inside emit() -- and in none of them does
it choose a step, compare a value, or decide a diagnosis.

Transport-agnostic on purpose: turn() takes a state and a string and returns a
state and a string. FastAPI, a CLI or a test harness all drive it the same way.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Optional, Tuple

from langgraph.graph import StateGraph, END

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import tools                                            # noqa: E402
from agent.guards import enforce                                   # noqa: E402
from agent.llm import LLM, MockLLM                                 # noqa: E402
from agent.state import (SessionState, Awaiting, EntryMode, Outcome,  # noqa: E402
                         Reading, Verdict, Emission, MAX_DEPTH, MAX_REASKS)
from core.logging import NullLogger, new_trace_id                  # noqa: E402


class Agent:
    """Holds the compiled graph plus the provider seam."""

    def __init__(self, llm: Optional[LLM] = None, logger=None):
        self.llm = llm or MockLLM()
        self.log = logger or NullLogger()
        self.graph = self._build()

    # ------------------------------------------------------------- emit
    def emit(self, st: SessionState, payload: dict, node: str,
             fact_ids=(), grounded_text=()) -> SessionState:
        """Phrase, cite, guard, record. The only way a message leaves."""
        fact_ids = [f for f in fact_ids if f]
        citations = tools.render_citations(fact_ids)
        text = self.llm.phrase(payload).text

        # Structural integers the phrasing legitimately uses for sequencing.
        extra = [str(payload.get("step") or ""), str(st.step_cursor),
                 str(len(st.readings))]
        safe, blocked, bad = enforce(
            text, citations=citations, fact_ids=fact_ids,
            code=st.active_code, model=st.model, serial=st.serial,
            technician_text=st.inbox or "", extra=[e for e in extra if e],
            grounded_text=grounded_text)

        st.use_facts(fact_ids)
        st.emissions = st.emissions + [Emission(
            text=safe, original_text=text, fact_ids=fact_ids,
            citations=citations, node=node, blocked=blocked,
            grounded_text=[g for g in grounded_text if g],
            block_reason=(f"ungrounded numbers: {bad}" if blocked else ""))]
        if blocked:
            self.log.event("validation", "message_blocked",
                           {"node": node, "offending": bad}, level="error")
        return st

    # ------------------------------------------------------------ nodes
    def n_receive(self, st: SessionState) -> SessionState:
        st.note("receive")
        return st

    def n_intake(self, st: SessionState) -> SessionState:
        st.note("intake")
        parsed = self.llm.parse_intake(st.inbox or "")
        if parsed.model and not st.model:
            st.model = parsed.model
        if parsed.serial and not st.serial:
            st.serial = parsed.serial
        known = [c for c in parsed.codes if tools.code_exists(c)]
        # A candidate that is not in golden/ is only treated as a failure code
        # if it is code-SHAPED -- carries a digit, @ or #. Otherwise it is an
        # uppercase word in a Roman Urdu sentence, and escalating on it would
        # turn "ACHHA" into an unknown-code report.
        unknown = [c for c in parsed.codes
                   if not tools.code_exists(c)
                   and any(ch.isdigit() or ch in "@#" for ch in c)]
        if known:
            st.entry_mode = EntryMode.CODE
            if st.active_code and known[0] != st.active_code:
                # Technician changed the code mid-session: restart traversal.
                st.step_cursor = 0
                st.diagnosis = None
            st.active_code = known[0]
            st.pending_codes = [c for c in known[1:] if c != st.active_code]
        elif unknown:
            # A code-shaped token that is not in this manual. Say so; do not
            # retrieve the nearest match and answer about a different fault.
            st.entry_mode = EntryMode.UNKNOWN
            st.symptom_text = None
            st.pending_codes = []
            st.diagnosis = None
            st.escalation_note = {"reason": "unknown_code", "codes": unknown}
        elif parsed.symptom:
            st.entry_mode = EntryMode.SYMPTOM
            st.symptom_text = parsed.symptom
        self.log.event("decision", "intake",
                       {"codes": parsed.codes, "model": parsed.model,
                        "serial": parsed.serial, "mode": st.entry_mode.value},
                       level="detail")
        return st

    def n_identify_machine(self, st: SessionState) -> SessionState:
        """The hard gate. Not a prompt instruction -- an edge in the graph."""
        st.note("identify_machine")
        if st.awaiting == Awaiting.MACHINE and st.inbox:
            parsed = self.llm.parse_intake(st.inbox)
            if parsed.model:
                st.model = parsed.model
            if parsed.serial:
                st.serial = parsed.serial
            if parsed.codes and not st.active_code:
                known = [c for c in parsed.codes if tools.code_exists(c)]
                if known:
                    st.entry_mode = EntryMode.CODE
                    st.active_code = known[0]
        if not st.machine_identified:
            st.awaiting = Awaiting.MACHINE
            return self.emit(st, {"kind": "ask_machine"}, "identify_machine")
        st.awaiting = Awaiting.NOTHING
        return st

    def n_resolve_entry(self, st: SessionState) -> SessionState:
        st.note("resolve_entry")
        if st.escalation_note and st.escalation_note.get("reason") == "unknown_code":
            return st
        if st.entry_mode == EntryMode.CODE and st.active_code:
            return st
        if st.entry_mode == EntryMode.SYMPTOM:
            hits = tools.search_symptoms(st.symptom_text or "")
            if not hits:
                st.escalation_note = {
                    "reason": "symptom_entry_unavailable",
                    "symptom": st.symptom_text,
                    "detail": ("H-Mode and S-Mode symptom trees are not "
                               "extracted, so there is no ground truth to "
                               "execute for a symptom.")}
                return st
        st.awaiting = Awaiting.ENTRY
        return self.emit(st, {"kind": "ask_entry"}, "resolve_entry")

    def n_preflight(self, st: SessionState) -> SessionState:
        """Ask what else is showing, then honour the manual's precondition."""
        st.note("preflight")
        if st.awaiting == Awaiting.OTHER_CODES and st.inbox is not None:
            parsed = self.llm.parse_intake(st.inbox)
            extra = [c for c in parsed.codes
                     if tools.code_exists(c) and c != st.active_code]
            st.pending_codes = list(dict.fromkeys(st.pending_codes + extra))
            st.preflight_done = True
            st.awaiting = Awaiting.NOTHING

        if not st.preflight_done:
            st.awaiting = Awaiting.OTHER_CODES
            return self.emit(st, {"kind": "ask_other_codes"}, "preflight")

        # Precondition: if the manual says solve X first and X is on the
        # monitor, go to X. Starting the wrong tree wastes the session.
        pres = tools.get_preconditions(st.active_code or "")
        for p in pres:
            if p in st.pending_codes and p not in st.visited_codes:
                st.pending_codes = [c for c in st.pending_codes if c != p] + \
                                   ([st.active_code] if st.active_code else [])
                # Both codes come from golden/ -- the precondition target from
                # the manual's own step text -- so their digits are grounded.
                st = self.emit(st, {"kind": "precondition", "solve_first": p,
                                    "then": st.active_code}, "preflight",
                               grounded_text=[p, st.active_code or ""])
                st.active_code = p
                st.step_cursor = 0
                break
        return st

    def n_resolve_pointer(self, st: SessionState) -> SessionState:
        """Nine codes are pure redirects. Seven cycles exist. Both handled."""
        st.note("resolve_pointer")
        code = st.active_code or ""
        rec = tools.lookup_code(code)
        if rec is None:
            st.escalation_note = {"reason": "unknown_code", "codes": [code]}
            return st

        if not rec.get("is_pointer_only"):
            st.visit(code)
            return st

        res = tools.resolve_redirect(code, visited=st.visited_codes,
                                     max_depth=MAX_DEPTH)
        st.redirect_chain = st.redirect_chain + res["chain"]
        st.depth = st.depth + res["depth"]
        for c in res["chain"]:
            st.visit(c)
        self.log.event("decision", "redirect",
                       {"from": code, "chain": res["chain"],
                        "reason": res["reason"]}, level="detail")
        if res["target"] is None:
            st.escalation_note = {"reason": res["reason"], "chain": res["chain"],
                                  "detail": "redirect did not terminate on a "
                                            "code with a procedure"}
            return st
        if res["target"] != code:
            st = self.emit(st, {"kind": "redirect", "from_code": code,
                                "to_code": res["target"]}, "resolve_pointer",
                           grounded_text=[code, res["target"]])
            st.active_code = res["target"]
            st.step_cursor = 0
        return st

    def n_execute_step(self, st: SessionState) -> SessionState:
        """ONE step. Never the tree."""
        st.note("execute_step")
        n = st.step_cursor + 1
        step = tools.get_step(st.active_code or "", n)
        if step is None:
            st.escalation_note = st.escalation_note or {
                "reason": "steps_exhausted", "code": st.active_code,
                "steps_run": st.step_cursor}
            return st

        st.step_cursor = n
        ms = step.get("measurements") or []
        proc = (step.get("procedure") or "")[:400]
        payload = {
            "kind": "step",
            "step": step["step"],
            "cause": step.get("cause") or "",
            "procedure": proc,
            "safety": tools.safety_precondition(step),
            # A step with a measurement wants a number. Everything else -- an
            # explicit YES/NO branch, or a sequential check with neither -- wants
            # a normal/not-normal confirmation.
            "expects": "value" if ms else "yes_no",
        }
        fact_ids = [step.get("fact_id")]
        grounded = [step.get("cause") or "", proc]
        if ms:
            payload["measurement"] = {k: ms[0][k] for k in
                                      ("quantity", "point", "criteria")}
            fact_ids.append(ms[0].get("fact_id"))
            grounded += [ms[0].get("point") or "", ms[0].get("criteria") or ""]
        st.awaiting = Awaiting.READING
        return self.emit(st, payload, "execute_step", fact_ids, grounded)

    def n_parse_reading(self, st: SessionState) -> SessionState:
        """Technician reply -> structured. Vague is rejected, never guessed."""
        st.note("parse_reading")
        step = tools.get_step(st.active_code or "", st.step_cursor)
        ms = (step or {}).get("measurements") or []
        expecting = "value" if ms else "yes_no"

        parsed = self.llm.parse_reading(st.inbox or "", expecting)
        rd = Reading(step=st.step_cursor, raw=st.inbox or "", kind=parsed.kind,
                     value=parsed.value, unit=parsed.unit)

        if parsed.kind == "vague":
            rd.verdict = Verdict.VAGUE
            st.readings = st.readings + [rd]
            st.reask_count = st.reask_count + 1
            if st.reask_count > MAX_REASKS:
                st.escalation_note = {"reason": "no_usable_reading",
                                      "step": st.step_cursor,
                                      "attempts": st.reask_count}
                return st
            st.awaiting = Awaiting.READING
            return self.emit(st, {"kind": "ask_exact"}, "parse_reading")

        st.reask_count = 0
        if parsed.kind == "yes_no":
            rd.verdict = Verdict.YES if parsed.yes_no == "YES" else Verdict.NO
        else:
            unit_norm = tools.parse_quantity(f"{parsed.value}{parsed.unit or ''}")
            verdict = tools.check_reading(
                ms[0]["criteria"] if ms else "",
                unit_norm[0] if unit_norm else None,
                unit_norm[1] if unit_norm else None)
            rd.verdict = Verdict(verdict)
            if verdict == "OUT_OF_RANGE":
                # Not vagueness -- the technician gave a precise number of the
                # wrong quantity. Saying "be more exact" would be both wrong and
                # insulting; name what the step actually measures.
                st.readings = st.readings + [rd]
                st.reask_count = st.reask_count + 1
                if st.reask_count > MAX_REASKS:
                    st.escalation_note = {"reason": "uncomparable_reading",
                                          "step": st.step_cursor}
                    return st
                st.awaiting = Awaiting.READING
                return self.emit(
                    st, {"kind": "ask_unit",
                         "quantity": ms[0].get("quantity") if ms else "",
                         "point": ms[0].get("point") if ms else ""},
                    "parse_reading", grounded_text=[ms[0].get("point", "")] if ms else [])
        st.readings = st.readings + [rd]
        st.awaiting = Awaiting.NOTHING
        return st

    def n_evaluate_step(self, st: SessionState) -> SessionState:
        """PURE CODE. The comparison already happened in tools; this routes."""
        st.note("evaluate_step")
        step = tools.get_step(st.active_code or "", st.step_cursor)
        if step is None:
            st.escalation_note = {"reason": "steps_exhausted"}
            return st
        rd = st.readings[-1]
        branches = step.get("branches") or {}

        taken = None
        if rd.verdict in (Verdict.YES, Verdict.PASS):
            taken = "YES"
        elif rd.verdict in (Verdict.NO, Verdict.FAIL):
            taken = "NO"

        self.log.event("decision", "evaluate",
                       {"code": st.active_code, "step": st.step_cursor,
                        "verdict": rd.verdict.value, "branch": taken},
                       level="detail")

        if taken and taken in branches:
            outcome = branches[taken]
            disp = tools.branch_disposition(outcome)
            bfid = (step.get("branch_fact_ids") or {}).get(taken)
            if disp == "CONCLUDE":
                st.diagnosis = outcome
                st.use_facts([bfid])
                return st
            st.use_facts([bfid])
            return st

        # Sequential (format B): a check that comes back not-normal at this step
        # IS the diagnosis. The manual's causes are ordered, so the first one
        # that fails is the answer -- there is nothing further to deduce.
        if rd.verdict in (Verdict.FAIL, Verdict.NO):
            st.diagnosis = step.get("cause") or step.get("procedure")
            st.use_facts([step.get("fact_id")])
            return st
        return st

    def n_conclude(self, st: SessionState) -> SessionState:
        st.note("conclude")
        st.outcome = Outcome.CONCLUDED
        st.awaiting = Awaiting.NOTHING
        action = ""
        if st.pending_codes:
            action = f"Next, work {st.pending_codes[0]}."
        # The diagnosis text and any hand-off code both come from golden/.
        grounded = [st.diagnosis or ""] + list(st.pending_codes)
        return self.emit(st, {"kind": "conclude", "diagnosis": st.diagnosis or "",
                              "action": action}, "conclude",
                         fact_ids=list(st.fact_ids_used)[-4:],
                         grounded_text=grounded)

    def n_escalate(self, st: SessionState) -> SessionState:
        st.note("escalate")
        st.outcome = Outcome.ESCALATED
        st.awaiting = Awaiting.NOTHING
        st.escalation_note = dict(st.escalation_note or {"reason": "exhausted"},
                                  code=st.active_code,
                                  model=st.model, serial=st.serial,
                                  steps_run=st.step_cursor,
                                  readings=[r.model_dump(mode="json")
                                            for r in st.readings],
                                  visited=list(st.visited_codes),
                                  fact_ids=list(st.fact_ids_used))
        return self.emit(st, {"kind": "escalate"}, "escalate")

    def n_pause(self, st: SessionState) -> SessionState:
        return st

    # ------------------------------------------------------------ edges
    @staticmethod
    def _names_other_code(st: SessionState) -> bool:
        """Technician named a different real code than the one in progress.

        Happens: the monitor scrolls, or they misread it the first time. The
        session must follow them rather than keep walking a tree for a fault
        that is not the one being reported.
        """
        import re as _re
        for tok in _re.findall(r"\b([A-Z0-9@#]{4,7})\b", st.inbox or ""):
            if tools.code_exists(tok) and tok != st.active_code:
                return True
        return False

    @staticmethod
    def e_receive(st: SessionState) -> str:
        if st.awaiting == Awaiting.MACHINE:
            return "identify_machine"
        if st.awaiting == Awaiting.OTHER_CODES:
            return "preflight"
        if st.awaiting == Awaiting.READING:
            return "intake" if Agent._names_other_code(st) else "parse_reading"
        return "intake"

    @staticmethod
    def e_after_machine(st: SessionState) -> str:
        return "pause" if st.awaiting == Awaiting.MACHINE else "resolve_entry"

    @staticmethod
    def e_after_entry(st: SessionState) -> str:
        if st.escalation_note:
            return "escalate"
        return "pause" if st.awaiting == Awaiting.ENTRY else "preflight"

    @staticmethod
    def e_after_preflight(st: SessionState) -> str:
        return "pause" if st.awaiting == Awaiting.OTHER_CODES else "resolve_pointer"

    @staticmethod
    def e_after_pointer(st: SessionState) -> str:
        return "escalate" if st.escalation_note else "execute_step"

    @staticmethod
    def e_after_execute(st: SessionState) -> str:
        return "escalate" if st.escalation_note else "pause"

    @staticmethod
    def e_after_parse(st: SessionState) -> str:
        if st.escalation_note:
            return "escalate"
        return "pause" if st.awaiting == Awaiting.READING else "evaluate_step"

    @staticmethod
    def e_after_evaluate(st: SessionState) -> str:
        if st.diagnosis:
            return "conclude"
        if st.escalation_note:
            return "escalate"
        if st.step_cursor >= tools.step_count(st.active_code or ""):
            return "escalate"
        return "execute_step"

    # ------------------------------------------------------------ build
    def _build(self):
        g = StateGraph(SessionState)
        for name in ("receive", "intake", "identify_machine", "resolve_entry",
                     "preflight", "resolve_pointer", "execute_step",
                     "parse_reading", "evaluate_step", "conclude", "escalate",
                     "pause"):
            g.add_node(name, getattr(self, f"n_{name}"))

        g.set_entry_point("receive")
        g.add_conditional_edges("receive", self.e_receive,
                                {"intake": "intake",
                                 "identify_machine": "identify_machine",
                                 "preflight": "preflight",
                                 "parse_reading": "parse_reading"})
        g.add_edge("intake", "identify_machine")
        g.add_conditional_edges("identify_machine", self.e_after_machine,
                                {"pause": "pause", "resolve_entry": "resolve_entry"})
        g.add_conditional_edges("resolve_entry", self.e_after_entry,
                                {"pause": "pause", "preflight": "preflight",
                                 "escalate": "escalate"})
        g.add_conditional_edges("preflight", self.e_after_preflight,
                                {"pause": "pause",
                                 "resolve_pointer": "resolve_pointer"})
        g.add_conditional_edges("resolve_pointer", self.e_after_pointer,
                                {"execute_step": "execute_step",
                                 "escalate": "escalate"})
        g.add_conditional_edges("execute_step", self.e_after_execute,
                                {"pause": "pause", "escalate": "escalate"})
        g.add_conditional_edges("parse_reading", self.e_after_parse,
                                {"pause": "pause", "escalate": "escalate",
                                 "evaluate_step": "evaluate_step"})
        g.add_conditional_edges("evaluate_step", self.e_after_evaluate,
                                {"execute_step": "execute_step",
                                 "conclude": "conclude",
                                 "escalate": "escalate"})
        for terminal in ("conclude", "escalate", "pause"):
            g.add_edge(terminal, END)
        return g.compile()

    # ------------------------------------------------------------- drive
    def turn(self, st: SessionState, message: str) -> Tuple[SessionState, str]:
        """One technician message in, one assistant message out.

        Transport-agnostic by design: FastAPI, a CLI and the test harness all
        drive the graph through this one function.
        """
        before = len(st.emissions)
        nodes_before = len(st.node_sequence)
        st.inbox = message
        st.trace_id = new_trace_id()
        self.log.trace(st.trace_id)

        t0 = time.perf_counter()
        out = self.graph.invoke(st, {"recursion_limit": 60})
        st = SessionState.model_validate(out)
        dt = (time.perf_counter() - t0) * 1000.0
        st.inbox = None
        new = st.emissions[before:]

        # One event per turn carrying the whole path and its effect. Enough to
        # reconstruct the answer without re-running it: which nodes ran, what
        # the technician said, which facts the reply rests on, and whether
        # anything was blocked on the way out.
        self.log.event(
            "decision", "turn",
            {"session_id": st.session_id,
             "technician": message,
             "nodes": st.node_sequence[nodes_before:],
             "active_code": st.active_code,
             "step_cursor": st.step_cursor,
             "awaiting": st.awaiting.value,
             "outcome": st.outcome.value,
             "fact_ids": [f for e in new for f in e.fact_ids],
             "citations": [c.get("manual_page") for e in new
                           for c in e.citations],
             "blocked": [e.node for e in new if e.blocked],
             "emitted_chars": sum(len(e.text) for e in new)},
            duration_ms=dt, level="detail")
        return st, "\n".join(e.text for e in new)

    def start(self, session_id: str) -> SessionState:
        return SessionState(session_id=session_id)


def draw() -> str:
    """ASCII rendering of the graph, for the README and for review."""
    return """
  START
    |
  receive ──┬─ awaiting=machine ─────────> identify_machine
            ├─ awaiting=other_codes ─────> preflight
            ├─ awaiting=reading ─────────> parse_reading
            └─ otherwise ────────────────> intake
                                             |
  intake ────────────────────────────────> identify_machine
                                             |
  identify_machine ─┬─ no model/serial ──> pause (ask)   [HARD GATE]
                    └─ identified ───────> resolve_entry
                                             |
  resolve_entry ─┬─ unknown code ────────> escalate
                 ├─ symptom, no tree ────> escalate
                 ├─ need entry ──────────> pause (ask)
                 └─ code known ──────────> preflight
                                             |
  preflight ─┬─ not asked yet ───────────> pause (ask other codes)
             └─ precondition? jump ──────> resolve_pointer
                                             |
  resolve_pointer ─┬─ cycle/dead end ────> escalate
                   └─ target ────────────> execute_step
                                             |
  execute_step ──────────────────────────> pause (await reading)   ONE step
                                             |
  parse_reading ─┬─ vague / uncomparable > pause (ask exact)
                 ├─ too many re-asks ────> escalate
                 └─ parsed ──────────────> evaluate_step
                                             |
  evaluate_step ─┬─ branch concludes ────> conclude
                 ├─ steps exhausted ─────> escalate
                 └─ advance ─────────────> execute_step
                                             |
  conclude / escalate / pause ───────────> END

  LLM touches exactly: intake, parse_reading, and phrasing inside emit().
  Step selection, value comparison and diagnosis are code.
"""


if __name__ == "__main__":
    print(draw())
