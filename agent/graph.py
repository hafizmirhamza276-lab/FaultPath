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
                         Reading, Verdict, Emission, TreeKind,
                         MAX_DEPTH, MAX_REASKS)
from core.run_log import NullLogger, new_trace_id                  # noqa: E402


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

        # EVERY PIECE OF EVIDENCE IS BUILT ONCE AND BOTH USED AND RECORDED.
        #
        # The guard here and the guard at the API boundary must see the SAME
        # evidence, or the API blocks messages the graph allowed and the two
        # diverge -- which is the one thing the boundary must never do. It is
        # not enough to pass extra evidence to enforce(); it has to go into the
        # Emission too, because that is all the boundary gets. Guarding on more
        # than is recorded is the same defect as recording more than is
        # guarded, and the E2E transcript comparison catches it either way.
        #
        # A symptom id is to a symptom session what the failure code is to a
        # code session: it comes from golden/, the model never types it, and
        # "HM28" carries a 28 that a candidate list legitimately prints. The
        # candidates' manual pages come from each record's title_provenance --
        # ground truth, not a number the phrasing invented. The 1..n printed
        # beside them are sequencing integers, the same category as "Step 3",
        # and bounded by the list actually offered rather than being a licence
        # for arbitrary small numbers.
        evidence = [g for g in grounded_text if g]
        if st.active_symptom:
            evidence.append(st.active_symptom)
        for i, c in enumerate(st.symptom_candidates, start=1):
            evidence += [str(c.get("symptom_id") or ""),
                         str(c.get("manual_page") or ""), str(i)]
        evidence = [e for e in evidence if e]

        safe, blocked, bad = enforce(
            text, citations=citations, fact_ids=fact_ids,
            code=st.active_code, model=st.model, serial=st.serial,
            technician_text=st.inbox or "", extra=[e for e in extra if e],
            grounded_text=evidence)

        st.use_facts(fact_ids)
        st.emissions = st.emissions + [Emission(
            text=safe, original_text=text, fact_ids=fact_ids,
            citations=citations, node=node, blocked=blocked,
            grounded_text=evidence,
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

        # RESOLVING AN ASK COMES FIRST, and must return before the code hunt.
        # Candidate ids are code-SHAPED -- "HM28" is four upper-case chars with
        # digits -- so the unknown-code branch below would read a technician
        # answering our own question as a failure code this manual does not
        # have, and escalate on it.
        if st.awaiting == Awaiting.SYMPTOM_CHOICE and st.symptom_candidates:
            # A real failure code still wins. A technician who produces one
            # mid-question has handed us the better entry point: a code is a
            # dict lookup, a symptom is a match.
            parsed = self.llm.parse_intake(st.inbox or "")
            known = [c for c in parsed.codes if tools.code_exists(c)]
            if known:
                st.entry_mode = EntryMode.CODE
                st.enter_code(known[0])
                st.pending_codes = [c for c in known[1:] if c != st.active_code]
                st.awaiting = Awaiting.NOTHING
                self.log.event("decision", "symptom_to_code",
                               {"code": known[0],
                                "abandoned": st.entry_switch_log[-1:]},
                               level="detail")
                return st
            pick = tools.resolve_symptom_choice(st.inbox or "",
                                                st.symptom_candidates)
            if pick:
                st.enter_symptom(pick, tools.tree_kind(pick), "confirmed")
                st.awaiting = Awaiting.NOTHING
                self.log.event("decision", "symptom_confirmed",
                               {"symptom_id": pick}, level="detail")
                return st
            # Still ambiguous. Ask again rather than take the first candidate;
            # guessing here would undo the entire reason for having asked.
            st.awaiting = Awaiting.SYMPTOM_CHOICE
            return self.emit(st, {"kind": "ask_symptom_choice",
                                  "candidates": st.symptom_candidates,
                                  "repeat": True}, "intake")

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
            # enter_code restarts traversal on a change and abandons any
            # symptom tree in progress, recording the switch either way.
            st.enter_code(known[0])
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
        elif (st.inbox or "").strip():
            # Candidates were proposed, none is real, and none is code-shaped:
            # they are upper-case WORDS, not codes. The manual's own S-Mode
            # title is 'Engine does not crank when starting switch is turned to
            # "START" position.' -- START is five upper-case characters, so the
            # extractor offers it and the whole title stops being a symptom.
            #
            # Disposed of here rather than in llm.py on purpose: the model
            # proposes candidates, code decides what they are. Tightening the
            # model's regex instead would drop DAFQKR and B@BAZG, which carry
            # no digit either.
            st.entry_mode = EntryMode.SYMPTOM
            st.symptom_text = st.inbox
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
        # Clearing unconditionally would wipe an unanswered SYMPTOM_CHOICE on
        # the way past, and resolve_entry would then re-run the matcher and ask
        # the same question a second time in the same turn.
        if st.awaiting != Awaiting.SYMPTOM_CHOICE:
            st.awaiting = Awaiting.NOTHING
        return st

    def n_resolve_entry(self, st: SessionState) -> SessionState:
        """Settle on a tree: a failure code, a symptom tree, a question, or no.

        THREE OUTCOMES FROM THE MATCHER, and only one of them is an answer.
        MATCHED enters. ASK presents every candidate with its manual page and
        waits -- it does not pick. UNMAPPED says so and offers what it can.

        Neither ASK nor UNMAPPED is an error path. A technician who is asked
        loses ten seconds; one sent silently into the wrong tree loses an hour.
        """
        st.note("resolve_entry")
        if st.escalation_note and st.escalation_note.get("reason") == "unknown_code":
            return st
        if st.entry_resolved:
            return st
        # The question is already out and unanswered. Re-running the matcher
        # here would ask it twice in one turn -- and worse, a differently
        # worded reply could quietly change the candidate list the technician
        # is answering about.
        if st.awaiting == Awaiting.SYMPTOM_CHOICE:
            return st
        if st.entry_mode == EntryMode.SYMPTOM:
            res = tools.search_symptoms(st.symptom_text or "")
            self.log.event("decision", "symptom_match",
                           {"query": st.symptom_text, "layer": res["layer"],
                            "outcome": res["outcome"],
                            "candidates": res["symptom_ids"]}, level="detail")

            if res["outcome"] == "MATCHED":
                sid = res["symptom_ids"][0]
                st.enter_symptom(sid, tools.tree_kind(sid), res["layer"])
                return st

            if res["outcome"] == "ASK":
                # Every candidate, with its page. The technician decides.
                st.symptom_candidates = list(res["candidates"])
                st.awaiting = Awaiting.SYMPTOM_CHOICE
                return self.emit(
                    st, {"kind": "ask_symptom_choice",
                         "candidates": res["candidates"], "repeat": False},
                    "resolve_entry",
                    grounded_text=[c["symptom"] for c in res["candidates"]])

            # UNMAPPED. Say so, and offer the honest alternatives: a failure
            # code, or the section where the topic actually lives. NEVER the
            # nearest tree -- "oil leak ho raha hai" belongs to Testing and
            # Adjusting, and answering it out of H-Mode would be confident,
            # fluent and wrong.
            st.escalation_note = {
                "reason": "symptom_unmapped",
                "symptom": st.symptom_text,
                "match_note": res.get("note") or "",
                "detail": ("No H-Mode or S-Mode symptom tree in SEN06867-13 "
                           "covers this. Not routed to the nearest tree.")}
            return self.emit(
                st, {"kind": "symptom_unmapped", "symptom": st.symptom_text or "",
                     "note": res.get("note") or ""}, "resolve_entry")

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

            # A REAL CODE NAMED DURING A SYMPTOM SESSION TAKES OVER.
            #
            # This is the manual's own instruction, not a preference: every
            # H-Mode tree's related_information opens "Pre-troubleshooting: If
            # a failure code is shown, do the troubleshooting for that code
            # first." 35 of the 182 prose pointers ARE that sentence.
            #
            # In a code session the same reply means something different --
            # those are concurrent codes to work afterwards, and pending_codes
            # is right for them. Only the symptom session hands over.
            if st.active_symptom and extra:
                took = extra[0]
                abandoned = st.active_symptom
                st.entry_mode = EntryMode.CODE
                st.enter_code(took)
                st.pending_codes = [c for c in st.pending_codes if c != took]
                self.log.event("decision", "symptom_to_code",
                               {"from_symptom": abandoned, "to_code": took,
                                "why": "manual: do the failure code first"},
                               level="detail")
                st = self.emit(st, {"kind": "code_takes_over",
                                    "code": took}, "preflight",
                               grounded_text=[took])

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
        # A symptom tree has no pointer-only form: the 182 prose pointers are
        # inside steps, not whole records, and they are surfaced where they
        # occur rather than followed. Nothing to resolve here.
        if st.active_symptom and not st.active_code:
            return st
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
        """ONE step. Never the tree.

        TREE KIND DECIDES WHAT MAY BE ASKED. A branching tree stores a YES and
        a NO outcome per step, so asking yes/no reads the manual back. A flat
        S-Mode tree stores cause / point to check / remedy and no branch
        outcomes at all; asking "is it normal, yes or no?" there invents an
        interaction the manual does not have.
        """
        st.note("execute_step")
        n = st.step_cursor + 1

        if st.tree_kind == TreeKind.FLAT:
            return self._execute_flat_row(st, n)
        if st.active_symptom:
            step = tools.get_symptom_step(st.active_symptom, n)
        else:
            step = tools.get_step(st.active_code or "", n)
        if step is None:
            st.escalation_note = st.escalation_note or {
                "reason": "steps_exhausted",
                "code": st.active_code or st.active_symptom,
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
        # A detail pointer is supplementary: the step is still executable, and
        # the manual is saying where the long form lives. Surfaced, not
        # followed, and never used to fabricate the detail it points at.
        if st.active_symptom:
            dp = tools.step_detail_pointer(st.active_symptom, step)
            if dp:
                payload["pointer"] = dp["text"]
                payload["pointer_page"] = (dp.get("provenance") or {}).get(
                    "manual_page")
                # The page too, not just the text. It is the extractor's own
                # provenance for this pointer, so it is ground truth -- but it
                # has to be SAID to be ground truth or the guard blocks the
                # message, which is the guard working, not the guard being
                # wrong.
                grounded += [dp["text"], str(payload["pointer_page"] or "")]

        st.awaiting = Awaiting.READING
        return self.emit(st, payload, "execute_step", fact_ids, grounded)

    def _execute_flat_row(self, st: SessionState, n: int) -> SessionState:
        """One row of a flat S-Mode tree: cause, point to check, and a wait.

        The REMEDY IS WITHHELD until the point to check is confirmed. The
        manual prints all three columns side by side, but a guided session that
        shows the remedy alongside the question is not guiding -- it is letting
        the technician read the answer off the back of the card.
        """
        step = tools.get_symptom_step(st.active_symptom or "", n)
        if step is None:
            st.escalation_note = st.escalation_note or {
                "reason": "rows_exhausted", "symptom": st.active_symptom,
                "rows_checked": st.step_cursor,
                "detail": ("Every cause the manual lists for this symptom was "
                           "checked and none matched.")}
            return st
        st.step_cursor = n
        payload = {
            "kind": "flat_row",
            "step": step["step"],
            "cause": step.get("cause") or "",
            "point_to_check": (step.get("point_to_check") or "")[:400],
            # NOT "yes_no". The reply confirms an observation, and the routing
            # it drives is the opposite of a branching step's.
            "expects": "observation",
        }
        fact_ids = [step.get("fact_id")]
        grounded = [payload["cause"], payload["point_to_check"]]
        st.awaiting = Awaiting.OBSERVATION
        return self.emit(st, payload, "execute_step", fact_ids, grounded)

    def n_parse_reading(self, st: SessionState) -> SessionState:
        """Technician reply -> structured. Vague is rejected, never guessed."""
        st.note("parse_reading")

        if st.tree_kind == TreeKind.FLAT:
            return self._parse_observation(st)

        if st.active_symptom:
            step = tools.get_symptom_step(st.active_symptom, st.step_cursor)
        else:
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

    def _parse_observation(self, st: SessionState) -> SessionState:
        """Flat row: does the technician see what this row describes.

        The LLM's contract is unchanged -- it still returns a yes_no parse,
        because "does this describe what you see" is answered the same way in
        Roman Urdu as "is it normal". What changes is the VERDICT recorded, and
        it is deliberately a different pair. Reusing YES/NO here would leave
        n_evaluate_step unable to tell an advance from a stop, and the two mean
        opposite things on the two tree kinds.
        """
        parsed = self.llm.parse_reading(st.inbox or "", "yes_no")
        rd = Reading(step=st.step_cursor, raw=st.inbox or "", kind=parsed.kind,
                     value=parsed.value, unit=parsed.unit)

        if parsed.kind != "yes_no":
            rd.verdict = Verdict.VAGUE
            st.readings = st.readings + [rd]
            st.reask_count = st.reask_count + 1
            if st.reask_count > MAX_REASKS:
                st.escalation_note = {"reason": "no_usable_observation",
                                      "step": st.step_cursor,
                                      "attempts": st.reask_count}
                return st
            st.awaiting = Awaiting.OBSERVATION
            return self.emit(st, {"kind": "ask_observation_again"},
                             "parse_reading")

        st.reask_count = 0
        rd.verdict = (Verdict.OBSERVED if parsed.yes_no == "YES"
                      else Verdict.NOT_OBSERVED)
        st.readings = st.readings + [rd]
        st.awaiting = Awaiting.NOTHING
        return st

    def n_evaluate_step(self, st: SessionState) -> SessionState:
        """PURE CODE. The comparison already happened in tools; this routes."""
        st.note("evaluate_step")

        if st.tree_kind == TreeKind.FLAT:
            return self._evaluate_flat_row(st)

        if st.active_symptom:
            step = tools.get_symptom_step(st.active_symptom, st.step_cursor)
        else:
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

    def _evaluate_flat_row(self, st: SessionState) -> SessionState:
        """Flat tree routing, WITH THE POLARITY THE MANUAL ACTUALLY HAS.

            branching step   YES = the check was normal  -> ADVANCE
            flat row         OBSERVED = you see the fault -> STOP, remedy

        These are opposite, and the reason the verdict pairs are separate. A
        flat tree walked with branching polarity runs every row, matches
        nothing, and hands over -- plausible output, no error, wrong answer.
        """
        step = tools.get_symptom_step(st.active_symptom or "", st.step_cursor)
        if step is None:
            st.escalation_note = {"reason": "rows_exhausted"}
            return st
        rd = st.readings[-1]

        self.log.event("decision", "evaluate_flat_row",
                       {"symptom": st.active_symptom, "row": st.step_cursor,
                        "verdict": rd.verdict.value}, level="detail")

        if rd.verdict != Verdict.OBSERVED:
            return st                      # not this cause; try the next row

        # This row is the answer. Cause and remedy are separate facts with
        # separate ids, because the manual gives them separate cells.
        st.diagnosis = step.get("cause") or ""
        st.remedy = step.get("remedy") or ""
        st.remedy_fact_id = step.get("remedy_fact_id")
        st.use_facts([step.get("fact_id"), st.remedy_fact_id])

        # The remedy may itself be one of the 182 prose pointers -- the manual
        # answering "go and look over there". Recorded as a pointer rather than
        # delivered as an instruction, and never followed.
        ptr = tools.step_prose_pointer(st.active_symptom or "", step)
        if ptr:
            st.prose_pointer = {
                "text": ptr["text"],
                "manual_page": (ptr.get("provenance") or {}).get("manual_page"),
                "pdf_page": (ptr.get("provenance") or {}).get("pdf_page"),
                "resolution": ptr.get("resolution") or "",
            }
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
        payload = {"kind": "conclude", "diagnosis": st.diagnosis or "",
                   "action": action}
        fact_ids = list(st.fact_ids_used)[-4:]

        # REMEDY IS A FIRST-CLASS FIELD, not a sentence appended to the
        # diagnosis. The manual gives it its own column and its own cell, so it
        # gets its own payload key, its own fact id and its own citation.
        if st.remedy:
            payload["remedy"] = st.remedy
            grounded.append(st.remedy)
            if st.remedy_fact_id and st.remedy_fact_id not in fact_ids:
                fact_ids.append(st.remedy_fact_id)
        if st.prose_pointer:
            # The manual refers you elsewhere. Say where, say that we do not
            # hold it, and stop. Following it would mean guessing a target.
            payload["pointer"] = st.prose_pointer["text"]
            payload["pointer_page"] = st.prose_pointer.get("manual_page")
            grounded += [st.prose_pointer["text"],
                         str(payload["pointer_page"] or "")]
        return self.emit(st, payload, "conclude", fact_ids=fact_ids,
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
        return self.emit(st, {"kind": "escalate",
                              "reason": st.escalation_note.get("reason") or "",
                              "symptom": st.active_symptom or ""}, "escalate")

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
        # An answer to our own ASK goes to intake, which resolves the pick (or
        # takes a real failure code instead) and falls through to
        # resolve_entry. No new edge: the choice is technician text becoming
        # entry state, which is exactly what intake is for.
        if st.awaiting == Awaiting.SYMPTOM_CHOICE:
            return "intake"
        if st.awaiting in (Awaiting.READING, Awaiting.OBSERVATION):
            return "intake" if Agent._names_other_code(st) else "parse_reading"
        return "intake"

    @staticmethod
    def e_after_machine(st: SessionState) -> str:
        return "pause" if st.awaiting == Awaiting.MACHINE else "resolve_entry"

    @staticmethod
    def e_after_entry(st: SessionState) -> str:
        if st.escalation_note:
            return "escalate"
        # ASK is a pause, not a failure: the question has been sent and the
        # session is waiting on a person, exactly like the machine gate.
        if st.awaiting in (Awaiting.ENTRY, Awaiting.SYMPTOM_CHOICE):
            return "pause"
        return "preflight"

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
        if st.awaiting in (Awaiting.READING, Awaiting.OBSERVATION):
            return "pause"
        return "evaluate_step"

    @staticmethod
    def e_after_evaluate(st: SessionState) -> str:
        if st.diagnosis:
            return "conclude"
        if st.escalation_note:
            return "escalate"
        total = (tools.symptom_step_count(st.active_symptom)
                 if st.active_symptom else tools.step_count(st.active_code or ""))
        if st.step_cursor >= total:
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
