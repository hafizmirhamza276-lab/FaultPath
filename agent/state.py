#!/usr/bin/env python3
"""
state.py
Session state, validated on every transition.

The state is the whole agent. Every decision the graph makes is a pure function
of this object plus the incoming message, which is what makes the replay test
possible: feed a recorded transcript, get an identical state sequence.

Nothing here is optional bookkeeping. visited_codes exists because seven
cross-reference cycles are documented in the manual and hop-expansion without a
visited set loops until it runs out of tokens. fact_ids_used exists because the
outgoing-message guard needs to know what the answer is allowed to say.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import List, Optional, Dict, Any

from pydantic import BaseModel, Field, field_validator


class EntryMode(str, Enum):
    CODE = "code"            # technician named a failure code -- dict lookup
    SYMPTOM = "symptom"      # no code; H-Mode / S-Mode tree via core.symptom_match
    UNKNOWN = "unknown"      # neither; must ask


class TreeKind(str, Enum):
    """What shape of tree is being executed, and therefore what may be asked.

    The values are golden/'s own tree_kind strings, not a re-encoding of them:
    a parallel vocabulary is a second source of truth waiting to drift.

    BRANCHING trees store a YES and a NO outcome per step, so asking yes/no is
    reading the manual back. FLAT trees store cause / point to check / remedy
    and NO branch outcomes at all -- asking "is it normal, yes or no?" there
    invents an interaction the manual does not have, and inverts the polarity
    while doing it: on a branching step YES means normal and advances, on a
    flat row confirming the point to check means the fault is FOUND.
    """
    BRANCHING = "SymptomTreeBranching"
    FLAT = "SymptomTreeFlat"


class Awaiting(str, Enum):
    NOTHING = "nothing"
    MACHINE = "machine"          # model + serial
    ENTRY = "entry"              # a code or a symptom
    SYMPTOM_CHOICE = "symptom_choice"   # matcher asked; technician picks
    OTHER_CODES = "other_codes"  # preflight
    READING = "reading"          # measured value or YES/NO on a branching step
    OBSERVATION = "observation"  # flat row: does this describe what you see


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    YES = "YES"
    NO = "NO"
    # Flat-tree row confirmation. Deliberately NOT reusing YES/NO: their
    # polarity is the opposite one. On a branching step YES means the check
    # was normal and the tree advances; on a flat row OBSERVED means the
    # technician is looking at the fault and the tree stops. Collapsing them
    # into one pair is how a flat tree would end up walked backwards.
    OBSERVED = "OBSERVED"
    NOT_OBSERVED = "NOT_OBSERVED"
    VAGUE = "VAGUE"              # not a reading; must ask again
    UNPARSED = "UNPARSED"


class Outcome(str, Enum):
    RUNNING = "running"
    CONCLUDED = "concluded"
    ESCALATED = "escalated"


class Reading(BaseModel):
    """One technician reply about one step. Raw text is kept deliberately.

    When a value is disputed six months later the question is what the
    technician actually typed, not what the parser made of it.
    """
    step: int
    raw: str
    kind: str = "none"                  # value | yes_no | vague | none
    value: Optional[float] = None
    unit: Optional[str] = None
    verdict: Verdict = Verdict.UNPARSED
    ts: float = Field(default_factory=time.time)


class Emission(BaseModel):
    """One outgoing message, with the facts it is allowed to rest on.

    original_text is what the agent tried to say before the guard ran. Keeping
    it matters for scoring: a gate that only inspects the delivered text would
    be defeated by the guard itself -- a blocked message carries no numbers, so
    "did it try to emit a value too early" becomes unanswerable. Behaviour is
    judged on the attempt; safety is judged on what was delivered.
    """
    text: str
    original_text: str = ""
    fact_ids: List[str] = Field(default_factory=list)
    citations: List[Dict[str, Any]] = Field(default_factory=list)
    # The ground-truth strings this message was composed from. Carried so the
    # process boundary can re-run the guard on the SAME evidence the graph used
    # -- an independent check, not a stricter one. A boundary guard holding less
    # evidence would block correctly-grounded text and make the API behave
    # differently from the graph, which is the one thing it must never do.
    grounded_text: List[str] = Field(default_factory=list)
    node: str = ""
    blocked: bool = False
    block_reason: str = ""


class SessionState(BaseModel):
    """Validated on every transition. Extra keys are a bug, not a feature."""
    model_config = {"extra": "forbid", "validate_assignment": True}

    session_id: str
    manual_id: str = "SEN06867-13"

    # --- machine identity. The hard gate.
    model: Optional[str] = None
    serial: Optional[str] = None

    # --- entry
    entry_mode: EntryMode = EntryMode.UNKNOWN
    active_code: Optional[str] = None
    symptom_text: Optional[str] = None

    # --- symptom entry. active_symptom is to a symptom session what
    # active_code is to a code session; both are never set at once, and
    # entry_switch_log records it if the technician moves between them.
    active_symptom: Optional[str] = None
    tree_kind: Optional[TreeKind] = None
    # What the matcher offered when it would not pick. Kept on the state so the
    # choice can be resolved on the NEXT turn without re-running retrieval --
    # re-running it would let a differently-worded reply silently change the
    # candidate list the technician is answering about.
    symptom_candidates: List[Dict[str, Any]] = Field(default_factory=list)
    symptom_match_layer: Optional[str] = None
    entry_switch_log: List[str] = Field(default_factory=list)

    # --- flat-tree outcome. A remedy is not a diagnosis and is not procedure
    # prose: the manual gives it its own column and its own cell, so it gets
    # its own field, its own fact id and its own citation.
    remedy: Optional[str] = None
    remedy_fact_id: Optional[str] = None
    # The manual sends the technician somewhere this repo does not hold.
    # Surfaced, never followed.
    prose_pointer: Optional[Dict[str, Any]] = None

    # --- traversal
    visited_codes: List[str] = Field(default_factory=list)
    # Every visit in order, repeats included. visited_codes is the protection
    # set and is deduped by construction, so it can never show a loop -- this is
    # the evidence cycle_safety is scored on.
    code_visit_log: List[str] = Field(default_factory=list)
    step_cursor: int = 0
    depth: int = 0
    pending_codes: List[str] = Field(default_factory=list)
    redirect_chain: List[str] = Field(default_factory=list)

    # --- record
    readings: List[Reading] = Field(default_factory=list)
    fact_ids_used: List[str] = Field(default_factory=list)
    emissions: List[Emission] = Field(default_factory=list)
    node_sequence: List[str] = Field(default_factory=list)

    # --- control
    awaiting: Awaiting = Awaiting.NOTHING
    inbox: Optional[str] = None
    outcome: Outcome = Outcome.RUNNING
    diagnosis: Optional[str] = None
    escalation_note: Optional[Dict[str, Any]] = None
    preflight_done: bool = False
    reask_count: int = 0
    trace_id: Optional[str] = None

    @field_validator("visited_codes", "pending_codes")
    @classmethod
    def _no_dupes(cls, v):
        seen, out = set(), []
        for x in v:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # -- helpers ---------------------------------------------------------
    @property
    def machine_identified(self) -> bool:
        """The gate. Both halves, not either.

        PC200 and PC490 share failure codes but not pin numbers, and serial
        ranges split the manual's applicability. A value released without both
        is a guess wearing a number's clothes.
        """
        return bool(self.model) and bool(self.serial)

    @property
    def entry_resolved(self) -> bool:
        """A tree to execute has been settled on -- code or symptom."""
        return bool(self.active_code) or bool(self.active_symptom)

    def enter_symptom(self, symptom_id: str, tree_kind: str, layer: str) -> None:
        """Commit to a symptom tree. Clears any code traversal in progress.

        The two entry modes share a cursor, so entering a symptom without
        resetting it resumes the new tree at whatever step the old one reached.
        """
        if self.active_symptom and self.active_symptom != symptom_id:
            self.entry_switch_log = self.entry_switch_log + [
                f"{self.active_symptom}->{symptom_id}"]
        self.active_symptom = symptom_id
        self.tree_kind = TreeKind(tree_kind)
        self.symptom_match_layer = layer
        self.active_code = None
        self.step_cursor = 0
        self.diagnosis = None
        self.remedy = None
        self.remedy_fact_id = None
        self.prose_pointer = None
        self.symptom_candidates = []

    def enter_code(self, code: str) -> None:
        """Commit to a failure-code tree, abandoning any symptom in progress.

        A technician who names a real code mid-symptom has given us the better
        entry point: a code is a dict lookup and a symptom is a match.
        """
        if self.active_symptom:
            self.entry_switch_log = self.entry_switch_log + [
                f"{self.active_symptom}->{code}"]
            self.active_symptom = None
            self.tree_kind = None
            self.symptom_candidates = []
            self.remedy = None
            self.remedy_fact_id = None
            self.prose_pointer = None
        if self.active_code and self.active_code != code:
            self.entry_switch_log = self.entry_switch_log + [
                f"{self.active_code}->{code}"]
        if self.active_code != code:
            self.step_cursor = 0
            self.diagnosis = None
        self.active_code = code

    def visit(self, code: str) -> bool:
        """Record a code. False if already visited -- that is a cycle."""
        self.code_visit_log = self.code_visit_log + [code]
        if code in self.visited_codes:
            return False
        self.visited_codes = self.visited_codes + [code]
        return True

    def note(self, node: str) -> None:
        self.node_sequence = self.node_sequence + [node]

    def use_facts(self, fact_ids) -> None:
        merged = list(self.fact_ids_used)
        for f in fact_ids:
            if f and f not in merged:
                merged.append(f)
        self.fact_ids_used = merged


MAX_DEPTH = 8          # redirect hops before we stop and hand over
MAX_REASKS = 3         # vague answers before escalating rather than nagging
