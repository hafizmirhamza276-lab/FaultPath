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
    SYMPTOM = "symptom"      # no code; H/S-mode retrieval (stubbed)
    UNKNOWN = "unknown"      # neither; must ask


class Awaiting(str, Enum):
    NOTHING = "nothing"
    MACHINE = "machine"          # model + serial
    ENTRY = "entry"              # a code or a symptom
    OTHER_CODES = "other_codes"  # preflight
    READING = "reading"          # result of the emitted step


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    OUT_OF_RANGE = "OUT_OF_RANGE"
    YES = "YES"
    NO = "NO"
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
