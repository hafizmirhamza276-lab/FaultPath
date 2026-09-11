#!/usr/bin/env python3
"""
llm.py
The model's entire job, and its boundary.

It does exactly two things:
  1. understand what the technician typed (Roman Urdu / Hinglish / English)
  2. phrase a step that code has already chosen

It never selects a step, never compares a measurement, never decides a
diagnosis, never emits a number or a page. Those are in tools.py and graph.py,
where they can be read, tested and argued with.

Every call returns a schema-validated object. A parse failure asks the
technician again -- it never falls through to free text, because free text is
exactly where an unvalidated model answer would re-enter the system.

MockLLM ships so the graph and its tests run with no network and no key. It is
rule-based and deterministic: same input, same output, forever.
"""
from __future__ import annotations

import re
from typing import List, Optional

from pydantic import BaseModel, Field


# ------------------------------------------------------------- schemas

class IntakeParse(BaseModel):
    """What the technician's opening message contained. Extraction only."""
    model_config = {"extra": "forbid"}
    codes: List[str] = Field(default_factory=list)
    model: Optional[str] = None
    serial: Optional[str] = None
    symptom: Optional[str] = None


class ReadingParse(BaseModel):
    """One reply about one step.

    kind is the whole decision surface the model gets: is this a number, a
    yes/no, or too vague to use. It does not get to say whether the number
    passes -- that is arithmetic, done in tools.check_reading.
    """
    model_config = {"extra": "forbid"}
    kind: str = "vague"                 # value | yes_no | vague
    value: Optional[float] = None
    unit: Optional[str] = None
    yes_no: Optional[str] = None        # YES | NO


class Phrasing(BaseModel):
    model_config = {"extra": "forbid"}
    text: str


class LLM:
    """Provider seam. A real client implements these three and nothing else."""

    name = "abstract"

    def parse_intake(self, text: str) -> IntakeParse:
        raise NotImplementedError

    def parse_reading(self, text: str, expecting: str) -> ReadingParse:
        raise NotImplementedError

    def phrase(self, payload: dict) -> Phrasing:
        raise NotImplementedError


# --------------------------------------------------------------- mock

# Komatsu failure codes are 4-7 chars of [A-Z0-9@#] and do NOT reliably start
# with a letter: 602KNX, 879AKA, 6AZ0ZG, 989L00 are all real. An earlier
# leading-letter pattern silently dropped every numeric-prefixed code, and the
# session fell through to symptom entry instead of a dict lookup.
CODE_RE = re.compile(r"\b([A-Z0-9@#]{4,7})\b")
MODEL_RE = re.compile(r"\b(PC\d{2,3}[A-Z]{0,2}-?\d{0,2}[A-Z]?\d?)\b", re.I)
SERIAL_RE = re.compile(r"\b(?:s/?n|serial|seri[ae]l)\s*[:\-]?\s*(\d{4,8})\b", re.I)
BARE_SERIAL_RE = re.compile(r"\b(\d{6,8})\b")

# Roman Urdu / Hinglish hedges. A technician who says "thoda kam tha" has not
# taken a measurement, and treating it as one is how a wrong value enters the
# record wearing the authority of a reading.
VAGUE_RE = re.compile(
    r"\b(thoda|zyada|kam|kuch|shayad|lagta|lag raha|normal hi|theek hi|"
    r"approx|around|roughly|about|maybe|seems|looks fine|ok lag)\b", re.I)

YES_RE = re.compile(r"\b(yes|haan|han|ji|ok|okay|done|ho gaya|sahi|correct|"
                    r"theek hai|yep|y)\b", re.I)
NO_RE = re.compile(r"\b(no|nahi|nahin|nope|galat|n)\b", re.I)

VALUE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(k|M|m|µ|u)?\s*"
    r"(ohm|ohms|Ω|Ω|ω|V|volts?|A|amps?|mA|kPa|MPa|rpm|Hz|%|°?C)\b", re.I)
BARE_NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.])")


class MockLLM(LLM):
    """Deterministic, rule-based, offline.

    Not a stand-in for a real model's quality -- a stand-in for its INTERFACE.
    The graph must behave identically whichever is plugged in, and the tests
    must run with neither network nor key.
    """

    name = "mock"

    def parse_intake(self, text: str) -> IntakeParse:
        t = text or ""
        mm = MODEL_RE.search(t)
        model = mm.group(1).upper() if mm else None

        # Strip the model designation before hunting for codes -- PC200 is
        # code-shaped and is not a code.
        code_hay = MODEL_RE.sub(" ", t)
        codes = []
        for c in CODE_RE.findall(code_hay):
            cu = c.upper()
            if cu != c:                      # codes are typed in upper case
                continue
            if cu not in codes:
                codes.append(cu)
        # Candidates only. The graph decides which are real by looking them up
        # in golden/ -- the model proposes, code disposes. Filtering by shape
        # here would drop DAFQKR and B@BAZG, which carry no digit at all.

        sm = SERIAL_RE.search(t)
        serial = sm.group(1) if sm else None
        if serial is None and model:
            # A bare 6-8 digit number alongside a model is a serial in practice.
            for cand in BARE_SERIAL_RE.findall(t):
                if cand not in (model or ""):
                    serial = cand
                    break

        symptom = None
        if not codes:
            s = t.strip()
            symptom = s or None
        return IntakeParse(codes=codes, model=model, serial=serial, symptom=symptom)

    def parse_reading(self, text: str, expecting: str) -> ReadingParse:
        t = (text or "").strip()
        if not t:
            return ReadingParse(kind="vague")

        m = VALUE_RE.search(t)
        if m:
            # A number with a unit is a reading even if the sentence hedges
            # around it: "thoda kam, 0.8 ohm" is still 0.8 ohm.
            return ReadingParse(kind="value", value=float(m.group(1)),
                                unit=(m.group(2) or "") + m.group(3))

        if VAGUE_RE.search(t):
            return ReadingParse(kind="vague")

        if expecting == "yes_no":
            if NO_RE.search(t):
                return ReadingParse(kind="yes_no", yes_no="NO")
            if YES_RE.search(t):
                return ReadingParse(kind="yes_no", yes_no="YES")
            return ReadingParse(kind="vague")

        bare = BARE_NUM_RE.search(t)
        if bare:
            # A bare number with no unit cannot be compared against a criterion
            # that has one. Treated as vague so the graph asks for the unit
            # rather than assuming which quantity was meant.
            return ReadingParse(kind="vague")

        if NO_RE.search(t):
            return ReadingParse(kind="yes_no", yes_no="NO")
        if YES_RE.search(t):
            return ReadingParse(kind="yes_no", yes_no="YES")
        return ReadingParse(kind="vague")

    def phrase(self, payload: dict) -> Phrasing:
        """Compose the outgoing sentence from fields code already decided.

        Every substantive string here is copied out of the payload, which came
        from golden/. The model is arranging, not authoring.
        """
        kind = payload.get("kind")
        if kind == "ask_machine":
            return Phrasing(text=(
                "Before I give you any values: which machine model and serial "
                "number is this? Values differ between models."))
        if kind == "ask_entry":
            return Phrasing(text=(
                "Which failure code is showing on the monitor? If there is no "
                "code, describe what the machine is doing."))
        if kind == "ask_other_codes":
            return Phrasing(text=(
                "Before we start: are any other failure codes showing on the "
                "monitor right now?"))
        if kind == "ask_exact":
            return Phrasing(text=(
                "I need the exact reading, not an impression. What value did "
                "the meter actually show, with its unit?"))
        if kind == "ask_unit":
            return Phrasing(text=(
                f"That reading does not match what this check measures. "
                f"This step measures {payload.get('quantity', 'the value')} at "
                f"{payload.get('point', 'the given point')}. Please give me "
                f"that reading."))
        if kind == "precondition":
            return Phrasing(text=(
                f"The manual says {payload['solve_first']} must be solved "
                f"before {payload['then']}. We will start there and come back."))
        if kind == "redirect":
            return Phrasing(text=(
                f"{payload['from_code']} has no procedure of its own -- it "
                f"points to {payload['to_code']}. Continuing there."))
        if kind == "step":
            bits = []
            if payload.get("safety"):
                bits.append(f"Safety first: {payload['safety']}.")
            bits.append(f"Step {payload['step']}: {payload['cause']}.")
            if payload.get("procedure"):
                bits.append(payload["procedure"])
            if payload.get("measurement"):
                mm = payload["measurement"]
                bits.append(f"Measure {mm['quantity']} at {mm['point']}. "
                            f"Standard value: {mm['criteria']}.")
                bits.append("Tell me the exact reading.")
            else:
                bits.append("Is it normal? Answer yes or no.")
            return Phrasing(text=" ".join(bits))
        if kind == "conclude":
            return Phrasing(text=(
                f"Diagnosis: {payload['diagnosis']} "
                f"{payload.get('action', '')}").strip())
        if kind == "escalate":
            return Phrasing(text=(
                "I have run out of checks the manual defines for this code. "
                "Handing over with everything recorded so far."))
        return Phrasing(text=payload.get("text", ""))
