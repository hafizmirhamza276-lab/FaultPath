#!/usr/bin/env python3
"""
guards.py
The last line between a model and a wrong number.

The design intent is that the model is structurally unable to emit a value: it
names fact ids, code renders the citation. This guard enforces that intent
rather than trusting it. Every outgoing message is scanned, and any number that
is not traceable to a cited fact blocks the message.

Blocking, not warning. A message that has already gone to a technician cannot be
unsent, and "we logged a warning" is not a control.

Allowed sources of a number, all of them things the model did not invent:
  - the verbatim text of a cited fact
  - the fact id itself (step indices)
  - the active failure code (CA451 contains 451)
  - the model designation and serial the technician supplied
  - what the technician typed this turn
  - small structural integers used for sequencing ("Step 3")
"""
from __future__ import annotations

import re
from typing import Iterable, List, Optional, Tuple

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _numbers(text: str) -> List[str]:
    return NUMBER_RE.findall(str(text or ""))


def allowed_numbers(citations: Iterable[dict],
                    fact_ids: Iterable[str] = (),
                    code: Optional[str] = None,
                    model: Optional[str] = None,
                    serial: Optional[str] = None,
                    technician_text: str = "",
                    extra: Iterable[str] = (),
                    grounded_text: Iterable[str] = ()) -> set:
    ok = set()
    for c in citations or []:
        ok |= set(_numbers(c.get("verbatim_text")))
        ok |= set(_numbers(c.get("manual_page")))
        ok |= set(_numbers(str(c.get("pdf_page") or "")))
    # Strings the message was BUILT from, straight out of golden/. A step's
    # procedure text is numbered ("1. Turn the starting switch...") and those
    # digits are the manual's, not the model's. Passing them explicitly keeps
    # the guard aimed at invention rather than at quotation -- and it is not
    # circular, because the caller may only put ground-truth strings here.
    for s in grounded_text or []:
        ok |= set(_numbers(s))
    for f in fact_ids or []:
        ok |= set(_numbers(f))
    for s in (code, model, serial, technician_text):
        ok |= set(_numbers(s))
    ok |= {str(x) for x in extra or []}
    return ok


def check_message(text: str,
                  citations: Iterable[dict],
                  fact_ids: Iterable[str] = (),
                  code: Optional[str] = None,
                  model: Optional[str] = None,
                  serial: Optional[str] = None,
                  technician_text: str = "",
                  extra: Iterable[str] = (),
                  grounded_text: Iterable[str] = ()) -> Tuple[bool, List[str]]:
    """(ok, offending_numbers). ok is False when the message must not be sent."""
    ok_set = allowed_numbers(citations, fact_ids, code, model, serial,
                             technician_text, extra, grounded_text)
    bad = [n for n in _numbers(text) if n not in ok_set]
    return (not bad), bad


BLOCKED_TEMPLATE = (
    "I cannot give you that value from what I have verified. "
    "Let me re-check the manual entry before I answer."
)


def enforce(text: str, **kw) -> Tuple[str, bool, List[str]]:
    """Return (text_to_send, blocked, offending).

    On a block the technician gets an honest refusal rather than a silently
    truncated answer. A half-message about a measurement is its own hazard.
    """
    ok, bad = check_message(text, **kw)
    if ok:
        return text, False, []
    return BLOCKED_TEMPLATE, True, bad
