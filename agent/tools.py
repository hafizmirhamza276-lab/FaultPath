#!/usr/bin/env python3
"""
tools.py
Every tool the agent can call. All deterministic, none LLM-backed.

The diagnosis is already written -- Komatsu wrote it, 174 codes and 996 ordered
steps. These functions execute that tree. A named failure code is a dict lookup,
not a similarity search; routing it through retrieval would convert a certainty
into a probability.

check_reading is the one that matters most. Comparing a measured value against a
criterion is arithmetic, and arithmetic is not a job for a language model.
"""
from __future__ import annotations

import glob
import json
import os
import re
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))

EXPECTED_SCHEMA = 2

_RECORDS: Optional[Dict[str, dict]] = None


class SchemaMismatch(RuntimeError):
    """golden/ is not the shape this agent was written against."""


def _load() -> Dict[str, dict]:
    recs = {}
    for p in sorted(glob.glob(os.path.join(GOLD, "failure_codes", "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        if r.get("schema_version") != EXPECTED_SCHEMA:
            raise SchemaMismatch(
                f"{r.get('code')}: schema_version {r.get('schema_version')!r}, "
                f"expected {EXPECTED_SCHEMA}. Re-run pipeline/extract_golden.py. "
                "Refusing to execute a tree whose shape is not the one this "
                "agent's step and provenance handling was written for.")
        recs[r["code"]] = r
    if not recs:
        raise SchemaMismatch(f"no failure codes under {GOLD}/failure_codes/")
    return recs


def records() -> Dict[str, dict]:
    global _RECORDS
    if _RECORDS is None:
        _RECORDS = _load()
    return _RECORDS


# --------------------------------------------------------------- lookup

def lookup_code(code: str) -> Optional[dict]:
    """Exact lookup. No fuzzy matching -- a near-miss is a different machine
    fault, and silently answering about the wrong one is worse than saying no."""
    return records().get((code or "").strip().upper())


def code_exists(code: str) -> bool:
    return lookup_code(code) is not None


def real_steps(rec: dict) -> List[dict]:
    """Steps a technician can actually carry out. Redirect rows are not checks."""
    return [s for s in rec.get("steps", []) if not s.get("redirect")]


def get_step(code: str, n: int) -> Optional[dict]:
    """1-based position in the code's real steps, not the printed step number.

    They usually coincide; on the 27 column-split repairs they can gap.
    """
    rec = lookup_code(code)
    if not rec:
        return None
    steps = real_steps(rec)
    if n < 1 or n > len(steps):
        return None
    return steps[n - 1]


def step_count(code: str) -> int:
    rec = lookup_code(code)
    return len(real_steps(rec)) if rec else 0


# ------------------------------------------------------------ redirects

def resolve_redirect(code: str, visited=None, max_depth: int = 8) -> dict:
    """Follow pointer-only codes to something with a real procedure.

    Cycle-safe by construction. Seven cycles are documented in the manual --
    D8AQKR -> DA2QKR -> D8AQKR among them -- so a visited set and a depth limit
    are not defensive programming, they are required for termination.
    """
    visited = list(visited or [])
    chain: List[str] = []
    cur = (code or "").strip().upper()
    depth = 0

    while True:
        rec = lookup_code(cur)
        if rec is None:
            return {"target": None, "chain": chain, "reason": "unknown_code",
                    "cycle": False, "depth": depth}
        if cur in visited:
            return {"target": None, "chain": chain, "reason": "cycle",
                    "cycle": True, "depth": depth}
        visited.append(cur)
        chain.append(cur)

        if not rec.get("is_pointer_only"):
            return {"target": cur, "chain": chain, "reason": "resolved",
                    "cycle": False, "depth": depth}

        refs = [r for r in rec.get("refs_failure_codes", []) if code_exists(r)]
        nxt = next((r for r in refs if r not in visited), None)
        if nxt is None:
            return {"target": None, "chain": chain,
                    "reason": "cycle" if refs else "dead_end",
                    "cycle": bool(refs), "depth": depth}
        depth += 1
        if depth > max_depth:
            return {"target": None, "chain": chain, "reason": "depth_limit",
                    "cycle": False, "depth": depth}
        cur = nxt


# --------------------------------------------------------- preconditions

PRECONDITION_RE = re.compile(
    r"If (?:the )?failure code.*?do the troubleshooting", re.I | re.S)
CODE_REF_RE = re.compile(r"\[([A-Z0-9@#]{4,7})\]")


def get_preconditions(code: str) -> List[str]:
    """Codes the manual says to solve before this one.

    Read out of the step text, taking the FIRST code named in the sentence in
    textual order. CA451 says "[CA227] or [CA187]" and CA123 says
    "[CA187] or [CA227]" -- they resolve differently, and only reading the
    sentence gets that right.
    """
    rec = lookup_code(code)
    if not rec:
        return []
    for st in rec.get("steps", []):
        m = PRECONDITION_RE.search(st.get("procedure") or "")
        if not m:
            continue
        refs = [c for c in CODE_REF_RE.findall(m.group(0))
                if c != rec["code"] and code_exists(c)]
        if refs:
            return refs
    return []


# ------------------------------------------------------- reading compare

# SI prefixes seen in this manual's criteria.
_MULT = {"": 1.0, "k": 1e3, "m": 1e-3, "M": 1e6, "µ": 1e-6, "u": 1e-6}

_UNIT_ALIASES = {
    "ohm": "ohm", "ohms": "ohm", "Ω": "ohm", "Ω": "ohm", "ω": "ohm",
    "v": "v", "volt": "v", "volts": "v",
    "a": "a", "amp": "a", "amps": "a", "ma": "ma",
    "kpa": "kpa", "mpa": "mpa", "rpm": "rpm", "hz": "hz", "%": "%",
    "s": "s", "sec": "s", "min": "min", "c": "c", "°c": "c",
}

_NUM = r"(\d+(?:\.\d+)?)"
_UNIT = r"\s*(k|M|m|µ|u)?\s*(ohm|ohms|Ω|Ω|ω|V|volts?|A|amps?|mA|kPa|MPa|rpm|Hz|%|°?C|s|sec|min)?"

_RANGE_RE = re.compile(_NUM + _UNIT + r"\s*to\s*" + _NUM + _UNIT, re.I)
_MAX_RE = re.compile(r"max\.?\s*" + _NUM + _UNIT, re.I)
_MIN_RE = re.compile(r"min\.?\s*" + _NUM + _UNIT, re.I)
_APPROX_RE = re.compile(r"approx(?:imately)?\.?\s*" + _NUM + _UNIT, re.I)
_BARE_RE = re.compile(_NUM + _UNIT, re.I)

# Tolerance applied to "Approx." criteria. The manual states no tolerance, so
# this is an engineering assumption made in code where it is visible and
# reviewable, rather than left to a model to invent per answer.
APPROX_TOLERANCE = 0.10


def _norm_unit(prefix: Optional[str], unit: Optional[str]) -> Tuple[float, str]:
    u = _UNIT_ALIASES.get((unit or "").strip().lower(), (unit or "").strip().lower())
    mult = _MULT.get(prefix or "", 1.0)
    # mA is a unit in its own right in the manual's tables.
    if u == "ma":
        return mult * 1e-3, "a"
    return mult, u


def parse_quantity(text: str) -> Optional[Tuple[float, str]]:
    """'100 kΩ' -> (100000.0, 'ohm'). None when there is no number."""
    if not text:
        return None
    m = _BARE_RE.search(text)
    if not m:
        return None
    mult, unit = _norm_unit(m.group(2), m.group(3))
    return float(m.group(1)) * mult, unit


def parse_criteria(criteria: str) -> Optional[dict]:
    """Criterion string -> a predicate description. None when not numeric."""
    if not criteria:
        return None
    c = criteria.strip()

    m = _RANGE_RE.search(c)
    if m:
        lo_mult, lo_u = _norm_unit(m.group(2), m.group(3))
        hi_mult, hi_u = _norm_unit(m.group(5), m.group(6))
        unit = hi_u or lo_u
        # '0.2 to 4.6V' carries the unit only on the upper bound.
        if not m.group(3) and hi_u:
            lo_mult = hi_mult
        return {"kind": "range", "lo": float(m.group(1)) * lo_mult,
                "hi": float(m.group(4)) * hi_mult, "unit": unit}

    for rx, kind in ((_MAX_RE, "max"), (_MIN_RE, "min"), (_APPROX_RE, "approx")):
        m = rx.search(c)
        if m:
            mult, unit = _norm_unit(m.group(2), m.group(3))
            return {"kind": kind, "value": float(m.group(1)) * mult, "unit": unit}

    if re.search(r"^\s*(no\s+)?continuity", c, re.I):
        return {"kind": "continuity",
                "expect_continuity": not re.search(r"^\s*no\s", c, re.I)}
    return None


def check_reading(step_or_criteria, value=None, unit=None) -> str:
    """PASS / FAIL / OUT_OF_RANGE. Pure arithmetic, no model.

    OUT_OF_RANGE is not a soft FAIL. It means the comparison could not be made
    -- unit mismatch, or a criterion with no numeric content -- and the caller
    must ask again rather than guess a verdict.
    """
    criteria = step_or_criteria
    if isinstance(step_or_criteria, dict):
        ms = step_or_criteria.get("measurements") or []
        criteria = ms[0]["criteria"] if ms else ""

    spec = parse_criteria(criteria)
    if spec is None or value is None:
        return "OUT_OF_RANGE"

    if spec["kind"] == "continuity":
        return "OUT_OF_RANGE"

    want_unit = spec.get("unit") or ""
    got_unit = (unit or "")
    if want_unit and got_unit and want_unit != got_unit:
        return "OUT_OF_RANGE"

    if spec["kind"] == "range":
        return "PASS" if spec["lo"] <= value <= spec["hi"] else "FAIL"
    if spec["kind"] == "max":
        return "PASS" if value <= spec["value"] else "FAIL"
    if spec["kind"] == "min":
        return "PASS" if value >= spec["value"] else "FAIL"
    if spec["kind"] == "approx":
        tol = abs(spec["value"]) * APPROX_TOLERANCE
        return "PASS" if abs(value - spec["value"]) <= tol else "FAIL"
    return "OUT_OF_RANGE"


# ---------------------------------------------------------- branch logic

_ADVANCE_RE = re.compile(
    r"go to the next check|next check item|go to next|proceed to the next", re.I)
_REPAIR_RE = re.compile(
    r"confirmation of repair|repair or replace|replace the|adjust the", re.I)


def branch_disposition(outcome_text: str) -> str:
    """Does this branch outcome continue the tree or end it.

    Read off the manual's own wording rather than inferred. 'Go to the next
    check item' continues; 'Repair or replace ... Go to confirmation of repair'
    is the end of the diagnosis.
    """
    t = outcome_text or ""
    if _ADVANCE_RE.search(t):
        return "ADVANCE"
    if _REPAIR_RE.search(t):
        return "CONCLUDE"
    return "CONCLUDE"


SAFETY_RE = re.compile(
    r"(turn the starting switch to the off position|stop the engine|"
    r"lower the work equipment|disconnect the (?:battery|connector)|"
    r"apply the (?:parking )?brake)", re.I)


def safety_precondition(step: dict) -> Optional[str]:
    """The precondition the technician must act on before touching anything."""
    m = SAFETY_RE.search((step or {}).get("procedure") or "")
    return m.group(1) if m else None


# ------------------------------------------------------------- citations

def render_citations(fact_ids: List[str]) -> List[dict]:
    """Citations come from eval/citations.py, rendered by code from golden/.

    The model names a fact; the system renders the page. What the model does not
    type, it cannot get wrong.
    """
    import sys
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    from eval.citations import render, UnknownFact
    from core.loader import load_verification, UNVERIFIED
    ver = load_verification().get("facts", {})
    out = []
    for fid in fact_ids or []:
        try:
            cits = render([fid])
        except UnknownFact:
            continue          # an invented id yields no citation, by design
        for c in cits:
            # How far this fact has been checked travels WITH the citation. A
            # technician reading a value deserves to know whether anything
            # confirmed it against the page, and dropping the status at the
            # boundary would make "verified" and "nobody looked" indistinguishable.
            c["verification"] = ver.get(fid, UNVERIFIED)
        out += cits
    return out


# --------------------------------------------------------------- symptom

def search_symptoms(text: str) -> List[dict]:
    """STUB. H-Mode (pp. 1288-1471) and S-Mode (pp. 1472-1498) symptom trees are
    not extracted, so symptom entry has no ground truth to execute.

    Returns [] rather than falling back to similarity search over the failure
    codes. A symptom is not a code, and answering one with the other is how a
    technician ends up troubleshooting the wrong subsystem.
    """
    return []
