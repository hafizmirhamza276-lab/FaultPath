#!/usr/bin/env python3
"""
sessions.py
Scripted diagnostic sessions and the reference walk they are scored against.

The expected step order is NOT hand-written. reference_walk() is an independent
implementation of the manual's own traversal -- read the step, apply the scripted
answer, follow the branch the manual names -- built straight from
golden/failure_codes/. If the agent and the reference disagree, one of them
departed from the tree, and the disagreement says which step.

Hand-written expectations would encode what I think the manual says. This
encodes what it does say.

Covers: single code, multiple codes with a precondition, pointer-only redirect,
a documented cycle, vague readings, a technician who changes code mid-session,
and an unknown code.
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent import tools  # noqa: E402

MODEL = "PC200-10M0"
SERIAL = "700123"


# ------------------------------------------------------- scripted answers

def answer_for(step: dict, policy: str) -> str:
    """What the technician types for this step under a given policy.

    'pass_all'  every check comes back normal -> the tree runs to exhaustion
    'fail_at_N' the Nth check comes back bad  -> the tree concludes there
    """
    ms = step.get("measurements") or []
    if not ms:
        return "haan, normal hai"
    crit = ms[0]["criteria"]
    spec = tools.parse_criteria(crit)
    if spec is None:
        return "haan, normal hai"
    unit = spec.get("unit") or ""
    disp = {"ohm": "ohm", "v": "V", "a": "A", "kpa": "kPa", "mpa": "MPa",
            "rpm": "rpm", "hz": "Hz", "%": "%"}.get(unit, unit)
    if spec["kind"] == "range":
        v = (spec["lo"] + spec["hi"]) / 2
    elif spec["kind"] == "max":
        v = spec["value"] / 2 if spec["value"] else 0.0
    elif spec["kind"] == "min":
        v = spec["value"] * 2 if spec["value"] else 1.0
    elif spec["kind"] == "approx":
        v = spec["value"]
    else:
        return "haan, normal hai"
    return f"{_fmt(v)} {disp}".strip()


def failing_answer_for(step: dict) -> str:
    ms = step.get("measurements") or []
    if not ms:
        return "nahi, theek nahi hai"
    spec = tools.parse_criteria(ms[0]["criteria"])
    if spec is None:
        return "nahi, theek nahi hai"
    unit = spec.get("unit") or ""
    disp = {"ohm": "ohm", "v": "V", "a": "A", "kpa": "kPa", "mpa": "MPa",
            "rpm": "rpm", "hz": "Hz", "%": "%"}.get(unit, unit)
    if spec["kind"] == "range":
        v = spec["hi"] * 10 + 1
    elif spec["kind"] == "max":
        v = spec["value"] * 10 + 1
    elif spec["kind"] == "min":
        v = spec["value"] / 10 if spec["value"] else 0.0
    elif spec["kind"] == "approx":
        v = spec["value"] * 5 + 1
    else:
        return "nahi, theek nahi hai"
    return f"{_fmt(v)} {disp}".strip()


def _fmt(v: float) -> str:
    if v >= 1000:
        return f"{v/1000:g} k"
    return f"{v:g}"


# ------------------------------------------------------- reference walk

def reference_walk(code: str, policy: str = "pass_all",
                   fail_at: Optional[int] = None,
                   max_steps: int = 40) -> Dict:
    """Execute the manual's tree directly. The agent is scored against this.

    Deliberately written without importing the graph, so a bug in the graph
    cannot make its own expectation agree with it.
    """
    visited: List[str] = []
    cur = code
    depth = 0

    # Pointer-only entry resolves first, cycle-safe.
    res = tools.resolve_redirect(cur, visited=[], max_depth=8)
    chain = res["chain"]
    if res["target"] is None:
        return {"code": code, "resolved": None, "chain": chain,
                "steps": [], "outcome": "escalate", "reason": res["reason"],
                "diagnosis": None}
    cur = res["target"]
    visited.extend(chain)
    depth = res["depth"]

    steps_taken: List[int] = []
    diagnosis = None
    outcome = "escalate"
    n_steps = tools.step_count(cur)

    i = 1
    while i <= n_steps and len(steps_taken) < max_steps:
        step = tools.get_step(cur, i)
        steps_taken.append(i)
        failing = (fail_at is not None and i == fail_at)
        ms = step.get("measurements") or []
        branches = step.get("branches") or {}

        if ms:
            ans = failing_answer_for(step) if failing else answer_for(step, policy)
            q = tools.parse_quantity(ans)
            verdict = tools.check_reading(ms[0]["criteria"],
                                          q[0] if q else None, q[1] if q else None)
            taken = "YES" if verdict == "PASS" else "NO" if verdict == "FAIL" else None
        else:
            taken = "NO" if failing else "YES"
            verdict = "YES" if taken == "YES" else "NO"

        if taken and taken in branches:
            if tools.branch_disposition(branches[taken]) == "CONCLUDE":
                diagnosis = branches[taken]
                outcome = "conclude"
                break
            i += 1
            continue
        if verdict in ("FAIL", "NO"):
            diagnosis = step.get("cause") or step.get("procedure")
            outcome = "conclude"
            break
        i += 1

    return {"code": code, "resolved": cur, "chain": chain, "depth": depth,
            "steps": steps_taken, "outcome": outcome, "diagnosis": diagnosis,
            "reason": None}


def scripted_turns(code: str, fail_at: Optional[int] = None,
                   include_machine=True, other_codes="koi aur code nahi",
                   opening: Optional[str] = None) -> List[str]:
    """The technician's side of a session that follows the reference walk."""
    ref = reference_walk(code, fail_at=fail_at)
    turns = [opening or f"{code} aa raha hai"]
    if include_machine:
        turns.append(f"{MODEL}, serial {SERIAL}")
    turns.append(other_codes)
    target = ref["resolved"]
    if target:
        for i in ref["steps"]:
            step = tools.get_step(target, i)
            turns.append(failing_answer_for(step) if (fail_at and i == fail_at)
                         else answer_for(step, "pass_all"))
    return turns


# --------------------------------------------------------------- corpus

def build_sessions() -> List[Dict]:
    """~40 sessions across every behaviour the graph has to get right."""
    recs = tools.records()
    codes = sorted(recs)
    fmt_a = [c for c in codes if recs[c]["format"] == "A"
             and tools.step_count(c) >= 2 and not recs[c]["is_pointer_only"]]
    fmt_b = [c for c in codes if recs[c]["format"] == "B"
             and tools.step_count(c) >= 2 and not recs[c]["is_pointer_only"]]
    pointers = [c for c in codes if recs[c]["is_pointer_only"]]
    with_pre = [c for c in codes if tools.get_preconditions(c)]

    out: List[Dict] = []

    def add(kind, code, turns, expect, note=""):
        out.append({"id": f"{kind}_{code}_{len(out):02d}", "kind": kind,
                    "code": code, "turns": turns, "expect": expect, "note": note})

    # 1. single code, runs to a conclusion (format A, branch-driven)
    for c in fmt_a[:8]:
        ref = reference_walk(c, fail_at=1)
        add("single_fail_first", c, scripted_turns(c, fail_at=1), ref,
            "first check fails; the NO branch is the answer")

    # 2. single code, every check passes -> tree exhausts -> escalate
    for c in fmt_a[8:13]:
        ref = reference_walk(c)
        add("single_pass_all", c, scripted_turns(c), ref,
            "every check normal; the manual runs out and we hand over")

    # 3. sequential format B
    for c in fmt_b[:6]:
        ref = reference_walk(c, fail_at=2)
        add("sequential", c, scripted_turns(c, fail_at=2), ref,
            "format B: the first cause that fails is the diagnosis")

    # 4. pointer-only redirect
    for c in pointers[:6]:
        ref = reference_walk(c, fail_at=1)
        add("pointer", c, scripted_turns(c, fail_at=1), ref,
            "code has no procedure; must hop and continue there")

    # 5. precondition: the other code is on the monitor too
    for c in with_pre[:5]:
        pre = tools.get_preconditions(c)[0]
        ref = reference_walk(pre, fail_at=1)
        turns = [f"{c} aa raha hai", f"{MODEL}, serial {SERIAL}",
                 f"haan {pre} bhi show ho raha hai"]
        for i in ref["steps"]:
            turns.append(failing_answer_for(tools.get_step(ref["resolved"], i))
                         if i == 1 else
                         answer_for(tools.get_step(ref["resolved"], i), "pass_all"))
        add("precondition", c, turns, ref,
            f"manual says solve {pre} before {c}")

    # 6. vague readings before a usable one
    for c in fmt_a[13:17]:
        base = scripted_turns(c, fail_at=1)
        turns = base[:3] + ["thoda kam lag raha hai", "normal hi hoga shayad"] + base[3:]
        add("vague", c, turns, reference_walk(c, fail_at=1),
            "hedged answers must be refused, then the real reading accepted")

    # 7. technician changes the code mid-session
    for c, d in zip(fmt_a[17:20], fmt_a[20:23]):
        ref = reference_walk(d, fail_at=1)
        turns = [f"{c} aa raha hai", f"{MODEL}, serial {SERIAL}",
                 "koi aur code nahi", f"ruko, actually {d} hai"]
        # After the switch the session must run D's tree, not C's, so the
        # answers that follow are D's.
        for i in ref["steps"]:
            step = tools.get_step(ref["resolved"], i)
            turns.append(failing_answer_for(step) if i == 1
                         else answer_for(step, "pass_all"))
        add("code_switch", c, turns, ref, f"switched from {c} to {d} mid-session")

    # 8. unknown code
    for fake in ("CA999", "B@BZZZ", "X0X0X0"):
        add("unknown_code", fake,
            [f"{fake} aa raha hai", f"{MODEL}, serial {SERIAL}"],
            {"code": fake, "resolved": None, "chain": [], "steps": [],
             "outcome": "escalate", "reason": "unknown_code", "diagnosis": None},
            "not in this manual; must say so, not retrieve a neighbour")

    # 9. a step carrying a safety precondition must actually be reached, or the
    # rule that checks it is scored on nothing.
    safety_added = 0
    for c in codes:
        if safety_added >= 4 or recs[c]["is_pointer_only"]:
            continue
        steps = tools.real_steps(recs[c])
        idx = next((i for i, s in enumerate(steps, 1)
                    if tools.safety_precondition(s)), None)
        if idx is None:
            continue
        ref = reference_walk(c, fail_at=idx)
        if len(ref["steps"]) < idx:
            continue
        add("safety", c, scripted_turns(c, fail_at=idx), ref,
            f"step {idx} carries a safety precondition that must be surfaced")
        safety_added += 1

    # 10. a documented cycle, entered directly
    for c in ("D8AQKR", "DA2QKR", "DAZQKR"):
        if c in recs:
            add("cycle", c, scripted_turns(c, fail_at=1),
                reference_walk(c, fail_at=1),
                "member of a documented cross-reference cycle")

    # 10. machine never identified -- the gate must hold forever
    for c in fmt_a[:2]:
        add("no_machine", c,
            [f"{c} aa raha hai", "pata nahi", "bas code batao"],
            {"code": c, "resolved": None, "chain": [], "steps": [],
             "outcome": "gate_held", "reason": "no_machine", "diagnosis": None},
            "no model/serial: no technical value may be emitted, ever")

    return out


if __name__ == "__main__":
    ss = build_sessions()
    import collections
    print(f"{len(ss)} sessions")
    for k, v in collections.Counter(s["kind"] for s in ss).most_common():
        print(f"  {k:20} {v}")
