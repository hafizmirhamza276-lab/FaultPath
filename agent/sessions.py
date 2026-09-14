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

from agent import holdout as _holdout  # noqa: E402
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


# ----------------------------------------------- symptom reference walk

# Confirming a flat row and denying one. Kept as constants because the whole
# flat-tree polarity argument turns on which of these ends the session, and a
# literal buried in a fixture is a bad place for that to live.
ROW_MATCHES = "haan, bilkul yahi dikh raha hai"
ROW_DOES_NOT = "nahi, aisa nahi hai"


def symptom_reference_walk(symptom_id: str, match_at: Optional[int] = None,
                           max_steps: int = 40) -> Dict:
    """Execute a symptom tree directly from golden/, without the graph.

    Written without importing agent.graph, for the same reason reference_walk
    is: an expectation computed by the thing it is checking agrees with itself
    by construction.

    THE TWO TREE KINDS ARE WALKED DIFFERENTLY, AND OPPOSITELY.

      branching  YES means the check was normal -> follow the YES branch, which
                 usually advances. This is reference_walk's logic.
      flat       there are no branches. Rows are candidate causes, and
                 confirming a row's point-to-check means the fault is FOUND:
                 the walk STOPS there and the row's remedy is the answer.

    match_at is the row the scripted technician confirms. For a branching tree
    it is the step that comes back not-normal, which is the same thing said in
    the other polarity.
    """
    rec = tools.lookup_symptom(symptom_id)
    if rec is None:
        return {"symptom_id": symptom_id, "tree_kind": None, "steps": [],
                "outcome": "escalate", "reason": "unknown_symptom",
                "diagnosis": None, "remedy": None, "pointer": None}

    kind = rec["tree_kind"]
    steps_taken: List[int] = []
    diagnosis = remedy = pointer = None
    outcome = "escalate"
    n = tools.symptom_step_count(symptom_id)

    if kind == "SymptomTreeFlat":
        for i in range(1, min(n, max_steps) + 1):
            steps_taken.append(i)
            if match_at is not None and i == match_at:
                step = tools.get_symptom_step(symptom_id, i)
                diagnosis = step.get("cause") or ""
                remedy = step.get("remedy") or ""
                p = tools.step_prose_pointer(symptom_id, step)
                pointer = p["text"] if p else None
                outcome = "conclude"
                break
        return {"symptom_id": symptom_id, "tree_kind": kind,
                "steps": steps_taken, "outcome": outcome,
                "diagnosis": diagnosis, "remedy": remedy, "pointer": pointer,
                "reason": None if outcome == "conclude" else "rows_exhausted"}

    i = 1
    while i <= n and len(steps_taken) < max_steps:
        step = tools.get_symptom_step(symptom_id, i)
        steps_taken.append(i)
        failing = (match_at is not None and i == match_at)
        ms = step.get("measurements") or []
        branches = step.get("branches") or {}

        if ms:
            ans = failing_answer_for(step) if failing else answer_for(step, "pass_all")
            q = tools.parse_quantity(ans)
            verdict = tools.check_reading(ms[0]["criteria"],
                                          q[0] if q else None, q[1] if q else None)
            taken = "YES" if verdict == "PASS" else "NO" if verdict == "FAIL" else None
        else:
            taken = "NO" if failing else "YES"
            verdict = "YES" if taken == "YES" else "NO"

        if taken and taken in branches:
            if tools.branch_disposition(branches[taken]) == "CONCLUDE":
                diagnosis, outcome = branches[taken], "conclude"
                break
            i += 1
            continue
        if verdict in ("FAIL", "NO"):
            diagnosis = step.get("cause") or step.get("procedure")
            outcome = "conclude"
            break
        i += 1

    return {"symptom_id": symptom_id, "tree_kind": kind, "steps": steps_taken,
            "outcome": outcome, "diagnosis": diagnosis, "remedy": None,
            "pointer": None,
            "reason": None if outcome == "conclude" else "steps_exhausted"}


def scripted_symptom_turns(symptom_id: str, opening: str,
                           match_at: Optional[int] = None,
                           pick: Optional[str] = None) -> List[str]:
    """The technician's side of a symptom session that follows the walk."""
    ref = symptom_reference_walk(symptom_id, match_at=match_at)
    turns = [opening, f"{MODEL}, serial {SERIAL}"]
    if pick:                      # an ASK had to be answered first
        turns.append(pick)
    turns.append("koi aur code nahi")
    for i in ref["steps"]:
        if ref["tree_kind"] == "SymptomTreeFlat":
            turns.append(ROW_MATCHES if i == match_at else ROW_DOES_NOT)
        else:
            step = tools.get_symptom_step(symptom_id, i)
            turns.append(failing_answer_for(step) if i == match_at
                         else answer_for(step, "pass_all"))
    return turns


def build_symptom_sessions() -> List[Dict]:
    """Six scenarios, each naming a behaviour the graph has to get right.

    Not a sample of trees -- one session per DISTINCT BEHAVIOUR. Running fifty
    branching symptom trees would grow the number and test one thing.
    """
    out: List[Dict] = []

    def add(kind, sid, turns, expect, note="", code=None):
        # code and symptom_id are separate fields even though a session has at
        # most one of them: the protocol rules ask "did it end up on the code
        # it was given" and "did it end up on the tree it was given", and those
        # are different questions about different corpora.
        out.append({"id": f"{kind}_{sid}_{len(out):02d}", "kind": kind,
                    "code": code, "symptom_id": sid, "turns": turns,
                    "expect": expect, "note": note})

    # 1. clear symptom entering a BRANCHING H-Mode tree. The opening is the
    #    manual's own title, which layer 1 matches byte for byte.
    hm = "HM01"
    ref = symptom_reference_walk(hm, match_at=1)
    add("symptom_branching", hm,
        scripted_symptom_turns(hm, tools.lookup_symptom(hm)["symptom"], match_at=1),
        ref, "exact title -> MATCHED on layer 1; branching tree, YES/NO is "
             "reading the manual back")

    # 2. clear symptom entering a FLAT S-Mode tree and reaching a real remedy.
    #    Row 3 of SM01 is 'Replace if the item is broken' -- an instruction, not
    #    a redirect.
    sm = "SM01"
    ref = symptom_reference_walk(sm, match_at=3)
    add("symptom_flat_remedy", sm,
        scripted_symptom_turns(sm, tools.lookup_symptom(sm)["symptom"], match_at=3),
        ref, "flat tree: no YES/NO exists; confirming a row STOPS the walk and "
             "its remedy is the answer")

    # 3. ambiguous symptom -> ASK -> resolved when the technician picks.
    #    'swing slow hai' is a curated synonym pointing at two trees, and the
    #    map records it that way rather than choosing one.
    ref = symptom_reference_walk("HM29", match_at=1)
    add("symptom_ask", "HM29",
        scripted_symptom_turns("HM29", "swing slow hai", match_at=1, pick="2"),
        ref, "two candidates; the agent presents both with pages and waits")

    # 4. unmapped symptom. A real technician phrase with an obvious nearest
    #    tree and no correct one.
    add("symptom_unmapped", "-",
        ["oil leak ho raha hai", f"{MODEL}, serial {SERIAL}"],
        {"symptom_id": None, "tree_kind": None, "steps": [],
         "outcome": "escalate", "reason": "symptom_unmapped",
         "diagnosis": None, "remedy": None, "pointer": None},
        "no leak tree exists; must say so, not route to the nearest")

    # 5. a symptom tree whose REMEDY IS A PROSE POINTER.
    #
    #    SM01 rows 1 and 2 are the ONLY two of the 182 prose pointers that are
    #    a step's remedy -- 145 sit in step procedure text as supplementary
    #    detail and 35 in related_information, which preflight already handles.
    #    So this fixture is the only real instance in the corpus, and one
    #    passing test here is not broad coverage of pointer handling.
    ref = symptom_reference_walk(sm, match_at=1)
    add("symptom_pointer", sm,
        scripted_symptom_turns(sm, tools.lookup_symptom(sm)["symptom"], match_at=1),
        ref, "the manual refers you elsewhere; surface the page, do not follow")

    # 6. a technician who produces a failure code partway through a symptom
    #    session. A code is a dict lookup and beats a match, so the session
    #    must switch to it and abandon the symptom tree.
    code = next(c for c in sorted(tools.records())
                if tools.records()[c]["format"] == "A"
                and not tools.records()[c]["is_pointer_only"]
                and tools.step_count(c) >= 2
                and c not in _holdout.holdout_set())
    cref = reference_walk(code, fail_at=1)
    turns = [tools.lookup_symptom(hm)["symptom"], f"{MODEL}, serial {SERIAL}",
             f"ruko, monitor pe {code} bhi aa gaya", "koi aur code nahi"]
    for i in cref["steps"]:
        step = tools.get_step(cref["resolved"], i)
        turns.append(failing_answer_for(step) if i == 1
                     else answer_for(step, "pass_all"))
    add("symptom_to_code", code, turns, cref,
        f"entered on a symptom, switched to {code} when the monitor produced it",
        code=code)

    return out


# --------------------------------------------------------------- corpus

def build_sessions(exclude=None) -> List[Dict]:
    """~40 sessions across every behaviour the graph has to get right.

    Held-out codes are excluded BY DEFAULT, not by the caller remembering to.
    A holdout a fixture can reach by accident stops measuring generalisation
    while still reporting a number, which is the failure mode this whole guard
    exists for.
    """
    recs = tools.records()
    exclude = set(exclude) if exclude is not None else _holdout.holdout_set()
    codes = [c for c in sorted(recs) if c not in exclude]
    fmt_a = [c for c in codes if recs[c]["format"] == "A"
             and tools.step_count(c) >= 2 and not recs[c]["is_pointer_only"]]
    fmt_b = [c for c in codes if recs[c]["format"] == "B"
             and tools.step_count(c) >= 2 and not recs[c]["is_pointer_only"]]
    pointers = [c for c in codes if recs[c]["is_pointer_only"]]
    with_pre = [c for c in codes if tools.get_preconditions(c)]

    out: List[Dict] = []

    def add(kind, code, turns, expect, note=""):
        # A session must not WALK a held-out tree either. Cross-references do
        # not respect the split: a train code's precondition or redirect target
        # can be held out, and following it would tune us to a code that is
        # supposed to be measuring generalisation.
        touched = {expect.get("resolved")} | set(expect.get("chain") or [])
        if touched & exclude:
            return
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
