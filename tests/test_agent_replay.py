#!/usr/bin/env python3
"""
test_agent_replay.py
Verifies the agent, not the manual.

    python tests/test_agent_replay.py

Four things:

  1. REPLAY DETERMINISM. Feed a recorded transcript back and assert an identical
     state sequence -- same nodes, same steps, same readings, same diagnosis,
     same emitted text. Same state plus same input must give the same next
     state, always, or nothing else here means anything.

  2. Every agent metric proves it can FAIL on known-bad input, the same
     discipline that caught audit checks E4 and H2 sitting at zero while being
     structurally incapable of returning anything else.

  3. The good agent passes every gate; the deliberately bad one fails EVERY
     gate. A gate the bad agent passes is not measuring what it claims.

  4. The outgoing guard blocks a value the model was not entitled to emit.

Runs offline with MockLLM: no network, no key.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from agent import tools                                   # noqa: E402
from agent.graph import Agent                             # noqa: E402
from agent.runner import (BadAgent, BadSymptomAgent,      # noqa: E402
                          run_all, run_session)
from agent.sessions import build_sessions, build_symptom_sessions  # noqa: E402
from agent.state import SessionState                      # noqa: E402
from eval.metrics.agent import (build as build_metrics,   # noqa: E402
                                AGENT_GATES, SYMPTOM_GATES)
from eval.metrics.base import Metric                      # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


sessions = build_sessions()
print(f"\n{len(sessions)} scripted sessions")


# ================================================ 1. replay determinism
print("\nreplay determinism")


def fingerprint(result):
    """Everything that must be reproducible. Timestamps are excluded -- they
    legitimately differ and would break the comparison for the wrong reason."""
    st = result["state"]
    return {
        "nodes": result["node_sequence"],
        "steps": result["steps_visited"],
        "codes": result["visited_codes"],
        "outcome": result["outcome"],
        "diagnosis": result["diagnosis"],
        "fact_ids": result["fact_ids_used"],
        "emitted": [e["text"] for e in result["emissions"]],
        "blocked": [e["blocked"] for e in result["emissions"]],
        "readings": [(r["step"], r["kind"], r["value"], r["unit"], r["verdict"])
                     for r in st["readings"]],
        "cursor": st["step_cursor"],
    }


run_a = run_all(Agent(), sessions)
run_b = run_all(Agent(), sessions)

fa = [fingerprint(r) for r in run_a]
fb = [fingerprint(r) for r in run_b]
check("two runs of all sessions produce identical state sequences", fa == fb,
      next((f"first divergence: {s['id']}" for s, x, y in
            zip(sessions, fa, fb) if x != y), ""))

# Replay from the recorded transcript rather than the script, so the test
# exercises the same path a stored session would take on re-execution.
replayed = []
for rec in run_a:
    sess = next(s for s in sessions if s["id"] == rec["session_id"])
    replay_session = dict(sess, turns=[t["technician"] for t in rec["transcript"]])
    replayed.append(run_session(Agent(), replay_session))
check("recorded transcripts replay to the same state sequence",
      [fingerprint(r) for r in replayed] == fa,
      next((f"first divergence: {r['session_id']}" for r, x in
            zip(replayed, fa) if fingerprint(r) != x), ""))

# A state round-tripped through JSON must behave identically -- this is what a
# FastAPI service will do between turns.
st = Agent().start("roundtrip")
ag = Agent()
st, _ = ag.turn(st, "CA451 aa raha hai")
revived = SessionState.model_validate(json.loads(st.model_dump_json()))
s1, o1 = ag.turn(st, "PC200-10M0, serial 700123")
s2, o2 = ag.turn(revived, "PC200-10M0, serial 700123")
check("state survives a JSON round-trip unchanged", o1 == o2 and
      s1.node_sequence == s2.node_sequence)


# ============================================ 2. metric self-tests
print("\nevery agent metric can fail on known-bad input")
metrics = build_metrics()
tested, bad = 0, []
for m in metrics:
    if type(m).self_test is Metric.self_test:
        continue
    tested += 1
    try:
        m.self_test()
    except AssertionError as exc:
        bad.append(f"{m.name}: {exc}")
check(f"{tested} metric self-tests pass", not bad, "; ".join(bad))

gate_names = {g[0] for g in AGENT_GATES}
missing = sorted(m.name for m in metrics if m.name in gate_names
                 and type(m).self_test is Metric.self_test)
check("every gate-bearing agent metric has a self-test", not missing,
      f"missing: {missing}")


# ============================================ 3. good/bad separation
print("\ngood passes every gate, bad fails every gate")


def score(rows):
    agg = {}
    for m in build_metrics():
        vals = [m.compute({"type": "agent_session", **s}, r)
                for s, r in zip(sessions, rows)]
        vals = [v for v in vals if v is not None]
        if vals:
            agg[m.name] = sum(vals) / len(vals)
    return agg


def gates(agg):
    out = []
    for name, op, th in AGENT_GATES:
        v = agg.get(name)
        ok = v is not None and (v >= th if op == ">=" else v <= th)
        out.append((name, ok, v))
    return out


good_gates = gates(score(run_a))
bad_gates = gates(score(run_all(BadAgent(), sessions)))

good_failed = [n for n, ok, _ in good_gates if not ok]
bad_passed = [n for n, ok, _ in bad_gates if ok]
check(f"good agent passes all {len(AGENT_GATES)} gates", not good_failed,
      f"failed: {good_failed}")
check("bad agent fails ALL gates", not bad_passed,
      f"bad PASSED {bad_passed} -- each of those gates has a hole")


# ==================================== 3b. the symptom path, scored apart
#
# SCORED APART FROM THE CODE SESSIONS, DELIBERATELY. Folding six symptom
# sessions into the 45 code sessions would let a 45-session majority carry a
# broken symptom path over every threshold, and the averages would keep looking
# healthy while the second entry point did not work.
print("\nsymptom path: good passes every symptom gate, bad fails every one")

sym_sessions = build_symptom_sessions()
print(f"  {len(sym_sessions)} symptom sessions: "
      + ", ".join(s["kind"] for s in sym_sessions))


def sym_score(agent_cls):
    rows = [run_session(agent_cls(), s) for s in sym_sessions]
    agg = {}
    for m in build_metrics():
        vals = [m.compute({"type": "agent_session", **s}, r)
                for s, r in zip(sym_sessions, rows)]
        vals = [v for v in vals if v is not None]
        if vals:
            agg[m.name] = (sum(vals) / len(vals), len(vals))
    return agg, rows


good_agg, good_rows = sym_score(Agent)
bad_agg, _ = sym_score(BadSymptomAgent)

print(f"  {'gate':32} {'n':>3} {'good':>7} {'bad':>7}")
sym_good_failed, sym_bad_passed, unscored = [], [], []
for name, op, th in SYMPTOM_GATES:
    gv, n = good_agg.get(name, (None, 0))
    bv, _ = bad_agg.get(name, (None, 0))
    if gv is None:
        unscored.append(name)
        print(f"  {name:32} {0:>3} {'--':>7} {'--':>7}")
        continue
    g_ok = gv >= th if op == ">=" else gv <= th
    b_ok = bv is not None and (bv >= th if op == ">=" else bv <= th)
    if not g_ok:
        sym_good_failed.append(f"{name}={gv:.4f}")
    if b_ok:
        sym_bad_passed.append(name)
    print(f"  {name:32} {n:>3} {gv:7.4f} {(bv if bv is not None else float('nan')):7.4f}")

check(f"good agent passes all {len(SYMPTOM_GATES)} symptom gates",
      not sym_good_failed, f"failed: {sym_good_failed}")
check("bad symptom agent fails ALL symptom gates", not sym_bad_passed,
      f"bad PASSED {sym_bad_passed} -- each of those gates has a hole")
# A gate scored on zero sessions reports nothing and passes everything. That is
# the E4/H2 shape and it must fail loudly rather than print a dash.
check("every symptom gate is scored on at least one session", not unscored,
      f"scored on nothing: {unscored}")

# The machine gate is not a code-session rule. A symptom session that released
# a value before model and serial were known would be the same defect.
_ge = next(m for m in build_metrics() if m.name == "gate_enforcement")
_ge_vals = [_ge.compute({"type": "agent_session", **s}, r)
            for s, r in zip(sym_sessions, good_rows)]
check("the machine gate holds on every symptom session",
      all(v == 1.0 for v in _ge_vals if v is not None), str(_ge_vals))
_ge_bad = [_ge.compute({"type": "agent_session", **s}, r)
           for s, r in zip(sym_sessions,
                           [run_session(BadAgent(), s) for s in sym_sessions])]
check("a symptom session that bypasses the machine gate is caught",
      any(v == 0.0 for v in _ge_bad if v is not None), str(_ge_bad))

# Symptom sessions must replay identically too.
check("symptom sessions replay to identical state sequences",
      [fingerprint(r) for r in sym_score(Agent)[1]]
      == [fingerprint(r) for r in good_rows])

# CITATIONS MUST RESOLVE, not merely render. A symptom fact that produces a
# citation-shaped object pointing at the wrong page is the failure the whole
# provenance design exists to prevent.
_sym_facts = sorted({f for r in good_rows for f in r["fact_ids_used"]})
_rendered = tools.render_citations(_sym_facts)
check("every fact id a symptom session used renders a citation",
      len(_rendered) == len(_sym_facts),
      f"{len(_rendered)} citations for {len(_sym_facts)} fact ids")
check("every symptom citation carries a manual page and a pdf page",
      all(c.get("manual_page") and c.get("pdf_page") for c in _rendered))
_remedy_cites = [c for c in _rendered if ":remedy:" in c["fact_id"]]
check("remedy facts cite their own cell, not the row's cause",
      len(_remedy_cites) >= 2, f"{len(_remedy_cites)} remedy citations")


# ============================================ 4. the outgoing guard
print("\noutgoing value guard")
from agent.guards import check_message, enforce      # noqa: E402

ok, offending = check_message(
    "The standard value is Max. 7 ohm.",
    citations=[{"verbatim_text": "Max. 1 Ω", "manual_page": "40-121"}],
    fact_ids=["CA451:6:meas:0"], code="CA451")
check("a number with no supporting fact is caught", not ok and "7" in offending)

ok2, _ = check_message(
    "The standard value is Max. 1 ohm.",
    citations=[{"verbatim_text": "Max. 1 Ω", "manual_page": "40-121"}],
    fact_ids=["CA451:6:meas:0"], code="CA451")
check("a number copied from a cited fact is allowed", ok2)

sent, blocked, _ = enforce(
    "Measure 99 volts.", citations=[], fact_ids=[], code="CA451")
check("a blocked message is replaced, not truncated",
      blocked and "99" not in sent)

# The guard must not be defeatable by citing a fact that does not exist.
cites = tools.render_citations(["CA999:1:meas:0"])
check("an invented fact id renders no citation and grounds nothing",
      cites == [])


# ============================= 5. MODULE: dispatch, states, polarity
print("\nMODULE -- tree_kind dispatch, ASK/UNMAPPED, verdict polarity")
from agent.state import TreeKind, Awaiting, Verdict        # noqa: E402

# tree_kind drives behaviour, and is read from golden/ rather than guessed.
check("both tree kinds exist in the corpus and are the manual's own strings",
      {r["tree_kind"] for r in tools.symptoms().values()}
      == {TreeKind.BRANCHING.value, TreeKind.FLAT.value})
_flat = [s for s, r in tools.symptoms().items()
         if r["tree_kind"] == TreeKind.FLAT.value]
check("no flat tree carries a single branch outcome",
      all(not (st.get("branches") or {})
          for s in _flat for st in tools.lookup_symptom(s)["steps"]),
      "a flat tree with branches would make the polarity rule meaningless")
check("every flat row carries a remedy of its own",
      all(st.get("remedy") and st.get("remedy_fact_id")
          for s in _flat for st in tools.lookup_symptom(s)["steps"]))

# The verdict pairs must not be collapsed. This is the quiet failure: a flat
# tree walked with branching polarity produces fluent output and no error.
check("OBSERVED/NOT_OBSERVED are distinct from YES/NO",
      len({Verdict.OBSERVED, Verdict.NOT_OBSERVED,
           Verdict.YES, Verdict.NO}) == 4)

_ag = Agent()


def _drive(turns):
    st = _ag.start("mod")
    outs = []
    for t in turns:
        st, o = _ag.turn(st, t)
        outs.append(o)
    return st, " ".join(outs).lower()


MACHINE = "PC200-10M0, serial 700123"
_st, _txt = _drive(["swing slow hai", MACHINE])
check("ASK is a pause, not an outcome",
      _st.awaiting == Awaiting.SYMPTOM_CHOICE
      and _st.outcome.value == "running" and _st.active_symptom is None)
check("ASK presents every candidate with its page",
      _txt.count("(page ") == len(_st.symptom_candidates) >= 2)
_st2, _ = _drive(["swing slow hai", MACHINE, "kuch samajh nahi aaya"])
check("an unclear reply to an ASK asks again rather than taking the first",
      _st2.active_symptom is None
      and _st2.awaiting == Awaiting.SYMPTOM_CHOICE)

_st3, _txt3 = _drive(["AC kaam nahi kar raha", MACHINE])
check("UNMAPPED refuses and names no tree",
      _st3.active_symptom is None
      and _st3.escalation_note.get("reason") == "symptom_unmapped"
      and "closest-looking" in _txt3)

_flat_title = tools.lookup_symptom("SM01")["symptom"]
_st4, _txt4 = _drive([_flat_title, MACHINE, "koi aur code nahi"])
check("a flat tree is asked for an observation, never yes/no",
      _st4.tree_kind == TreeKind.FLAT
      and _st4.awaiting == Awaiting.OBSERVATION
      and "is that what you are seeing" in _txt4
      and "answer yes or no" not in _txt4)
_st5, _ = _drive([_flat_title, MACHINE, "koi aur code nahi", "haan yahi hai"])
check("confirming a flat row STOPS the walk (opposite of a branching YES)",
      _st5.outcome.value == "concluded" and _st5.step_cursor == 1
      and _st5.readings[-1].verdict == Verdict.OBSERVED)
_st6, _ = _drive([_flat_title, MACHINE, "koi aur code nahi", "nahi"])
check("denying a flat row ADVANCES it",
      _st6.step_cursor == 2 and _st6.outcome.value == "running"
      and _st6.readings[-1].verdict == Verdict.NOT_OBSERVED)
check("the remedy is a field of its own, not procedure prose",
      _st5.remedy and _st5.remedy_fact_id
      and _st5.remedy_fact_id.endswith(":remedy:0"))
check("a prose-pointer remedy is surfaced with its page and not followed",
      _st5.prose_pointer and _st5.prose_pointer.get("manual_page")
      and _st5.remedy == _st5.prose_pointer["text"])


# ====================================== 6. OVERFITTING GUARD: mutations
print("\nOVERFITTING GUARD")
print("\n  SYMPTOM MUTATION TABLE")

_sym_cases = [{"type": "agent_session", **s} for s in sym_sessions]


def _catches(agent_cls, gate_name):
    """Does gate_name FAIL for this agent. The point of a planted fault."""
    rows = [run_session(agent_cls(), s) for s in sym_sessions]
    m = next(x for x in build_metrics() if x.name == gate_name)
    vals = [m.compute(c, r) for c, r in zip(_sym_cases, rows)]
    vals = [v for v in vals if v is not None]
    op, th = next((o, t) for n, o, t in SYMPTOM_GATES if n == gate_name)
    if not vals:
        return False, None
    avg = sum(vals) / len(vals)
    ok = avg >= th if op == ">=" else avg <= th
    return (not ok), avg


_MUTATIONS = [
    ("agent_picks_on_ambiguity", BadSymptomAgent, "symptom_asks_on_ambiguity"),
    ("unmapped_routed_to_nearest_tree", BadSymptomAgent,
     "symptom_unmapped_not_routed"),
    ("flat_tree_walked_with_branching_polarity", BadSymptomAgent,
     "symptom_flat_polarity"),
    ("prose_pointer_followed_by_guessing", BadSymptomAgent,
     "symptom_pointer_surfaced"),
    ("remedy_buried_in_procedure_text", BadSymptomAgent, "remedy_correct"),
    ("symptom_session_bypasses_machine_gate", BadAgent, "gate_enforcement"),
    ("code_ignored_mid_symptom_session", BadSymptomAgent,
     "symptom_code_takes_over"),
]
_missed = []
for _name, _cls, _gate in _MUTATIONS:
    if _gate == "gate_enforcement":
        rows = [run_session(_cls(), s) for s in sym_sessions]
        m = next(x for x in build_metrics() if x.name == "gate_enforcement")
        vals = [v for v in (m.compute(c, r) for c, r in zip(_sym_cases, rows))
                if v is not None]
        caught, avg = (sum(vals) / len(vals) < 1.0), sum(vals) / len(vals)
    else:
        caught, avg = _catches(_cls, _gate)
    print(f"    {_name:44} {'caught' if caught else 'MISSED':8} "
          f"{_gate} = {avg if avg is None else round(avg, 4)}")
    if not caught:
        _missed.append(_name)
check("every planted symptom fault is caught", not _missed, str(_missed))

# NEGATIVE COVERAGE. Each symptom gate needs a case that fails it, and the
# control must NOT fail -- a gate that fires on everything is as useless as
# one that fires on nothing.
_neg_missing = [n for n, _, _ in SYMPTOM_GATES
                if not _catches(BadSymptomAgent, n)[0]]
check("every symptom gate has a failing case", not _neg_missing,
      str(_neg_missing))
check("the control passes every symptom gate -- gates are not always-on",
      not [n for n, _, _ in SYMPTOM_GATES if _catches(Agent, n)[0]])

# INDEPENDENT DERIVATION, over the parsed AST.
print("\n  independent derivation")
import ast as _ast                                        # noqa: E402

_sess_src = open(os.path.join(REPO_ROOT, "agent", "sessions.py"),
                 encoding="utf-8").read()
_sess_imports = {n.module.split(".")[0]
                 for n in _ast.walk(_ast.parse(_sess_src))
                 if isinstance(n, _ast.ImportFrom) and n.module}
check("the symptom reference walk does not import the graph it checks",
      "agent" in _sess_imports and "graph" not in _sess_src.split("import")[0]
      and not any(isinstance(n, _ast.ImportFrom) and n.module
                  and n.module.endswith("graph")
                  for n in _ast.walk(_ast.parse(_sess_src))),
      "an expectation computed by the thing it checks agrees with itself")

for _mod in ("core/symptom_match.py", "agent/tools.py", "agent/runner.py"):
    _src = open(os.path.join(REPO_ROOT, _mod), encoding="utf-8").read()
    _names = {(n.module or "").split(".")[0]
              for n in _ast.walk(_ast.parse(_src))
              if isinstance(n, _ast.ImportFrom)}
    _names |= {a.name.split(".")[0] for n in _ast.walk(_ast.parse(_src))
               if isinstance(n, _ast.Import) for a in n.names}
    check(f"{_mod} imports no model or network client",
          not (_names & {"openai", "anthropic", "requests", "httpx",
                         "urllib", "boto3", "transformers"}),
          str(sorted(_names)))

_match_src = open(os.path.join(REPO_ROOT, "core", "symptom_match.py"),
                  encoding="utf-8").read()
check("symptom matching imports clean(), it does not reimplement it",
      "from extract_golden import clean" in _match_src
      and "def clean(" not in _match_src,
      "two normalisation pipelines is the bug that moved three titles")


# ==================================================================== summary
print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"OK: agent verified over {len(sessions)} sessions")
sys.exit(0)
