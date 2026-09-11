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
from agent.runner import BadAgent, run_all, run_session   # noqa: E402
from agent.sessions import build_sessions                 # noqa: E402
from agent.state import SessionState                      # noqa: E402
from eval.metrics.agent import build as build_metrics, AGENT_GATES  # noqa: E402
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


# ==================================================================== summary
print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print(f"OK: agent verified over {len(sessions)} sessions")
sys.exit(0)
