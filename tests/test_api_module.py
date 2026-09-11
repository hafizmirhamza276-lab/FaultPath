#!/usr/bin/env python3
"""
test_api_module.py
MODULE level: contracts, serialisation, guard, error mapping, rate limiting.
No graph, no network, no PDF.

Fast on purpose -- this is the level that should fail first when a contract
changes, before anything spends a minute walking 49 sessions.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from pydantic import ValidationError                    # noqa: E402
from agent.guards import check_message, enforce         # noqa: E402
from agent.state import SessionState, Awaiting          # noqa: E402
from api import contracts as C                          # noqa: E402
from api.metrics import Metrics                         # noqa: E402
from api.store import (SessionStore, RateLimited,       # noqa: E402
                       TurnConflict, SessionLimit)

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:
        return False
    return False


# ------------------------------------------------------------- contracts
print("\ncontracts reject what they should")

check("unknown field on a request is rejected",
      raises(ValidationError, C.MessageRequest,
             message="hi", turn_index=0, model="PC490LC-11"),
      "an ignored field is how a client believes it sent a filter it did not")
check("empty message is rejected",
      raises(ValidationError, C.StartSessionRequest, message=""))
check("oversized message is rejected",
      raises(ValidationError, C.StartSessionRequest,
             message="x" * (C.MAX_MESSAGE_CHARS + 1)))
check("negative turn_index is rejected",
      raises(ValidationError, C.MessageRequest, message="hi", turn_index=-1))
check("bad awaiting literal is rejected",
      raises(ValidationError, C.AgentResponse, session_id="s", trace_id="t",
             turn_index=0, message="m", awaiting="whenever",
             session_status="running"))
check("a valid request is accepted",
      C.MessageRequest(message="haan", turn_index=3).turn_index == 3)
check("error body has no field for internals",
      set(C.ErrorResponse.model_fields) == {"error", "detail", "trace_id"},
      "an error body is an output channel; paths and traces must not fit in it")


# --------------------------------------------------------- serialisation
print("\nstate survives the wire")

st = SessionState(session_id="rt")
st.model, st.serial, st.active_code = "PC200-10M0", "700123", "CA451"
st.step_cursor, st.awaiting = 3, Awaiting.READING
st.use_facts(["CA451:3:meas:0"])
st.visit("CA451")
revived = SessionState.model_validate_json(st.model_dump_json())
check("round-trip is lossless", revived.model_dump() == st.model_dump())
check("round-trip preserves enum types", revived.awaiting is Awaiting.READING)
check("a stray field cannot be injected into state",
      raises(ValidationError, SessionState.model_validate,
             {**json.loads(st.model_dump_json()), "diagnosis_override": "x"}),
      "extra=forbid must hold on the rehydration path, not only at construction")


# ---------------------------------------------------------------- guard
print("\nungrounded-value guard")

cites = [{"verbatim_text": "Max. 1 Ω", "manual_page": "40-121"}]
ok, bad = check_message("It should read Max. 7 ohm.", citations=cites,
                        fact_ids=["X:1:meas:0"], code="CA451")
check("an invented value is caught", not ok and "7" in bad)
ok2, _ = check_message("It should read Max. 1 ohm.", citations=cites,
                       fact_ids=["X:1:meas:0"], code="CA451")
check("a cited value is allowed", ok2)
ok3, _ = check_message("Step 1: check the harness.", citations=[],
                       fact_ids=[], code="CA451", extra=["1"])
check("a structural step number is allowed", ok3)
ok4, bad4 = check_message("Measure 33 kPa.", citations=[], fact_ids=[],
                          code="CA451", grounded_text=["boost 33 kPa"])
check("a value quoted from ground truth is allowed", ok4, str(bad4))
sent, blocked, _ = enforce("Value is 99 V.", citations=[], fact_ids=[])
check("a blocked message is replaced, not truncated",
      blocked and "99" not in sent)


# ---------------------------------------------------------------- store
print("\nstore: order, duplication, caps, rate")

s = SessionStore(max_sessions=3, max_turns=4, rate_limit=3, window_s=1000.0)
rec = s.create("a", SessionState(session_id="a").model_dump_json())
s.check_turn(rec, 0)
s.commit(rec, rec.state_json, "hi", "there", "running", "t0", False)
check("an out-of-order turn is a conflict",
      raises(TurnConflict, s.check_turn, rec, 0),
      "a duplicate turn must not silently re-run")
check("the expected index is reported", rec.turn_index == 1)
s.check_turn(rec, 1)
s.commit(rec, rec.state_json, "b", "c", "running", "t1", False)
s.check_turn(rec, 2)
s.commit(rec, rec.state_json, "d", "e", "running", "t2", False)
check("rate limit fires", raises(RateLimited, s.check_turn, rec, 3))

s2 = SessionStore(max_sessions=2, max_turns=2, rate_limit=99, window_s=1000.0)
s2.create("x", SessionState(session_id="x").model_dump_json())
s2.create("y", SessionState(session_id="y").model_dump_json())
check("session cap fires",
      raises(SessionLimit, s2.create, "z",
             SessionState(session_id="z").model_dump_json()))

r2 = s2.get("x")
s2.check_turn(r2, 0)
s2.commit(r2, r2.state_json, "1", "2", "running", None, False)
s2.check_turn(r2, 1)
s2.commit(r2, r2.state_json, "3", "4", "running", None, False)
check("turn cap fires", raises(SessionLimit, s2.check_turn, r2, 2))
check("sessions do not share state",
      s2.get("x").turn_index == 2 and s2.get("y").turn_index == 0)


# -------------------------------------------------------------- metrics
print("\nmetrics")

m = Metrics()
m.inc("agent_turns_total")
m.inc("agent_turns_total")
m.inc("agent_gate_failures_total", gate="ungrounded_value")
for v in (10, 20, 30, 40, 50, 60, 70, 80, 90, 100):
    m.observe("execute_step", v)
text = m.render()
check("counters render", "agent_turns_total 2" in text)
check("labelled counters render",
      'agent_gate_failures_total{gate="ungrounded_value"} 1' in text)
check("percentiles render",
      'agent_node_latency_ms{node="execute_step",quantile="0.95"}' in text)
check("HELP lines present", "# HELP agent_turns_total" in text)


print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: module level")
sys.exit(0)
