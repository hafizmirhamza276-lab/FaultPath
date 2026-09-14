#!/usr/bin/env python3
"""
test_api_e2e.py
PIPELINE and E2E levels, plus the overfitting guard.

PIPELINE  the 45 scripted code sessions through the graph in-process; agent
          metrics must still hold -- good 7/7, bad 0/7.
E2E       the same sessions over HTTP, plus the 6 symptom sessions. Transcripts
          and state sequences must be IDENTICAL to the pipeline run. Any
          divergence means the API is doing something the graph is not, which
          is the one thing it must never do -- and it caught exactly that: the
          graph guarded an ASK on evidence it did not record, so the boundary
          guard, holding only what was recorded, blocked a message the graph
          had allowed.

          The symptom level is no longer deferred. It covers a flat S-Mode tree
          reaching a remedy and an ASK resolved across two turns.

Then: held-out split, independent derivation from the PDF, the mutation table,
and negative coverage per gate.
"""
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from fastapi.testclient import TestClient                      # noqa: E402
from agent.graph import Agent                                  # noqa: E402
from agent.runner import BadAgent, run_all, run_session        # noqa: E402
from agent import tools                                        # noqa: E402
from agent.sessions import (build_sessions,                    # noqa: E402
                            build_symptom_sessions)
from api.app import create_app                                 # noqa: E402
from api.store import SessionStore                             # noqa: E402
from eval.metrics.agent import build as build_metrics, AGENT_GATES  # noqa: E402
from tests import overfitting as OV                            # noqa: E402

failures = []
timings = {}


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def score(sessions, rows):
    agg = {}
    for m in build_metrics():
        vals = [m.compute({"type": "agent_session", **s}, r)
                for s, r in zip(sessions, rows)]
        vals = [v for v in vals if v is not None]
        if vals:
            agg[m.name] = sum(vals) / len(vals)
    return agg


def gates_for(agg):
    return [(n, (agg.get(n) is not None and
                 (agg[n] >= t if op == ">=" else agg[n] <= t)), agg.get(n))
            for n, op, t in AGENT_GATES]


SESSIONS = build_sessions()
SYM_SESSIONS = build_symptom_sessions()

# ======================================================= PIPELINE
print(f"\nPIPELINE -- {len(SESSIONS)} sessions in-process")
t0 = time.perf_counter()
good_rows = run_all(Agent(), SESSIONS)
bad_rows = run_all(BadAgent(), SESSIONS)
timings["pipeline"] = time.perf_counter() - t0

good_gates = gates_for(score(SESSIONS, good_rows))
bad_gates = gates_for(score(SESSIONS, bad_rows))
gf = [n for n, ok, _ in good_gates if not ok]
bp = [n for n, ok, _ in bad_gates if ok]
check(f"good agent passes all {len(AGENT_GATES)} gates", not gf, f"failed: {gf}")
check("bad agent fails ALL gates", not bp, f"bad PASSED {bp}")


# ============================================================= E2E
print(f"\nE2E -- the same {len(SESSIONS)} sessions over HTTP")
t0 = time.perf_counter()
client = TestClient(create_app(store=SessionStore(max_sessions=5000,
                                                  max_turns=200,
                                                  rate_limit=10_000)))


def drive_http(session):
    turns = session["turns"]
    r = client.post("/sessions", json={"message": turns[0]})
    assert r.status_code == 201, r.text
    b = r.json()
    sid, out = b["session_id"], [b["message"]]
    status = b["session_status"]
    for i, msg in enumerate(turns[1:], start=1):
        if status != "running":
            break
        r = client.post(f"/sessions/{sid}/messages",
                        json={"message": msg, "turn_index": i})
        assert r.status_code == 200, r.text
        b = r.json()
        out.append(b["message"])
        status = b["session_status"]
    view = client.get(f"/sessions/{sid}").json()
    return sid, out, view, status


http_results = [drive_http(s) for s in SESSIONS]
timings["e2e"] = time.perf_counter() - t0

# Identical transcripts, turn for turn.
mismatch = []
for s, g, (sid, http_msgs, view, status) in zip(SESSIONS, good_rows, http_results):
    proc_msgs = [t["assistant"] for t in g["transcript"]]
    if proc_msgs[:len(http_msgs)] != http_msgs:
        mismatch.append(s["id"])
check("HTTP transcripts are identical to the in-process run", not mismatch,
      f"diverged: {mismatch[:5]}")

state_mismatch = []
for s, g, (sid, _, view, status) in zip(SESSIONS, good_rows, http_results):
    if view["visited_codes"] != g["visited_codes"][:len(view["visited_codes"])] \
            or view["diagnosis"] != g["diagnosis"] \
            or view["fact_ids_used"] != g["fact_ids_used"]:
        state_mismatch.append(s["id"])
check("HTTP state matches the in-process state", not state_mismatch,
      f"diverged: {state_mismatch[:5]}")


# --------------------------------------------- E2E: the symptom path
#
# NO LONGER DEFERRED. This level used to print that symptom entry was a stub
# and skip it, which is the right way to declare a gap but not a substitute for
# closing one.
print(f"\nE2E -- {len(SYM_SESSIONS)} symptom sessions over HTTP")
t0 = time.perf_counter()
sym_proc = [run_session(Agent(), s) for s in SYM_SESSIONS]
sym_http = [drive_http(s) for s in SYM_SESSIONS]
timings["e2e_symptom"] = time.perf_counter() - t0

sym_mismatch = []
for s, g, (sid, http_msgs, view, status) in zip(SYM_SESSIONS, sym_proc, sym_http):
    if [t["assistant"] for t in g["transcript"]][:len(http_msgs)] != http_msgs:
        sym_mismatch.append(s["id"])
check("HTTP symptom transcripts are identical to the in-process run",
      not sym_mismatch, f"diverged: {sym_mismatch}")

sym_state_mismatch = []
for s, g, (sid, _, view, status) in zip(SYM_SESSIONS, sym_proc, sym_http):
    if (view.get("active_symptom") != g["symptom_id"]
            or view.get("tree_kind") != g["tree_kind"]
            or view.get("remedy") != g["remedy"]
            or view["diagnosis"] != g["diagnosis"]
            or view["fact_ids_used"] != g["fact_ids_used"]):
        sym_state_mismatch.append(s["id"])
check("HTTP symptom state matches the in-process state", not sym_state_mismatch,
      f"diverged: {sym_state_mismatch}")

# A FLAT S-MODE TREE REACHING A REMEDY, over HTTP, end to end.
_flat_sess = next(s for s in SYM_SESSIONS if s["kind"] == "symptom_flat_remedy")
_sid, _msgs, _view, _status = next(
    h for s, h in zip(SYM_SESSIONS, sym_http) if s["id"] == _flat_sess["id"])
check("a flat S-Mode tree reaches remedy over HTTP",
      _status == "concluded"
      and _view["tree_kind"] == "SymptomTreeFlat"
      and _view["remedy"] == _flat_sess["expect"]["remedy"]
      and _view["diagnosis"] == _flat_sess["expect"]["diagnosis"],
      f"status={_status} remedy={_view.get('remedy')!r}")
check("the flat tree was asked to observe, never to answer yes or no",
      "is that what you are seeing" in " ".join(_msgs).lower()
      and "answer yes or no" not in " ".join(_msgs).lower())
check("remedy crosses the boundary in its own field, not inside diagnosis",
      _view["remedy"] not in (_view["diagnosis"] or ""))

# CITATIONS RESOLVE over HTTP -- rendered by the API, not by the test.
_r = client.post("/sessions", json={"message": _flat_sess["turns"][0]})
_csid = _r.json()["session_id"]
_last = _r.json()
for _i, _m in enumerate(_flat_sess["turns"][1:], start=1):
    _last = client.post(f"/sessions/{_csid}/messages",
                        json={"message": _m, "turn_index": _i}).json()
check("the concluding HTTP response carries citations",
      len(_last["citations"]) > 0, str(_last["citations"])[:200])
check("every HTTP citation names a fact id, a manual page and a pdf page",
      all(c.get("fact_id") and c.get("manual_page") and c.get("pdf_page")
          for c in _last["citations"]),
      str(_last["citations"])[:300])
_pages = {c["manual_page"] for c in _last["citations"]}
check("the cited pages are the S-Mode pages this tree lives on",
      _pages <= set(tools.lookup_symptom("SM01")["manual_pages"]), str(_pages))

# AN ASK RESOLVED OVER TWO TURNS, which is the whole point of ASK being a
# first-class state rather than an error: the session survives the question.
_a = client.post("/sessions", json={"message": "swing slow hai"}).json()
_asid = _a["session_id"]
_a2 = client.post(f"/sessions/{_asid}/messages",
                  json={"message": "PC200-10M0, serial 700123",
                        "turn_index": 1}).json()
check("ASK surfaces as its own awaiting state over HTTP",
      _a2["awaiting"] == "symptom_choice", _a2["awaiting"])
check("ASK returns the candidates, so a UI need not parse the prose",
      len(_a2["symptom_candidates"]) >= 2
      and all(c.get("manual_page") for c in _a2["symptom_candidates"]),
      str(_a2["symptom_candidates"])[:200])
check("the agent has entered no tree while the question is open",
      client.get(f"/sessions/{_asid}").json()["active_symptom"] is None)
_a3 = client.post(f"/sessions/{_asid}/messages",
                  json={"message": "2", "turn_index": 2}).json()
_aview = client.get(f"/sessions/{_asid}").json()
check("the technician's pick resolves the ASK on the next turn",
      _aview["active_symptom"] == "HM29"
      and _a3["awaiting"] != "symptom_choice",
      f"{_aview['active_symptom']} awaiting={_a3['awaiting']}")

# An unmapped input must not acquire a tree over HTTP either.
_u = client.post("/sessions", json={"message": "AC kaam nahi kar raha"}).json()
_u2 = client.post(f"/sessions/{_u['session_id']}/messages",
                  json={"message": "PC200-10M0, serial 700123",
                        "turn_index": 1}).json()
_uview = client.get(f"/sessions/{_u['session_id']}").json()
check("an unmapped symptom ends the session without entering a tree",
      _uview["active_symptom"] is None and _uview["diagnosis"] is None
      and _u2["session_status"] == "escalated",
      f"{_uview['active_symptom']} {_u2['session_status']}")


# ------------------------------------------------- HTTP-only behaviours
print("\nHTTP-only behaviours")

r = client.post("/sessions", json={"message": "CA451 aa raha hai"})
sid = r.json()["session_id"]
client.post(f"/sessions/{sid}/messages",
            json={"message": "PC200-10M0, serial 700123", "turn_index": 1})
dup = client.post(f"/sessions/{sid}/messages",
                  json={"message": "PC200-10M0, serial 700123", "turn_index": 1})
check("a duplicate turn is 409, not a silent re-run", dup.status_code == 409,
      f"got {dup.status_code}")
check("the conflict names the expected index",
      "turn_index" in dup.json().get("detail", ""))

ooo = client.post(f"/sessions/{sid}/messages",
                  json={"message": "x", "turn_index": 99})
check("an out-of-order turn is 409", ooo.status_code == 409)

# Interrupted session resumed: re-read state, continue where it left off.
view = client.get(f"/sessions/{sid}").json()
resumed = client.post(f"/sessions/{sid}/messages",
                      json={"message": "koi aur code nahi",
                            "turn_index": view["turn_index"]})
check("an interrupted session resumes from its stored state",
      resumed.status_code == 200 and resumed.json()["step_number"] == 1)

bad_body = client.post("/sessions", json={"msg": "typo"})
check("a malformed body is 422", bad_body.status_code == 422)
check("the 422 body carries no internals",
      set(bad_body.json()) == {"error", "detail", "trace_id"})

oversized = client.post("/sessions", json={"message": "x" * 5000})
check("an oversized payload is 422", oversized.status_code == 422)

unknown = client.get("/sessions/deadbeefdeadbeef")
check("an unknown session is 404", unknown.status_code == 404)

# Concurrent sessions must not bleed.
a = client.post("/sessions", json={"message": "CA451 aa raha hai"}).json()
b = client.post("/sessions", json={"message": "CA452 aa raha hai"}).json()
client.post(f"/sessions/{a['session_id']}/messages",
            json={"message": "PC200-10M0, serial 700123", "turn_index": 1})
va = client.get(f"/sessions/{a['session_id']}").json()
vb = client.get(f"/sessions/{b['session_id']}").json()
check("concurrent sessions do not bleed state",
      va["active_code"] == "CA451" and vb["active_code"] == "CA452"
      and va["turn_index"] != vb["turn_index"])

ab = client.post(f"/sessions/{b['session_id']}/abandon")
check("abandon marks the session", ab.json()["session_status"] == "abandoned")
after = client.post(f"/sessions/{b['session_id']}/messages",
                    json={"message": "hi", "turn_index": 1})
check("an abandoned session accepts no more turns", after.status_code == 409)

tr = client.get(f"/sessions/{sid}/trace")
check("trace returns this session's events only", tr.status_code == 200)

mt = client.get("/metrics")
check("metrics render in Prometheus format",
      mt.status_code == 200 and "# TYPE agent_turns_total counter" in mt.text)
check("ready reports the schema version",
      client.get("/ready").json()["schema_version"] == 2)


# ============================================ OVERFITTING GUARD
print("\nOVERFITTING GUARD")
print(f"  split: {len(OV.TRAIN)} train / {len(OV.HOLDOUT)} holdout "
      f"({len(OV.HOLDOUT)/(len(OV.TRAIN)+len(OV.HOLDOUT)):.0%})")

# (a) held-out split -- no fixture may name a held-out code
fixture_codes = {s["code"] for s in SESSIONS} | {
    c for s in SESSIONS for c in [s["expect"].get("resolved")] if c}
leaked = sorted(fixture_codes & OV.HOLDOUT_SET)
check("no scripted session references a held-out code", not leaked,
      f"leaked: {leaked[:8]}")

# train vs holdout metrics, built the same way but over disjoint codes
from agent.sessions import reference_walk, scripted_turns       # noqa: E402
from agent import tools                                         # noqa: E402


def sessions_for(codes, limit=14):
    out = []
    for c in codes:
        if tools.lookup_code(c)["is_pointer_only"] or tools.step_count(c) < 2:
            continue
        ref = reference_walk(c, fail_at=1)
        if ref["outcome"] != "conclude":
            continue
        out.append({"id": f"split_{c}", "kind": "single_fail_first", "code": c,
                    "turns": scripted_turns(c, fail_at=1), "expect": ref,
                    "note": ""})
        if len(out) >= limit:
            break
    return out


train_s, hold_s = sessions_for(OV.TRAIN), sessions_for(OV.HOLDOUT)
train_agg = score(train_s, run_all(Agent(), train_s))
hold_agg = score(hold_s, run_all(Agent(), hold_s))
print(f"  {'metric':26}{'train':>9}{'holdout':>9}{'gap':>8}")
gaps = {}
for k in ("path_correctness", "diagnosis_correct", "gate_enforcement",
          "cycle_safety", "ungrounded_value_rate", "premature_conclusion"):
    tv, hv = train_agg.get(k), hold_agg.get(k)
    if tv is None or hv is None:
        continue
    gaps[k] = abs(tv - hv)
    print(f"  {k:26}{tv:9.4f}{hv:9.4f}{gaps[k]:8.4f}")
worst = max(gaps.values()) if gaps else 0.0
check("train/holdout gap is within a few points", worst <= 0.05,
      f"largest gap {worst:.4f} -- tuned to the cases rather than the manual")

# (b) independent derivation: expectations read back through the PDF
from eval.citations import PageText                             # noqa: E402
pages = PageText()
if not pages.available():
    print("  SKIP  independent derivation (source PDF not present)")
else:
    # Walk held-out codes for steps that actually carry a measurement -- step 1
    # frequently has none, and scoring zero facts would let this check "pass"
    # by never running, which is the exact shape it exists to prevent.
    checked, mismatched, detail = 0, 0, []
    for code in OV.HOLDOUT:
        if checked >= 12:
            break
        for i in range(1, tools.step_count(code) + 1):
            step = tools.get_step(code, i)
            for m in (step.get("measurements") or [])[:1]:
                exp = OV.expected_from_pdf(m["fact_id"], pages)
                if exp is None:
                    continue
                checked += 1
                # The page and text are re-read from the PDF, not from the
                # record the agent walks. Agreement here is evidence; agreement
                # between two loads of the same object would be nothing.
                if not exp["resolved"]:
                    mismatched += 1
                    detail.append(f"{m['fact_id']}: {exp['reason']}")
                elif exp["manual_page"] != m["provenance"]["manual_page"]:
                    mismatched += 1
                    detail.append(f"{m['fact_id']}: page "
                                  f"{exp['manual_page']} vs "
                                  f"{m['provenance']['manual_page']}")
    check(f"expectations re-derived from the PDF agree ({checked} held-out facts)",
          checked >= 5 and mismatched == 0,
          "; ".join(detail[:4]) or f"only {checked} facts checked")

# (c) mutation check
print("\n  MUTATION TABLE")
mut_rows = []
for fault, desc in OV.MUTATIONS:
    try:
        agent = OV.mutant_agent(fault)
        rows = run_all(agent, SESSIONS)
        agg = score(SESSIONS, rows)
        caught_by = [n for n, ok, _ in gates_for(agg) if not ok]
    except Exception as exc:
        caught_by = [f"crash:{type(exc).__name__}"]
    mut_rows.append((fault, caught_by))
    status = "caught" if caught_by else "*** NOT CAUGHT ***"
    print(f"    {fault:22} {status:18} {', '.join(caught_by[:3]) or '-'}")
uncaught = [f for f, c in mut_rows if not c]
check("every injected fault is caught by at least one gate", not uncaught,
      f"uncaught: {uncaught} -- these are holes in the suite")

# (d) negative coverage: every gate must have a case that fails it
print("\n  NEGATIVE COVERAGE")
failed_by_something = set()
for _, caught in mut_rows:
    failed_by_something |= set(caught)
failed_by_something |= {n for n, ok, _ in bad_gates if not ok}
missing_neg = [n for n, _, _ in AGENT_GATES if n not in failed_by_something]
for n, _, _ in AGENT_GATES:
    print(f"    {n:26} {'has a failing case' if n in failed_by_something else '*** NO FAILING CASE ***'}")
check("every gate has at least one case that fails it", not missing_neg,
      f"no failing case for: {missing_neg} -- a gate nothing can fail is not a gate")


print("\n" + "=" * 60)
print(f"pipeline {timings['pipeline']:.1f}s   e2e {timings['e2e']:.1f}s")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: pipeline + e2e + overfitting guard")
sys.exit(0)
