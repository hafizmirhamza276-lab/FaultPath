#!/usr/bin/env python3
"""
app.py
Thin transport over the diagnostic graph.

The API must not be able to change agent behaviour. No route makes a diagnostic
decision -- if a decision is needed it belongs in agent/graph.py. Handlers do
three things: validate the request, call turn(), serialise the result.

State crosses every turn as JSON and is rehydrated through SessionState. The
replay test already proves that round-trip is lossless, so the wire path and the
in-process path cannot diverge -- and the E2E suite asserts that transcripts and
state sequences match the in-process run exactly.

The ungrounded-value guard runs AGAIN here, at the boundary, after the graph has
already run it inside emit(). Not redundancy for its own sake: emit() protects
the agent's own composition, and this protects the process boundary. A value
must not be able to leave, whatever path produced it.
"""
from __future__ import annotations

import os
import sys
import time
import uuid
from typing import Optional

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import tools                                          # noqa: E402
from agent.graph import Agent                                    # noqa: E402
from agent.guards import check_message                           # noqa: E402
from agent.llm import MockLLM                                    # noqa: E402
from agent.state import SessionState, Awaiting, Outcome          # noqa: E402
from api import contracts as C                                   # noqa: E402
from api.metrics import METRICS                                  # noqa: E402
from api.store import (SessionStore, RateLimited, TurnConflict,  # noqa: E402
                       SessionLimit)
from core.run_log import RunLogger, read_events                  # noqa: E402

LOGS_DIR = os.path.join(REPO_ROOT, "eval_out", "logs")

_AWAIT_MAP = {
    Awaiting.MACHINE: "machine",
    Awaiting.OTHER_CODES: "other_codes",
    Awaiting.READING: "reading",
    Awaiting.ENTRY: "none",
    Awaiting.NOTHING: "none",
}
_STATUS_MAP = {Outcome.RUNNING: "running", Outcome.CONCLUDED: "concluded",
               Outcome.ESCALATED: "escalated"}


def create_app(agent: Optional[Agent] = None,
               store: Optional[SessionStore] = None,
               logger: Optional[RunLogger] = None,
               quiet: bool = True) -> FastAPI:
    """Explicit dependencies so tests drive the same app the server runs."""
    log = logger or RunLogger(log_dir=LOGS_DIR, quiet=quiet)
    ag = agent or Agent(llm=MockLLM(), logger=log)
    st_store = store or SessionStore()

    app = FastAPI(title="Komatsu Diagnostic Agent", version="1.0.0",
                  docs_url="/docs", redoc_url=None)
    app.state.agent = ag
    app.state.store = st_store
    app.state.log = log

    # ---------------------------------------------------------- errors
    def _err(status: int, error: str, detail: str, trace_id=None) -> JSONResponse:
        METRICS.inc("agent_errors_total", **{"class": error})
        return JSONResponse(
            status_code=status,
            content=C.ErrorResponse(error=error, detail=detail,
                                    trace_id=trace_id).model_dump())

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError):
        # Field names and reasons only. Never the submitted values, never a
        # path, never a traceback: an error body is an output channel too.
        fields = []
        for e in exc.errors()[:8]:
            loc = ".".join(str(p) for p in e.get("loc", []) if p != "body")
            fields.append(f"{loc or 'body'}: {e.get('type', 'invalid')}")
        return _err(422, "invalid_request", "; ".join(fields) or "invalid body")

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        # The class name is logged; the client is told nothing about internals.
        log.event("decision", "unhandled_error",
                  {"type": type(exc).__name__}, level="error")
        return _err(500, "internal_error",
                    "The request could not be completed.")

    # ---------------------------------------------------------- helpers
    def _load(rec) -> SessionState:
        return SessionState.model_validate_json(rec.state_json)

    def _respond(state: SessionState, rec, message: str, trace_id: str,
                 turn_index: int, blocked: bool) -> C.AgentResponse:
        total = tools.step_count(state.active_code or "") or None
        last = state.emissions[-1] if state.emissions else None
        cites = [C.Citation(**c) for c in (last.citations if last else [])]
        return C.AgentResponse(
            session_id=state.session_id,
            trace_id=trace_id,
            turn_index=turn_index,
            message=message,
            awaiting=_AWAIT_MAP.get(state.awaiting, "none"),
            citations=cites,
            fact_ids=list(state.fact_ids_used),
            step_number=state.step_cursor or None,
            total_steps=total,
            diagnosis=state.diagnosis,
            session_status=("abandoned" if rec.status == "abandoned"
                            else _STATUS_MAP.get(state.outcome, "running")),
            blocked=blocked,
        )

    BLOCK_TEXT = ("I cannot give you that value from what I have verified. "
                  "Let me re-check the manual entry before I answer.")

    def _boundary_guard(state: SessionState, emissions, technician: str) -> tuple:
        """Last check before anything leaves the process.

        Runs per emission, on the same evidence the graph used to compose it.
        Independent, not stricter: a boundary holding less evidence would block
        correctly-grounded text and make the API diverge from the graph.

        emit() protects the agent's composition; this protects egress. It is the
        check nothing runs after, so it is the one that decides what a
        technician actually sees.
        """
        parts, blocked_any = [], False
        for e in emissions:
            if e.blocked:
                parts.append(e.text)          # already replaced upstream
                blocked_any = True
                continue
            ok, offending = check_message(
                e.text, citations=e.citations, fact_ids=e.fact_ids,
                code=state.active_code, model=state.model, serial=state.serial,
                technician_text=technician,
                extra=[str(state.step_cursor)],
                grounded_text=list(e.grounded_text))
            if ok:
                parts.append(e.text)
                continue
            blocked_any = True
            parts.append(BLOCK_TEXT)
            METRICS.inc("agent_blocked_messages_total")
            METRICS.inc("agent_gate_failures_total", gate="ungrounded_value")
            log.event("validation", "boundary_block",
                      {"session_id": state.session_id, "node": e.node,
                       "offending": offending, "original_text": e.text},
                      level="error")
        return "\n".join(parts), blocked_any

    def _turn(state: SessionState, rec, text: str):
        before = len(state.emissions)
        t0 = time.perf_counter()
        METRICS.inc("agent_llm_calls_total", role="turn")
        state, _ = ag.turn(state, text)
        dt = (time.perf_counter() - t0) * 1000.0
        METRICS.observe("turn", dt)
        METRICS.inc("agent_llm_latency_ms_total", dt)
        tail = state.node_sequence[-6:]
        for node in tail:
            METRICS.observe(node, dt / max(1, len(tail)))
        message, blocked = _boundary_guard(state, state.emissions[before:], text)
        return state, message, blocked

    # ---------------------------------------------------------- routes
    @app.post("/sessions", response_model=C.AgentResponse, status_code=201)
    async def start_session(body: C.StartSessionRequest):
        sid = uuid.uuid4().hex[:16]
        try:
            rec = st_store.create(sid, ag.start(sid).model_dump_json())
        except SessionLimit as exc:
            return _err(429, "session_limit", str(exc))
        METRICS.inc("agent_sessions_total")
        state = _load(rec)
        state, message, blocked = _turn(state, rec, body.message)
        idx = rec.turn_index
        st_store.commit(rec, state.model_dump_json(), body.message, message,
                        _STATUS_MAP.get(state.outcome, "running"),
                        state.trace_id, blocked)
        return _respond(state, rec, message, state.trace_id or "", idx, blocked)

    @app.post("/sessions/{session_id}/messages", response_model=C.AgentResponse)
    async def post_message(session_id: str, body: C.MessageRequest):
        rec = st_store.get(session_id)
        if rec is None:
            return _err(404, "not_found", "No such session.")
        if rec.status == "abandoned":
            return _err(409, "session_abandoned",
                        "This session was abandoned and accepts no more turns.")
        if rec.status in ("concluded", "escalated"):
            return _err(409, "session_closed",
                        f"This session is {rec.status} and accepts no more turns.")
        try:
            st_store.check_turn(rec, body.turn_index)
        except TurnConflict as exc:
            # Not a silent re-run: replaying a turn against a mutated state
            # would give the same conversation two different transcripts.
            return _err(409, "turn_conflict",
                        f"Out-of-order or duplicate turn. Expected turn_index "
                        f"{exc.expected}.")
        except RateLimited as exc:
            return _err(429, "rate_limited", str(exc))
        except SessionLimit as exc:
            return _err(429, "session_limit", str(exc))

        METRICS.inc("agent_turns_total")
        state = _load(rec)
        state, message, blocked = _turn(state, rec, body.message)
        idx = rec.turn_index
        st_store.commit(rec, state.model_dump_json(), body.message, message,
                        _STATUS_MAP.get(state.outcome, "running"),
                        state.trace_id, blocked)
        return _respond(state, rec, message, state.trace_id or "", idx, blocked)

    @app.get("/sessions/{session_id}", response_model=C.SessionView)
    async def get_session(session_id: str):
        rec = st_store.get(session_id)
        if rec is None:
            return _err(404, "not_found", "No such session.")
        state = _load(rec)
        return C.SessionView(
            session_id=session_id,
            session_status=("abandoned" if rec.status == "abandoned"
                            else _STATUS_MAP.get(state.outcome, "running")),
            model=state.model, serial=state.serial, manual_id=state.manual_id,
            active_code=state.active_code, entry_mode=state.entry_mode.value,
            awaiting=_AWAIT_MAP.get(state.awaiting, "none"),
            step_number=state.step_cursor,
            total_steps=tools.step_count(state.active_code or "") or None,
            visited_codes=list(state.visited_codes),
            pending_codes=list(state.pending_codes),
            fact_ids_used=list(state.fact_ids_used),
            diagnosis=state.diagnosis,
            turn_index=rec.turn_index,
            transcript=[C.TurnRecord(**t) for t in rec.transcript])

    @app.get("/sessions/{session_id}/trace", response_model=C.TraceView)
    async def get_trace(session_id: str):
        rec = st_store.get(session_id)
        if rec is None:
            return _err(404, "not_found", "No such session.")
        # The session is usually still open, so the sink's buffer holds the
        # most recent events -- exactly the ones being asked for.
        log.flush()
        wanted = set(rec.trace_ids)
        events = [e for e in read_events(LOGS_DIR, log.run_id)
                  if e.get("trace_id") in wanted]
        return C.TraceView(session_id=session_id,
                           events=[C.TraceEvent(**e) for e in events])

    @app.post("/sessions/{session_id}/abandon", response_model=C.SessionView)
    async def abandon(session_id: str):
        rec = st_store.abandon(session_id)
        if rec is None:
            return _err(404, "not_found", "No such session.")
        return await get_session(session_id)

    @app.get("/health", response_model=C.HealthResponse)
    async def health():
        return C.HealthResponse(status="ok")

    @app.get("/ready", response_model=C.ReadyResponse)
    async def ready():
        """Ready means the ground truth is loaded AND is the shape this build
        was written against. A process that answers on a schema it does not
        understand is worse than one that refuses traffic."""
        try:
            recs = tools.records()
            ver = next(iter(recs.values())).get("schema_version")
            return C.ReadyResponse(
                status="ready", codes_loaded=len(recs), schema_version=ver,
                expected_schema_version=tools.EXPECTED_SCHEMA)
        except Exception as exc:
            return JSONResponse(status_code=503, content=C.ReadyResponse(
                status="not_ready", codes_loaded=0, schema_version=None,
                expected_schema_version=tools.EXPECTED_SCHEMA,
                detail=type(exc).__name__).model_dump())

    @app.get("/metrics")
    async def metrics():
        active, turns = st_store.stats()
        METRICS.counters["agent_sessions_active"] = active
        return PlainTextResponse(METRICS.render(),
                                 media_type="text/plain; version=0.0.4")

    return app


app = create_app()
