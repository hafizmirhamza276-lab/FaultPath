#!/usr/bin/env python3
"""
contracts.py
The wire format. Pydantic v2, strict, no free-form dicts.

Every request model forbids unknown fields. A typo in a client payload is a 422
rather than a silently ignored field, because a silently ignored field is how a
client ends up believing it sent a model filter it did not send.

Nothing here contains logic. These are shapes; the graph owns behaviour.
"""
from __future__ import annotations

from typing import List, Optional, Literal

from pydantic import BaseModel, Field, ConfigDict

STRICT = ConfigDict(extra="forbid", str_strip_whitespace=True)

MAX_MESSAGE_CHARS = 2000


class Citation(BaseModel):
    """Rendered by code from golden/. Never typed by a model."""
    model_config = STRICT
    fact_id: str
    manual_id: str
    revision: int
    model: str
    section: str
    manual_page: Optional[str] = None
    pdf_page: Optional[int] = None
    verbatim_text: str
    # How far this fact has been checked against the source document.
    # "unverified" is surfaced, never hidden: a caller must be able to tell a
    # confirmed value from one nobody has looked at.
    verification: Literal["human_verified", "resolver_verified",
                          "unverified"] = "unverified"


class StartSessionRequest(BaseModel):
    model_config = STRICT
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)


class MessageRequest(BaseModel):
    model_config = STRICT
    message: str = Field(min_length=1, max_length=MAX_MESSAGE_CHARS)
    # The index of the turn this message answers. A mismatch is a 409, not a
    # silent re-run: replaying a turn against a mutated state would produce a
    # different transcript for the same conversation, and the client would have
    # no way to know which one is real.
    turn_index: int = Field(ge=0)


class AgentResponse(BaseModel):
    model_config = STRICT
    session_id: str
    trace_id: str
    turn_index: int
    message: str
    awaiting: Literal["machine", "other_codes", "reading", "none"]
    citations: List[Citation] = Field(default_factory=list)
    fact_ids: List[str] = Field(default_factory=list)
    step_number: Optional[int] = None
    total_steps: Optional[int] = None
    diagnosis: Optional[str] = None
    session_status: Literal["running", "concluded", "escalated", "abandoned"]
    blocked: bool = False


class TurnRecord(BaseModel):
    model_config = STRICT
    turn_index: int
    trace_id: Optional[str] = None
    technician: str
    assistant: str
    blocked: bool = False


class SessionView(BaseModel):
    model_config = STRICT
    session_id: str
    session_status: Literal["running", "concluded", "escalated", "abandoned"]
    model: Optional[str] = None
    serial: Optional[str] = None
    manual_id: str
    active_code: Optional[str] = None
    entry_mode: str
    awaiting: Literal["machine", "other_codes", "reading", "none"]
    step_number: int
    total_steps: Optional[int] = None
    visited_codes: List[str] = Field(default_factory=list)
    pending_codes: List[str] = Field(default_factory=list)
    fact_ids_used: List[str] = Field(default_factory=list)
    diagnosis: Optional[str] = None
    turn_index: int
    transcript: List[TurnRecord] = Field(default_factory=list)


class TraceEvent(BaseModel):
    model_config = STRICT
    run_id: str
    trace_id: Optional[str] = None
    ts: float
    stage: str
    event: str
    duration_ms: Optional[float] = None
    level: str = "info"
    payload: dict = Field(default_factory=dict)


class TraceView(BaseModel):
    model_config = STRICT
    session_id: str
    events: List[TraceEvent] = Field(default_factory=list)


class HealthResponse(BaseModel):
    model_config = STRICT
    status: Literal["ok"]
    service: str = "komatsu-diagnostic-agent"


class ReadyResponse(BaseModel):
    model_config = STRICT
    status: Literal["ready", "not_ready"]
    codes_loaded: int
    schema_version: Optional[int] = None
    expected_schema_version: int
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    """Errors say what the client did wrong and nothing about this process.

    No paths, no stack traces, no prompt text. An error body is an output
    channel like any other, and it is the one people forget to guard.
    """
    model_config = STRICT
    error: str
    detail: str
    trace_id: Optional[str] = None
