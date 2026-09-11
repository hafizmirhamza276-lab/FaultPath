#!/usr/bin/env python3
"""
store.py
Session storage, rate limiting and caps. No agent behaviour lives here.

State is held as JSON and rehydrated through SessionState on every turn, not
kept as a live object. That is deliberate: it is the same round-trip the replay
test already proves, so the wire path and the in-process path cannot drift.
Keeping a live object would make the API faster and the guarantee weaker.
"""
from __future__ import annotations

import json
import threading
import time
from typing import Dict, List, Optional, Tuple

MAX_SESSIONS = 500
MAX_TURNS_PER_SESSION = 80
RATE_LIMIT_TURNS = 30           # per session
RATE_LIMIT_WINDOW_S = 60.0
SESSION_TTL_S = 60 * 60 * 4


class RateLimited(Exception):
    pass


class TurnConflict(Exception):
    """Client sent a turn index that does not match the session."""

    def __init__(self, expected: int, got: int):
        self.expected, self.got = expected, got
        super().__init__(f"expected turn_index {expected}, got {got}")


class SessionLimit(Exception):
    pass


class Record:
    __slots__ = ("session_id", "state_json", "transcript", "turn_index",
                 "status", "created", "updated", "hits", "trace_ids")

    def __init__(self, session_id: str, state_json: str):
        now = time.time()
        self.session_id = session_id
        self.state_json = state_json
        self.transcript: List[dict] = []
        self.turn_index = 0
        self.status = "running"
        self.created = now
        self.updated = now
        self.hits: List[float] = []
        self.trace_ids: List[str] = []


class SessionStore:
    """In-memory, thread-safe. A real deployment swaps this for Redis; the
    interface is deliberately small enough that the swap touches nothing else."""

    def __init__(self, max_sessions: int = MAX_SESSIONS,
                 max_turns: int = MAX_TURNS_PER_SESSION,
                 rate_limit: int = RATE_LIMIT_TURNS,
                 window_s: float = RATE_LIMIT_WINDOW_S,
                 clock=time.time):
        self._lock = threading.RLock()
        self._sessions: Dict[str, Record] = {}
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.rate_limit = rate_limit
        self.window_s = window_s
        self._clock = clock

    # -- lifecycle -------------------------------------------------------
    def create(self, session_id: str, state_json: str) -> Record:
        with self._lock:
            self._evict_expired()
            if len(self._sessions) >= self.max_sessions:
                raise SessionLimit(
                    f"server is holding its maximum of {self.max_sessions} "
                    "concurrent sessions")
            rec = Record(session_id, state_json)
            self._sessions[session_id] = rec
            return rec

    def get(self, session_id: str) -> Optional[Record]:
        with self._lock:
            rec = self._sessions.get(session_id)
            if rec and self._clock() - rec.updated > SESSION_TTL_S:
                del self._sessions[session_id]
                return None
            return rec

    def abandon(self, session_id: str) -> Optional[Record]:
        with self._lock:
            rec = self._sessions.get(session_id)
            if rec:
                rec.status = "abandoned"
                rec.updated = self._clock()
            return rec

    # -- per-turn checks -------------------------------------------------
    def check_turn(self, rec: Record, turn_index: int) -> None:
        """Order, duplication, caps and rate, in that order.

        Turn order is checked before the rate limit so a client that
        accidentally replays an old turn gets told what is actually wrong
        rather than being throttled for it.
        """
        with self._lock:
            if turn_index != rec.turn_index:
                raise TurnConflict(rec.turn_index, turn_index)
            if rec.turn_index >= self.max_turns:
                raise SessionLimit(
                    f"session reached its maximum of {self.max_turns} turns")
            now = self._clock()
            rec.hits = [t for t in rec.hits if now - t < self.window_s]
            if len(rec.hits) >= self.rate_limit:
                raise RateLimited(
                    f"more than {self.rate_limit} turns in "
                    f"{int(self.window_s)}s for this session")
            rec.hits.append(now)

    def commit(self, rec: Record, state_json: str, technician: str,
               assistant: str, status: str, trace_id: Optional[str],
               blocked: bool) -> int:
        with self._lock:
            rec.state_json = state_json
            rec.transcript.append({
                "turn_index": rec.turn_index,
                "trace_id": trace_id,
                "technician": technician,
                "assistant": assistant,
                "blocked": blocked,
            })
            if trace_id:
                rec.trace_ids.append(trace_id)
            rec.turn_index += 1
            rec.status = status
            rec.updated = self._clock()
            return rec.turn_index

    # -- housekeeping ----------------------------------------------------
    def _evict_expired(self) -> None:
        now = self._clock()
        stale = [k for k, r in self._sessions.items()
                 if now - r.updated > SESSION_TTL_S]
        for k in stale:
            del self._sessions[k]

    def stats(self) -> Tuple[int, int]:
        with self._lock:
            return (len(self._sessions),
                    sum(r.turn_index for r in self._sessions.values()))

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
