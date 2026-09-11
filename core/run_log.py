#!/usr/bin/env python3
"""
run_log.py
Structured, replayable observability.

NAMED run_log.py, NOT logging.py, AND IT MATTERS. Python resolves the directory
of the executed script first, so `python core/orchestrator.py` puts core/ at
sys.path[0]. A module here called logging.py becomes THE logging module for that
whole process -- including for third-party libraries that import the standard
one. pdfplumber does, and died on `logging.getLogger` with:

    AttributeError: module 'logging' has no attribute 'getLogger'

That was latent for as long as every orchestrator stage shelled out to a
subprocess; the first stage to import pdfplumber in-process hit it immediately.
No compatibility shim was left behind: a shim keeps the shadowing filename on
disk, and the filename is the defect. Do not rename this back, and do not add a
core/logging.py beside it.

The bar: log enough to reconstruct any answer WITHOUT re-running it. Query as
received and as normalised, filters applied, every chunk returned with score and
rank, which chunks entered the context, which fact_ids the answer used, every
citation rendered, and every gate result. If a number is questioned six months
from now, the log is the answer -- not a re-run against a repo that has moved.

Two sinks:
  terminal  human, colourised, live. Respects --quiet and --verbose.
  central   eval_out/logs/<run_id>.jsonl -- one event per line
            eval_out/logs/index.jsonl    -- append-only, one summary row per run

index.jsonl is what turns this from a snapshot into a record. Six months of rows
showing hallucination rate falling is a far stronger argument than any single
number.

TIMINGS ARE LOGGED SEPARATELY FROM METRICS, deliberately. Wall time varies run
to run; Tier-1 metrics must not. Mixing them would make a determinism check fail
for a reason that has nothing to do with determinism.

No LLM. This module writes lines; it never interprets them.
"""
import json
import os
import sys
import time
import uuid

STAGES = ("retrieval", "rerank", "generation", "citation", "validation", "decision")

_COLOUR = {
    "retrieval": "\033[36m", "rerank": "\033[35m", "generation": "\033[32m",
    "citation": "\033[34m", "validation": "\033[33m", "decision": "\033[95m",
    "PASS": "\033[32m", "FAIL": "\033[31m", "WARN": "\033[33m",
    "dim": "\033[2m", "bold": "\033[1m", "reset": "\033[0m",
}


def _supports_colour(stream):
    if os.environ.get("NO_COLOR"):
        return False
    return hasattr(stream, "isatty") and stream.isatty()


def new_run_id():
    """Sortable and unique: a UTC stamp plus a short random suffix.

    Deliberately NOT derived from run content -- two identical runs are distinct
    events and must not collide in index.jsonl. This is the one place randomness
    is allowed, and it never touches a metric.
    """
    return time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + "_" + uuid.uuid4().hex[:8]


def new_trace_id():
    return uuid.uuid4().hex[:12]


class TerminalSink:
    """Human-readable, live. Never the record of truth -- that is the JSONL."""

    def __init__(self, stream=None, quiet=False, verbose=False):
        self.stream = stream or sys.stderr
        self.quiet = quiet
        self.verbose = verbose
        self.colour = _supports_colour(self.stream)

    def _c(self, key, text):
        if not self.colour:
            return text
        return f"{_COLOUR.get(key, '')}{text}{_COLOUR['reset']}"

    def write(self, event):
        if self.quiet:
            return
        name = event.get("event", "")
        stage = event.get("stage", "")
        # Per-case chatter only at --verbose; milestones always.
        if not self.verbose and event.get("level") == "detail":
            return
        ms = event.get("duration_ms")
        dur = self._c("dim", f" {ms:.1f}ms") if isinstance(ms, (int, float)) else ""
        head = self._c(stage, f"[{stage or '-':10}]") if stage else " " * 12
        line = f"{head} {name}{dur}"
        p = event.get("payload") or {}
        if p and (self.verbose or event.get("level") != "detail"):
            bits = []
            for k, v in list(p.items())[:6]:
                if isinstance(v, float):
                    v = f"{v:.4f}"
                elif isinstance(v, (list, dict)):
                    v = f"<{len(v)}>"
                bits.append(f"{k}={v}")
            if bits:
                line += self._c("dim", "  " + " ".join(bits))
        print(line, file=self.stream)

    def rule(self, text=""):
        if self.quiet:
            return
        print(self._c("bold", text or "-" * 68), file=self.stream)

    def gate(self, name, status, value, op, threshold):
        if self.quiet:
            return
        v = "n/a" if value is None else f"{value:.4f}"
        tag = self._c(status if status in ("PASS", "FAIL") else "WARN",
                      f"{status:7}")
        print(f"  [{tag}] {name:26} {v} {op} {threshold}", file=self.stream)


class JsonlSink:
    """The record of truth. One event per line.

    Flushed every FLUSH_EVERY lines rather than every line. Per-line flushing
    cost 11.8% of wall time on a 1,393-case run -- over the 10% budget -- and
    bought little: the common reason to read this file is that the run died,
    and a 200-line tail still identifies the case that killed it. Anything at
    level "error" flushes immediately, and close() always flushes, so a clean
    exit or a handled failure loses nothing.
    """

    FLUSH_EVERY = 200

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8", buffering=1 << 16)
        self._since_flush = 0

    def write(self, event):
        self._fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._since_flush += 1
        if event.get("level") == "error" or self._since_flush >= self.FLUSH_EVERY:
            self._fh.flush()
            self._since_flush = 0

    def flush(self):
        if self._fh:
            self._fh.flush()
            self._since_flush = 0

    def close(self):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None


class RunLogger:
    """One run. Emits to every sink; owns run_id and per-request trace_id."""

    def __init__(self, run_id=None, log_dir=None, quiet=False, verbose=False,
                 enabled=True):
        self.run_id = run_id or new_run_id()
        self.enabled = enabled
        self.terminal = TerminalSink(quiet=quiet, verbose=verbose)
        self.jsonl = None
        if enabled and log_dir:
            self.jsonl = JsonlSink(os.path.join(log_dir, f"{self.run_id}.jsonl"))
            self.index_path = os.path.join(log_dir, "index.jsonl")
        else:
            self.index_path = None
        self._trace = None

    # -- trace scope -----------------------------------------------------
    def trace(self, trace_id=None):
        self._trace = trace_id or new_trace_id()
        return self._trace

    def event(self, stage, event, payload=None, duration_ms=None, level="info"):
        if not self.enabled:
            return
        rec = {
            "run_id": self.run_id,
            "trace_id": self._trace,
            "ts": time.time(),
            "stage": stage,
            "event": event,
            "duration_ms": duration_ms,
            "level": level,
            "payload": payload or {},
        }
        self.terminal.write(rec)
        if self.jsonl:
            self.jsonl.write(rec)

    def timed(self, stage, event, payload=None, level="info"):
        return _Timer(self, stage, event, payload, level)

    # -- run-level -------------------------------------------------------
    def summary(self, row):
        """Append one row to index.jsonl. This is the longitudinal record."""
        if not self.enabled or not self.index_path:
            return
        row = dict(row, run_id=self.run_id)
        with open(self.index_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")

    def flush(self):
        """Force buffered events to disk.

        Needed by any reader that wants the current tail -- the trace endpoint
        serves a session that is still in progress, and a 200-line buffer would
        show it as empty.
        """
        if self.jsonl:
            self.jsonl.flush()

    def close(self):
        if self.jsonl:
            self.jsonl.close()


class _Timer:
    def __init__(self, logger, stage, event, payload, level):
        self.logger, self.stage, self.event = logger, stage, event
        self.payload = payload or {}
        self.level = level
        self.t0 = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def add(self, **kw):
        self.payload.update(kw)
        return self

    def __exit__(self, *exc):
        ms = (time.perf_counter() - self.t0) * 1000.0
        self.logger.event(self.stage, self.event, self.payload, duration_ms=ms,
                          level=self.level)
        return False


class NullLogger(RunLogger):
    """Logging disabled. Used to measure the overhead of logging itself."""

    def __init__(self):
        super().__init__(run_id="null", log_dir=None, quiet=True, enabled=False)

    def event(self, *a, **kw):
        return

    def summary(self, row):
        return


def read_index(log_dir):
    path = os.path.join(log_dir, "index.jsonl")
    if not os.path.isfile(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_events(log_dir, run_id):
    path = os.path.join(log_dir, f"{run_id}.jsonl")
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
