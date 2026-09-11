#!/usr/bin/env python3
"""
overfitting.py
The guard against a test suite that agrees with itself.

Our fixtures and our implementation both derive from golden/. That is exactly
how citation_accuracy read 1.0000 while 635 of 846 citations named the wrong
page: the test set and the system shared a mistake, so the metric confirmed the
error instead of catching it. Four defences:

  a) HELD-OUT SPLIT. 20% of failure codes are reserved. No fixture, scripted
     session or threshold may reference them. Metrics are reported on train and
     holdout separately; a gap means we tuned to the cases, not the manual.

  b) INDEPENDENT DERIVATION. Expected values are read back from the PDF through
     the citation resolver, not from the in-memory records the agent walks. If
     both sides load the same object the test proves only self-consistency.

  c) MUTATION CHECK. Deliberate faults are injected and each must be caught by
     at least one test. A fault nothing catches is a hole, and it gets named.

  d) NEGATIVE COVERAGE. Every gate must have a case that fails it. A gate no
     test can fail is not a gate -- the E4 and H2 finding, applied to ourselves.
"""
from __future__ import annotations

import hashlib
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from agent import tools  # noqa: E402

# The split itself lives in agent/holdout.py so fixture builders exclude it by
# construction rather than by remembering to.
from agent import holdout as _holdout  # noqa: E402

TRAIN = _holdout.train_codes()
HOLDOUT = _holdout.holdout_codes()
HOLDOUT_SET = _holdout.holdout_set()


def assert_no_holdout_leak(names, where=""):
    """Fixtures must not mention a held-out code. Raises with the offender."""
    leaked = sorted({n for n in names if n in HOLDOUT_SET})
    if leaked:
        raise AssertionError(
            f"held-out codes referenced by {where or 'a fixture'}: {leaked}. "
            "The holdout only measures generalisation while nothing tunes to it.")


# --------------------------------------------------- b) independent derivation

def expected_from_pdf(fact_id, pages=None):
    """Read a fact's expected page and text back through the PDF.

    Deliberately does NOT consult the in-memory record the agent walks. It
    renders the citation, then resolves it against the source page, so a fixture
    built from this is anchored to the document rather than to our parse of it.

    Returns None when the PDF is unavailable -- the caller must skip rather than
    quietly fall back to the in-memory value, which would reintroduce exactly
    the self-agreement this exists to prevent.
    """
    from eval.citations import render, resolve, PageText, UnknownFact
    pages = pages or PageText()
    if not pages.available():
        return None
    try:
        cit = render([fact_id])[0]
    except UnknownFact:
        return None
    res = resolve(cit, pages)
    return {"fact_id": fact_id, "manual_page": cit["manual_page"],
            "pdf_page": cit["pdf_page"], "verbatim_text": cit["verbatim_text"],
            "resolved": res["resolved"], "reason": res["reason"]}


# ------------------------------------------------------------ c) mutations

MUTATIONS = [
    ("skip_machine_gate", "agent answers before model+serial are known"),
    ("dump_all_steps", "agent emits the whole tree in one message"),
    ("accept_vague_reading", "agent treats 'thoda kam' as a passing reading"),
    ("ignore_precondition", "agent starts the named code, not the one the "
                            "manual says to solve first"),
    ("drop_visited_set", "agent follows cross-references with no cycle guard"),
    ("model_typed_page", "agent types a page number instead of naming a fact"),
    ("off_by_one_cursor", "agent skips a step when advancing"),
]


def mutant_agent(fault):
    """An Agent with exactly one deliberate fault."""
    from agent.graph import Agent
    from agent.runner import BadAgent
    from agent.state import Awaiting, Reading, Verdict

    class _M(Agent):
        pass

    if fault == "skip_machine_gate":
        _M.n_identify_machine = BadAgent.n_identify_machine
    elif fault == "dump_all_steps":
        _M.n_execute_step = BadAgent.n_execute_step
    elif fault == "accept_vague_reading":
        _M.n_parse_reading = BadAgent.n_parse_reading
    elif fault == "ignore_precondition":
        _M.n_preflight = BadAgent.n_preflight
    elif fault == "drop_visited_set":
        _M.n_resolve_pointer = BadAgent.n_resolve_pointer
    elif fault == "model_typed_page":
        def _typed(self, st, payload, node, fact_ids=(), grounded_text=()):
            # Types a page and a value with nothing behind them.
            from agent.state import Emission
            txt = "See manual page 40-999. Standard value: Max. 7 ohm."
            st.emissions = st.emissions + [Emission(
                text=txt, original_text=txt, node=node)]
            if node == "execute_step":
                st.awaiting = Awaiting.READING
            return st
        _M.emit = _typed
    elif fault == "off_by_one_cursor":
        base = Agent.n_execute_step

        def _skip(self, st):
            st = base(self, st)
            if st.step_cursor and st.step_cursor < tools.step_count(
                    st.active_code or ""):
                st.step_cursor = st.step_cursor + 1     # silently skips a check
            return st
        _M.n_execute_step = _skip
    else:
        raise ValueError(fault)
    return _M()
