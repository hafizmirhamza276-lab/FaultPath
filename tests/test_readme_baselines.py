#!/usr/bin/env python3
"""
test_readme_baselines.py
The README's baseline tables, re-derived and compared digit for digit.

    python tests/test_readme_baselines.py

WHY THIS EXISTS
---------------
README.md carried the claim that `--corpus section40` reproduced a documented
table "digit for digit". It stopped being true in 9d9b7c1, when symptom cases
took qa_set from 1,318 to 1,865 cases and a Section-40-only corpus could no
longer serve a symptom case. `ceiling_recall` went from 1.0000 to 0.7204.
Nothing checked the claim, so nothing noticed, and the sentence sat there
asserting exactness for months.

That is the FOURTH check in this sequence that agreed with reality by luck
rather than by construction:

  1. the shared NOT_CEILING reason      -- one string explaining 30 metrics
  2. the frozen verification artefact   -- derived once, then trusted forever
  3. the English-only refusal detector  -- right only while the model spoke
                                           English
  4. this table                         -- exact only until the case set moved

Each looked like corroboration. A number that matches is not evidence that
anything checked it.

WHAT IS VERIFIED AND WHAT IS NOT
--------------------------------
This file verifies the two LOCAL BASELINE tables, which are anchored in
README.md by HTML comments so the parse is anchored rather than heuristic.
Every cell is re-derived: chunk counts, six retrieval metrics, and the gate
tally.

It does NOT verify the historical comparison sections -- "What the 57 symptom
trees cost" and the rest. Those are records of a measurement taken at a
particular commit against a corpus state that no longer exists. They are
labelled HISTORICAL in README.md and this file asserts that the label is still
there, because a historical table that loses its label becomes a false claim of
exactness again.

Deterministic. No LLM, no network.
"""
from __future__ import annotations

import os
import re
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval import chunkers                                   # noqa: E402
from eval import run_eval                                   # noqa: E402
from eval.adapters import LocalBM25Retriever                # noqa: E402
from eval.metrics.base import Registry                      # noqa: E402
from eval.metrics import retrieval as m_retrieval           # noqa: E402
from eval.metrics import generation as m_generation         # noqa: E402
from eval.metrics import safety as m_safety                 # noqa: E402
from eval.metrics import conversation as m_conversation     # noqa: E402
from eval import citations as citemod                       # noqa: E402
from core.run_log import NullLogger                          # noqa: E402

README = os.path.join(REPO_ROOT, "README.md")

# The anchors. Parsing by "the table after this heading" is how a doc check
# quietly stops covering the table when someone adds a paragraph.
ANCHOR = re.compile(r"<!--\s*BASELINE\s+corpus=(\S+)\s*-->")
HISTORICAL = re.compile(r"<!--\s*HISTORICAL\s+(\S+)\s*-->")

CHUNKERS = ("structural", "fixed_2000", "fixed_512")

# Rows this file knows how to re-derive. A row in the README that is not here
# fails the run rather than being skipped -- an unrecognised row is the shape
# of a claim nobody checks.
METRIC_ROWS = ("ceiling_recall", "recall@5", "answer_coverage@5",
               "fragmentation_gap@10", "context_precision", "mrr")

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


# ------------------------------------------------------------------ parsing

def parse_tables(text):
    """{corpus: {row_label: [cell, cell, cell]}} from the anchored tables."""
    out = {}
    for m in ANCHOR.finditer(text):
        corpus = m.group(1)
        rows = {}
        header_seen = False
        for line in text[m.end():].splitlines():
            line = line.strip()
            if not line.startswith("|"):
                if rows:
                    break            # table ended
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue             # the ---|---|--- separator
            label = cells[0].strip("*").strip()
            if not header_seen:
                header_seen = True
                got = [c.strip("*").strip() for c in cells[1:]]
                if got != list(CHUNKERS):
                    raise SystemExit(
                        f"ERROR: table '{corpus}' columns are {got}, "
                        f"expected {list(CHUNKERS)}")
                continue
            rows[label] = [c.strip("*").strip() for c in cells[1:]]
        out[corpus] = rows
    return out


def as_number(cell):
    return float(cell.replace(",", ""))


# ------------------------------------------------------- re-derivation

def run_one(corpus_scope, chunker, cases, pages):
    """One full evaluation, in process, through run_eval's own evaluate().

    Driven in process rather than by subprocess so the PDF and the case set are
    loaded once instead of six times. Measured, not assumed: 625s here against
    roughly 13 minutes as six separate CLI runs. Most of the cost is the scoring
    itself and no arrangement avoids it -- this check is genuinely expensive,
    which is the honest price of the claim it verifies.

    It is the same code path -- run_eval.evaluate, run_eval.aggregate,
    run_eval.check_gates -- so a reproduction here is a reproduction of the
    documented command rather than of a lookalike.
    """
    records = run_eval.load_records(corpus_scope)
    corpus = chunkers.build(chunker, list(records.values()))
    retriever = LocalBM25Retriever(corpus)

    registry = Registry()
    registry.extend(m_retrieval.build(corpus))
    registry.extend(m_generation.build(records, pages))
    registry.extend(m_safety.build(records))
    registry.extend(m_conversation.build())

    system = run_eval.SYSTEMS["good"](records)
    all_cases = (list(cases)
                 + m_safety.injection_cases(records)
                 + m_conversation.build_scenarios(records, cases))

    def run_case(case):
        ctx = system.retrieve(case, retriever, 20)
        if case["type"] == "conversation":
            replies = system.converse(case, ctx)
            turns = [{"role": "technician", "text": case["turns"][0]}]
            for i, rep in enumerate(replies):
                turns.append({"role": "assistant", "text": rep})
                if i + 1 < len(case["turns"]):
                    turns.append({"role": "technician",
                                  "text": case["turns"][i + 1]})
            return {"transcript": turns, "retrieved": ctx,
                    "answer": " ".join(replies), "citations": [],
                    "citations_rendered": [], "refused": False}
        out = system.answer(case, ctx)
        out["retrieved"] = ctx
        rendered = []
        for fid in (out.get("fact_ids") or []):
            try:
                rendered += citemod.render([fid])
            except citemod.UnknownFact:
                pass
        out["citations_rendered"] = rendered
        if not rendered and out.get("citations"):
            out["citations_rendered"] = [
                {"fact_id": None, "manual_page": c, "pdf_page": None,
                 "verbatim_text": ""} for c in out["citations"]]
        elif rendered:
            out["citations"] = [c["manual_page"] for c in rendered]
        return out

    rows, _ = run_eval.evaluate(all_cases, registry, run_case, NullLogger())
    metrics = run_eval.aggregate(registry, rows)
    gates = run_eval.check_gates(metrics[1])
    passed = sum(1 for g in gates if g["status"] == "PASS")
    return {"chunks": len(corpus),
            "metrics": {k: v["value"] for k, v in metrics[1].items()},
            "gates": f"{passed}/{len(gates)}"}


def main():
    started = time.time()
    text = open(README, encoding="utf-8").read()
    tables = parse_tables(text)

    print("\nREADME baseline tables: anchored, and re-derived cell by cell")
    check("both baseline tables are anchored in README.md",
          set(tables) == {"all", "section40"},
          f"found anchors for {sorted(tables)}; expected all + section40")
    if failures:
        return 1

    # The historical sections must stay LABELLED. An unlabelled historical
    # table reads as a current measurement, which is the original defect.
    labelled = HISTORICAL.findall(text)
    check(f"{len(labelled)} historical section(s) still carry their label",
          bool(labelled),
          "no <!-- HISTORICAL ... --> marker found; a historical table with no "
          "label is indistinguishable from a verified one")

    cases = run_eval.load_cases()
    pages = citemod.PageText()

    for corpus_scope in ("all", "section40"):
        rows = tables[corpus_scope]
        unknown = sorted(set(rows) - set(METRIC_ROWS) - {"chunks", "gates"})
        check(f"[{corpus_scope}] every documented row is one this file checks",
              not unknown,
              f"unchecked rows {unknown}: add them here or they are claims "
              f"nothing verifies")

        for i, chunker in enumerate(CHUNKERS):
            got = run_one(corpus_scope, chunker, cases, pages)
            bad = []
            if "chunks" in rows:
                want = as_number(rows["chunks"][i])
                if got["chunks"] != want:
                    bad.append(f"chunks: README {want:.0f}, got {got['chunks']}")
            for row in METRIC_ROWS:
                if row not in rows:
                    continue
                want = as_number(rows[row][i])
                have = got["metrics"].get(row)
                if have is None:
                    bad.append(f"{row}: not produced by this run")
                elif f"{have:.4f}" != f"{want:.4f}":
                    bad.append(f"{row}: README {want:.4f}, got {have:.4f}")
            if "gates" in rows:
                if rows["gates"][i] != got["gates"]:
                    bad.append(f"gates: README {rows['gates'][i]}, "
                               f"got {got['gates']}")
            check(f"[{corpus_scope}/{chunker}] reproduces digit for digit",
                  not bad, "\n          ".join(bad))

    print(f"\n  {time.time() - started:.0f}s")
    if failures:
        print(f"\n{len(failures)} README baseline check(s) FAILED.\n"
              "A documented table that no longer reproduces is a false claim "
              "of exactness. Either the code changed and the table must be\n"
              "regenerated, or the table was never right. Do not edit the "
              "number to match without knowing which.")
        return 1
    print("\nREADME baselines reproduce.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
