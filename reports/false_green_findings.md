# Finding — a red test level sat unnoticed for four commits

Recorded before the fix, and separately from it, so the sequence stays legible:
this is what was wrong, not what was done about it.

## What happened

`tests/test_human_verify.py` failed from `cc6706d` onward. It was noticed at
`ce26881`, four commits later, and only because the full suite was run by hand
while investigating something else.

Across those four commits the orchestrator reported **11 stages green, 36/36
gates pass**.

That is a false green, which is the single outcome `core/orchestrator.py` exists
to prevent. Its own module docstring says so:

> A green end-to-end report built on a broken extraction is the worst output
> this could produce.

## The two assertions that failed

```
FAIL  unresolved_facts.json still holds 39
FAIL  12 KNOWN_LIMITATION and 27 NEEDS_HUMAN_VERIFICATION
```

`reports/unresolved_facts.json` across the relevant commits:

| commit | total | breakdown |
|---|---|---|
| `3ec6b64` | 39 | 12 KL + 27 NHV |
| `cc6706d` | 552 | 504 KL + 27 NHV + **21 DEFECT** — test went red here |
| `9e60cc9` | 531 | 504 KL + 27 NHV |
| `ce26881` | 531 | 504 KL + 27 NHV + 0 DEFECT |

## The data is right; the expectation was under-specified

This is worth stating precisely, because "the test is stale" is too generous to
the test and too harsh on the number.

The original 39 is **still there, unchanged**:

| section | kind | classification | n |
|---|---|---|---|
| section40 | cause | NEEDS_HUMAN_VERIFICATION | 27 |
| section40 | cause | KNOWN_LIMITATION | 12 |
| section40 | step_procedure | KNOWN_LIMITATION | 347 |
| hmode | step_procedure | KNOWN_LIMITATION | 145 |

`12 + 27 = 39` is exactly the `cause` bucket, and it has not moved. What changed
is that fidelity began enumerating a **fact kind that did not previously exist
in the enumeration at all** — `step_procedure` — and 492 of those appeared.

So the test asserted an unqualified total while meaning a per-kind one. When a
new kind entered, the total it pinned absorbed the new kind and the assertion
became false without anything it cared about having changed.

That is the same defect class as the `qa_set.json` staleness this repo already
documented: **an expectation with no producer, pinned to a number whose
definition later widened underneath it.**

## Why nothing caught it

`tests/test_human_verify.py` is a level in `tests/run_all.py` and has **no
orchestrator stage**. The orchestrator is presented as the one controlled entry
point — CLAUDE.md and every recent instruction say everything runs through it —
and it does not run this level.

Audited across all of `run_all.py`:

| level | orchestrator stage |
|---|---|
| ground-truth | `extract` runs `test_extraction.py` as its guard |
| fidelity | **none** — `s_fidelity` runs the *module*, not the test |
| human-verify | **none** |
| harness | `harness` |
| agent | `agent` |
| module | `api` |
| pipeline+e2e | `api` |
| orchestrator | **none** — and cannot be, it is the runner |

Two levels had no stage. One of them was red.

This is the `orphan_detection` shape the structural checks already look for in
the manual's own cross-references: something with no one watching it. The
orchestrator had it in its own test surface.

## What the fix has to be

Adding a `human_verify` stage closes this instance and not the class. The class
closes only when the two lists cannot drift:

- every level in `run_all.py` has a stage, and every stage has a level
- both lists derived programmatically, never hand-maintained — a hand-written
  list of levels drifts exactly the way this expectation did
- the parity check carries a self-test that plants an orphan and confirms the
  run fails, because a check that cannot fail is not a check

Fix and parity check follow in separate commits.
