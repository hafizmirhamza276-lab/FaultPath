# Stored derived values — a deliberate sweep

**Date:** 2026-09-16 · **Scope:** `pipeline/ eval/ core/ agent/ api/ knowledge/ tests/ golden/ reports/`

## The defect class

A value that **could be computed** from a more primary source is instead
persisted — to disk, to a cache, to a committed artefact, or to a field inside a
record — and the stored copy then **silently substitutes for the derivation**.
When the logic or the primary source changes, the stored copy goes stale and
nothing notices, because the stored copy is what everything reads.

Five instances have been found so far, **every one of them by accident, while
doing something else**:

| # | instance | found while |
|---|---|---|
| 1 | shared `NOT_CEILING` reason — one string standing in for 30 arguments | generalising the ceiling guard |
| 2 | `reports/fact_verification.json` — derived once, then trusted forever | wiring verification |
| 3 | README baseline tables — derived by running an eval, then frozen as prose | reading the README |
| 4 | `eval_out/model_cache.jsonl` storing `refused` — replayed on every hit | fixing the refusal detector |
| 5 | `numbers_in()` treating page tokens, machine identity and step ordinals as values | three separate metrics |

This document is the first time the class has been searched for **on purpose**.

## Method

Every `json.dump`, every committed `.json`/`.tsv`, every module-level constant
that restates data living elsewhere, every keyed-lookup cache, and every field
inside a `golden/` record that summarises other fields. Each candidate was
**re-derived and compared**, not eyeballed.

Three candidates were dismissed only after the comparison contradicted a first
reading — recorded here because "I checked and I was wrong" is the useful half:

- `golden/index.json` looked 307-way inconsistent until the probe used the real
  field names (`manual_pages[0]`, `standalone_measurements`). Actual: 0.
- The METRICS.md case table looked stale by 535 cases until the section filter
  was applied. It is Section-40-scoped and exact.
- `fact_id` looked 2,260-way inconsistent until the probe stopped treating
  `branch_fact_ids` as a list and the step index as positional. Actual: 0.

---

## DANGEROUS — fixed in this commit

### 1. `fact_id`, stored in every `golden/failure_codes/*.json`

**Stored:** `<code>:<step>:<kind>:<index>` on every step, measurement and branch
— 3,305 of them.
**Derivable from:** the record's own structure. The id is defined as a function
of position and nothing else.
**Would a stale copy be detected?** **Only partly, and not in the way that
matters.** `tests/test_extraction.py` checked uniqueness, shape against
`FACT_RE`, and **one** hand-picked spot (`CA451:6:meas:0`). An id that is
unique, well-formed and pointing at the **wrong fact** passed all three.

**Why this is the dangerous one.** `eval/citations.py` resolves citations *by
fact id*. The project's central claim is that the model names a `fact_id` and
code renders the page — "what the model does not generate it cannot get wrong".
That claim rests entirely on the id corresponding to the fact. A mismatched id
does not fail loudly; it renders a **confident, correctly-formatted, wrong
citation**.

**State when swept:** all 3,305 agree with position. 0 disagreements.
**Fix:** `tests/test_extraction.py` now re-derives every id from position and
compares, replacing the single spot check. Proven to fail on a planted
mismatch.

---

## MODERATE — listed, not fixed

### 2. `golden/index.json`

**Stored:** `format`, `action_level`, `manual_page`, `n_steps`,
`n_measurements`, `refs` for all 174 codes.
**Derivable from:** the 174 records it summarises. Every field is a projection.
**Would a stale copy be detected?** **Nothing.** And it is worse than unchecked
— it is also **unread**. Every consumer skips it by name: `build_qa_set.py:90`,
`chunkers.py:113`, `citations.py:63`, `fidelity.py:238`,
`audit_symptoms.py:56`, `symptom_match.py:109`, `agent/tools.py:355`,
`test_extraction.py:284`.

So it is a committed artefact, presented in README.md and CLAUDE.md as ground
truth, that nothing produces a check for and nothing consumes. Blast radius is
zero **today**, entirely because no code trusts it; a human reading it, or the
first consumer that stops skipping it, gets whatever was last written.

**State when swept:** consistent. 0 mismatches across 174 codes × 6 fields.

### 3. The audit baseline — `21 findings / 6 HIGH / 8 MEDIUM / 5 LOW / 2 INFO`

**Stored:** as prose in `CLAUDE.md:122`, `README.md:245`, `METRICS.md:251`.
**Derivable from:** `reports/audit_findings.json`, by counting.
**Would a stale copy be detected?** **Nothing** — and this one is pointed.
`core/orchestrator.py:186-196` **already computes** the severity histogram into
the run record, then gates only on `audit_exit_zero`. The derivation exists and
its result is discarded. Turning it into a gate is close to free.

**State when swept:** consistent. 21 findings, 6/8/5/2.

### 4. The METRICS.md case-type table

**Stored:** `METRICS.md:159-170`, eight rows and a total.
**Derivable from:** `golden/qa_set.json`, by grouping.
**Would a stale copy be detected?** **Nothing.** Identical in shape to the
README reproduction claim fixed in `d3384c4` — and that one *had* rotted.

**State when swept:** every row exact, **for Section 40**. Two gaps that are
labelling rather than staleness: the table never says it is Section-40-scoped,
and the 535 symptom cases and the `symptom_remedy` type are absent from it
entirely. A reader takes 1,330 for the size of the corpus, which is 1,865.

### 5. `FACT_KINDS` — `pipeline/fidelity.py:33`

**Stored:** a tuple of the eight enumerated fact kinds.
**Derivable from:** the kinds `build_facts()` actually emits.
**Would a stale copy be detected?** **Nothing, and nothing would happen
either.** It has **no consumer** — only `GATED_KINDS` is read, at line 641.

The trap is the inverse of the usual one. A stale copy causes no failure
because the value is never used, so anyone adding a fact kind updates
`FACT_KINDS`, observes no effect, and reasonably concludes the list is
authoritative and current. It is neither.

### 6. `numbers_in()` counting non-values — *different class, same consequence*

Not a stored derived value; a **wrong predicate**, included because it was
asked for and because it is the one item here that is **corrupting a reported
number right now**.

`numbers_in()` returns every numeric token, and three metrics have read that as
"measurement values delivered":

| site | counted as a value | status |
|---|---|---|
| `fabricated_values` | page tokens (`40-179`) | fixed earlier |
| `injection_resistance` (`uncited`) | step ordinals (`Step 1`) | fixed in `d812f4f`, now uses `CRIT_VALUE` |
| `clean_refusal` | machine identity (`PC200-10M0`, `SEN06867-13`) and page numbers | **open** |

**Live consequence:** `clean_refusal` reads **0.0000** on the model while all
five of its refusals are correctly detected and none leaks a measurement value.
The number is false, not merely imprecise. `CRIT_VALUE` is the repo's one
deterministic definition of a measurement value and is the obvious fix, but it
changes a reported metric and belongs in its own commit with its own
before/after.

---

## BENIGN — checked and dismissed, with the reason

| candidate | why it is not this defect |
|---|---|
| `knowledge/symptom_synonyms.json`, `symptom_heldout.json` | the `symptom_map` stage **regenerates both before reading them** (`core/orchestrator.py:260-271`). Regeneration is the check; a stale commit is overwritten, and git shows the drift. |
| `PageText._cache` (`citations.py:236`), `_normalise_cached` (`base.py:34`), `sealed._cache` | in-process, keyed on the input, no persistence across runs, and `sealed` exposes `reset_cache()`. A pure function memoised for one process cannot go stale. |
| the eight counts in `tests/test_extraction.py` | a **deliberate tripwire**: compared against the live derivation on every run. The stored number is the alarm, not the answer. |
| `reports/transcription/**` | human-entered transcriptions. Primary data, not derived from anything. |
| thresholds — `SEALED_FRACTION`, `LEAK_RUN`, gate values | configuration. Someone chose them; there is no truer answer elsewhere to drift from. |

---

## The pattern worth keeping

Four of the five live candidates are **currently consistent**. That is the point
rather than a reassurance: every one of the five historical instances was also
consistent right up until it was not, and in four of five cases nothing would
have reported the transition. Consistency observed once is not a mechanism.

The distinguishing question is not *"is this value correct?"* but ***"what
would tell me if it stopped being?"*** — and for items 2 through 5 the answer is
still "nothing".
