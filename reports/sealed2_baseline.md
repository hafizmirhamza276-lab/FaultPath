# sealed2 — the second holdout, and the blind baseline on both sides

**Date:** 2026-09-21 · **Derived at:** `3804598` (before any selector work) ·
**Status:** UNSPENT — one measurement remains

## Why

`eval/sealed.py` is spent: measured on 2026-09-17 at `2e8edb2`, and a holdout is
spent the moment a result from it is known. Since then the prompt selector has
been rewritten twice and is about to be rewritten a third time, every round
scored on the same 1,511 cases. `reports/oracle_selection_finding.md` is what
that produces — a selector that reached 7/7 by reading the answer key, on the
only cases anyone was looking at.

So this set was derived **before** the question-derived selector was designed.
At the moment of derivation the design did not exist, so it cannot have
influenced the split.

## The rule

`eval/sealed2.py`. **The algorithm is `sealed.derive_split`, imported, not
copied.** The record unit, the `(section, type)` stratification, the coverage
repair and the loose-case handling are the first split's — a second
implementation of a judgement is how the two drift apart, and this repo has
already paid for that once (`eval/refusal.py`).

Two things differ, and only two:

| | |
|---|---|
| input | the **1,511 cases `sealed.py` did not take**, never the full 1,865 |
| salt | `komatsu-sealed2-cases-v1` — a new one, so the draw is independent of the first rather than a re-run of it on a smaller pool |

Everything else holds: 20% target, **record-level** so no record straddles the
boundary, stratified by what each record generates, every `(section, type)`
guaranteed present on both sides, deterministic, no RNG, no stored id list, and
**it reads no eval result** — section, type, record identity, content hash, and
nothing else. Asserted in `tests/test_eval_harness.py` over the parsed AST.

Two properties only a *second* split can violate are asserted too: `sealed2`
shares no case with the spent set, and `dev ∪ sealed2 == train` exactly.

## Counts

**1,511 pool → dev 1,183 / sealed2 328 (21.7%), 38 records.**

| section/type | pool | sealed2 | share |
|---|---:|---:|---:|
| section40/adversarial_model | 5 | 1 | 20.0% |
| section40/adversarial_unknown | 5 | 1 | 20.0% |
| section40/branch_following | 92 | 19 | 20.7% |
| section40/cross_ref_hop | 7 | 2 | 28.6% |
| section40/direct_lookup | 138 | 29 | 21.0% |
| section40/numeric_exactness | 709 | 156 | 22.0% |
| section40/precondition | 8 | 2 | 25.0% |
| section40/step_ordering | 131 | 27 | 20.6% |
| symptoms/branch_following | 30 | 6 | 20.0% |
| symptoms/numeric_exactness | 180 | 32 | 17.8% |
| symptoms/step_ordering | 46 | 9 | 19.6% |
| symptoms/symptom_remedy | 160 | 44 | 27.5% |

## What sealed2 cannot measure — stated before it is spent

**`injection_resistance`: 0 of 15.** Injection cases are generated at run time
by `safety.injection_cases`, which wraps every seventh code of
`sorted(records)`; none of those 15 landed on the sealed2 side. The coverage
guarantee works on qa_set `(section, type)` pairs and cannot see generated
cases, and teaching it to would mean importing `eval.metrics` into a split rule
— the one import class this module is forbidden, because a split that can see a
metric can be adjusted to flatter it.

This is not silent: `run_eval` lists `injection_resistance` under
`not_measured` on every sealed2 run. But **a sealed2 number is a number about
answering, not about resisting.** The injection gate is measured on dev, n=15.

`refusal_correctness` / `clean_refusal` are thin for a structural reason —
adversarial cases carry no record, so they split per case: **n=2** on sealed2
against n=8 on dev. Directionally useful, not a rate.

## Blind-selector baseline, both sides

`SELECTOR_BLIND=1`, budget 3,500, `gpt-4.1-2025-04-14`, structural/bm25/k=20.
Served from the cache built by the `3804598` blind run (1 fresh call each), so
this is a re-partition of an existing measurement, not a new one.

Runs: `eval_out/runs/20260921T075834_blindA_dev.json`,
`eval_out/runs/20260921T080033_blindA_sealed2.json`.

| gate | dev (1,223 cases) | sealed2 (344 cases) | |
|---|---:|---:|---|
| `numeric_exactness` ≥ 0.98 | 0.8588 | 0.8617 | FAIL / FAIL |
| `citation_accuracy` ≥ 0.95 | 0.8722 | 0.8190 | FAIL / FAIL |
| `fabricated_values` ≤ 0.005 | 0.0006 | 0.0033 | PASS / PASS |
| `refusal_correctness` ≥ 1.0 | 1.0000 (n=8) | 1.0000 (n=2) | PASS / PASS |
| `over_refusal` ≤ 0.05 | 0.0043 | 0.0092 | PASS / PASS |
| `hit_rate@5` ≥ 0.95 | 0.9974 | 1.0000 | PASS / PASS |
| `filter_correctness` ≥ 1.0 | 1.0000 | 1.0000 | PASS / PASS |
| **total** | **5/7** | **5/7** | |

| metric | dev n | dev | sealed2 n | sealed2 |
|---|---:|---:|---:|---:|
| `numeric_exactness` | 701 | 0.8588 | 188 | 0.8617 |
| `citation_accuracy` | 1174 | 0.8722 | 326 | 0.8190 |
| `content_recall` | 1175 | 0.7938 | 326 | 0.7914 |
| `contradiction` | 701 | 0.0670 | 188 | 0.0426 |
| `groundedness` | 1221 | 0.9901 | 344 | 0.9871 |
| `completeness` | 237 | 0.2419 | 63 | 0.2612 |
| `fabricated_values` | 1223 | 0.0006 | 344 | 0.0033 |
| `injection_resistance` | 15 | 0.8667 | **0** | **NOT MEASURED** |

The two sides agree closely on the metric that matters most here —
`numeric_exactness` 0.8588 dev vs 0.8617 sealed2, a 0.3-point gap on n=701 and
n=188 — which is what a representative split looks like. `citation_accuracy` is
5.3 points lower on sealed2; on the blind selector that tracks which records
happened to land there, and it is recorded now so that a later sealed2 number
is compared against **this** baseline rather than against dev's.

## The one rule for using it

One measurement, once, on a frozen selector. After that it is spent and this
file gets the SPENT banner `eval/sealed.py` carries. Re-deriving the same ids
will not restore its unseenness.
