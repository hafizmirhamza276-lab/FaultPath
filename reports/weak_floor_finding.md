# Finding — the floor system had no floor on 57% of `numeric_exactness`

**Date:** 2026-09-16 · **Component:** `eval/synthetic.py:_nudge` · **Status:** reported, not yet fixed

## The defect

`WeakSystem` answers a `numeric_exactness` case by taking the correct criterion
and passing it through `_nudge`, which moves every number by 10% and rounds:

```python
def _nudge(text, factor=1.1):
    """Move every number by 10%. Deterministic, and enough to fail exactness."""
```

The docstring's claim — "enough to fail exactness" — is false for small
integers, and this manual is made of small integers:

```
'Max. 1 Ω'      -> 'Max. 1 Ω'        1 x 1.1 = 1.1, rounds back to 1
'Max. 2 Ω'      -> 'Max. 2 Ω'
'Max. 4 V'      -> 'Max. 4 V'
'Min. 1 MΩ'     -> 'Min. 1 MΩ'
'Min. 5 Ω'      -> 'Min. 6 Ω'        the first integer that does move
'0.2 to 4.6V'   -> '0.2 to 5.1V'     decimals move
```

Every integer from 1 to 4 is a fixed point of `round(n * 1.1)`. So is `0`.

## Scale

Measured over the committed ground truth, not sampled:

| population | unchanged by `_nudge` | share |
|---|---:|---:|
| all golden criteria | 650 / 1,146 | **56.7%** |
| `numeric_exactness` cases, non-sealed | 506 / 889 | **56.9%** |
| `numeric_exactness` cases, sealed | 100 / 189 | 52.9% |
| `numeric_exactness` cases, all | 606 / 1,078 | 56.2% |

The survivors are not an exotic tail. They are the manual's commonest criteria:

| count | criterion |
|---:|---|
| 235 | `Max. 1 Ω` |
| 128 | `Min. 1 MΩ` (U+2126) |
| 116 | `Min. 1 MΩ` (U+03A9) |
| 37 | `0 MPa {0 kgf/cm2}` |
| 21 | `Max. 1 V` |

## The consequence, stated plainly

`WeakSystem` is **the floor**. Its entire job is to fail, so that a gate which
it passes is known to have a hole — `tests/test_eval_harness.py` asserts
"weak fails EVERY gate" for exactly this reason.

On 57% of `numeric_exactness` cases it did not fail. It emitted the correct
criterion, verbatim, and scored 1.0.

So:

- **Every good-vs-weak separation recorded on `numeric_exactness` was partly
  fake.** The separation that was measured was the separation on the 43% of
  cases where the arithmetic happened to move a number. On the rest, the
  reference system and the deliberately-bad system produced the same answer.
- **That includes the separations used to validate this harness.** The claim
  "good 7/7, weak 0/7" still holds in aggregate — weak's `numeric_exactness`
  average stays well under the 0.98 gate — but the aggregate was carried by a
  minority of cases while the majority silently agreed.
- **A real system could score 0.569 on the metric METRICS.md calls "the single
  most important metric in the system" while being no better than the system
  built to be bad.** There was no lower bound to compare against.

## Why it went unnoticed

The gate-level assertion is aggregate. `weak` fails `numeric_exactness` overall,
so every test stayed green and the summary line read correctly. Nothing asserted
the property the floor actually needs, which is not "fails on average" but
**"fails on every case, by construction"**.

## The pattern — seventh instance, and the first to reach the reference systems

This is a number that agreed with reality by luck rather than by construction.
Six before it:

1. the shared `NOT_CEILING` reason — one string standing in for 30 arguments
2. `reports/fact_verification.json` — derived once, then trusted forever
3. the README reproduction claim — exact until the case set moved
4. the English-only refusal detector — right only while the model spoke English
5. `_COMPLIANCE` — wrong in both directions, in English
6. `numbers_in()` — identifiers counted as values, three separate call sites

All six were **measurement apparatus**: detectors, artefacts, predicates. This
one is different in kind. It is in the **reference systems themselves** — the
fixtures every other number in the harness is calibrated against. The previous
six could produce a wrong number about a system. This one made the yardstick
wrong, so the numbers it validated were wrong about the harness.

`_nudge`'s docstring asserted the property ("enough to fail exactness") and
nothing checked it. That is the whole class in one line.

## What the fix has to be

Not a different constant. `factor=1.3` would move 1 to 1.3 to 1, and picking a
factor that happens to work on today's corpus is the same defect with a
different number.

The floor must be **provably different from the original for every criterion in
the corpus**, asserted over all 1,146 — not sampled — and wrong the way a bad
system is wrong rather than in a way that advertises itself as synthetic.

Expect baselines to move wherever `weak` appears, and expect good-vs-weak
separation to **widen**. That is the metric becoming honest, not an improvement.

---

## FIXED — 2026-09-16

`_nudge` is replaced by `wrong_criterion`, which substitutes **a different real
criterion from the manual**, chosen by hash of the original and walked forward
until the result does not contain it. Derived from the corpus rather than
computed from the value, so there is no arithmetic for a fixed point to hide in.

Asserted over all 1,146 criteria and all three corpus scopes — `all` (pool 124),
`section40` (90), `symptoms` (34): **0 survivors** in each. The guarantee is
*containment*, not inequality, because `numeric_exactness` asks whether the
answer contains the verbatim criterion, so `1 Ω` inside `Max. 1 Ω` would pass
while being a different string.

It does not advertise itself as synthetic: the output is a real measurement
value from this manual, formatted like the right one. That is how a bad
retrieval system actually fails — wrong row, full confidence — rather than a
sentinel a scorer could special-case.

### `weak`, before → after (train split)

| metric | before | after |
|---|---:|---:|
| `numeric_exactness` | 0.4994 | **0.0000** |
| `contradiction` | 0.3858 | 0.8234 |
| `groundedness` | 0.7904 | 0.5049 |
| `faithfulness_det` | 0.8317 | 0.6523 |
| `uncited_claim_rate` | 0.6534 | 0.8276 |
| `fabricated_values` | 0.0906 | 0.1762 |
| `content_recall` | 0.3106 | 0.2116 |

All seven move in the direction of *worse*, which is what a floor should be.
No other Tier-1 metric moved, and `good` is untouched.

### How to read historical run records

Run records in `eval_out/runs/` are append-only and **cannot be rewritten**.
Any record written before 2026-09-16 with `"system": "weak"` carries a
`numeric_exactness` near **0.50 that is not a measurement of anything** — it is
roughly the share of criteria whose numbers the 10% nudge failed to move. Read
those records as follows:

- `weak`'s `numeric_exactness`, and the six metrics above, are **void** in any
  pre-fix record. Do not diff them against a post-fix run; `eval/compare.py`
  will report a large regression that is the fix, not a change in behaviour.
- Any **good-vs-weak separation** quoted from a pre-fix record understates the
  true separation, and was computed over a mixed population where 57% of cases
  had no floor at all.
- `good` and `model` numbers in those records are **unaffected** — neither
  system touches `_nudge` — so pre-fix records remain valid for those.

The `git` commit that fixes this is the boundary. Records are identified by
their `timestamp` and by the `git` block each record already carries.
