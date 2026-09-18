# Finding — the model is shown 34.3% of the corpus

**Date:** 2026-09-18 · **Component:** `eval/model_system.py:94` (`build_prompt`) · **Status:** reported, not yet fixed

## The defect

`build_prompt` slices every retrieved chunk at a fixed offset:

```python
f"{(c.get('text') or '')[:MAX_CHUNK_CHARS]}"      # MAX_CHUNK_CHARS = 1800
```

The `structural` chunker emits **one chunk per record**, and records render in a
fixed order:

```
header (code, model, manual, serial, pages, action level, details, controller)
Step 1 … Step N          (cause + procedure prose)
Step N measurement.  Measuring point: …  Standard value: …
```

The measurement table is **last**. So the first 1,800 characters are reliably
the header and the step prose, and what falls off the end is reliably the
measuring points and their standard values — which is precisely what
`numeric_exactness`, the metric METRICS.md calls "the single most important
metric in the system", asks about.

## Scale

| | |
|---|---:|
| `MAX_CHUNK_CHARS` | 1,800 |
| structural chunks (one per record) | 231 |
| chunk chars — median / mean / max | 3,969 / 5,029 / 27,108 |
| chunks **longer** than the cap | **201 of 231 = 87.0%** |
| share of corpus text the model can **ever** see | **34.3%** |
| `MAX_CONTEXT_CHUNKS` | 6 |
| effective prompt ceiling | 10,800 chars of extract |

## Evidence from the `false_absence` diagnosis

Of 237 non-sealed `numeric_exactness` answers that engage, do not contain the
criterion, and state **no value at all**:

| category | n | share |
|---|---:|---:|
| **(a)** the (point, value) pair was in the retrieved chunk and **cut by the slice** | **233** | **98.3%** |
| **(b)** the pair was in the extracts and the model missed it | 4 | 1.7% |
| **(c)** present but under a label the model did not recognise | **0** | — |

**Retrieval never missed — not once in 237.** The correct record was retrieved
and the measurement was in the chunk every time. The failure is entirely
downstream of retrieval, in what the prompt builder chose to show.

Verbatim, case `98a140c83d` (CA122), asking for `Between ECM (female) (37) and
(44)`. Present in the retrieved chunk:

```
Step 4 measurement. Resistance. Measuring point: Between ECM (female) (37) and (44). Standard value: Min. 100kΩ (page 40-181)
```

In the extracts the model saw: **False**. Full chunk 22,451 chars; extracts
11,119. The model answered *"is measurement ya standard value ka mention nahi
hai"* — a true statement about what it was given.

## The consequence, stated plainly

**`numeric_exactness 0.5450` is substantially a measurement of the truncation,
not of the model.** So are `content_recall`, `faithfulness_det` and the
citation metrics, in proportion.

**The spent sealed measurement must from now on be read as "this model on 34.3%
of context".** Its 4-of-7 gate result is not wrong, but it is not a measurement
of the model's capability either — it is a measurement of the model under a
context budget that discards two thirds of the ground truth, most of it the
part the failing gates ask about. Recorded beside the spent marker in
`eval/sealed.py` so the number is never quoted without it.

This does not retroactively invalidate the sealed run. It fixes what the run
was a measurement *of*.

## Why this went unnoticed

The two constants are plausible on their face — 1,800 characters and 6 chunks
read like sensible prompt hygiene, and nothing about them says "discards 66% of
the ground truth". Nothing measured the relationship between the cap and the
chunk-size distribution, and no metric reports context coverage. Every symptom
surfaced somewhere else: as the model refusing, as `numeric_exactness` being
low, as eight of nine supposed partial deliveries. Each was investigated as a
model or detector problem.

It is the same shape as the rest of this sequence — a number that looked like a
property of the thing being measured and was a property of the apparatus — with
the difference that this one sits in the input path rather than in a scorer.
