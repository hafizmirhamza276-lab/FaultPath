# Question-derived row selection — replacing the oracle

**Date:** 2026-09-21 · **Component:** `eval/selector.py` (new),
`eval/model_system.py` `select_rows` · **Holdout:** `sealed2`, derived at
`0f633ab` before any of this existed

## What was wrong

`3804598` recorded it: the prompt builder read `case["expected"]`. `_anchors()`
promoted the row carrying `expected.point`; the row order came from
`bool(expected.criteria)`. `expected` is the scoring payload and does not exist
at inference time, so the harness was locating the answer row and then scoring
the model on reading the row it had been handed — 7/7 gates with it, 5/7
without.

## What replaces it

`eval/selector.py`. It receives the **question** and the **record's rows**, and
never the case object. A function that cannot see the answer key cannot read
it, which is a stronger guarantee than a convention about which fields to
touch.

**Discriminativeness is IDF over the record's own rows.** Not over the corpus,
and not a weighting anyone picked. A token in every row of a record identifies
nothing inside it — `resistance` in a record that is forty resistance
measurements scores ~0. A token in one row carries the match: a pin number, a
connector designation, a one-off component name.

Three consequences fall out of that choice rather than being engineered:

- **No stop-word list.** The questions are romanised Hinglish over English
  technical content; the scaffold words (`ke`, `liye`, `par`, `ki`, `kya`)
  appear in no row of any record, so they score zero without anyone
  enumerating them. A stop list would be a phrase list wearing a different hat,
  and would need a new entry per language.
- **Abbreviation robustness for free.** "res" for "resistance" loses a token
  that was worth almost nothing, because that record's rows all said
  Resistance. No synonym table.
- **No phrase list anywhere**, in either language. Rule 4 is satisfied by
  construction: every judgement is "does this question token appear in this
  row", matched against text this repo extracted.

Three priorities, filled until the budget runs out, then emitted in record
order:

| | |
|---|---|
| **0** | header rows, **plus the first row of each other kind** — the record's entry points. Structural and case-independent. "Which check comes first" is answered by the record's own order, which no question token points at. Costs ~4 rows and is what takes dev reachability from 99.66% to 100%. |
| **1** | rows the question matches, ranked by `score / sqrt(row length)`. The budget is spent in characters, so a long row must earn its cost. Raw score leaves 7 dev answers outside the budget; density leaves 4; density plus entry points leaves 0. |
| **2+** | everything else by kind, in an order the question also decides — kinds ranked by the total match they attracted, ties broken by `(step, remedy, branch, measurement)`. A question that matches nothing specific therefore gets **step prose first**, derived from the absence of measurement evidence rather than from a flag. |

**Order is derived, not read.** `selector.wants_value()` exposes the same
signal — does the match concentrate in measurement rows — so its accuracy can
be *reported*. Against `bool(expected.criteria)` over the 1,484 delivered
non-sealed cases it agrees **91.24%** of the time. That number is a
measurement, never an input: scoring may read the answer key, selection may
not.

## The caveat that has to come first

**On this corpus the question contains its expected measuring point verbatim
701 of 701 times — 100%.** `eval/build_qa_set.py` templates it in.

So question-anchoring is being measured under conditions no technician
reproduces. A clean reachability number on template wording is not evidence
that the anchoring works; it is mostly evidence that the template works. This
is why the perturbation set below exists and why its numbers, not the clean
ones, are the honest estimate.

## Guards

Each has a planted fault proving it fires.

| guard | what it asserts | proved by |
|---|---|---|
| AST over the **call graph** out of `build_prompt` | no reachable function reads `expected`, `gold`, or any key of either. 9 functions walked. The forbidden names are **derived from qa_set** — every key any case's `expected` dict actually has, 18 of them — so adding a field to `build_qa_set.py` forbids it the same day with nobody updating a list. | a planted `.get("point")` inside `selector.select` is caught |
| **mutation** | the `cb990e1` oracle anchor, re-inserted verbatim into `select_rows`, is caught. | caught by the same no-answer-key AST check, naming `expected`, `point`, `point_to_check` |
| signature | `selector.select` takes a question, never a case | — |
| reachability, gated at 100% | the selector drops none of the answers retrieval delivered | a starved budget (300 chars) is detected as dropping rows |
| **phrasing robustness**, gated at 100% | the same check under seven deterministic English rewrites | each rewrite is asserted to change a real template question, so a perturbed measurement is not the unperturbed one under another name |

**A note on scope, stated rather than left quiet:** the reachability gate runs
over all non-sealed cases, which includes `sealed2`. It contains no model
output — ground truth, retrieval, selector — so it is not an eval result and
does not spend the holdout. It is nonetheless the one piece of sealed2-derived
information available while the selector was being built. Narrowing it to dev
would have weakened a standing gate to keep a bookkeeping rule tidy.

## Reachability at 3,500

Share of the answers **retrieval delivered** that survive the slice.

### Non-sealed (1,484 delivered) — the gate's own scope

| bucket | n | blind | **question** | oracle |
|---|---:|---:|---:|---:|
| `numeric_exactness` | 889 | 0.8684 | **1.0000** | 1.0000 |
| `branch_following` | 122 | 0.9508 | **1.0000** | 1.0000 |
| `step_ordering` | 175 | 0.9543 | **1.0000** | 1.0000 |
| `direct_lookup` | 138 | 1.0000 | 1.0000 | 1.0000 |
| `symptom_remedy` | 160 | 1.0000 | 1.0000 | 1.0000 |
| **all** | **1484** | **0.9117** | **1.0000** | **1.0000** |

Question-derived selection matches the oracle on reachability without reading
anything. The 131 answers blind selection lost are all recovered.

### Under perturbation (dev, 1,162 delivered clean)

| rewrite | delivered | blind | **question** |
|---|---:|---:|---:|
| *(none)* | 1162 | 0.9088 | **1.0000** |
| `lower` | 1162 | 0.9088 | **1.0000** |
| `strip_punct` | 1157 | 0.9084 | **1.0000** |
| `loose_pins` | 1162 | 0.9088 | **1.0000** |
| `short_names` | 1162 | 0.9088 | **1.0000** |
| `abbreviate` | 1162 | 0.9088 | **1.0000** |
| `typo` | 1156 | 0.9083 | **1.0000** |
| `combined` | 1152 | 0.9097 | **1.0000** |

The **delivered** column is the honest part of this table. It falls 1,162 →
1,152 under `combined` because a rewritten question *retrieves* slightly
differently, and 10 answers stop being delivered at all. That is retrieval
degrading, not the selector, and it is why the two are reported separately.

**Diagnostic, so the robustness is not overclaimed.** Every rewrite preserves
the pin numbers — a wrong pin number is a different question, not a
perturbation of this one — and pin numbers are an obvious thing for the anchor
to be resting on. Measured: with **every purely numeric question token removed
from the selector**, dev reachability is still **1.0000** on all 1,162. The
anchor rests on connector and component names and on the entry-point rule, not
on the numbers alone.

## What this misses

Stated here and in the module docstring, not discovered later.

- A question naming the measuring point in words the manual never uses — a
  different name for the same connector. There is no synonym source in this
  repo that is not a model.
- A question whose only discriminating token is numeric and appears in many
  rows. IDF drives it near zero and the tail order decides, correctly but
  weakly.
- Free-form questions. Every perturbation keeps the question's structure and
  changes its surface; none of them rewrites word order or drops the technical
  vocabulary, and the 100% verbatim rate above means nothing here tests a
  technician describing a measuring point in their own words.
