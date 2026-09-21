# Finding — the prompt selector reads the answer key

**Date:** 2026-09-21 · **Component:** `eval/model_system.py` `_anchors()` /
`select_rows()` · **Status:** reported and measured, not fixed

## The defect

`build_prompt` decides which rows of each retrieved record the model is shown.
Since `6cbd647` that decision reads `case["expected"]` — the scoring payload.

```python
# eval/model_system.py:170   (inside _anchors)
exp = case.get("expected") or {}
point = exp.get("point") or exp.get("point_to_check") or None
step = exp.get("step")
```

```python
# eval/model_system.py:232   (inside select_rows)
wants_value = bool((case.get("expected") or {}).get("criteria"))
order = (("header", "anchored", "measurement", "remedy", "branch", "step")
         if wants_value else
         ("header", "anchored", "step", "remedy", "branch", "measurement"))
```

`expected.point` is the measuring point whose value is the answer.
`expected.step` is the step the answer is on. `expected.criteria` is the answer
itself. **`expected` does not exist at inference time.** A deployed assistant
receives a technician's question and nothing else; it has no field telling it
which of a record's rows to look at.

So the harness locates the answer row, promotes it above the budget cut, and
then scores the model on whether it read the row the harness handed it. That is
§4 of `PROJECT_STATE.md`: the harness doing the model's work and reporting it as
the model's score.

The `_anchors` docstring argues — correctly — that selecting on overlap with the
question string would make the prompt depend on phrasing. That reasoning is
sound and is also how this got in: the case's own fields include its answer, so
"derive from the case, not from the question" walked straight into the answer
key.

## Every read of `expected` / `gold` in `eval/`, by path

Scoring code reading `expected` is not a leak. Prompt-building and
context-selecting code reading it is. The split:

### On the prompt / context path — LEAKS

| file:line | what it reads | effect |
|---|---|---|
| `eval/model_system.py:170` | `expected.point`, `expected.point_to_check`, `expected.step` | promotes the answer row to priority 2 |
| `eval/model_system.py:232` | `expected.criteria` | selects measurement-first vs prose-first row order |

Those are the only two. Both are reached from
`build_prompt` → `select_rows` → `_anchors`, and therefore from
`ModelSystem.answer` (`:459`) and `ModelSystem.converse` (`:509`).

### On the prompt / context path — CLEAN

| file:line | what it passes | why it is not a leak |
|---|---|---|
| `eval/model_system.py:456` `ModelSystem.retrieve` | `case["question"]`, `case["filters"]` | the question is the runtime input; `filters` is `{model, manual_id}`, which a technician supplies about their own machine |
| `eval/model_system.py:268` `build_prompt` | `filters`, `case["question"]`, chunk text | no `expected` read of its own |
| `eval/adapters.py` `LocalBM25Retriever.search` | query + filters only | never sees the case object |
| `eval/chunkers.py` | records only | corpus construction, case-independent |

### Reads `expected`, but is not on the prompt path

| file:line | context |
|---|---|
| `eval/synthetic.py:159`, `:231` | `GoodSystem.answer` / `.converse` — the REFERENCE answers out of the ground truth **by declared category**. Its perfection is asserted, not measured. |
| `eval/synthetic.py:326`, `:362` | `WeakSystem.answer` / `.converse` — the FLOOR, same construction inverted. |
| `eval/refusal.py:132` (via `golden_facts`) | post-hoc judgement about an answer already produced. Reaches the model path only through `ModelSystem._derive`, which runs **after** the call and changes no prompt. |
| `eval/build_qa_set.py`, `eval/adversarial.py` | case construction — `expected` is the thing being written |
| `eval/metrics/*` | scoring |

`GoodSystem` and `WeakSystem` are not defects. A reference system that answers
out of the answer key is the definition of a reference system, and it is
declared on the class (`CATEGORY = "reference"` / `"floor"`). What makes the two
`model_system.py` lines different is that `ModelSystem.CATEGORY` is
`under_test`: nothing about it is asserted, its score **is** the measurement,
and a measurement that consults the answer key is not one.

## Scale

`SELECTOR_BLIND=1` (added in this commit, default off) hides `expected` from the
selector: no anchors, and one fixed order — `header, measurement, remedy,
branch, step` — that does not depend on the case. Budget unchanged at 3,500.

Reachability of the expected content in the extracts, over the 1,484 non-sealed
cases whose answer **retrieval delivered** (same measurement as the budget
sweep, so the two are comparable):

| bucket | n | oracle | blind | delta |
|---|---:|---:|---:|---:|
| `numeric_exactness` | 889 | 1.0000 | **0.8684** | **−0.1316** |
| `branch_following` | 122 | 1.0000 | 0.9508 | −0.0492 |
| `step_ordering` | 175 | 1.0000 | 0.9543 | −0.0457 |
| `direct_lookup` | 138 | 1.0000 | 1.0000 | 0.0000 |
| `symptom_remedy` | 160 | 1.0000 | 1.0000 | 0.0000 |
| **overall** | **1484** | **1.0000** | **0.9117** | **−0.0883** |

**131 of 1,484 answers are in the retrieved text and are cut by the blind
slice.** The anchor is what put them back. `direct_lookup` and `symptom_remedy`
are unaffected because their answers are header rows or short records that fit
regardless — the leak is concentrated exactly where the gated metrics are.

## Evidence — one case, verbatim

Case `d3464de2ea`, `numeric_exactness`, record `D8AQKR`.

```
Q:  D8AQKR ke liye Between CM02 (female) (9), CP01 (female) (64),
    CE02 (female) (47), AC01 (female) (1), N08 (female) (10),
    CK05 (female) (64) or K02 (female) (B) and ground par resistance ki
    standard value kya honi chahiye?

expected: {'quantity': 'Resistance',
           'point': 'Between CM02 (female) (9), CP01 (female) (64), ...
                     K02 (female) (B) and ground',
           'criteria': 'Min. 1 MΩ', 'step': None}
```

The answer row, present in the retrieved text under both selectors:

```
Measurement. Resistance. Measuring point: Between CM02 (female) (9),
CP01 (female) (64), CE02 (female) (47), AC01 (female) (1),
N08 (female) (10), CK05 (female) (64) or K02 (female) (B) and ground.
Standard value: Min. 1 MΩ (page 40-291)
```

| | chars |
|---|---:|
| full retrieved text (6 chunks) | 89,147 |
| oracle extracts | 20,855 |
| blind extracts | 20,853 |

The two prompts are the same size to within two characters. **The blind one does
not contain the answer row and the oracle one does.** `D8AQKR` carries dozens of
near-identical `Min. 1 MΩ` continuity rows; the blind selector takes them in
record order until the budget runs out, and the one that was asked about is not
near the front. `_anchors` matches `expected.point` against each line and lifts
that row to priority 2.

This is the clearest possible shape of the defect: identical budget, identical
retrieval, identical record — the only difference is that one selector was told
the answer.

## Measured — blind vs oracle, same model, same retriever, same cache

Both runs: `--system model --split train --chunker structural --retriever bm25
--k 20`, 1,511 non-sealed qa_set cases + 15 injection + 41 conversation = 1,567
scored. `gpt-4.1` / `gpt-4.1-2025-04-14`, api-version `2025-01-01-preview`.

- oracle: `eval_out/runs/20260918T132414_b3500_model.json` (`cb990e1`), 1,557
  fresh calls
- blind: `eval_out/runs/20260921T051003_blind_model.json`, **1,011 fresh calls
  and 567 cache hits**

Those 567 hits are not stale. The cache key covers the exact prompt text, so a
hit means the blind selector produced a **byte-identical prompt** — the record
fit inside 3,500 characters and there was nothing for the anchor to promote.
Measured directly over the 1,511 non-sealed cases: **541 (35.8%) get the same
prompt either way; 970 (64.2%) do not.**

### Gates

| gate | oracle | blind | delta | | |
|---|---:|---:|---:|---|---|
| `numeric_exactness` ≥ 0.98 | 0.9831 | **0.8594** | **−0.1237** | PASS | **FAIL** |
| `citation_accuracy` ≥ 0.95 | 0.9747 | **0.8607** | **−0.1140** | PASS | **FAIL** |
| `fabricated_values` ≤ 0.005 | 0.0013 | 0.0012 | −0.0001 | PASS | PASS |
| `refusal_correctness` ≥ 1.0 | 1.0000 | 1.0000 | 0.0000 | PASS | PASS |
| `over_refusal` ≤ 0.05 | 0.0007 | 0.0053 | +0.0047 | PASS | PASS |
| `hit_rate@5` ≥ 0.95 | 0.9980 | 0.9980 | 0.0000 | PASS | PASS |
| `filter_correctness` ≥ 1.0 | 1.0000 | 1.0000 | 0.0000 | PASS | PASS |
| **total** | **7/7** | **5/7** | | | |

**Both gates the selector can reach, it carries.** The five it cannot —
retrieval, filters, refusal, fabrication — are identical or near-identical,
which is the control: the blind flag changes only what is shown, and nothing
that does not depend on what is shown moved.

### Headline metrics

| metric | n | oracle | blind | delta |
|---|---:|---:|---:|---:|
| `numeric_exactness` | 889 | 0.9831 | **0.8594** | **−0.1237** |
| `citation_accuracy` | 1500 | 0.9747 | **0.8607** | **−0.1140** |
| `contradiction` (lower better) | 889 | 0.0146 | **0.0619** | **+0.0472** |
| `content_recall` | 1501 | 0.8353 | 0.7932 | −0.0421 |
| `groundedness` | 1565 | 0.9951 | 0.9895 | −0.0057 |
| `over_refusal` (lower better) | 1501 | 0.0007 | 0.0053 | +0.0047 |
| `uncited_claim_rate` | 1548 | 0.7374 | 0.7393 | +0.0019 |
| `completeness` | 300 | 0.2448 | 0.2459 | +0.0011 |
| `fabricated_values` (lower better) | 1567 | 0.0013 | 0.0012 | −0.0001 |
| `faithfulness_det` | 1560 | 1.0000 | 1.0000 | 0.0000 |
| `citation_presence` | 1501 | 0.9993 | 0.9993 | 0.0000 |
| `injection_resistance` | 15 | 0.8667 | 0.8667 | 0.0000 |
| `refusal_correctness` / `clean_refusal` | 10 | 1.0000 | 1.0000 | 0.0000 |
| `pii_leakage` / `model_leakage` | — | 0.0000 | 0.0000 | 0.0000 |
| `context_precision`, `hit_rate@5`, `filter_correctness` | — | — | — | 0.0000 |

`contradiction` more than quadrupling is the honest reading of the same event:
blind, the model is more often shown a *different* `Min. 1 MΩ` row than the one
asked about, and answering from it is a contradiction rather than an absence.

### Per-bucket

`numeric_exactness` is scored only on its own bucket (n=889): **0.9831 →
0.8594, −0.1237.**

**`citation_accuracy`**

| bucket | n | oracle | blind | delta |
|---|---:|---:|---:|---:|
| `numeric_exactness` | 888 | 0.9921 | 0.8525 | **−0.1396** |
| `step_ordering` | 177 | 0.9605 | 0.7966 | **−0.1638** |
| `branch_following` | 122 | 0.9590 | 0.7951 | **−0.1639** |
| `symptom_remedy` | 160 | 0.8812 | 0.8938 | +0.0125 |
| `direct_lookup` | 137 | 1.0000 | 1.0000 | 0.0000 |
| `cross_ref_hop` | 7 | 1.0000 | 1.0000 | 0.0000 |
| `precondition` | 8 | 1.0000 | 1.0000 | 0.0000 |
| **all** | **1499** | **0.9746** | **0.8606** | **−0.1141** |

**`content_recall`**

| bucket | n | oracle | blind | delta |
|---|---:|---:|---:|---:|
| `step_ordering` | 177 | 0.7175 | 0.5706 | −0.1469 |
| `numeric_exactness` | 889 | 0.9741 | 0.9258 | −0.0484 |
| `symptom_remedy` | 160 | 0.4740 | 0.4406 | −0.0333 |
| `direct_lookup` | 138 | 0.5580 | 0.5664 | +0.0085 |
| `branch_following` | 122 | 0.7623 | **0.8443** | **+0.0820** |
| `cross_ref_hop` | 7 | 1.0000 | 1.0000 | 0.0000 |
| `precondition` | 8 | 1.0000 | 1.0000 | 0.0000 |
| **all** | **1501** | **0.8353** | **0.7932** | **−0.0421** |

`branch_following` `content_recall` is **better blind**. The oracle order for a
value question ranks `branch` fifth, so anchoring on a measurement pushed branch
prose out; the blind order gives branch rows a fixed slot. This is the one place
where the oracle selector was actively *costing* the model something, and it is
also the one `cb990e1` regression I reported — 0.8525 → 0.7623. Blind puts it at
0.8443. **That regression was caused by the leak, not by the budget.**

### Cases that changed outcome

Per-case values on the five gated per-case metrics:

| gated metric | n | worse blind | better blind | unchanged |
|---|---:|---:|---:|---:|
| `citation_accuracy` | 1499 | 176 | 5 | 1318 |
| `numeric_exactness` | 889 | **110** | **0** | 779 |
| `over_refusal` | 1501 | 7 | 0 | 1494 |
| `fabricated_values` | 1567 | 3 | 4 | 1560 |
| `refusal_correctness` | 10 | 0 | 0 | 10 |

**210 of 1,567 cases (13.4%) changed on at least one gated metric** —
`numeric_exactness` 142, `step_ordering` 31, `branch_following` 28,
`symptom_remedy` 5, `cross_ref_hop` 4.

`numeric_exactness` changed in **one direction only: 110 worse, 0 better.** A
selection rule that helps and never hurts on the metric it was anchored to is
the signature of an oracle rather than of a heuristic.

### Reachability, blind, at 3,500

The reachability table above is the same measurement at blind settings: overall
**1.0000 → 0.9117**, `numeric_exactness` **1.0000 → 0.8684**.

Reachability 0.8684 and scored `numeric_exactness` 0.8594 on the same bucket
means the model converts nearly all of what it is shown: the 12.4-point drop in
the score is almost exactly the 13.2-point drop in what reaches the prompt. The
leak is in the input path, not in the model's reading of it.

## The consequence, stated plainly

**What the `cb990e1` numbers measure:** this model, on a prompt where the
harness has already found the row the question is about, given a corpus where
the correct record was retrieved.

**What they do not measure:** the model's ability to find the right row in a
record it was handed. That is a real capability, it is the one a deployed
assistant needs, and the oracle selector supplies it from the answer key.

Concretely:

- `numeric_exactness 0.9831` and `citation_accuracy 0.9747` are **upper
  bounds**, not measurements. Blind they are **0.8594** and **0.8607**.
- **7/7 gates is not a claim about the model. Blind it is 5/7**, and the two
  that fall are exactly the two the selector can reach.
- The improvement recorded across `b0c26fa` → `cb990e1` splits cleanly. On
  `numeric_exactness`:

  | prompt builder | budget | anchors | `numeric_exactness` |
  |---|---:|---|---:|
  | head slice `text[:N]` (`b0c26fa`) | 1,800 | no | 0.5450 |
  | anchored, case-shaped order (`6cbd647`) | 1,800 | yes | 0.9010 |
  | anchored, case-shaped order (`cb990e1`) | 3,500 | yes | **0.9831** |
  | **blind, fixed order** | **3,500** | **no** | **0.8594** |

  **0.5450 → 0.8594 is real** — a bigger budget and putting the measurement
  table ahead of the header prose are both things a system can do without the
  answer key. **0.8594 → 0.9831 is oracle.** Roughly 71% of the total gain
  survives blind selection and 29% does not.
- The **spent sealed measurement is unaffected**: it was taken at `2e8edb2`,
  before `6cbd647`, with no anchors and a head slice. Its caveat is still "this
  model on 34.3% of context" and no oracle caveat is owed on it.
- `good` and `weak` are unaffected in a different way: they never call
  `select_rows`. Their scores did not move and could not.

None of this makes the harness wrong about the model being better at 3,500 than
at 1,800. It makes the *size* of that improvement unattributable until the blind
column is read alongside it.

## Why this went unnoticed

The leak arrived inside a fix for the opposite problem. `6cbd647` was written
because the head slice discarded the measurement table on 87% of chunks
(`reports/prompt_truncation_finding.md`), and the obvious repair for "the
prompt drops the answer" is "keep the answer". The docstring even says so, in
priority 2, and then argues explicitly for deriving from the case rather than
from the question — an argument about *phrasing independence* that never asked
whether the case's fields are available at inference time.

The gate added in `cb990e1` measures exactly the wrong thing for catching it:
*"the selector drops none of the 1,484 answers retrieval delivered"* is
trivially satisfied by a selector that reads the answer key, and the closer the
selector gets to being an oracle the greener that check goes. It was built to
detect truncation and it does; it is structurally blind to this.

It is the same family as the rest of the sequence — a number that looks like a
property of the thing measured and is a property of the apparatus — with the
distinction that the previous instance made the model look worse than it is and
this one makes it look better.
