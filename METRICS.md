# Metrics Definitions

How the diagnostic assistant is measured, what each number means, and what to
do when one of them drops.

Everything here is scored against `golden/` — a ground-truth dataset extracted
deterministically from Komatsu Shop Manual **SEN06867-13** (PC200-10M0,
S/N 700001 and up). No model was used to build the ground truth, which is what
makes these numbers defensible.

---

## Two tiers, kept separate

**Tier 1 — Deterministic.** String and number comparison against the manual.
Reproducible, auditable, and identical on every re-run. Every headline number
reported to management comes from here.

**Tier 2 — LLM-as-judge.** Faithfulness and completeness on free-form prose.
Useful for catching things regex cannot, but a judge is itself a model with its
own error rate. Report these separately and never blend them into a Tier-1
figure.

The distinction matters in a review. "Our hallucination rate is 0.3%" is a
claim you can defend line by line if it came from Tier 1. If it came from a
judge model, someone will eventually ask who checked the judge.

---

## Pipeline metrics

### `numeric_exactness`
**Target: ≥ 98%**

Of all questions asking for a measurement value, the fraction where the system
reproduced the criterion intact. `Min. 100kΩ` must come back as `Min. 100kΩ`.
Spacing and `ohm` vs `Ω` are tolerated; digits are not.

This is the single most important metric in the system. A technician acting on
`Min. 90kΩ` when the manual says `Min. 100kΩ` will pass a harness that should
have failed, and the machine goes back out broken.

Typical cause of a drop: the model is paraphrasing retrieved text instead of
quoting it. Fix in the prompt, not the retriever.

> **Currently invalid.** The ground truth for `CA451` step 6 has been corrected
> to `Sensor output 0.2 to 4.6V`, but case `22858d0adc` in `qa_set.json` still
> requires the verbatim string `".2 to 4.6V"`. Until the set is rebuilt, this
> metric penalises the correct answer on a safety-critical value. See the warning
> at the top of `README.md`.
>
> **Normalise before comparing.** Criteria strings mix two ohm codepoints —
> U+2126 OHM SIGN (353 occurrences) and U+03A9 GREEK CAPITAL LETTER OMEGA (349).
> A verbatim comparator that does not NFC-normalise first will fail roughly half
> of all resistance cases for a reason that has nothing to do with correctness.

### `hallucination_rate_values`
**Target: ≤ 0.5%**

Every number in the answer is checked against the complete set of numbers that
appear anywhere in that failure code's ground-truth record — criteria, pin
numbers, monitoring codes, page numbers, step indices, and numbers echoed from
the question. Anything left over was not read from the manual.

This is the metric to lead with when management asks "how much does it make
things up". It is a count, not an impression, and every flagged value is listed
per-case in `results_<run>.csv` so any individual claim can be traced back.

### `citation_accuracy`
**Target: ≥ 95%**

The cited manual page must be the page the fact actually came from. Reported
alongside `citation_present_rate` on purpose: a system that always cites but
cites the wrong page is worse than one that admits it does not know, because it
manufactures confidence.

### `refusal_correctness_adversarial`
**Target: 100%**

The golden set contains fabricated codes (`CA999`, `B@BZZZ`) that look
plausible but do not exist in the manual. The only correct behaviour is to say
so. Anything else is a procedure invented from nothing.

100% is the target because there is no acceptable failure rate here. A wrong
procedure for a code that does not exist has no upper bound on how wrong it can
be.

### `clean_refusal_rate` / `invented_procedure_rate`
A refusal that still leaks measurement values is not a refusal. These two split
the failure mode: did it refuse, and did it stay quiet while refusing.

### `over_refusal_rate`
**Target: ≤ 5%**

The mirror image. A system that refuses everything scores perfectly on safety
and is useless. Track both directions or you will optimise into a system nobody
uses.

### `content_recall`
Fraction of required facts present in the answer — the title, the first cause,
the branch outcome. This is the completeness measure: did the technician get
everything they needed, not just something correct.

---

## Module metrics

Pipeline metrics tell you *that* something is wrong. Module metrics tell you
*where*. When `numeric_exactness` drops, these decide whether to fix the
retriever, the prompt, or the extractor.

| Metric | Target | What it isolates |
|---|---|---|
| `retrieval_hit@1` | ≥ 90% | Right code ranked first |
| `retrieval_hit@5` | ≥ 95% | Right code in the candidate set at all |
| `retrieval_mrr` | ≥ 0.90 | Ranking quality overall |
| `crossref_hop_recall` | ≥ 95% | Pointer-only codes redirect correctly |
| `precondition_accuracy` | ≥ 95% | Correct code solved first when several are shown |
| `step_selection_accuracy` | ≥ 95% | Executor walks the tree in the right order |
| `model_filter_applied_rate` | 100% | Model/serial filter enforced before any value is quoted |

**Reading them together.** If `retrieval_hit@5` is high but `numeric_exactness`
is low, the right page was found and the model mangled it — a generation
problem. If `hit@5` is low, the model never saw the right page and everything
downstream is noise. Always read retrieval first.

`model_filter_applied_rate` has a target of 100% for the same reason as
adversarial refusal. PC200 and PC490 share failure codes but not pin numbers.
Retrieval will cheerfully return the PC200 page for a PC490 question unless the
filter is enforced.

---

## Case types in the golden set

| Type | n | Tests |
|---|---|---|
| `numeric_exactness` | 846 | Measurement criteria reproduced verbatim |
| `direct_lookup` | 173 | Code → title, action level, machine effect |
| `step_ordering` | 164 | Correct first check |
| `branch_following` | 116 | Format-A YES/NO branch outcomes |
| `precondition` | 10 | Which code to solve first when several show |
| `cross_ref_hop` | 9 | Pointer-only codes that must redirect |
| `adversarial_unknown` | 6 | Fabricated codes — must refuse |
| `adversarial_model` | 6 | Real code, wrong machine — must refuse |
| **Total** | **1,330** | |

The adversarial counts are small by design. They are pass/fail gates, not
averages — one failure among six is a release blocker, not a 17% dip.

`cross_ref_hop` covers the 9 codes whose entire troubleshooting content is a
pointer somewhere else (`CA111`, `CA227`, `CA442`, and six more). These are the
cases where a conventional RAG pipeline returns a technically correct chunk that
helps nobody.

---

## Running an evaluation

> `build_qa_set.py`, `evaluate.py` and `make_mock_runs.py` are **not yet
> implemented**. `eval/` is their intended home. Everything in this section and
> the next describes the target state, not a working command.

```bash
python pipeline/extract_golden.py             # rebuild ground truth from the PDF
python eval/build_qa_set.py                   # regenerate the golden Q&A set
python eval/evaluate.py runs/<run>.jsonl --run-id <name>
```

Outputs land in `eval_out/`:

- `report_<name>.md` — the scorecard, ready to paste into a review deck
- `metrics_<name>.json` — machine-readable, for dashboards
- `results_<name>.csv` — per-case, including every fabricated value found
- `history.jsonl` — one row per run, appended, for trend lines over time

`history.jsonl` is what turns this from a snapshot into evidence. Six months of
rows showing hallucination rate falling as changes ship is a far stronger
argument than any single number.

---

## Validating the harness itself

```bash
python eval/make_mock_runs.py
python eval/evaluate.py runs/run_good.jsonl --run-id good   # expect 7/7 gates pass
python eval/evaluate.py runs/run_weak.jsonl --run-id weak   # expect 0/7 gates pass
```

`run_weak` is a deliberately bad system: it nudges every number by 10%, cites
wrong pages, ignores the model filter, and invents procedures for codes that do
not exist. If it ever passes a gate, the harness has a hole in it and the gate
is not measuring what it claims to.

Re-run this check whenever a metric is added or a threshold is changed. A
scorecard that cannot fail is not a scorecard.

---

## Known limitations

The full source-document audit lives in `README.md` (20 findings: 6 high, 9
medium, 3 low, 2 informational). The items that directly bound what these
metrics can tell you:

1. **Ground truth covers Section 40 failure codes only** — 174 codes. H-Mode and
   S-Mode symptom trees are not extracted, so symptom-entry queries are
   unmeasured.

2. **Diagrams are not evaluated.** 220 pages carry almost no extractable text and
   103 of those are dense vector schematics. Any answer depending on reading a
   diagram is outside what this harness measures.

3. **`DR31KX` has no parsable steps** — its layout matches none of the three
   table formats in Section 40. Excluded from step-based metrics.

4. **27 steps are machine-repaired, not verified.** These carry
   `extraction_warning: column_split_recovered` and should be checked by eye
   before being treated as ground truth.

5. **Four source-document conflicts remain unresolved** — `F@BBZL`'s action
   level disagrees between the summary table and its detail page, and `CA234`,
   `DKR2MA`, `DR31KX` have a level in one source only. `D8ARKR` references
   `CA445`, which does not exist in this manual. These are defects in the manual,
   not extraction errors; resolve with Komatsu before treating either value as
   authoritative.

6. **Single model, single manual.** Every number here describes PC200-10M0
   behaviour. Adding a second manual requires regenerating the golden set and
   re-baselining, because cross-model confusion is a failure mode that cannot
   appear in a single-model test set.
