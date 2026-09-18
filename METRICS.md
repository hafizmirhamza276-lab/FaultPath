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

> **The expected value is the whole criterion string, not a fragment of it.**
> `CA451` step 6 expects `Sensor output 0.2 to 4.6V`; an evaluator that pins
> `0.2 to 4.6V` is testing a different string than the manual prints. This
> metric once required `.2 to 4.6V` — a corrupt extraction that survived in the
> test set after the ground truth was fixed, penalising correct answers on a
> common-rail sensor voltage. `tests/test_extraction.py` now asserts every
> pinned string still exists in `golden/`.
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

### Deterministic provenance

The model refers to facts by `fact_id`; the system renders the citation from the
ground truth. Three metrics measure whether that holds.

`citation_resolvability` — every emitted citation re-opens the source PDF at the
page it names and finds its text there. A citation that cannot be resolved is
not a citation, it is a claim about one.

> **⚠ This metric separates ARCHITECTURES, not quality. Do not read a low score
> as "cites badly".**
>
> It can only score a system that **names `fact_id`s**. A system that answers
> from prose and types page numbers scores **0.0000 regardless of how good its
> answers are** — `citations_rendered` falls back to entries synthesised from
> the typed pages, and `resolve()` returns *"citation carries no pdf_page"*
> without ever opening the PDF.
>
> On the sealed run the real model scored **0.0000** here and **0.7435** on
> `citation_accuracy`. Those are not the same measurement and reading them as
> one understates the model by 0.74. It is never shown a `fact_id`:
> `build_prompt` renders `--- extract (page N, record CODE) ---` and no
> `fact_id` appears anywhere in the prompt.
>
> `GoodSystem` scores 0.9886 by returning `list(case["fact_ids"])` — the case's
> own answer key. It is not resolving anything; it is echoing ground truth it
> was handed, which makes its score an **existence proof that the metric can
> reach 1.0**, not evidence a real system can.
>
> Consequence, stated plainly: **it cannot discriminate between two real
> systems of the same architecture.** Both score 0.0000 without a fact-id-
> carrying retrieval layer and both score near 1.0 with one, whatever the
> answers say. The same caveat applies to `citation_span_precision` below,
> which also keys on `fact_id` and reported **NOT MEASURED** for the model.

`citation_span_precision` — the cited page is the page the fact is *on*, not
merely a page belonging to that code. 75% of measurements are not on their
code's first page, so a per-code citation is right by accident a quarter of the
time. **Also fact_id-keyed**: it scores only citations carrying a `fact_id` the
ground truth knows, so a system that names none is reported as NOT MEASURED
rather than scored — see the caveat above.

`uncited_claim_rate` — factual statements carrying no `fact_id`. These are the
model speaking on its own account, which the design is meant to make impossible.

On the local baseline these separate cleanly by mechanism: every case type that
names fact ids resolves at **1.0000** (1,126 cases), and every type that types
its own page resolves at **0.0000**. The metric is measuring adoption of the
mechanism, not luck.

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

> Tier 1 is implemented in `eval/`. Tier 2 is not wired: judge metrics register
> through the same `Metric` interface with `TIER = 2` and are aggregated into a
> separate block, so they cannot be quoted as Tier-1 figures.

```bash
python pipeline/extract_golden.py             # rebuild ground truth from the PDF
python eval/build_qa_set.py                   # regenerate the golden Q&A set
python eval/run_eval.py --chunker structural --system good --label baseline
python eval/compare.py eval_out/runs/<a>.json eval_out/runs/<b>.json
```

Runs land in `eval_out/runs/<timestamp>_<label>.json` with the git SHA, a
dirty-tree flag, the full config, every metric, per-case rows and wall time.

**Tier 1 has no noise floor.** It is deterministic, so `compare.py` treats any
movement as real and exits non-zero on a regression. There is no tolerance band
to hide a small regression in. A Tier-2 comparison will need one; the two must
never share a threshold.

**Case ids are frozen.** Runs are matched case by case on `qa_set.json` ids. If
`build_qa_set.py` ever changes its id scheme, `compare.py` refuses with
`ids changed, runs not comparable` rather than reporting every case as new.

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
python tests/test_eval_harness.py
```

Three things, in order of how badly they bite:

1. **Every metric proves it can fail** on known-bad input, and the run aborts if
   one cannot. Audit checks `E4` and `H2` each sat at zero for months while
   being structurally incapable of returning anything else, and both zeros were
   read as statements about the document. A metric nobody has watched fail is
   not a measurement.
2. **Good passes 7/7 gates; weak fails 0/7.** If weak passes one, that gate has
   a hole. Each gate is additionally checked for direction — weak must be
   strictly worse, not merely different.
3. **Two runs of one config produce identical numbers**, asserted rather than
   assumed.

`run_weak` is a deliberately bad system: it reports **a different real criterion
from the manual**, cites wrong pages, ignores the model filter, and invents
procedures for codes that do not exist. If it ever passes a gate, the harness
has a hole in it and the gate is not measuring what it claims to.

> **The floor must fail BY CONSTRUCTION, and until 2026-09-16 it did not.**
> It used to nudge every number by 10% and round. Every integer from 0 to 4 is
> a fixed point of `round(n × 1.1)`, and this manual is made of small integers,
> so **650 of 1,146 golden criteria came back unchanged** — including
> `Max. 1 Ω` (235×) and `Min. 1 MΩ` (244×). On 57% of `numeric_exactness` cases
> the deliberately-bad system emitted the **correct** answer and scored 1.0.
>
> `weak`'s `numeric_exactness` was **0.4994**; it is now **0.0000**. Every
> good-vs-weak separation previously recorded on that metric was carried by the
> 43% of cases where the arithmetic happened to move a number. The gate-level
> claim ("weak fails 0/7") always held, because it is an aggregate — which is
> exactly why it did not catch this. A floor does not need to fail on average;
> it needs to fail on every case.
>
> `tests/test_eval_harness.py` now asserts that weak's answer never contains
> the criterion, over all 1,146 criteria and all three corpus scopes, and
> carries a self-test showing the old `_nudge` fails that assertion on 650 of
> them. Full account: `reports/weak_floor_finding.md`.

Re-run this check whenever a metric is added or a threshold is changed. A
scorecard that cannot fail is not a scorecard.

---

## Known limitations

The full source-document audit lives in `README.md` (21 findings: 6 high, 8
medium, 5 low, 2 informational). The items that directly bound what these
metrics can tell you:

1. **Provenance is verified at PAGE granularity, never at row granularity.**
   Every fact records `{manual_page, pdf_page, table_index, row_index}`, but
   `row_index` and `table_index` are **recorded by every kind and checked by
   nothing.** Both checks that could see them are page-granular: the resolver
   asks whether the text is on the cited page, and `page_containment` asks
   whether the page is in a legitimate range. Measured directly — mutating a
   fact's `row_index` or `table_index` by one is invisible to both:

   ```
   measurement row_index  +1  -> resolution 1.0000  containment 1.0000  MISSED
   measurement table_index +1  -> resolution 1.0000                      MISSED
   title       row_index  +1  -> resolution 1.0000  containment 1.0000  MISSED
   ```

   A wrong row on the *right* page therefore passes. Moving a fact to a
   different page is caught (resolution drops), and moving it outside its
   kind's legitimate range is caught (`page_containment` drops) — it is only
   the same-page case that is open.

   This is a property of the provenance schema, not of any one kind; titles
   did not introduce it, enumerating them is what prompted the measurement.
   Closing it would require **independent re-derivation from the PDF at check
   time** — re-reading the table and confirming the row, the way
   `pipeline/structural.py` re-derives step structure rather than trusting the
   extractor. That is a separate decision, not yet taken.

2. **Ground truth covers Section 40 and the symptom trees** — 174 failure codes
   plus 57 H-Mode and S-Mode trees. It does not cover Testing and Adjusting,
   which is where several things a technician asks about actually live: there
   is no leak tree, no air-conditioner tree and no horn tree, and
   `knowledge/symptom_unmapped.tsv` records 14 such phrasings rather than
   forcing them to the nearest match.

3. **Diagrams are not evaluated.** 220 pages carry almost no extractable text and
   103 of those are dense vector schematics. Any answer depending on reading a
   diagram is outside what this harness measures.

4. **`DR31KX` has no parsable steps** — its layout matches none of the three
   table formats in Section 40. Excluded from step-based metrics.

5. **27 steps are machine-repaired, not verified.** These carry
   `extraction_warning: column_split_recovered` and should be checked by eye
   before being treated as ground truth.

6. **Four source-document conflicts remain unresolved** — `F@BBZL`'s action
   level disagrees between the summary table and its detail page, and `CA234`,
   `DKR2MA`, `DR31KX` have a level in one source only. `D8ARKR` references
   `CA445`, which does not exist in this manual. These are defects in the manual,
   not extraction errors; resolve with Komatsu before treating either value as
   authoritative.

7. **Single model, single manual.** Every number here describes PC200-10M0
   behaviour. Adding a second manual requires regenerating the golden set and
   re-baselining, because cross-model confusion is a failure mode that cannot
   appear in a single-model test set.

---

## Verification status of the ground truth

Every metric in this document is computed against `golden/`. How far `golden/`
itself has been verified against the source PDF bounds what any of them can
claim.

| status | facts | what it means |
|---|---:|---|
| `resolver_verified` | 6,023 | text found on the cited PDF page by `eval/citations.py` |
| `human_verified` | **0** | a person read the page — **never performed** |
| `unverified` | 531 | 504 known limitations + 27 machine-repaired causes |

**These are DERIVED every run, not maintained here.** The fidelity stage calls
`loader.build_verification()` on its own result and gates on the artefact
matching it, so the numbers above are a snapshot of a computed value rather
than a figure anyone keeps up to date.

They were 3,266 / 39 until `1cdf075`, because
`reports/fact_verification.json` had no producer and had not been rebuilt since
symptom trees existed. `load_ground_truth` falls through to `unverified` for a
fact it cannot find, so **2,757 facts the resolver verifies were reported
`unverified` to API callers** — the system understating its own grounding. The
pin here and the one in `tests/test_orchestrator.py` both read that same frozen
file, so they agreed with each other and the agreement read as corroboration.
See `reports/frozen_verification_findings.md`.

**The human round was built and not run.** `numeric_exactness`,
`fabricated_values` and `citation_accuracy` all rest on facts that resolve
exactly — 872/872 measurements, 1437/1437 branches — so those figures carry
external confirmation. What has no independent reading is reassembled cause
text on 27 steps, which feeds `content_recall` and `path_correctness` but no
numeric metric.

The resolver and the extractor share normalisation rules, so a misreading common
to both is invisible to the resolver. Quoting `resolver_verified` as if it were
independent human confirmation would overstate it.
