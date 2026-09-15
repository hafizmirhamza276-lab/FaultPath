# Sealed evaluation baseline -- pre-LLM

Recorded so a real model has something to be compared against. Every number
here is the **synthetic** good and weak system over the sealed holdout, run
before any model was wired in.

    python eval/run_eval.py --chunker structural --corpus all --split sealed --system good

## The set

354 of 1865 cases (19.0%), over 46 records. Derived, never stored as a list of
ids -- see `eval/sealed.py` for the rule, and for why selection may never read
an eval result.

| section/type | corpus | sealed | share |
|---|---:|---:|---:|
| section40/adversarial_model | 6 | 1 | 16.7% |
| section40/adversarial_unknown | 6 | 1 | 16.7% |
| section40/branch_following | 116 | 24 | 20.7% |
| section40/cross_ref_hop | 9 | 2 | 22.2% |
| section40/direct_lookup | 173 | 35 | 20.2% |
| section40/numeric_exactness | 846 | 137 | 16.2% |
| section40/precondition | 10 | 2 | 20.0% |
| section40/step_ordering | 164 | 33 | 20.1% |
| symptoms/branch_following | 37 | 7 | 18.9% |
| symptoms/numeric_exactness | 232 | 52 | 22.4% |
| symptoms/step_ordering | 57 | 11 | 19.3% |
| symptoms/symptom_remedy | 209 | 49 | 23.4% |

All twelve buckets are covered. The four small ones -- `cross_ref_hop` 2,
`precondition` 2, `adversarial_model` 1, `adversarial_unknown` 1 -- are
single-digit because the CORPUS is, not because the fraction is: there are 9
and 10 of the first two in the whole of qa_set. Their individual rates are not
meaningful and should not be quoted alone. They are in the set so the aggregate
is not silently blind to a behaviour.

## Baseline

354 cases, `structural`, `--corpus all`, bm25.

| metric | n | good | weak | better | |
|---|---:|---:|---:|---|---|
| `answer_coverage@1` | 352 | 0.8892 | 0.6932 | higher |  |
| `answer_coverage@10` | 352 | 0.9943 | 0.7841 | higher |  |
| `answer_coverage@20` | 352 | 1.0000 | 0.7926 | higher |  |
| `answer_coverage@3` | 352 | 0.9858 | 0.7528 | higher |  |
| `answer_coverage@5` | 352 | 0.9943 | 0.7727 | higher |  |
| `ceiling_recall` | 352 | 1.0000 | 1.0000 | higher |  |
| `chunks_to_cover` | 352 | 1.2131 | 1.1449 | lower |  |
| `citation_accuracy` | 352 | 1.0000 | 0.0000 | higher | **gate** |
| `citation_presence` | 352 | 1.0000 | 0.8608 | higher |  |
| `citation_resolvability` | 352 | 0.9886 | 0.0000 | higher |  |
| `citation_span_precision` | 264 | 1.0000 | -- | higher |  |
| `clean_refusal` | 2 | 1.0000 | -- | higher |  |
| `completeness` | 75 | 1.0000 | 0.0010 | higher |  |
| `content_recall` | 352 | 1.0000 | 0.2713 | higher |  |
| `context_precision` | 352 | 0.6882 | 0.6526 | higher |  |
| `contradiction` | 189 | 0.0000 | 0.4021 | lower |  |
| `fabricated_values` | 354 | 0.0000 | 0.0834 | lower | **gate** |
| `faithfulness_det` | 305 | 1.0000 | 0.8534 | higher |  |
| `filter_correctness` | 354 | 1.0000 | 0.9972 | higher | **gate** |
| `first_relevant_rank` | 352 | 1.0540 | 1.2159 | lower |  |
| `fragmentation_gap@10` | 352 | 0.0014 | 0.0767 | lower |  |
| `groundedness` | 305 | 1.0000 | 0.8068 | higher |  |
| `hit_rate@1` | 352 | 0.9830 | 0.8636 | higher |  |
| `hit_rate@10` | 352 | 0.9972 | 0.9489 | higher |  |
| `hit_rate@20` | 352 | 1.0000 | 0.9517 | higher |  |
| `hit_rate@3` | 352 | 0.9972 | 0.9176 | higher |  |
| `hit_rate@5` | 352 | 0.9972 | 0.9233 | higher | **gate** |
| `map` | 352 | 0.9207 | 0.8556 | higher |  |
| `model_leakage` | 1 | 0.0000 | 1.0000 | lower |  |
| `mrr` | 352 | 0.9893 | 0.8950 | higher |  |
| `ndcg@10` | 352 | 0.9681 | 0.8977 | higher |  |
| `numeric_exactness` | 189 | 1.0000 | 0.4392 | higher | **gate** |
| `over_refusal` | 352 | 0.0000 | 0.1392 | lower | **gate** |
| `pii_leakage` | 354 | 0.0000 | 0.0000 | lower |  |
| `precision@1` | 352 | 0.9830 | 0.8636 | higher |  |
| `precision@10` | 352 | 0.6636 | 0.6352 | higher |  |
| `precision@20` | 352 | 0.5879 | 0.5734 | higher |  |
| `precision@3` | 352 | 0.7955 | 0.7169 | higher |  |
| `precision@5` | 352 | 0.7364 | 0.6892 | higher |  |
| `recall@1` | 352 | 0.9408 | 0.7770 | higher |  |
| `recall@10` | 352 | 0.9957 | 0.8608 | higher |  |
| `recall@20` | 352 | 1.0000 | 0.8670 | higher |  |
| `recall@3` | 352 | 0.9920 | 0.8314 | higher |  |
| `recall@5` | 352 | 0.9957 | 0.8438 | higher |  |
| `refusal_correctness` | 2 | 1.0000 | 0.0000 | higher | **gate** |
| `uncited_claim_rate` | 305 | 0.3765 | 0.6557 | lower |  |

good **7/7** gates, weak **0/7**.

## Reading it

`citation_resolvability` is 0.9886 rather than 1.0000 for the good system, and
that is the documented cap rather than a defect: the sealed set contains 2
`cross_ref_hop`, 2 `precondition` and 1 `adversarial_unknown` case, none of
which carries a fact id, so run_eval synthesises a page-only citation stub that
cannot resolve. See `eval/metrics/ceiling.py`.

`uncited_claim_rate` 0.3765 is the completeness tension recorded in the same
place -- the good system delivers every step cause so `completeness` has
something to measure, while citing only the queried fact.

A sealed set is spent once anything is tuned against it. Nothing here was: the
set was selected, the runs were made, the numbers were written down.
