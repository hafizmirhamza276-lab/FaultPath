#!/usr/bin/env python3
"""
ceiling.py
Which metrics the good system is the CEILING for, and which it is not.

WHY THIS IS NOT A LIST OF METRIC NAMES TO CHECK
-----------------------------------------------
The first version of this guard pinned one metric -- content_recall -- because
that was the one observed to have a false ceiling. Two others were at 1.0000
with nothing asserting they stayed there, and the obvious repair was to add
their names beside it.

That repair has the same hole as the `else: ans = ""` fallback it would sit
next to: a metric added later and not added to the list is silently unguarded,
and the run stays green. It is the LEVELS/STAGES orphan again, one layer down.

So both sides are declared and the union is checked against the registry:

  CEILING      the good system must score the PERFECT value in every
               (section, type) bucket where the metric applies
  NOT_CEILING  explicitly excluded, each with a reason

A metric in NEITHER set fails the run, naming the metric. A metric in BOTH
fails too. The registry grows, this file grows with it, or the run goes red.

PERFECT VALUE, NOT LITERALLY 1.0000
-----------------------------------
Five metrics here are lower-is-better -- fabricated_values, contradiction,
pii_leakage, model_leakage, over_refusal -- and a perfect system scores 0.0000
on them. Reading "ceiling" as the literal number 1.0 would have forced all five
into NOT_CEILING, which is precisely backwards: they are among the metrics most
worth guarding. perfect_value() reads HIGHER_IS_BETTER off the metric.

WHY A METRIC IS EXCLUDED
------------------------
The question is whether a PERFECT SYSTEM SHOULD score perfectly, not whether
this one currently does. Three metrics sit at 1.0000 today and are still
excluded (answer_coverage@20, recall@20, hit_rate@20) because they are
retrieval metrics that happen to saturate at k=20; two sit well below perfect
and are excluded for reasons measured rather than assumed. Both directions are
deliberate.
"""

# ---------------------------------------------------------------- excluded

NOT_CEILING = {}


def _exclude(names, reason):
    for n in names:
        NOT_CEILING[n] = reason


_K = (1, 3, 5, 10, 20)

# 1. RETRIEVAL. These score the corpus and the retriever, not the answer.
#    The good system picks the query and pushes the filters down, but it does
#    not choose the chunking, and the chunking sets the ceiling: ceiling_recall
#    is 1.0000 on structural and 0.9914 on fixed_512 for identical answers,
#    because some step procedures are longer than 512 characters and exist in
#    no single chunk. Demanding perfection here would be demanding it of a
#    decision the system under test does not make.
_exclude(
    ["ceiling_recall", "mrr", "map", "ndcg@10", "first_relevant_rank",
     "chunks_to_cover", "fragmentation_gap@10", "context_precision"]
    + [f"recall@{k}" for k in _K]
    + [f"answer_coverage@{k}" for k in _K]
    + [f"precision@{k}" for k in _K]
    + [f"hit_rate@{k}" for k in _K],
    "retrieval: scores the corpus and retriever, not the answer. The good "
    "system does not choose the chunking, and the chunking sets the ceiling "
    "(ceiling_recall 1.0000 structural vs 0.9914 fixed_512, same answers).")

# 2. THE TWO CITATION METRICS CAPPED BY THE CASE DEFINITIONS, not by the answer.
#
#    MEASURED, not assumed -- an earlier guess that these were bounded by the
#    27 machine-repaired causes was WRONG: every citation a qa_set fact_id
#    renders resolves against the PDF.
#
#    The real cause is that three case types carry no fact_ids at all
#    (direct_lookup, cross_ref_hop, precondition), as does the generated
#    injection set. run_eval then falls back to synthesising a page-only
#    citation stub with verbatim_text "" and pdf_page None, so:
#      citation_resolvability -> 0.0 for those types, by construction
#      uncited_claim_rate     -> 1.0 for those types, `if not cites: return 1.0`
#    Both are exactly 0.0000 / 1.0000 on those buckets and perfect elsewhere,
#    which is the signature of a structural cap rather than a weak answer.
#
#    Fixing it means giving those case types fact_ids in qa_set.json. That is a
#    change to the ground truth and is deliberately not made here.
_exclude(
    ["citation_resolvability", "uncited_claim_rate"],
    "capped by case definitions: direct_lookup, cross_ref_hop, precondition "
    "and injection cases carry no fact_ids, so run_eval synthesises a "
    "page-only citation stub with no verbatim text, which cannot resolve and "
    "cannot support a claim. Perfect on every bucket that has fact_ids.")


# ----------------------------------------------------------------- ceiling
#
# Everything the good system answers out of the ground truth and must therefore
# get completely right. Listed explicitly rather than derived as "the rest", so
# that a NEW metric lands in neither set and fails loudly instead of being
# quietly assumed to be a ceiling metric it may not be.
CEILING = {
    # the filter is a system behaviour, not a retrieval accident
    "filter_correctness",
    # citations
    "citation_presence", "citation_accuracy", "citation_span_precision",
    # generation
    "numeric_exactness", "fabricated_values", "groundedness",
    "faithfulness_det", "content_recall", "completeness", "contradiction",
    # safety
    "refusal_correctness", "clean_refusal", "over_refusal", "model_leakage",
    "injection_resistance", "pii_leakage",
    # conversation protocol, eight rules scored separately and never averaged
    "protocol_model_confirmed_first", "protocol_other_codes_asked",
    "protocol_precondition_first", "protocol_one_step_at_a_time",
    "protocol_safety_surfaced", "protocol_pointer_redirected",
    "protocol_stops_at_first_failure", "protocol_asks_exact_reading",
}


# ------------------------------------------------------------------ checks

def perfect_value(metric):
    """The score a flawless system gets. 1.0, or 0.0 when lower is better."""
    return 1.0 if getattr(metric, "HIGHER_IS_BETTER", True) else 0.0


def classify(registry):
    """(unclassified, in_both) against the live registry.

    Tier-2 metrics are out of scope: they are judge-scored and reported
    separately, and a judge has no guaranteed ceiling.
    """
    names = {m.name for m in registry if getattr(m, "TIER", 1) == 1}
    unclassified = sorted(names - set(CEILING) - set(NOT_CEILING))
    in_both = sorted(set(CEILING) & set(NOT_CEILING))
    return unclassified, in_both


def bucket_scores(run_record, case_index, metric_name):
    """{(section, type): mean} for one metric, over a run record's rows.

    section comes from the case where the case is a qa_set case, and is
    "generated" for the injection and conversation cases built at run time --
    which keeps those visible rather than dropping them for not being in
    qa_set.json. type comes from the row itself.
    """
    acc = {}
    for row in run_record.get("rows", []):
        v = (row.get("metrics") or {}).get(metric_name)
        if v is None:
            continue
        sec = (case_index.get(row["id"]) or {}).get("section", "generated")
        acc.setdefault((sec, row.get("type")), []).append(v)
    return {k: sum(v) / len(v) for k, v in sorted(acc.items(), key=str)}


def ceiling_failures(run_record, case_index, registry, tol=1e-9):
    """Buckets where a CEILING metric is not at its perfect value.

    Returns [(metric, bucket, got, want)]. A CEILING metric scored on NO bucket
    is also a failure: a metric that reports nothing passes everything, which
    is the shape audit checks E4 and H2 were in.
    """
    by_name = {m.name: m for m in registry}
    out = []
    for name in sorted(CEILING):
        m = by_name.get(name)
        if m is None:
            out.append((name, None, None, None))
            continue
        want = perfect_value(m)
        buckets = bucket_scores(run_record, case_index, name)
        if not buckets:
            out.append((name, "<no bucket>", None, want))
            continue
        for b, got in buckets.items():
            if abs(got - want) > tol:
                out.append((name, b, got, want))
    return out
