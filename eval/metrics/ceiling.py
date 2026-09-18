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

# 2. citation_resolvability -- capped by three buckets left uncited ON PURPOSE.
#
#    This reason has been wrong twice and is now measured. The first version
#    blamed the 27 machine-repaired causes: WRONG, every citation a qa_set
#    fact_id renders resolves against the PDF. The second blamed four case
#    types carrying no fact_ids, which was right for THIS metric and wrong for
#    uncited_claim_rate below -- the two were sharing one reason and only one
#    of them fitted it.
#
#    direct_lookup now cites machine_effect and scores 1.0000. What remains at
#    0.0 is cross_ref_hop (9), precondition (10) and the generated injection
#    set (15), each uncited after being checked, with the reason recorded in
#    the case itself under `uncited_reason`:
#      cross_ref_hop  3 of 9 would cite synthetic "Redirect" text that cannot
#                     resolve, and 4 more a verbatim string not naming the
#                     target. Only 2 of 9 would be sound.
#      precondition   the ordering is DERIVED from word order inside one
#                     sentence; the manual states no ordering.
#      injection      an adversarial prompt has nothing to cite, which is the
#                     same precedent the adversarial_* cases set.
#    AND A THIRD THING THE REASON MISSED UNTIL THE SEALED RUN. Everything above
#    explains why GOOD stops at 0.9886. It does not explain what the metric
#    measures about anyone else, and the sealed run made that unavoidable: the
#    model scored 0.0000 -- exactly the floor, the only metric where it sat
#    there.
#
#    That is not a quality result. citation_resolvability requires the answer
#    to name fact_ids, and the model is never shown one: build_prompt renders
#    "--- extract (page N, record CODE) ---" and no fact_id appears anywhere in
#    the prompt. With none named, citations_rendered falls back to entries
#    synthesised from the pages the model typed, and resolve() returns
#    "citation carries no pdf_page" without opening the PDF. 0.0000 is the
#    fallback reporting that it has nothing to resolve.
#
#    GoodSystem scores 0.9886 by returning list(case["fact_ids"]) -- the case's
#    own answer key. It is not resolving anything; it is echoing ground truth
#    it was handed.
#
#    So the metric SEPARATES ARCHITECTURES, NOT QUALITY. A system whose
#    retrieval layer carries fact-id-tagged spans through to its output scores
#    near 1.0; one that answers from prose scores 0.0000, however good its
#    answers are. It CANNOT DISCRIMINATE BETWEEN TWO REAL SYSTEMS of the same
#    architecture, and it must not be read as "the model cites badly". The
#    model's actual citation behaviour is citation_accuracy (0.7435 sealed),
#    which scores the pages it does type. The two are different measurements
#    and reading them as one understates the model by 0.74.
_exclude(
    ["citation_resolvability"],
    "ARCHITECTURAL, not quality: it requires the answer to name fact_ids, and "
    "a system that is never shown one scores 0.0000 however good its answers "
    "are -- the model does, while citation_accuracy reads 0.7435. good's "
    "0.9886 is the reference echoing case['fact_ids'], not a resolvable "
    "pipeline. Its own cap is three buckets uncited after being checked: "
    "cross_ref_hop (would cite unresolvable synthetic text), precondition "
    "(the claim is derived, not stated) and injection (nothing to cite), each "
    "recording its reason in the case.")

# 3. uncited_claim_rate -- capped by a DELIBERATE TENSION BETWEEN TWO METRICS,
#    not by missing fact_ids. This is the correction: step_ordering scores
#    0.9630 WITH fact_ids, so "no fact_ids" never explained this metric.
#
#    The good system answers procedure-shaped questions by delivering every
#    step cause -- "Full procedure: Step 1: ... Step N: ..." -- so completeness
#    has something to measure, while naming only the QUERIED fact's id. The
#    other steps' atoms are then uncited by construction. The buckets with that
#    addendum are the high ones (step_ordering 0.9630, branch_following 0.6626,
#    cross_ref_hop 0.6481); the ones without it are low (numeric_exactness
#    0.0041, symptom_remedy 0.0000).
#
#    THE WEAKER OF THE TWO EXCLUSIONS, and said so rather than dressed up. A
#    perfect system arguably SHOULD ground every claim it makes -- that is the
#    stated design premise -- which would put this in CEILING. Reaching 0.0
#    means citing every delivered step's fact id, which is a real change to the
#    good system and to the completeness/uncited balance, not a reclassification.
#    Revisit together, not separately.
#    AND IT IS ALSO ARCHITECTURE-GATED, which the above does not say. The
#    sealed run read 0.7179 for the model against 0.3905 for good, and that gap
#    is not a quality gap. The supported-atom set is built from each citation's
#    verbatim_text; a system that names no fact_ids has citations synthesised
#    from its typed pages, whose verbatim_text is "", so support collapses to
#    the QUESTION ALONE and the rate is high BY CONSTRUCTION.
#
#    Unlike citation_resolvability this reads as an ordinary quality number --
#    both systems get a mid-range score, so nothing on its face says the two
#    are not comparable. They are not.
#
#    (a) OR (b) -- ARGUED AND MEASURED, n=1,511 non-sealed, before deciding:
#          current  (verbatim_text "" -> question only)   0.6910
#          (b1)     support = the retrieved context       0.0004
#          (b2)     support = text of the pages cited     0.0882
#
#    (a), architecture-gated, for three reasons:
#      1. (b1) is the natural reading of "support the harness can see" and it
#         COLLAPSES INTO faithfulness_det, which measures exactly that and
#         reads 0.0019 as a rate. A metric that cannot differ from its
#         neighbour is not measuring anything -- the fragmentation_gap lesson.
#      2. (b2) is distinct, but it answers a DIFFERENT QUESTION: "is the claim
#         on the page you cited" rather than "is the claim backed by a fact
#         id". That is a new metric wearing this one's name, and the honest
#         way to have it is to add it, not to redefine this.
#      3. (b2)'s residue is partly an artefact of its own: 142 cited pages had
#         no manual->pdf mapping, so some of the 8.82% is missing map entries
#         rather than unsupported claims. Swapping one construction artefact
#         for another is not a fix.
#
#    So the number stays and the CAVEAT is recorded instead. Both exclusions
#    now apply and they are different in kind: the completeness tension caps
#    GOOD, and the empty support set inflates ANY SYSTEM WITHOUT FACT_IDS.
_exclude(
    ["uncited_claim_rate"],
    "TWO causes, different in kind. (i) capped by a deliberate tension with "
    "completeness: the good system delivers every step cause so completeness "
    "can be measured, while citing only the queried fact, leaving the rest "
    "uncited by construction -- NOT missing fact_ids, since step_ordering "
    "scores 0.9630 with them. (ii) ARCHITECTURE-GATED as well: support is "
    "built from each citation's verbatim_text, which is empty for a system "
    "that names no fact_ids, so support collapses to the question and the "
    "rate is inflated by construction -- model 0.7179 vs good 0.3905 on "
    "sealed is not a quality gap. Measured alternatives: support from "
    "retrieved context reads 0.0004 and duplicates faithfulness_det; support "
    "from cited-page text reads 0.0882 and is a different question.")


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

# ------------------------------------------- which systems the guard applies to
#
# THE CEILING IS A PROPERTY OF A REFERENCE, NOT OF EVERY SYSTEM.
#
# Requiring 1.0000 of a real model would be asserting that the thing being
# measured has already succeeded -- the measurement and the assertion swapped
# places. So the guard applies to REFERENCE systems only:
#
#   reference   perfection asserted. A bucket below 1.0000 is a false ceiling
#               and the metric has no headroom there.
#   floor       failure asserted. Passing a gate means the gate has a hole.
#   under_test  nothing asserted. The score IS the measurement.
#
# The categories are DERIVED from each system's declared CATEGORY, and their
# union is checked against the registry -- a system in no category fails naming
# itself, exactly as an unclassified metric does above. Hardcoding "good" would
# stop working the moment a second reference existed and would silently treat
# it as a system under test.

CATEGORIES = ("reference", "floor", "under_test")


def classify_systems(systems):
    """(uncategorised, unknown_category) for a {name: system} registry."""
    uncategorised, unknown = [], []
    for name, sysobj in sorted(systems.items()):
        cat = getattr(sysobj, "CATEGORY", None)
        if cat is None:
            uncategorised.append(name)
        elif cat not in CATEGORIES:
            unknown.append(f"{name}={cat}")
    return uncategorised, unknown


def ceiling_systems(systems):
    """The systems the ceiling guard applies to. References only."""
    return sorted(n for n, s in systems.items()
                  if getattr(s, "CATEGORY", None) == "reference")


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
