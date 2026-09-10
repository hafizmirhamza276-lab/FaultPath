#!/usr/bin/env python3
"""
retrieval.py
Ranking and coverage. Deterministic.

RELEVANCE IS CONTENT, NEVER ID. A chunk is relevant to a case when it contains
one of that case's verbatim golden facts. No chunk id is ever consulted. That
definition works against any index -- ours, a vendor's, one built by an
ingestion pipeline we did not write -- so a local BM25 run and a hosted run are
scored by identical code and are genuinely comparable.

Two families here, and the second matters more for this product:

  recall@k          per-FACT. Of the facts this question needs, how many were
                    retrieved.
  answer_coverage@k all-or-nothing. Were ALL of them there.

A technician does not get 70% of a repair. The gap between these two is
fragmentation -- facts scattered across chunks such that no single top-k window
holds a complete answer. No reranker closes it, because reranking reorders what
retrieval already found; it cannot join two half-answers into one. Only a
chunking change or a multi-hop assembly step does that.
"""
from .base import Metric, contains, golden_facts, is_adversarial
from ..adapters import tokenize

KS = (1, 3, 5, 10, 20)

# Adversarial cases have no correct chunk anywhere in the corpus -- that is the
# point of them. Scoring retrieval on them would measure nothing and would drag
# every ranking average down by a constant.
RETRIEVAL_TYPES = ("numeric_exactness", "direct_lookup", "step_ordering",
                   "branch_following", "precondition", "cross_ref_hop")


def _relevant_flags(case, chunks):
    """Per retrieved chunk: does it carry at least one golden fact."""
    facts = golden_facts(case)
    return [any(contains(c.get("text", ""), f) for f in facts) for c in chunks]


def _facts_found(case, chunks):
    """Per golden fact: is it present in this set of chunks."""
    return [any(contains(c.get("text", ""), f) for c in chunks)
            for f in golden_facts(case)]


class _RetrievalMetric(Metric):
    APPLIES_TO = RETRIEVAL_TYPES

    def applies(self, case):
        return not is_adversarial(case) and case["type"] in self.APPLIES_TO \
            and bool(golden_facts(case))


# ------------------------------------------------------------------- ceiling

class CeilingRecall(_RetrievalMetric):
    """Facts present ANYWHERE in the corpus, regardless of ranking.

    Report this FIRST. It is set by ingestion and chunking, not by the
    retriever, and every other retrieval number lives underneath it. Tuning a
    reranker when ceiling_recall is 0.85 is optimising inside a box that is
    already missing 15% of the answers.
    """
    name = "ceiling_recall"

    def __init__(self, corpus):
        self.corpus = list(corpus)

    def compute(self, case, result):
        if not self.applies(case):
            return None
        found = _facts_found(case, self.corpus)
        return sum(found) / len(found) if found else None

    def self_test(self):
        m = CeilingRecall([{"text": "Max. 1 ohm"}])
        good = {"type": "numeric_exactness", "must_contain_verbatim": ["Max. 1 Ω"]}
        bad = {"type": "numeric_exactness", "must_contain_verbatim": ["Min. 100kΩ"]}
        assert m.compute(good, None) == 1.0, "ceiling_recall cannot pass"
        assert m.compute(bad, None) == 0.0, "ceiling_recall cannot fail"


# ---------------------------------------------------------------- ranking

class RecallAtK(_RetrievalMetric):
    """Per-fact recall in the top k."""

    def __init__(self, k):
        self.k = k
        self.name = f"recall@{k}"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        found = _facts_found(case, result["retrieved"][:self.k])
        return sum(found) / len(found) if found else None

    def self_test(self):
        m = RecallAtK(1)
        case = {"type": "numeric_exactness", "must_contain_verbatim": ["Max. 1 Ω"]}
        assert m.compute(case, {"retrieved": [{"text": "Max. 1 Ω"}]}) == 1.0
        assert m.compute(case, {"retrieved": [{"text": "nothing"}]}) == 0.0, \
            "recall@k cannot fail"


class PrecisionAtK(_RetrievalMetric):
    """Share of the top k that carries any golden fact."""

    def __init__(self, k):
        self.k = k
        self.name = f"precision@{k}"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        top = result["retrieved"][:self.k]
        if not top:
            return 0.0
        return sum(_relevant_flags(case, top)) / len(top)


class HitRateAtK(_RetrievalMetric):
    """Was anything relevant in the top k at all."""

    def __init__(self, k):
        self.k = k
        self.name = f"hit_rate@{k}"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        return float(any(_relevant_flags(case, result["retrieved"][:self.k])))

    def self_test(self):
        m = HitRateAtK(5)
        case = {"type": "direct_lookup", "must_contain": ["Engine Overspeed"]}
        assert m.compute(case, {"retrieved": [{"text": "xxx"}] * 5}) == 0.0, \
            "hit_rate@k cannot fail"
        assert m.compute(case, {"retrieved": [{"text": "Engine Overspeed"}]}) == 1.0


class FirstRelevantRank(_RetrievalMetric):
    """1-based rank of the first relevant chunk; 0 when none was retrieved.

    Lower is better, so this is excluded from any "higher is better" reading.
    Reported because an average rank of 4 and an average rank of 1 can share an
    identical hit_rate@5 while feeling completely different to use.
    """
    name = "first_relevant_rank"
    HIGHER_IS_BETTER = False

    def compute(self, case, result):
        if not self.applies(case):
            return None
        for i, rel in enumerate(_relevant_flags(case, result["retrieved"]), 1):
            if rel:
                return float(i)
        return 0.0


class MRR(_RetrievalMetric):
    name = "mrr"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        for i, rel in enumerate(_relevant_flags(case, result["retrieved"]), 1):
            if rel:
                return 1.0 / i
        return 0.0

    def self_test(self):
        case = {"type": "direct_lookup", "must_contain": ["alpha"]}
        m = MRR()
        assert m.compute(case, {"retrieved": [{"text": "alpha"}]}) == 1.0
        assert m.compute(case, {"retrieved": [{"text": "zzz"}, {"text": "alpha"}]}) == 0.5
        assert m.compute(case, {"retrieved": [{"text": "zzz"}]}) == 0.0, \
            "mrr cannot fail"


class MAP(_RetrievalMetric):
    """Mean average precision over the retrieved ranking."""
    name = "map"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        flags = _relevant_flags(case, result["retrieved"])
        hits, total = 0, 0.0
        for i, rel in enumerate(flags, 1):
            if rel:
                hits += 1
                total += hits / i
        return total / hits if hits else 0.0


class NDCGAt10(_RetrievalMetric):
    """Binary-relevance nDCG@10."""
    name = "ndcg@10"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        import math
        flags = _relevant_flags(case, result["retrieved"][:10])
        dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(flags, 1) if r)
        n_rel = sum(flags)
        idcg = sum(1.0 / math.log2(i + 1) for i in range(1, n_rel + 1))
        return dcg / idcg if idcg else 0.0


# --------------------------------------------------------------- coverage

class AnswerCoverageAtK(_RetrievalMetric):
    """Were ALL facts for this question inside the top k. All or nothing."""

    def __init__(self, k):
        self.k = k
        self.name = f"answer_coverage@{k}"

    def compute(self, case, result):
        if not self.applies(case):
            return None
        found = _facts_found(case, result["retrieved"][:self.k])
        return float(all(found)) if found else None

    def self_test(self):
        m = AnswerCoverageAtK(5)
        case = {"type": "direct_lookup", "must_contain": ["alpha", "beta"]}
        half = {"retrieved": [{"text": "alpha only"}]}
        assert m.compute(case, half) == 0.0, \
            "answer_coverage cannot fail on a partial answer"
        whole = {"retrieved": [{"text": "alpha and beta"}]}
        assert m.compute(case, whole) == 1.0


class ChunksToCover(_RetrievalMetric):
    """How many chunks down the ranking before the answer is complete.

    0 means never covered within the retrieved window. Lower is better. This is
    the operational cost of fragmentation: it is the size of context window a
    generator needs before it can answer at all.
    """
    name = "chunks_to_cover"
    HIGHER_IS_BETTER = False

    def compute(self, case, result):
        if not self.applies(case):
            return None
        facts = golden_facts(case)
        need = set(range(len(facts)))
        got = set()
        for i, c in enumerate(result["retrieved"], 1):
            for j in need - got:
                if contains(c.get("text", ""), facts[j]):
                    got.add(j)
            if got == need:
                return float(i)
        return 0.0


class FragmentationGap(_RetrievalMetric):
    """recall@10 minus answer_coverage@10.

    Its own number because it has its own fix. A large gap means the facts were
    found but scattered -- reranking cannot help, since it reorders what was
    already retrieved rather than assembling across chunks. Chunking or a
    multi-hop step is the only lever.
    """
    name = "fragmentation_gap@10"
    HIGHER_IS_BETTER = False

    def compute(self, case, result):
        if not self.applies(case):
            return None
        found = _facts_found(case, result["retrieved"][:10])
        if not found:
            return None
        return (sum(found) / len(found)) - float(all(found))

    def self_test(self):
        m = FragmentationGap()
        # Two facts, one retrieved: recall 0.5, coverage 0. Gap must be 0.5.
        case = {"type": "direct_lookup", "must_contain": ["alpha", "beta"]}
        r = {"retrieved": [{"text": "alpha only"}]}
        assert m.compute(case, r) == 0.5, \
            "fragmentation_gap cannot fire -- if every case carries one fact, " \
            "recall and answer_coverage are identical by construction"
        whole = {"retrieved": [{"text": "alpha and beta"}]}
        assert m.compute(case, whole) == 0.0


class ContextPrecision(_RetrievalMetric):
    """Share of retrieved tokens that sit in a chunk carrying a golden fact.

    Wasted context is paid for twice: once at the token meter, and again in
    diluted attention over a longer prompt.
    """
    name = "context_precision"

    def __init__(self, k=10):
        self.k = k

    def compute(self, case, result):
        if not self.applies(case):
            return None
        top = result["retrieved"][:self.k]
        flags = _relevant_flags(case, top)
        total = sum(len(tokenize(c.get("text", ""))) for c in top)
        if not total:
            return 0.0
        useful = sum(len(tokenize(c.get("text", "")))
                     for c, rel in zip(top, flags) if rel)
        return useful / total


class FilterCorrectness(Metric):
    """Was the model/manual filter actually enforced.

    PC200 and PC490 share failure codes but not pin numbers. A retriever that
    ignores the filter will cheerfully return the wrong machine's page, and the
    generator downstream has no way to know.
    """
    name = "filter_correctness"

    def applies(self, case):
        return bool(case.get("filters"))

    def compute(self, case, result):
        if not self.applies(case):
            return None
        filters = case["filters"]
        for c in result.get("retrieved", []):
            for key, want in filters.items():
                if want is not None and key in c and c[key] != want:
                    return 0.0
        return 1.0

    def self_test(self):
        m = FilterCorrectness()
        case = {"type": "direct_lookup", "filters": {"model": "PC200-10M0"}}
        bad = {"retrieved": [{"model": "PC490LC-11"}]}
        assert m.compute(case, bad) == 0.0, "filter_correctness cannot fail"
        good = {"retrieved": [{"model": "PC200-10M0"}]}
        assert m.compute(case, good) == 1.0


def build(corpus):
    """All retrieval metrics, ceiling first -- the order they should be read."""
    metrics = [CeilingRecall(corpus)]
    metrics += [RecallAtK(k) for k in KS]
    metrics += [AnswerCoverageAtK(k) for k in KS]
    metrics += [PrecisionAtK(k) for k in KS]
    metrics += [HitRateAtK(k) for k in KS]
    metrics += [MRR(), MAP(), NDCGAt10(), FirstRelevantRank(),
                ChunksToCover(), FragmentationGap(), ContextPrecision(),
                FilterCorrectness()]
    return metrics
