#!/usr/bin/env python3
"""
base.py
Metric protocol, registry, and the text/number primitives every Tier-1 metric
shares.

TIER 1 IS DETERMINISTIC. No model, no network, no randomness. Same inputs
produce the same numbers on every machine, every time. That is the whole reason
Tier-1 figures can go in front of management: each one is defensible line by
line against golden/.

Tier 2 (judge model) registers metrics through the same interface. It must set
TIER = 2 so aggregation can keep the two apart. Nothing in this file may ever
import a model client.
"""
import re
import json
import functools
import unicodedata

# ---------------------------------------------------------------- normalisation

# Criteria strings in the manual mix two ohm codepoints -- U+2126 OHM SIGN and
# U+03A9 GREEK CAPITAL LETTER OMEGA -- roughly half and half. A comparator that
# does not fold them fails half of all resistance cases for a reason that has
# nothing to do with correctness. "ohm" spelled out is tolerated too.
_OHM_FORMS = ["Ω", "Ω", "ω", "ohms", "ohm"]


# Chunk texts are normalised over and over -- once per fact, per metric, per
# case. Caching makes the run tractable and changes no result: normalise is a
# pure function. Sized above the largest corpus (1,708 fixed_512 chunks) plus
# the fact strings.
@functools.lru_cache(maxsize=8192)
def _normalise_cached(s):
    for form in _OHM_FORMS:
        s = re.sub(re.escape(form), " ohm ", s, flags=re.I)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def normalise(text):
    """Fold whitespace, case and ohm spellings. NEVER touches digits.

    Digits are the payload. `Min. 100kΩ` and `Min. 90kΩ` must stay different
    strings after normalisation or numeric_exactness is measuring nothing.
    """
    if text is None:
        return ""
    return _normalise_cached(unicodedata.normalize("NFC", str(text)))


def _normalise_uncached(text):
    """Reference implementation, kept so the cache can be proven equivalent."""
    if text is None:
        return ""
    s = unicodedata.normalize("NFC", str(text))
    for form in _OHM_FORMS:
        s = re.sub(re.escape(form), " ohm ", s, flags=re.I)
    s = s.replace("’", "'").replace("“", '"').replace("”", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def contains(haystack, needle):
    """Containment under normalisation. Used for every relevance judgement."""
    n = normalise(needle)
    return bool(n) and n in normalise(haystack)


# -------------------------------------------------------------------- numbers

# A bare number, optionally decimal. Deliberately not unit-aware: the point is to
# catch every numeric token an answer emits, then ask where each one came from.
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def numbers_in(text):
    """Every numeric token, as strings so 0.20 and 0.2 stay distinguishable."""
    return NUMBER_RE.findall(str(text or ""))


def numbers_in_record(rec):
    """Every number appearing anywhere in a golden record.

    Serialising the whole record is deliberate. A number is legitimate if it
    appears in the ground truth AT ALL -- as a criterion, a pin, a monitoring
    code, a page, a step index. Enumerating only the fields we expect would
    flag correct answers that quoted a field we forgot about.
    """
    return set(numbers_in(json.dumps(rec, ensure_ascii=False)))


# ---------------------------------------------------------------------- facts

# Fields of `expected` that are bookkeeping rather than content a technician
# needs retrieved.
_NON_FACT_KEYS = {"step", "branch", "behaviour", "note"}
_PLACEHOLDERS = {"", "-", "--", "none", "null"}


def golden_facts(case):
    """Every string that must be retrievable for this question to be answerable.

    Relevance throughout this harness is judged by CONTENT containment of these
    strings -- never by chunk id. An id-based judgement only works against an
    index whose ids we minted; content works against any index, including a
    vendor one we have never seen. It also means the same metric code scores a
    local BM25 run and a hosted Azure run without modification.

    The fact set is the WHOLE expected answer, not just the single pinned
    string. A numeric_exactness case needs the measuring point as well as the
    criterion; a direct_lookup needs the title and the machine effect. Scoring
    against one fact per case would make answer_coverage identical to recall by
    construction -- and a metric that cannot differ from its neighbour is not
    measuring anything, which is precisely how this harness's own
    fragmentation_gap first read 0.0000 on every chunking.
    """
    facts, seen = [], set()

    def push(v):
        if not isinstance(v, str):
            return
        s = v.strip()
        if s.lower() in _PLACEHOLDERS or len(s) < 3:
            return
        if s not in seen:
            seen.add(s)
            facts.append(s)

    for v in case.get("must_contain_verbatim", []) or []:
        push(v)
    for v in case.get("must_contain", []) or []:
        push(v)
    for k, v in (case.get("expected") or {}).items():
        if k not in _NON_FACT_KEYS:
            push(v)
    return facts


def is_adversarial(case):
    return case["type"].startswith("adversarial")


def sentences(text):
    """Split an answer into claim-sized units. Deterministic, no NLP model."""
    parts = re.split(r"(?<=[.!?])\s+|\n+|(?:^|\s)[•\-•]\s*", str(text or ""))
    return [p.strip() for p in parts if p and p.strip()]


# --------------------------------------------------------------------- metric

class Metric:
    """One measurement.

    name       stable identifier; run records and compare.py key on it
    TIER       1 = deterministic, 2 = judge model. Never blend them.
    HIGHER_IS_BETTER  direction, so compare.py can classify a delta without a
                      per-metric lookup table
    APPLIES_TO tuple of case types, or None for all

    compute(case, result) -> float | bool | None
        None means "not applicable to this case" and is excluded from the
        aggregate rather than counted as zero. Scoring an inapplicable case as 0
        silently drags every headline number down and hides which cases actually
        failed.

    aggregate(rows) -> float
        rows are the non-None compute() values.
    """
    name = "unnamed"
    TIER = 1
    HIGHER_IS_BETTER = True
    APPLIES_TO = None

    def applies(self, case):
        if self.APPLIES_TO is None:
            return True
        return case["type"] in self.APPLIES_TO

    def compute(self, case, result):
        raise NotImplementedError

    def aggregate(self, rows):
        vals = [float(v) for v in rows if v is not None]
        return sum(vals) / len(vals) if vals else 0.0

    def self_test(self):
        """Assert this metric can return a failing score on known-bad input.

        E4 and H2 both sat at zero for months because they were incapable of
        returning anything else, and both zeros were read as statements about
        the document. Every gate-bearing metric here carries one of these.
        Return None to declare no self-test (non-gate metrics only).
        """
        return None


class Registry:
    """Metrics register here. Tier 2 appends to the same list."""

    def __init__(self):
        self._metrics = []

    def add(self, metric):
        if any(m.name == metric.name for m in self._metrics):
            raise ValueError(f"duplicate metric name: {metric.name}")
        self._metrics.append(metric)
        return metric

    def extend(self, metrics):
        for m in metrics:
            self.add(m)
        return self

    def tier(self, n):
        return [m for m in self._metrics if m.TIER == n]

    def __iter__(self):
        return iter(self._metrics)

    def __len__(self):
        return len(self._metrics)
