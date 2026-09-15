#!/usr/bin/env python3
"""
test_eval_harness.py
Verifies the evaluation harness, not the product.

    python tests/test_eval_harness.py

Three things, in order of how badly they bite:

  1. Every metric proves it can FAIL on known-bad input. Audit checks E4 and H2
     each sat at zero for months while being structurally incapable of returning
     anything else, and both zeros were read as statements about the document.
     A metric nobody has watched fail is not a measurement.

  2. Good passes every gate; weak fails EVERY gate. If weak passes one, that
     gate has a hole and is not measuring what it claims.

  3. compare.py refuses to diff runs whose case ids were regenerated under a
     different scheme, rather than silently reporting every case as new.

Deterministic throughout. Two invocations produce identical numbers; that is
asserted rather than assumed.
"""
import ast
import collections
import json
import os
import subprocess
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from eval import chunkers                                # noqa: E402
from eval.adapters import LocalBM25Retriever             # noqa: E402
from eval.metrics.base import Metric, Registry           # noqa: E402
from eval.metrics import retrieval as m_retrieval        # noqa: E402
from eval.metrics import generation as m_generation      # noqa: E402
from eval.metrics import conversation as m_conversation  # noqa: E402
from eval.metrics import safety as m_safety              # noqa: E402
from eval.metrics import ceiling                         # noqa: E402
from eval import run_eval                                # noqa: E402

failures = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


def build_registry(corpus, records):
    reg = Registry()
    reg.extend(m_retrieval.build(corpus))
    reg.extend(m_generation.build(records))
    reg.extend(m_safety.build(records))
    reg.extend(m_conversation.build())
    return reg


# ============================================================ 1. self-tests
print("\nevery metric can fail on known-bad input")

records = run_eval.load_records()
corpus = chunkers.build("structural", list(records.values()))
registry = build_registry(corpus, records)

n_tested, self_test_failures = run_eval.run_self_tests(registry)
check(f"{n_tested} metric self-tests pass", not self_test_failures,
      "; ".join(self_test_failures))

# Gate-bearing metrics must be among those with a self-test -- these are the
# numbers that decide whether a release ships.
gate_names = {g[0] for g in run_eval.GATES}
without = sorted(
    m.name for m in registry
    if m.name in gate_names and type(m).self_test is Metric.self_test)
check("every gate-bearing metric has a self-test", not without,
      f"missing: {without}")


# ============================================ 1b. the sealed case holdout
print("\nsealed holdout: derived, stratified, and reads no eval result")
from eval import sealed as sealed_mod                     # noqa: E402

try:
    sealed_mod.self_test()
    check("sealed self-test passes (determinism, coverage, no straddle)", True)
except AssertionError as exc:
    check("sealed self-test passes", False, str(exc))

_sd = sealed_mod.describe()
check("the sealed fraction is near the declared 20%",
      0.15 <= _sd["fraction"] <= 0.25, f"{_sd['fraction']:.3f}")
check("every (section, type) bucket has sealed cases",
      all(v["sealed"] > 0 for v in _sd["coverage"].values()),
      f"blind buckets: {[k for k, v in _sd['coverage'].items() if not v['sealed']]}")
check("every bucket still has train cases",
      all(v["corpus"] - v["sealed"] > 0 for v in _sd["coverage"].values()))
print(f"    {_sd['n_sealed']} sealed / {_sd['n_cases']} cases "
      f"({_sd['fraction']:.1%}) over {_sd['sealed_records']} records, "
      f"{len(_sd['coverage'])} buckets all covered")

# SELECTION MUST NOT READ AN EVAL RESULT. Asserted over the parsed AST, not
# trusted to review. This is the specific way the 91 held-out symptom phrasings
# became contaminated: scoring was revised after seeing which of them failed.
# A set chosen with any knowledge of outcomes is spent at the moment of
# choosing.
_sealed_src = open(os.path.join(REPO_ROOT, "eval", "sealed.py"),
                   encoding="utf-8").read()
_sealed_imports = {(n.module or "").split(".")[-1]
                   for n in ast.walk(ast.parse(_sealed_src))
                   if isinstance(n, ast.ImportFrom)}
_sealed_imports |= {a.name.split(".")[-1]
                    for n in ast.walk(ast.parse(_sealed_src))
                    if isinstance(n, ast.Import) for a in n.names}
_forbidden = _sealed_imports & {"run_eval", "synthetic", "adapters", "metrics",
                                "retrieval", "generation", "compare"}
check("sealed selection imports nothing that could carry an eval result",
      not _forbidden, f"imports {sorted(_forbidden)}")
check("sealed selection reads no run record",
      "eval_out" not in _sealed_src and "runs/" not in _sealed_src)

# The three holdouts are different things and must not be confused for one
# another. Asserted by unit: codes, phrasings, case ids.
from agent import holdout as code_holdout                 # noqa: E402
_code_hold = code_holdout.holdout_set()
_sealed = sealed_mod.sealed_ids()
check("the code holdout and the sealed case holdout are different units",
      not (_code_hold & _sealed),
      "a code id and a case id should never collide; if they do, one of the "
      "two holdouts is being read as the other")
print(f"    agent/holdout.py {len(_code_hold)} codes | "
      f"eval/sealed.py {len(_sealed)} cases | "
      f"symptom_heldout.json sealed/contaminated phrasings -- three units")


# ==================================================== 2. good/weak separation
print("\ngood passes every gate, weak fails every gate")


def run(system, label):
    out = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "eval", "run_eval.py"),
         "--chunker", "structural", "--system", system,
         "--label", label, "--quiet"],
        cwd=REPO_ROOT, capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"run_eval failed for {system}:\n{out.stdout}\n{out.stderr}")
    path = out.stdout.strip().splitlines()[-1].split("written to", 1)[1].strip()
    with open(os.path.join(REPO_ROOT, path), encoding="utf-8") as f:
        return json.load(f)


good = run("good", "selftest_good")
weak = run("weak", "selftest_weak")

good_pass = [g["gate"] for g in good["gates"] if g["status"] == "PASS"]
good_fail = [g["gate"] for g in good["gates"] if g["status"] != "PASS"]
weak_pass = [g["gate"] for g in weak["gates"] if g["status"] == "PASS"]

check(f"good passes all {len(good['gates'])} gates", not good_fail,
      f"failed: {good_fail}")
check("weak fails ALL gates", not weak_pass,
      f"weak PASSED {weak_pass} -- each of those gates has a hole and is not "
      f"measuring what it claims")

# ------------------------------------------- the ceiling, EVERY metric
#
# GOOD SYSTEM IS THE CEILING, AND THE CEILING IS ASSERTED, NOT ASSUMED.
#
# A bucket where the good system is below perfect is a bucket where the metric
# has no headroom: it cannot separate a good real system from a mediocre one,
# and the aggregate quietly averages two different ceilings. content_recall sat
# at 0.6699 on symptom_remedy while every other bucket was 1.0000, and the
# aggregate read 0.9628 and looked fine.
#
# Generalised from one metric to all of them via eval/metrics/ceiling.py, which
# declares CEILING and NOT_CEILING and is checked against the live registry --
# the LEVELS/STAGES parity pattern. Naming three metrics here instead would
# leave the fourth silently unguarded.
print("\nthe good system is the ceiling, per metric per bucket")

unclassified, in_both = ceiling.classify(registry)
check("every Tier-1 metric is declared CEILING or NOT_CEILING",
      not unclassified,
      f"unclassified: {unclassified}\n"
      "          A new metric must be declared in eval/metrics/ceiling.py. "
      "Landing in\n          neither set means nothing asserts whether the "
      "good system should be\n          perfect on it, which is how a false "
      "ceiling gets in unnoticed.")
check("no metric is declared both CEILING and NOT_CEILING", not in_both,
      f"in both: {in_both}")
print(f"    {len(ceiling.CEILING)} ceiling / {len(ceiling.NOT_CEILING)} "
      f"excluded / {len([m for m in registry if getattr(m, 'TIER', 1) == 1])} "
      f"Tier-1 metrics in the registry")

QA_BY_ID = {c["id"]: c for c in json.load(
    open(os.path.join(REPO_ROOT, "golden", "qa_set.json"), encoding="utf-8"))}

failed_ceilings = ceiling.ceiling_failures(good, QA_BY_ID, registry)
check("the good system scores the perfect value in EVERY bucket of EVERY "
      "ceiling metric", not failed_ceilings,
      "\n          ".join(
          f"{m} {b}: got {g if g is None else round(g, 4)}, want {w}"
          for m, b, g, w in failed_ceilings[:10]))

# The other half of a usable ceiling: the weak system must be strictly worse
# somewhere. A metric both systems max out cannot discriminate either.
#
# A bucket the weak system NEVER SCORED is not a bucket it maxed out. Defaulting
# an absent score to the perfect value read clean_refusal and
# citation_span_precision as undiscriminating when in fact the weak system
# simply produces nothing for them -- it never refuses, so clean_refusal has no
# refusal to score. Absent and perfect are different facts and are reported as
# different facts.
non_discriminating, unscored_by_weak = [], []
for name in sorted(ceiling.CEILING):
    m = next((x for x in registry if x.name == name), None)
    if m is None:
        continue
    want = ceiling.perfect_value(m)
    gb = ceiling.bucket_scores(good, QA_BY_ID, name)
    wb = ceiling.bucket_scores(weak, QA_BY_ID, name)
    shared = [k for k in gb if k in wb]
    if not gb:
        continue
    if not shared:
        unscored_by_weak.append(name)
    elif all(abs(wb[k] - want) <= 1e-9 for k in shared):
        non_discriminating.append(name)
print(f"    ceiling metrics the weak system also maxes out: "
      f"{non_discriminating or 'none'}")
print(f"    ceiling metrics the weak system never produces a score for: "
      f"{unscored_by_weak or 'none'}")

# content_recall in full, the metric this guard started from.
print(f"\n  content_recall  {'section':10} {'type':20} {'good':>8} {'weak':>8}")
_gcr = ceiling.bucket_scores(good, QA_BY_ID, "content_recall")
_wcr = ceiling.bucket_scores(weak, QA_BY_ID, "content_recall")
for k, v in _gcr.items():
    print(f"  {'':16} {str(k[0]):10} {str(k[1]):20} {v:8.4f} "
          f"{_wcr.get(k, float('nan')):8.4f}")

# ------------------------------------------------------------ self-tests
#
# Three, in the same style as the LEVELS/STAGES parity check's three. A guard
# that reports zero on a healthy repo has to be shown capable of reporting
# something else.
print("\n  CEILING GUARD SELF-TESTS")


class _FakeMetric:
    def __init__(self, name, hib=True):
        self.name, self.HIGHER_IS_BETTER, self.TIER = name, hib, 1


_probe_reg = list(registry) + [_FakeMetric("brand_new_metric")]
_unc, _ = ceiling.classify(_probe_reg)
check("  a metric in neither set fails the run, naming it",
      _unc == ["brand_new_metric"], str(_unc))

# A ceiling metric falling in ONE bucket must fail, and the failure must name
# the bucket -- an aggregate would hide a single bucket behind nine good ones.
_planted = json.loads(json.dumps(good))
_hit = 0
for row in _planted["rows"]:
    case = QA_BY_ID.get(row["id"])
    if case and case.get("section") == "symptoms" \
            and case["type"] == "symptom_remedy" \
            and (row.get("metrics") or {}).get("content_recall") is not None:
        row["metrics"]["content_recall"] = 0.5
        _hit += 1
_planted_fail = ceiling.ceiling_failures(_planted, QA_BY_ID, registry)
_named = [(m, b) for m, b, _, _ in _planted_fail
          if m == "content_recall" and b == ("symptoms", "symptom_remedy")]
check("  a ceiling metric falling in one bucket fails, naming the bucket",
      bool(_named) and len(_planted_fail) == 1,
      f"planted in {_hit} rows -> failures {_planted_fail[:3]}")
# ...and the aggregate over that same planted run is still 0.94, which is why
# this is asserted per bucket and not on the mean.
_agg = ceiling.bucket_scores(_planted, QA_BY_ID, "content_recall")
_mean = sum(_agg.values()) / len(_agg)
check("  the same fault is invisible in the aggregate", _mean > 0.9,
      f"aggregate {_mean:.4f} -- a per-bucket check is the only thing that "
      f"sees it")

check("  control: the unmodified run has no ceiling failure and no "
      "unclassified metric",
      not ceiling.ceiling_failures(good, QA_BY_ID, registry)
      and not ceiling.classify(registry)[0])


# Direction check per gate: the weak system must be worse, not merely different.
for gate, op, _ in run_eval.GATES:
    ga = good["metrics"].get(gate, {}).get("value")
    wa = weak["metrics"].get(gate, {}).get("value")
    if ga is None or wa is None:
        continue
    worse = (wa < ga) if op == ">=" else (wa > ga)
    check(f"weak is strictly worse on {gate}", worse, f"good={ga} weak={wa}")


# ==================================================== 3. determinism
print("\nTier 1 is byte-reproducible")

good2 = run("good", "selftest_good2")
check("two runs of the same config produce identical metrics",
      good["metrics"] == good2["metrics"],
      "Tier 1 must be deterministic; differing numbers mean a bug in the harness")
check("two runs produce identical per-case rows",
      [r["metrics"] for r in good["rows"]] == [r["metrics"] for r in good2["rows"]])


# ==================================================== 4. compare.py behaviour
print("\ncompare.py")


def run_compare(pa, pb):
    out = subprocess.run(
        [sys.executable, os.path.join(REPO_ROOT, "eval", "compare.py"), pa, pb],
        cwd=REPO_ROOT, capture_output=True, text=True)
    return out.returncode, out.stdout + out.stderr


def write_tmp(rec):
    fd, p = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(rec, f)
    return p


p_good, p_good2, p_weak = write_tmp(good), write_tmp(good2), write_tmp(weak)

rc, _ = run_compare(p_good, p_good2)
check("identical runs compare clean and exit 0", rc == 0)

rc, out = run_compare(p_good, p_weak)
check("a regression exits non-zero", rc == 1, f"exit={rc}")
check("a regression is reported as REGRESSED", "REGRESSED" in out)
check("a broken gate is called out", "GATE BROKEN" in out)

# id-freeze: a rebuilt qa_set under a new scheme must be refused, not diffed.
drifted = json.loads(json.dumps(good2))
drifted["case_id_scheme"] = "md5(question)[:10]"
p_drift = write_tmp(drifted)
rc, out = run_compare(p_good, p_drift)
check("an id-scheme change is refused, not silently diffed", rc == 2, f"exit={rc}")
check("the refusal names the cause", "ids changed, runs not comparable" in out)

# The same must hold when the scheme string is unchanged but the ids are not.
renamed = json.loads(json.dumps(good2))
for i, r in enumerate(renamed["rows"]):
    r["id"] = f"newid{i}"
p_renamed = write_tmp(renamed)
rc, out = run_compare(p_good, p_renamed)
check("wholesale id replacement is also refused", rc == 2, f"exit={rc}")

for p in (p_good, p_good2, p_weak, p_drift, p_renamed):
    os.unlink(p)


# ==================================================================== summary
print("\n" + "=" * 60)
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: harness verified")
sys.exit(0)
