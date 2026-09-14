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

# ------------------------------------------------- the ceiling, per bucket
#
# GOOD SYSTEM IS THE CEILING, AND THE CEILING IS ASSERTED, NOT ASSUMED.
#
# content_recall measures whether an answer carries every fact the question
# requires. GoodSystem answers straight out of the ground truth, so it should
# carry all of them in EVERY bucket. Where it does not, the ceiling for that
# bucket is below 1.0 and the metric has no headroom there: it cannot separate
# a good real system from a mediocre one, and the aggregate quietly averages
# two different ceilings.
#
# That is exactly what happened to symptom_remedy, which sat at 0.6699 while
# every other bucket was 1.0000 -- GoodSystem named the cause and the remedy
# but not the observation that triggers it. The aggregate read 0.9628 and
# looked fine.
#
# ASSERTED PER (section, type), NOT ON THE AGGREGATE. An aggregate cannot show
# a single bucket falling, which is the whole failure mode. Same reasoning as
# the raise on GoodSystem's unregistered-type fallback: a case type the good
# system cannot fully answer must fail loudly rather than lower the ceiling in
# silence.
QA_BY_ID = {c["id"]: c for c in json.load(
    open(os.path.join(REPO_ROOT, "golden", "qa_set.json"), encoding="utf-8"))}


def content_recall_buckets(run_record):
    acc = collections.defaultdict(list)
    for row in run_record["rows"]:
        case = QA_BY_ID.get(row["id"])
        if case is None:
            continue
        v = (row.get("metrics") or {}).get("content_recall")
        if v is not None:
            acc[(case.get("section"), case["type"])].append(v)
    return {k: sum(v) / len(v) for k, v in sorted(acc.items())}


good_cr = content_recall_buckets(good)
weak_cr = content_recall_buckets(weak)
print(f"\n  content_recall ceiling  {'section':10} {'type':20} {'n':>5} "
      f"{'good':>8} {'weak':>8}")
below = []
for k, v in good_cr.items():
    n = sum(1 for row in good["rows"]
            if QA_BY_ID.get(row["id"], {}).get("section") == k[0]
            and QA_BY_ID.get(row["id"], {}).get("type") == k[1]
            and (row.get("metrics") or {}).get("content_recall") is not None)
    print(f"  {'':22} {k[0]:10} {k[1]:20} {n:>5} {v:8.4f} "
          f"{weak_cr.get(k, float('nan')):8.4f}")
    if v < 1.0:
        below.append(f"{k[0]}/{k[1]}={v:.4f}")
check("the good system scores content_recall 1.0000 in EVERY bucket",
      not below,
      f"buckets below the ceiling: {below}\n"
      "          The good system answers out of the ground truth, so a bucket "
      "under 1.0\n          means its answer is missing a required fact -- a "
      "defect in the answer,\n          and a ceiling the metric cannot "
      "measure above.")
check("content_recall still discriminates in every bucket the weak system "
      "answers", all(weak_cr.get(k, 0.0) < v for k, v in good_cr.items()),
      f"no separation in: "
      f"{[k for k, v in good_cr.items() if weak_cr.get(k, 0.0) >= v]}")


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
