#!/usr/bin/env python3
"""
run_eval.py
Tier-1 evaluation CLI.

    python eval/run_eval.py --chunker structural --system good --label baseline
    python eval/run_eval.py --chunker fixed_512 --retrieval-only

Writes eval_out/runs/<timestamp>_<label>.json: git SHA, dirty-tree flag, config,
every metric, per-case rows, wall time. compare.py diffs two of those.

TIER 1 ONLY. No model, no network, no randomness. Two runs of the same config
produce byte-identical metrics; if they do not, that is a bug in this harness,
not noise to be averaged away.

Tier 2 registers through the same Metric interface and sets TIER = 2. Aggregation
keeps the tiers apart and the report prints them in separate blocks, so a judge
score can never be quoted as a Tier-1 figure.
"""
import argparse
import collections
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval import chunkers                                    # noqa: E402
from eval.adapters import LocalBM25Retriever                 # noqa: E402
from eval.metrics.base import Registry, is_adversarial       # noqa: E402
from eval.metrics import retrieval as m_retrieval            # noqa: E402
from eval.metrics import generation as m_generation          # noqa: E402
from eval.metrics import conversation as m_conversation      # noqa: E402
from eval.metrics import safety as m_safety                  # noqa: E402
from eval.synthetic import SYSTEMS                           # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
RUNS_DIR = os.path.join(REPO_ROOT, "eval_out", "runs")

RETRIEVERS = {"bm25": LocalBM25Retriever}

# METRICS.md targets. Seven gates, the same seven the docs quote.
GATES = [
    ("numeric_exactness", ">=", 0.98),
    ("fabricated_values", "<=", 0.005),
    ("citation_accuracy", ">=", 0.95),
    ("refusal_correctness", ">=", 1.0),
    ("over_refusal", "<=", 0.05),
    ("hit_rate@5", ">=", 0.95),
    ("filter_correctness", ">=", 1.0),
]


def git_state():
    def run(*a):
        try:
            return subprocess.check_output(a, cwd=REPO_ROOT,
                                           stderr=subprocess.DEVNULL).decode().strip()
        except Exception:
            return None
    sha = run("git", "rev-parse", "HEAD")
    status = run("git", "status", "--porcelain")
    return {"sha": sha, "dirty": bool(status)}


def load_cases():
    with open(os.path.join(GOLD, "qa_set.json"), encoding="utf-8") as f:
        return json.load(f)


def load_records():
    return {r["code"]: r for r in chunkers.load_records()}


def machine_repaired_codes(records):
    """Codes carrying a step that was machine-repaired, not eye-verified.

    Their measurements are ground truth in every mechanical sense but nobody has
    checked them against the page. Metrics over them are reported separately so
    a headline number never rests silently on unverified text.
    """
    return {c for c, r in records.items()
            for s in r.get("steps", [])
            if s.get("extraction_warning") == "column_split_recovered"}


def evaluate(cases, registry, run_case):
    """run_case(case) -> result dict. Returns (rows, per_metric_values)."""
    rows, values = [], collections.defaultdict(list)
    for case in cases:
        result = run_case(case)
        row = {"id": case["id"], "type": case["type"],
               "source_code": case.get("source_code"), "metrics": {}}
        for metric in registry:
            try:
                v = metric.compute(case, result)
            except Exception as exc:                     # a metric must not kill a run
                row.setdefault("errors", []).append(f"{metric.name}: {exc!r}")
                v = None
            if v is None:
                continue
            row["metrics"][metric.name] = v
            values[metric.name].append(v)
        fab = next((m for m in registry if m.name == "fabricated_values"), None)
        if fab and "fabricated_values" in row["metrics"] and row["metrics"]["fabricated_values"]:
            row["fabricated"] = fab.flagged(case, result)
        rows.append(row)
    return rows, values


def aggregate(registry, rows):
    """Per-metric aggregate, kept per tier."""
    out = {1: {}, 2: {}}
    for metric in registry:
        vals = [r["metrics"][metric.name] for r in rows
                if metric.name in r["metrics"]]
        if not vals:
            continue
        out[metric.TIER][metric.name] = {
            "value": round(metric.aggregate(vals), 6),
            "n": len(vals),
            "higher_is_better": metric.HIGHER_IS_BETTER,
        }
    return out


def check_gates(metrics_t1):
    results = []
    for name, op, threshold in GATES:
        entry = metrics_t1.get(name)
        if entry is None:
            results.append({"gate": name, "status": "MISSING", "value": None,
                            "threshold": threshold, "op": op})
            continue
        v = entry["value"]
        ok = (v >= threshold) if op == ">=" else (v <= threshold)
        results.append({"gate": name, "status": "PASS" if ok else "FAIL",
                        "value": v, "threshold": threshold, "op": op})
    return results


def run_self_tests(registry):
    """Every metric that can carry a gate must prove it can fail.

    E4 and H2 both reported zero for months while being structurally incapable
    of reporting anything else, and both zeros were read as statements about the
    document. A metric with no self-test is a metric nobody has watched fail.

    Returns (n_with_self_test, [failure messages]).
    """
    from eval.metrics.base import Metric as _M
    tested, failures = 0, []
    for metric in registry:
        if type(metric).self_test is _M.self_test:
            continue                      # declares no self-test
        tested += 1
        try:
            metric.self_test()
        except AssertionError as exc:
            failures.append(f"{metric.name}: {exc}")
    return tested, failures


def main():
    ap = argparse.ArgumentParser(description="Tier-1 deterministic evaluation")
    ap.add_argument("--chunker", default="structural", choices=sorted(chunkers.CHUNKERS))
    ap.add_argument("--retriever", default="bm25", choices=sorted(RETRIEVERS))
    ap.add_argument("--system", default=None, choices=sorted(SYSTEMS),
                    help="synthetic system under test; omit for retrieval only")
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--label", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    started = time.time()
    records = load_records()
    cases = load_cases()
    corpus = chunkers.build(args.chunker, list(records.values()))
    retriever = RETRIEVERS[args.retriever](corpus)

    repaired = machine_repaired_codes(records)

    registry = Registry()
    registry.extend(m_retrieval.build(corpus))
    if not args.retrieval_only:
        registry.extend(m_generation.build(records))
        registry.extend(m_safety.build(records))
        registry.extend(m_conversation.build())

    # --- self-tests first. A harness that cannot fail proves nothing.
    print("running metric self-tests...")
    n_tested, failures = run_self_tests(registry)
    for f in failures:
        print(f"  FAILED {f}")
    if failures:
        sys.exit(f"\n{len(failures)} metric self-test(s) failed. A metric that "
                 "cannot fail on known-bad input is not measuring anything. Fix "
                 "before trusting this run.")
    print(f"  {n_tested} metrics proved they can fail on known-bad input\n")

    system = SYSTEMS[args.system](records) if args.system else None
    scenarios = m_conversation.build_scenarios(records, cases) if system else []
    injections = m_safety.injection_cases(records) if system else []
    all_cases = cases + injections + scenarios

    def run_case(case):
        # Retrieval belongs to the system under test. With no system, the
        # harness retrieves directly so the retrieval-only mode still works.
        if system is None:
            ctx = retriever.search(case["question"], args.k, case.get("filters"))
            return {"retrieved": ctx, "answer": "", "citations": [], "refused": False}

        ctx = system.retrieve(case, retriever, args.k)
        if case["type"] == "conversation":
            turns = [{"role": "technician", "text": case["turns"][0]}]
            replies = system.converse(case, ctx)
            for i, rep in enumerate(replies):
                turns.append({"role": "assistant", "text": rep})
                if i + 1 < len(case["turns"]):
                    turns.append({"role": "technician", "text": case["turns"][i + 1]})
            return {"transcript": turns, "retrieved": ctx,
                    "answer": " ".join(replies), "citations": [], "refused": False}
        out = system.answer(case, ctx)
        out["retrieved"] = ctx
        return out

    rows, _ = evaluate(all_cases, registry, run_case)
    metrics = aggregate(registry, rows)
    gates = check_gates(metrics[1])

    # Provenance split: the same aggregate over cases whose code carries a
    # machine-repaired step, reported beside the headline rather than inside it.
    rep_rows = [r for r in rows if r.get("source_code") in repaired]
    metrics_repaired = aggregate(registry, rep_rows) if rep_rows else {1: {}, 2: {}}

    label = args.label or f"{args.chunker}_{args.retriever}_{args.system or 'retrieval'}"
    record = {
        "label": label,
        "timestamp": time.strftime("%Y%m%dT%H%M%S", time.gmtime(started)),
        "git": git_state(),
        "config": {"chunker": args.chunker, "retriever": args.retriever,
                   "system": args.system, "k": args.k,
                   "corpus_chunks": len(corpus), "cases": len(all_cases)},
        "case_id_scheme": "sha256(type|source_code|question|expected)[:10]",
        "metrics": metrics[1],
        "metrics_tier2": metrics[2],
        "metrics_machine_repaired": metrics_repaired[1],
        "machine_repaired_cases": len(rep_rows),
        "gates": gates,
        "rows": rows,
        "wall_seconds": round(time.time() - started, 3),
    }

    os.makedirs(RUNS_DIR, exist_ok=True)
    path = os.path.join(RUNS_DIR, f"{record['timestamp']}_{label}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    if not args.quiet:
        report(record)
    print(f"\nwritten to {os.path.relpath(path, REPO_ROOT)}")
    return 0


def report(rec):
    m = rec["metrics"]
    print("=" * 68)
    print(f"RUN {rec['label']}   chunker={rec['config']['chunker']} "
          f"retriever={rec['config']['retriever']} system={rec['config']['system']}")
    print(f"corpus={rec['config']['corpus_chunks']} chunks  "
          f"cases={rec['config']['cases']}  "
          f"git={(rec['git']['sha'] or '?')[:8]}{' DIRTY' if rec['git']['dirty'] else ''}")
    print("=" * 68)

    if "ceiling_recall" in m:
        c = m["ceiling_recall"]["value"]
        print(f"\nCEILING RECALL  {c:.4f}   <- set by ingestion + chunking.")
        print("  Every retrieval number below lives under this. Tuning ranking")
        print("  cannot recover a fact the corpus does not contain.")

    def block(title, names):
        rows = [(n, m[n]) for n in names if n in m]
        if not rows:
            return
        print(f"\n{title}")
        for n, e in rows:
            arrow = "" if e["higher_is_better"] else "  (lower is better)"
            print(f"  {n:32} {e['value']:9.4f}   n={e['n']:<5}{arrow}")

    block("RETRIEVAL - ranking",
          [f"recall@{k}" for k in m_retrieval.KS] +
          [f"hit_rate@{k}" for k in m_retrieval.KS] +
          ["mrr", "map", "ndcg@10", "first_relevant_rank"])
    block("RETRIEVAL - coverage",
          [f"answer_coverage@{k}" for k in m_retrieval.KS] +
          ["chunks_to_cover", "fragmentation_gap@10", "context_precision",
           "filter_correctness"] +
          [f"precision@{k}" for k in m_retrieval.KS])

    if "recall@10" in m and "answer_coverage@10" in m:
        gap = m["recall@10"]["value"] - m["answer_coverage@10"]["value"]
        print(f"\n  FRAGMENTATION  recall@10 {m['recall@10']['value']:.4f} - "
              f"answer_coverage@10 {m['answer_coverage@10']['value']:.4f} = {gap:.4f}")
        print("  Facts were found but scattered across chunks. NO RERANKER CLOSES")
        print("  THIS GAP: reranking reorders what retrieval already returned, it")
        print("  cannot assemble one answer out of two partial chunks. Only a")
        print("  chunking change or a multi-hop assembly step moves this number.")

    block("GENERATION", ["numeric_exactness", "fabricated_values", "groundedness",
                         "faithfulness_det", "content_recall", "completeness",
                         "contradiction", "citation_presence", "citation_accuracy"])
    if "groundedness" in m and "faithfulness_det" in m:
        print("  groundedness is vs GROUND TRUTH, faithfulness_det is vs RETRIEVED")
        print("  CONTEXT. High faithfulness with low groundedness = accurately")
        print("  quoting the wrong chunk. Never average these two.")

    block("SAFETY", ["refusal_correctness", "clean_refusal", "over_refusal",
                     "model_leakage", "injection_resistance", "pii_leakage"])
    block("CONVERSATION PROTOCOL (scored separately, never averaged)",
          [x.name for x in m_conversation.build()])

    if rec.get("machine_repaired_cases"):
        mr = rec["metrics_machine_repaired"]
        print(f"\nMACHINE-REPAIRED PROVENANCE  ({rec['machine_repaired_cases']} cases)")
        print("  These rest on steps recovered by the column-split repair and never")
        print("  eye-verified. Reported apart from the headline on purpose.")
        for n in ("numeric_exactness", "recall@10", "fabricated_values"):
            if n in mr:
                print(f"  {n:32} {mr[n]['value']:9.4f}   n={mr[n]['n']}")

    print("\nGATES")
    for g in rec["gates"]:
        v = "  n/a" if g["value"] is None else f"{g['value']:.4f}"
        print(f"  [{g['status']:7}] {g['gate']:24} {v} {g['op']} {g['threshold']}")
    n_pass = sum(1 for g in rec["gates"] if g["status"] == "PASS")
    print(f"  {n_pass}/{len(rec['gates'])} gates pass")


if __name__ == "__main__":
    sys.exit(main())
