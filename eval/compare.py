#!/usr/bin/env python3
"""
compare.py
Run-to-run regression diff. The reason the run records exist.

    python eval/compare.py eval_out/runs/<a>.json eval_out/runs/<b>.json

TIER 1 HAS NO NOISE FLOOR. It is deterministic: same inputs, same numbers. So
any movement at all is a real behaviour change and is reported as one. There is
no "within tolerance" band to hide a small regression in -- if a number moved,
something moved it. A Tier-2 comparison would need a band; this one must not
have one, and the two must never share a threshold.

Exits non-zero when anything regressed, so it can gate a commit.

CASE IDS ARE FROZEN. Runs are matched case by case on qa_set ids. If the id
scheme ever changes, every case looks new and every metric looks like it moved
from nothing -- a diff that is silently meaningless. That is refused loudly
instead.
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGS_DIR = os.path.join(REPO_ROOT, "eval_out", "logs")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def index_rows():
    sys.path.insert(0, REPO_ROOT)
    from core.logging import read_index
    return read_index(LOGS_DIR)


def resolve_run(token):
    """Accept a path, a run_id, or a label. Latest wins for a label."""
    if os.path.isfile(token):
        return load(token)
    rows = index_rows()
    hit = [r for r in rows if r.get("run_id") == token]
    if not hit:
        hit = [r for r in rows if r.get("label") == token]
    if not hit:
        raise SystemExit(
            f"no run matching {token!r}. `--list` shows what is on record.")
    row = hit[-1]
    path = os.path.join(REPO_ROOT, row["run_file"])
    if not os.path.isfile(path):
        raise SystemExit(f"index names {row['run_file']} but it is missing")
    return load(path)


def list_runs(limit=25):
    rows = index_rows()
    if not rows:
        print(f"no runs on record in {os.path.relpath(LOGS_DIR, REPO_ROOT)}")
        return 0
    print(f"{'run_id':26} {'label':22} {'git':9} {'gates':7} "
          f"{'num_exact':>10} {'fabricated':>11}  when")
    for r in rows[-limit:]:
        h = r.get("headline") or {}
        sha = (r.get("git_sha") or "?")[:7] + ("*" if r.get("git_dirty") else "")
        ne = h.get("numeric_exactness")
        fv = h.get("fabricated_values")
        print(f"{r.get('run_id',''):26} {(r.get('label') or '')[:22]:22} {sha:9} "
              f"{str(r.get('gates_passed'))+'/'+str(r.get('gates_total')):7} "
              f"{(f'{ne:.4f}' if ne is not None else '-'):>10} "
              f"{(f'{fv:.4f}' if fv is not None else '-'):>11}  "
              f"{r.get('timestamp','')}")
    print(f"\n{len(rows)} runs on record. This is the trend line: a metric moving")
    print("across months of rows is a stronger argument than any single number.")
    return 0


def check_comparable(a, b):
    """Refuse to diff runs that are not actually comparable.

    Returns a list of fatal reasons. Config differences are reported but not
    fatal -- comparing two chunkings is a legitimate use. Id-scheme drift is
    fatal, because it silently turns every case into a new case.
    """
    fatal = []

    scheme_a = a.get("case_id_scheme")
    scheme_b = b.get("case_id_scheme")
    if scheme_a != scheme_b:
        fatal.append(
            f"ids changed, runs not comparable: case_id_scheme differs\n"
            f"    A: {scheme_a}\n    B: {scheme_b}\n"
            f"  qa_set.json ids were regenerated under a different scheme. Every\n"
            f"  case in B is a new case as far as this diff is concerned, so the\n"
            f"  comparison would be meaningless rather than merely noisy.")

    ids_a = {r["id"] for r in a.get("rows", [])}
    ids_b = {r["id"] for r in b.get("rows", [])}
    if ids_a and ids_b:
        overlap = len(ids_a & ids_b) / max(len(ids_a | ids_b), 1)
        if overlap < 0.5:
            fatal.append(
                f"ids changed, runs not comparable: only {overlap:.1%} of case ids\n"
                f"  are shared ({len(ids_a & ids_b)} of {len(ids_a | ids_b)}).\n"
                f"  Either qa_set.json was rebuilt under a new id scheme, or these\n"
                f"  runs evaluated different case sets.")
    return fatal


def compare(a, b):
    ma, mb = a["metrics"], b["metrics"]
    rows = []
    for name in sorted(set(ma) | set(mb)):
        ea, eb = ma.get(name), mb.get(name)
        if ea is None or eb is None:
            rows.append({"metric": name, "a": ea and ea["value"],
                         "b": eb and eb["value"], "delta": None,
                         "status": "ADDED" if ea is None else "REMOVED"})
            continue
        d = eb["value"] - ea["value"]
        hib = ea.get("higher_is_better", True)
        if d == 0:
            status = "UNCHANGED"
        elif (d > 0) == hib:
            status = "IMPROVED"
        else:
            status = "REGRESSED"
        rows.append({"metric": name, "a": ea["value"], "b": eb["value"],
                     "delta": d, "status": status,
                     "higher_is_better": hib})
    return rows


def is_orchestrator(rec):
    return rec.get("kind") == "orchestrator"


def stage_diff(a, b):
    """Stage-by-stage diff of two orchestrator runs."""
    sa = {s["stage"]: s for s in a.get("stages", [])}
    sb = {s["stage"]: s for s in b.get("stages", [])}
    rows = []
    for name in sorted(set(sa) | set(sb)):
        x, y = sa.get(name), sb.get(name)
        rows.append({
            "stage": name,
            "a": x["status"] if x else "ABSENT",
            "b": y["status"] if y else "ABSENT",
            "changed": (x or {}).get("status") != (y or {}).get("status"),
            "a_secs": (x or {}).get("seconds"), "b_secs": (y or {}).get("seconds"),
            "skip_reason": (y or {}).get("skip_reason"),
        })
    return rows


def orchestrator_gate_diff(a, b):
    """Every gate, keyed by stage.gate, with its value on each side."""
    ga = {f"{g['stage']}.{g['gate']}": g for g in a.get("gates", [])}
    gb = {f"{g['stage']}.{g['gate']}": g for g in b.get("gates", [])}
    rows = []
    for k in sorted(set(ga) | set(gb)):
        x, y = ga.get(k), gb.get(k)
        av = x["value"] if x else None
        bv = y["value"] if y else None
        rows.append({"gate": k, "a": av, "b": bv,
                     "a_status": (x or {}).get("status", "ABSENT"),
                     "b_status": (y or {}).get("status", "ABSENT"),
                     "moved": av != bv})
    return rows


def report_orchestrator(a, b):
    print("=" * 72)
    print(f"A  {a['run_id']}  {(a['git']['sha'] or '?')[:8]}"
          f"{' DIRTY' if a['git']['dirty'] else ''}")
    print(f"B  {b['run_id']}  {(b['git']['sha'] or '?')[:8]}"
          f"{' DIRTY' if b['git']['dirty'] else ''}")
    print("=" * 72)

    print(f"\n{'stage':14}{'A':10}{'B':10}  note")
    for r in stage_diff(a, b):
        note = "CHANGED" if r["changed"] else ""
        if r["skip_reason"]:
            note = (note + "  " if note else "") + r["skip_reason"]
        print(f"{r['stage']:14}{r['a']:10}{r['b']:10}  {note}")

    moved = [g for g in orchestrator_gate_diff(a, b) if g["moved"]]
    print(f"\nGATES THAT MOVED: {len(moved)}")
    for g in moved:
        print(f"  {g['gate']:40} {g['a']} -> {g['b']}  "
              f"({g['a_status']} -> {g['b_status']})")
    broke = [g for g in orchestrator_gate_diff(a, b)
             if g["b_status"] == "FAIL" and g["a_status"] != "FAIL"]
    if broke:
        print("\nGATES BROKEN")
        for g in broke:
            print(f"  {g['gate']}: {g['a']} -> {g['b']}  <-- GATE BROKEN")

    ka, kb = a.get("known_gaps", {}), b.get("known_gaps", {})
    if ka != kb:
        print("\nKNOWN GAPS CHANGED")
        for k in sorted(set(ka) | set(kb)):
            if ka.get(k) != kb.get(k):
                print(f"  {k}: {ka.get(k)} -> {kb.get(k)}")
    else:
        print(f"\nknown gaps unchanged (human_verified: "
              f"{kb.get('human_verified')})")

    rc_a, rc_b = a.get("regression_counts", {}), b.get("regression_counts", {})
    if rc_a != rc_b:
        print("\nREGRESSION COUNTS MOVED -- investigate before accepting")
        for k in sorted(set(rc_a) | set(rc_b)):
            if rc_a.get(k) != rc_b.get(k):
                print(f"  {k}: {rc_a.get(k)} -> {rc_b.get(k)}")

    return 1 if (broke or rc_a != rc_b) else 0


def gate_diff(a, b):
    ga = {g["gate"]: g["status"] for g in a.get("gates", [])}
    gb = {g["gate"]: g["status"] for g in b.get("gates", [])}
    out = []
    for name in sorted(set(ga) | set(gb)):
        sa, sb = ga.get(name, "MISSING"), gb.get(name, "MISSING")
        if sa != sb:
            out.append((name, sa, sb))
    return out


def changed_cases(a, b, metric):
    """Per-case movement on one metric, matched by frozen id."""
    ra = {r["id"]: r["metrics"].get(metric) for r in a.get("rows", [])}
    rb = {r["id"]: r["metrics"].get(metric) for r in b.get("rows", [])}
    out = []
    for cid in sorted(set(ra) & set(rb)):
        if ra[cid] != rb[cid] and ra[cid] is not None and rb[cid] is not None:
            out.append((cid, ra[cid], rb[cid]))
    return out


def divergence(a, b, limit=20):
    """Which specific cases changed verdict, across every metric.

    When a metric drops the first question is always "which cases broke", and
    the answer should be one command away rather than a scripting exercise
    against two run files.
    """
    ra = {r["id"]: r for r in a.get("rows", [])}
    rb = {r["id"]: r for r in b.get("rows", [])}
    per_metric = {}
    for cid in sorted(set(ra) & set(rb)):
        ma, mb = ra[cid]["metrics"], rb[cid]["metrics"]
        for name in set(ma) | set(mb):
            va, vb = ma.get(name), mb.get(name)
            if va is None or vb is None or va == vb:
                continue
            per_metric.setdefault(name, []).append(
                {"case_id": cid, "type": ra[cid].get("type"),
                 "source_code": ra[cid].get("source_code"), "a": va, "b": vb,
                 "trace_a": ra[cid].get("trace_id"),
                 "trace_b": rb[cid].get("trace_id")})
    return per_metric


def main():
    ap = argparse.ArgumentParser(description="Tier-1 run-to-run regression diff")
    ap.add_argument("run_a", nargs="?", help="path, run_id, or label")
    ap.add_argument("run_b", nargs="?", help="path, run_id, or label")
    ap.add_argument("--list", action="store_true",
                    help="list runs on record in eval_out/logs/index.jsonl")
    ap.add_argument("--show-cases", metavar="METRIC", default=None,
                    help="list per-case movement for one metric")
    ap.add_argument("--divergence", action="store_true",
                    help="which specific cases changed verdict, across all metrics")
    args = ap.parse_args()

    if args.list:
        return list_runs()
    if not (args.run_a and args.run_b):
        ap.error("need two runs (path, run_id or label), or --list")

    a, b = resolve_run(args.run_a), resolve_run(args.run_b)

    # Orchestrator runs have their own shape: stages and gates, not per-case
    # rows. Diffing them through the case-level path would report every stage
    # as an added metric.
    if is_orchestrator(a) or is_orchestrator(b):
        if not (is_orchestrator(a) and is_orchestrator(b)):
            print("REFUSING TO COMPARE: one run is an orchestrator run and the "
                  "other is not. They record different things.")
            return 2
        return report_orchestrator(a, b)

    fatal = check_comparable(a, b)
    if fatal:
        print("REFUSING TO COMPARE\n")
        for f in fatal:
            print(f"  {f}\n")
        return 2

    print("=" * 70)
    print(f"A  {a['label']:28} {a['timestamp']}  "
          f"{(a['git']['sha'] or '?')[:8]}{' DIRTY' if a['git']['dirty'] else ''}")
    print(f"B  {b['label']:28} {b['timestamp']}  "
          f"{(b['git']['sha'] or '?')[:8]}{' DIRTY' if b['git']['dirty'] else ''}")
    if a["config"] != b["config"]:
        print("\nconfig differs (expected when comparing chunkings):")
        for k in sorted(set(a["config"]) | set(b["config"])):
            if a["config"].get(k) != b["config"].get(k):
                print(f"  {k}: {a['config'].get(k)} -> {b['config'].get(k)}")
    print("=" * 70)

    rows = compare(a, b)
    order = {"REGRESSED": 0, "ADDED": 1, "REMOVED": 1, "IMPROVED": 2, "UNCHANGED": 3}
    rows.sort(key=lambda r: (order[r["status"]], r["metric"]))

    print(f"\n{'metric':34} {'A':>10} {'B':>10} {'delta':>10}  status")
    for r in rows:
        av = "-" if r["a"] is None else f"{r['a']:.4f}"
        bv = "-" if r["b"] is None else f"{r['b']:.4f}"
        dv = "-" if r["delta"] is None else f"{r['delta']:+.4f}"
        print(f"{r['metric']:34} {av:>10} {bv:>10} {dv:>10}  {r['status']}")

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    gd = gate_diff(a, b)
    if gd:
        print("\nGATE CHANGES")
        for name, sa, sb in gd:
            marker = "  <-- GATE BROKEN" if sb == "FAIL" else ""
            print(f"  {name:26} {sa} -> {sb}{marker}")

    if args.show_cases:
        cases = changed_cases(a, b, args.show_cases)
        print(f"\nPER-CASE MOVEMENT on {args.show_cases} ({len(cases)} cases)")
        for cid, va, vb in cases[:40]:
            print(f"  {cid:14} {va} -> {vb}")
        if len(cases) > 40:
            print(f"  ... and {len(cases) - 40} more")

    if args.divergence:
        div = divergence(a, b)
        print(f"\nPER-CASE DIVERGENCE ({len(div)} metrics with case-level change)")
        for name in sorted(div, key=lambda n: -len(div[n])):
            rows_ = div[name]
            worse = [r for r in rows_ if r["b"] < r["a"]]
            print(f"\n  {name}  ({len(rows_)} cases changed, {len(worse)} worse)")
            for r in rows_[:8]:
                mark = "WORSE" if r["b"] < r["a"] else "better"
                print(f"    {r['case_id']:12} {r['source_code'] or '-':8} "
                      f"{r['type']:20} {r['a']} -> {r['b']}  {mark}")
                if r.get("trace_b"):
                    print(f"       trace {r['trace_b']} in the run's log")
            if len(rows_) > 8:
                print(f"    ... and {len(rows_) - 8} more")

    regressed = counts.get("REGRESSED", 0)
    broken = [g for g in gd if g[2] == "FAIL"]
    if regressed or broken:
        print(f"\nFAIL: {regressed} metric(s) regressed, {len(broken)} gate(s) broken.")
        print("Tier 1 is deterministic -- there is no noise floor, so every one of")
        print("these is a real behaviour change. Explain it or fix it.")
        return 1

    print("\nOK: no regressions.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
