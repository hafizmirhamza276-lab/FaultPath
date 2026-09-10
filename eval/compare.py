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


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


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


def main():
    ap = argparse.ArgumentParser(description="Tier-1 run-to-run regression diff")
    ap.add_argument("run_a")
    ap.add_argument("run_b")
    ap.add_argument("--show-cases", metavar="METRIC", default=None,
                    help="list per-case movement for one metric")
    args = ap.parse_args()

    a, b = load(args.run_a), load(args.run_b)

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
