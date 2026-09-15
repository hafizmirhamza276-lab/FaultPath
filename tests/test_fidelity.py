#!/usr/bin/env python3
"""
test_fidelity.py
Extraction fidelity: MODULE, PIPELINE and E2E, plus the overfitting guard.

The thing being defended against here is a suite that agrees with itself. The
structural checks must not reuse a single helper the extractor uses, or they
inherit its bugs and confirm them -- which is exactly how citation_accuracy read
1.0000 over 635 wrong pages.
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from core import loader                                  # noqa: E402
from pipeline import fidelity, structural                # noqa: E402
from tests import overfitting as OV                      # noqa: E402

failures = []
timings = {}


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}" + (f"\n          {detail}" if detail else ""))
        failures.append(name)


# ========================================================= MODULE
print("\nMODULE -- self-tests, classification, loader")
t0 = time.perf_counter()

try:
    fidelity.self_test()
    check("fidelity gates prove they can fail", True)
except AssertionError as exc:
    check("fidelity gates prove they can fail", False, str(exc))

try:
    structural.self_test()
    check("structural checks prove they can fail", True)
except AssertionError as exc:
    check("structural checks prove they can fail", False, str(exc))

proof = structural.prove_orphan_fires()
check("orphan_detection is silent on a complete tree", proof["silent_before"])
check("orphan_detection fires on a truncated tree (steps 6,7 cut)",
      proof["fires"],
      f"got {[o['step'] for o in proof['truncated_tree_orphans']]}")

# classification never falls through to silence
c = fidelity.classify({"fact_id": "X:1:step:0", "redirect": False,
                       "warn": None, "reason": "something new"})
check("an unclassifiable fact defaults to DEFECT",
      c["classification"] == fidelity.DEFECT,
      "silence must never be the safe option here")
check("a synthetic Redirect label is a KNOWN_LIMITATION",
      fidelity.classify({"fact_id": "X:1:step:0", "redirect": True,
                         "warn": None, "reason": ""})["classification"]
      == fidelity.KNOWN_LIMITATION)
check("a machine-repaired cause needs human verification",
      fidelity.classify({"fact_id": "X:1:step:0", "redirect": False,
                         "warn": "column_split_recovered",
                         "reason": ""})["classification"] == fidelity.NEEDS_HUMAN)

# loader API separates the two roles
kb = loader.load_knowledge_base()
gt = loader.load_ground_truth()
check("knowledge base carries no verification annotation",
      "verification" not in kb["CA451"]["steps"][0],
      "the agent must behave the same however well we have audited ourselves")
check("ground truth carries verification on every step",
      all("verification" in s for r in gt.values() for s in r["steps"]))
check("ground truth carries verification on every measurement",
      all("verification" in m for r in gt.values() for s in r["steps"]
          for m in s["measurements"]))
check("verification statuses are from the declared set",
      {s["verification"] for r in gt.values() for s in r["steps"]}
      <= set(loader.STATUSES))
summary = loader.summarise()
check("no fact is human_verified yet",
      summary[loader.HUMAN_VERIFIED]["n"] == 0,
      "claiming human verification nobody performed is the same self-agreement")
timings["module"] = time.perf_counter() - t0


# ========================================================= PIPELINE
# ------------------------------------------- every fact kind is reachable
#
# kind == "header" was dead for as long as it existed: _fact() looked up the
# step before dispatching on kind and raised for step 0, where a record-level
# fact necessarily sits. Nothing minted header ids, so nothing exercised the
# path, and render_citations turns UnknownFact into "no citation" -- correct
# for an invented id and indistinguishable from a branch that cannot fire.
# See reports/resolver_dead_branch.md.
#
# Derived from the fact-id grammar rather than a hand-written list, so a kind
# added to FACT_ID_RE and not made reachable fails here.
print("\nevery fact kind the grammar admits is reachable")
from eval.citations import (FACT_ID_RE, _fact,            # noqa: E402
                            UnknownFact, records as cite_records)

_KINDS = set(FACT_ID_RE.pattern.split("kind>")[1].split(")")[0].split("|"))


def _first_probe(pred, mk):
    """A real fact id of some kind, found in the corpus rather than guessed.

    Hand-picked probes go stale: the first attempt used CA131 step 1 for a
    branch and that step has none, so the check failed on the probe rather than
    on the thing it was probing.
    """
    for key in sorted(cite_records()):
        rec = cite_records()[key]
        for st in rec.get("steps", []):
            if pred(rec, st):
                return mk(key, st)
    return None


_PROBES = {
    "step": _first_probe(lambda r, s: s.get("fact_id"),
                         lambda k, s: f"{k}:{s['step']}:step:0"),
    "meas": "CA451:6:meas:0",
    "branch": _first_probe(lambda r, s: (s.get("branches") or {}),
                           lambda k, s: f"{k}:{s['step']}:branch:0"),
    "header": "CA131:0:header:4",
    "remedy": _first_probe(lambda r, s: s.get("remedy_fact_id"),
                           lambda k, s: s["remedy_fact_id"]),
}
print(f"  probes: {_PROBES}")
check("a probe exists for every kind in the fact-id grammar",
      set(_PROBES) == _KINDS, f"grammar {sorted(_KINDS)} probes {sorted(_PROBES)}")
_unreachable = []
for _kind, _fid in sorted(_PROBES.items()):
    try:
        _rec, _text, _prov = _fact(_fid)
        if not _prov.get("manual_page"):
            _unreachable.append(f"{_kind}: resolved with no page")
    except UnknownFact as exc:
        _unreachable.append(f"{_kind} ({_fid}): {exc}")
check("every fact kind resolves to a record, text and a page", not _unreachable,
      "; ".join(_unreachable))

# A header belongs to the record, not to a step. The dead branch invited the
# opposite reading, so the repaired one refuses a non-zero step explicitly.
try:
    _fact("CA131:1:header:0")
    check("a header fact id with a non-zero step is refused", False,
          "returned a record-level field as though it belonged to step 1")
except UnknownFact:
    check("a header fact id with a non-zero step is refused", True)

# header:0 IS THE TITLE, AND THE TITLE IS NOT ON THE HEADER BLOCK'S PAGE for
# 132 of 174 codes -- it is read from the failure-code index table. Asserted
# over the whole corpus rather than sampled, because the whole point of
# capturing title_provenance was that the obvious page is the wrong one.
_wrong_page, _no_prov = [], []
for _code, _rec in sorted(cite_records().items()):
    if "code" not in _rec:
        continue                       # symptom records have no header block
    _tp = _rec.get("title_provenance")
    if _tp is None:
        # Correct only where the title did not come from the index table.
        if _rec.get("in_code_table"):
            _no_prov.append(_code)
        continue
    _, _text, _prov = _fact(f"{_code}:0:header:0")
    if _prov.get("manual_page") != _tp["manual_page"]:
        _wrong_page.append(_code)
check("header:0 resolves against title_provenance, not the header block",
      not _wrong_page, f"{len(_wrong_page)} codes cite the header page for a "
                       f"title that is not on it: {_wrong_page[:5]}")
check("every code in the index table carries title_provenance", not _no_prov,
      f"missing: {_no_prov[:5]}")

# The one code outside the index table keeps the header page, and that is the
# right page for it -- its title really is printed there.
_off = [c for c, r in cite_records().items()
        if "code" in r and not r.get("in_code_table")]
check("a code absent from the index table falls back to the header block",
      len(_off) == 1 and cite_records()[_off[0]].get("title_provenance") is None
      and _fact(f"{_off[0]}:0:header:0")[2].get("manual_page")
      == cite_records()[_off[0]]["header_provenance"]["manual_page"],
      f"codes outside the index table: {_off}")

# --------------------------- every enumerated kind has a containment rule
#
# The CEILING/NOT_CEILING union principle, applied to page containment. A kind
# with no boundary rule must FAIL naming itself rather than default to
# permissive -- which is also why there is ONE containment check deriving its
# boundary per kind rather than two checks that do not know about each other.
# Two would let a third kind land under neither and both stay green.
print("\npage containment: one check, boundary derived per kind")

_enumerated = sorted({f["kind"] for f in fidelity.enumerate_facts(
    {k: v for k, v in fidelity.load_records().items()})}
    | {f["kind"] for f in fidelity.enumerate_symptom_facts(
        fidelity.load_symptom_records())})
_no_rule = [k for k in _enumerated if k not in fidelity.CONTAINMENT_BOUNDS]
check("every enumerated fact kind has a containment rule", not _no_rule,
      f"kinds with no boundary: {_no_rule}")
check("the index-table range is derived from the parser, not a literal",
      fidelity.code_table_pages() == (655, 662),
      f"got {fidelity.code_table_pages()} -- if parse_code_table's range moved, "
      f"this follows it, which is the point")

# FAULT 3: a kind with no rule fails, naming the kind. Not permissive.
_ok, _why = fidelity.containment_ok("brand_new_kind", {"pdf_pages": [1, 9]}, 5)
check("  FAULT 3  a kind with no boundary rule fails, naming it",
      _ok is False and "brand_new_kind" in _why, f"{_ok} {_why}")
check("  control  a known kind on a legitimate page passes",
      fidelity.containment_ok("measurement", {"pdf_pages": [1, 9]}, 5)[0] is True)
check("  control  a known kind on an illegitimate page fails",
      fidelity.containment_ok("measurement", {"pdf_pages": [1, 9]}, 50)[0] is False)
# A title may sit in the index table OR on its own detail page, and nowhere else.
check("  a title inside the index-table range passes",
      fidelity.containment_ok("title", {"pdf_pages": [700, 702]}, 656)[0] is True)
check("  a title on its own detail page passes (DAF8KB's case)",
      fidelity.containment_ok("title", {"pdf_pages": [700, 702]}, 701)[0] is True)
check("  a title on neither fails",
      fidelity.containment_ok("title", {"pdf_pages": [700, 702]}, 900)[0] is False)

print("\nPIPELINE -- fidelity and structure over the full corpus")
t0 = time.perf_counter()
from eval.citations import PageText                      # noqa: E402
pages = PageText()
if not pages.available():
    print(f"  SKIP  PDF not at {pages.path}; fidelity cannot be inferred")
    fid_res = None
else:
    fid_res = fidelity.check_facts(pages=pages)
    print(f"  {'kind':16}{'resolved':>10}{'total':>8}{'rate':>10}")
    for k, v in fid_res["by_kind"].items():
        print(f"  {k:16}{v['resolved']:>10}{v['total']:>8}{v['rate']:>10.4f}")
    for g in fidelity.gates(fid_res):
        check(f"gate {g['gate']} ({g['value']:.4f} {g['op']} {g['threshold']})",
              g["status"] == "PASS")
    check("page_containment is total", fid_res["page_containment"] >= 1.0)

    # ------------------------------- verification is derived, and staleness fails
    #
    # reports/fact_verification.json had no producer and sat frozen at e907cf4
    # while the corpus tripled. load_ground_truth falls through to UNVERIFIED for a
    # fact it cannot find, so 2,757 facts the resolver verifies were reported
    # unverified to API callers. See reports/frozen_verification_findings.md.
    #
    # Both sides derived, as with LEVELS/STAGES and CEILING/NOT_CEILING: what is on
    # disk is compared against what the current run produces. Regenerating once
    # would not stop it going stale again.
    print("\nverification is derived from the run, not read as a given")

    check("build_verification has a caller",
      "loader.build_verification" in open(
          os.path.join(REPO_ROOT, "core", "orchestrator.py"),
          encoding="utf-8").read(),
      "it had none, which is how the artefact froze")

    _drift = loader.verification_drift(fid_res)
    check("the committed verification artefact matches the current run",
          not _drift["stale"],
          f"missing={_drift['n_missing']} ghost={_drift['n_ghost']} "
          f"disagree={_drift['n_disagree']}\n"
          "          Run the fidelity stage and commit "
          "reports/fact_verification.json.")

    _ver = loader.load_verification()
    check("no fact is human_verified",
          _ver["counts"].get(loader.HUMAN_VERIFIED, 0) == 0,
          "nobody has read the pages; claiming otherwise is the self-agreement "
          "this exercise exists to avoid")

    # PLANTED FAULT: a stale artefact must be detected, not merely regenerated.
    # An artefact that rebuilds but cannot be shown stale is decoration.
    _stale = {"facts": {k: v for i, (k, v) in
                        enumerate(_ver["facts"].items()) if i % 2 == 0}}
    _d = loader.verification_drift(fid_res, verification=_stale)
    check("  FAULT  a stale artefact is detected and names the gap",
          _d["stale"] and _d["n_missing"] > 0
          and _d["missing"][0] not in _stale["facts"],
          f"{_d['n_missing']} missing, {_d['n_ghost']} ghost")
    # ...and the other direction: a fact the artefact claims and the run does
    # not produce. Dropping a corpus would otherwise look clean.
    _ghosted = {"facts": dict(_ver["facts"], **{"GHOST:9:step:0": loader.RESOLVER_VERIFIED})}
    _dg = loader.verification_drift(fid_res, verification=_ghosted)
    check("  FAULT  a ghost fact in the artefact is detected",
          _dg["stale"] and _dg["ghost"] == ["GHOST:9:step:0"], str(_dg["ghost"]))
    # A human_verified stamp must NOT be reported as drift -- the resolver
    # cannot make that claim and a rerun must not silently downgrade it.
    _human = {"facts": dict(_ver["facts"])}
    _first = next(iter(_human["facts"]))
    _human["facts"][_first] = loader.HUMAN_VERIFIED
    _dh = loader.verification_drift(fid_res, verification=_human)
    check("  a human_verified stamp is not treated as drift",
          _first not in _dh["disagree"],
          "a rerun must not downgrade a human reading to resolver_verified")
    check("  control: the real artefact shows no drift",
          not loader.verification_drift(fid_res)["stale"])


    reader = structural.PageReader()
    st_res = structural.run(reader=reader)
    for g in structural.gates(st_res):
        check(f"structural {g['gate']} ({g['value']:.4f})",
              g["status"] == "PASS",
              json.dumps(st_res[g["gate"]])[:200])
    # branch_completeness is REPORTED, not gated: DY20KA step 7 carries a NO
    # with no YES, which is audit finding D4 -- a defect in the manual, not in
    # the parser. Gating it would fail the build over a known source defect.
    print(f"  branch_completeness {st_res['branch_completeness']['rate']:.4f} "
          f"(reported, not gated -- {len(st_res['branch_completeness']['incomplete'])} "
          f"known source defect)")
    reader.close()
timings["pipeline"] = time.perf_counter() - t0


# ========================================================= E2E
print("\nE2E -- verification status must reach the wire")
t0 = time.perf_counter()
from fastapi.testclient import TestClient                # noqa: E402
from api.app import create_app                           # noqa: E402
from api.store import SessionStore                       # noqa: E402

client = TestClient(create_app(store=SessionStore(max_sessions=100,
                                                  max_turns=100,
                                                  rate_limit=10_000)))
b = client.post("/sessions", json={"message": "CA451 aa raha hai"}).json()
sid = b["session_id"]
seen = []
for i, msg in enumerate(["PC200-10M0, serial 700123", "koi aur code nahi",
                         "haan normal hai", "haan normal hai", "haan normal hai",
                         "haan normal hai", "haan normal hai"], start=1):
    r = client.post(f"/sessions/{sid}/messages",
                    json={"message": msg, "turn_index": i}).json()
    seen += r.get("citations") or []
    if r["session_status"] != "running":
        break

check("citations carry a verification field",
      seen and all("verification" in c for c in seen))
check("verification values are from the declared set",
      {c["verification"] for c in seen} <= set(loader.STATUSES),
      str({c["verification"] for c in seen}))
unverified = [c for c in seen if c["verification"] == loader.UNVERIFIED]
check("an unverified fact reaches the response as unverified",
      bool(unverified),
      "CA451 step 6 is machine-repaired and must not be reported as verified")
if unverified:
    print(f"    e.g. {unverified[0]['fact_id']} -> {unverified[0]['verification']}")
timings["e2e"] = time.perf_counter() - t0


# ================================================ OVERFITTING GUARD
print("\nOVERFITTING GUARD")

# a) independent derivation -- structural must share no helper with the parser.
# Checked over the parsed AST, not the raw text: the module's own docstring
# names those helpers in order to say it does not use them, and a substring grep
# cannot tell an import from a sentence about an import.
import ast                                               # noqa: E402

tree = ast.parse(open(os.path.join(REPO_ROOT, "pipeline", "structural.py"),
                      encoding="utf-8").read())
imported, attrs = set(), set()
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        imported |= {a.name.split(".")[0] for a in node.names}
    elif isinstance(node, ast.ImportFrom):
        imported.add((node.module or "").split(".")[0])
        imported |= {a.name for a in node.names}
    elif isinstance(node, ast.Name):
        attrs.add(node.id)
    elif isinstance(node, ast.Attribute):
        attrs.add(node.attr)

banned_modules = {"pdfplumber", "extract_golden"}
banned_names = {"parse_causes", "table_kind", "CRIT_VALUE", "GLUED_STEP",
                "_fuse_split_decimals", "parse_measurement_table", "norm_label"}
shared = sorted((imported & banned_modules) | (attrs & banned_names))
check("structural checks share no helper with the extractor", not shared,
      f"actually imports/calls: {shared} -- shared helpers share their bugs")
print(f"    structural imports: {sorted(imported - {'annotations'})}")

# b) mutation check
print("\n  MUTATION TABLE")


def mutate(kind, recs):
    import copy
    r = copy.deepcopy(recs)
    target = "CA451"
    if kind == "drop_a_step":
        r[target]["steps"] = [s for s in r[target]["steps"] if s["step"] != 6]
    elif kind == "renumber_1_2_4":
        for s in r[target]["steps"]:
            if s["step"] == 3:
                s["step"] = 4
    elif kind == "remove_a_yes_branch":
        fa = next(c for c, x in r.items() if x["format"] == "A"
                  and any("YES" in (s.get("branches") or {}) for s in x["steps"]))
        for s in r[fa]["steps"]:
            if "YES" in (s.get("branches") or {}):
                del s["branches"]["YES"]
                break
    elif kind == "shift_fact_page":
        r[target]["steps"][0]["provenance"]["pdf_page"] += 1
    elif kind == "corrupt_a_digit":
        for s in r[target]["steps"]:
            for m in s.get("measurements", []):
                m["criteria"] = m["criteria"].replace("0.2", "0.9")
    elif kind == "alter_a_title":
        # FAULT 1: the title no longer appears on the page its provenance
        # names. Titles were enumerated precisely so this fails.
        r["CA131"]["title"] = "Throttle Sensor Extremely High Error"
    elif kind == "move_title_off_the_index_table":
        # FAULT 2c: provenance moved outside both legitimate ranges.
        r["CA131"]["title_provenance"] = dict(
            r["CA131"]["title_provenance"], pdf_page=900)
    elif kind == "move_title_to_another_index_page":
        # FAULT 2b: still inside the index table, wrong page. Containment
        # cannot see it; text resolution can, because the title is not printed
        # there.
        tp = r["CA131"]["title_provenance"]
        r["CA131"]["title_provenance"] = dict(tp, pdf_page=tp["pdf_page"] + 3)
    return r


MUT = ["drop_a_step", "renumber_1_2_4", "remove_a_yes_branch",
       "shift_fact_page", "corrupt_a_digit", "false_verified",
       "alter_a_title", "move_title_off_the_index_table",
       "move_title_to_another_index_page"]
mut_rows = []
if fid_res is not None:
    base = fidelity.load_records()
    reader = structural.PageReader()
    for kind in MUT:
        caught = []
        if kind == "false_verified":
            # Marking an unverified fact as resolver_verified must not survive:
            # rebuilding verification from a fidelity run overwrites the claim.
            fake = {"facts": {"CA451:6:step:0": loader.RESOLVER_VERIFIED},
                    "counts": {}, "total": 1}
            rebuilt = fidelity.check_facts(pages=pages)
            truth = {r["fact_id"]: r["resolved"] for r in rebuilt["rows"]}
            if truth.get("CA451:6:step:0") is not True:
                caught.append("verification_rebuild")
        else:
            m = mutate(kind, base)
            f = fidelity.check_facts(recs=m, pages=pages)
            if any(g["status"] == "FAIL" for g in fidelity.gates(f)):
                caught.append("fidelity_gates")
            # Ungated kinds still have to move. Tier 1 is deterministic, so any
            # drop against the baseline is real -- the same reasoning compare.py
            # uses for having no noise floor. A fault that only touches an
            # ungated fact kind must not pass unnoticed just because no
            # threshold covers it.
            for k, v in f["by_kind"].items():
                baseline = fid_res["by_kind"].get(k, {}).get("rate")
                if baseline is not None and v["rate"] < baseline:
                    caught.append(f"resolution_drop[{k}]")
            s = structural.run(recs=m, reader=reader, codes=list(m))
            if any(g["status"] == "FAIL" for g in structural.gates(s)):
                caught += [g["gate"] for g in structural.gates(s)
                           if g["status"] == "FAIL"]
            if s["branch_completeness"]["rate"] < 1.0 and \
                    kind == "remove_a_yes_branch":
                caught.append("branch_completeness")
        mut_rows.append((kind, caught))
        print(f"    {kind:22} {'caught' if caught else '*** NOT CAUGHT ***':18} "
              f"{', '.join(dict.fromkeys(caught)) or '-'}")
    reader.close()
    uncaught = [k for k, c in mut_rows if not c]
    check("every planted fault is caught", not uncaught,
          f"uncaught: {uncaught} -- named, not fixed quietly")

# c) negative coverage is proven by the self-tests above (each gate has a
#    known-bad input it must fail on, run before the gate is trusted).

# d) held-out split
if fid_res is not None:
    hold = set(OV.HOLDOUT)
    h_rows = [r for r in fid_res["rows"] if r["code"] in hold]
    t_rows = [r for r in fid_res["rows"] if r["code"] not in hold]

    def rate(rows, kind=None):
        rs = [r for r in rows if kind is None or r["kind"] == kind]
        return (sum(1 for r in rs if r["resolved"]) / len(rs)) if rs else None
    print(f"\n  TRAIN vs HOLDOUT ({len(t_rows)} / {len(h_rows)} facts)")
    print(f"  {'kind':16}{'train':>10}{'holdout':>10}{'gap':>9}")
    worst = 0.0
    for k in ("measurement", "branch", "cause", None):
        tv, hv = rate(t_rows, k), rate(h_rows, k)
        if tv is None or hv is None:
            continue
        gap = abs(tv - hv)
        worst = max(worst, gap)
        print(f"  {(k or 'ALL'):16}{tv:>10.4f}{hv:>10.4f}{gap:>9.4f}")
    check("fidelity generalises to held-out codes", worst <= 0.05,
          f"largest gap {worst:.4f}")

if pages.available():
    pages.close()

print("\n" + "=" * 62)
print(f"module {timings.get('module', 0):.1f}s  "
      f"pipeline {timings.get('pipeline', 0):.1f}s  "
      f"e2e {timings.get('e2e', 0):.1f}s")
if failures:
    print(f"FAILED: {len(failures)} check(s)")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("OK: extraction fidelity")
sys.exit(0)
