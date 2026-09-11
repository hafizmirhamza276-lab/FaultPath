#!/usr/bin/env python3
"""Module tests and overfitting guard for the symptom map.

    python tests/test_symptom_map.py

E2E IS NOT COVERED HERE and is not silently skipped: a symptom session over
HTTP needs the agent's symptom path, which is still a stub. It is listed as
deferred at the end of this run rather than quietly omitted.
"""
import ast
import json
import os
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from core.symptom_match import (                                   # noqa: E402
    SymptomMatcher, normalise, tokens, score, gates,
    MATCHED, ASK, UNMAPPED, EXACT, NORMALISED, SYNONYM, RETRIEVAL)

FAILED = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"\n          {detail}"
                                                     if not ok else ""))
    if not ok:
        FAILED.append(name)


M = SymptomMatcher()
SMAP = json.load(open(os.path.join(REPO_ROOT, "knowledge",
                                   "symptom_synonyms.json"), encoding="utf-8"))
HELD = json.load(open(os.path.join(REPO_ROOT, "knowledge",
                                   "symptom_heldout.json"), encoding="utf-8"))

print("\nMODULE -- each layer in isolation")

# layer 1 -------------------------------------------------------------------
title = M.trees["HM06"]["symptom"]
r = M.match(title)
check("layer 1 matches the manual's phrasing byte for byte",
      r["layer"] == EXACT and r["outcome"] == MATCHED
      and r["symptom_ids"] == ["HM06"], str(r["symptom_ids"]))

# layer 2 -------------------------------------------------------------------
r = M.match("  BOOM   speed or POWER is low!! ")
check("layer 2 matches through case, spacing and punctuation",
      r["layer"] == NORMALISED and r["symptom_ids"] == ["HM06"], str(r))
r = M.match("Fine Control Performance or Re-\nsponse is Unsatisfactory")
check("layer 2 matches through the documented de-hyphenation",
      r["layer"] == NORMALISED and r["symptom_ids"] == ["HM03"], str(r))

# NORMALISATION SYMMETRY. Applying it on one side only is the quiet defect:
# it produces confident misses, not errors.
for s in (title, "Boom Speed or POWER is Low", "boom  speed or power is low"):
    check(f"normalise is idempotent on {s[:28]!r}",
          normalise(normalise(s)) == normalise(s))
check("normalise maps query and title to the SAME form",
      normalise("  BOOM   speed or POWER is low!! ") == normalise(title),
      f"{normalise('  BOOM   speed or POWER is low!! ')!r} vs {normalise(title)!r}")

# layer 3 -------------------------------------------------------------------
r = M.match("boom power kam hai")
check("layer 3 resolves a Roman Urdu synonym",
      r["layer"] == SYNONYM and r["outcome"] == MATCHED
      and r["symptom_ids"] == ["HM06"], str(r))
check("layer 3 is reached only after 1 and 2 fail",
      M.match(title)["layer"] == EXACT, "a synonym must not shadow the manual")

# layer 4 -------------------------------------------------------------------
r = M.match("bucket ki speed kam lag rahi hai aaj")
check("layer 4 proposes candidates for an unseen phrasing",
      r["layer"] == RETRIEVAL and r["outcome"] == ASK and r["symptom_ids"],
      str(r))
check("layer 4 NEVER returns MATCHED, even with one candidate",
      all(M.match(q)["outcome"] != MATCHED
          for q in ("bucket ki speed kam lag rahi hai aaj",
                    "arm ki taqat kam lag rahi hai",
                    "fan ki awaaz bahut tez aa rahi hai"))
      , "retrieval has a score and an opinion; it does not have knowledge")

# ambiguity -----------------------------------------------------------------
r = M.match("swing slow hai")
check("an ambiguous synonym asks instead of picking",
      r["outcome"] == ASK and set(r["symptom_ids"]) == {"HM28", "HM29"}, str(r))
r = M.match("awaaz aa rahi hai")
check("a four-way ambiguous phrase offers all four",
      r["outcome"] == ASK
      and set(r["symptom_ids"]) == {"HM04", "HM33", "HM37", "SM18"}, str(r))
check("every ambiguous candidate carries its page for the technician to read",
      all(c["manual_page"] for c in r["candidates"]), str(r["candidates"]))

# unmapped ------------------------------------------------------------------
r = M.match("AC kaam nahi kar raha")
check("a known-uncovered input is reported unmapped, not routed",
      r["outcome"] == UNMAPPED and not r["symptom_ids"], str(r))
r = M.match("zzzz qqqq wwww")
check("gibberish is unmapped rather than forced to a tree",
      r["outcome"] == UNMAPPED, str(r))

# the citation requirement --------------------------------------------------
bad = [e["input"] for e in SMAP["entries"]
       if not e["cites"] or any(not c.get("manual_phrasing")
                                or not c.get("manual_page") for c in e["cites"])]
check("every synonym entry cites a manual phrasing and a page",
      not bad, str(bad[:5]))
wrong = [e["input"] for e in SMAP["entries"] for c in e["cites"]
         if M.trees[c["symptom_id"]]["symptom"] != c["manual_phrasing"]]
check("every cited phrasing equals the tree's own title",
      not wrong, str(wrong[:5]))
check("every target id exists in golden/symptoms/",
      all(i in M.trees for e in SMAP["entries"]
          for i in e["target_symptom_ids"]))


print("\nMETRICS -- built-from vs held out")
built = [{"input": e["input"], "expected": e["target_symptom_ids"]}
         for e in SMAP["entries"]]
titles = [{"input": M.trees[s]["symptom"], "expected": [s]} for s in M.trees]
for name, cases in (("manual titles", titles), ("built-from", built),
                    ("held-out", HELD["cases"]),
                    ("sealed", HELD["sealed_cases"])):
    s = score(M, cases)
    g = gates(s)
    print(f"    {name:15} n={s['n']:<4} entry={s['symptom_entry_accuracy']:.4f} "
          f"ask={s['symptom_ask_rate']:.4f} "
          f"wrong_tree={s['symptom_wrong_tree_rate']:.4f} "
          f"unmapped={s['unmapped_rate']:.4f}")
    check(f"{name}: no wrong tree", s["symptom_wrong_tree_rate"] == 0.0)
    check(f"{name}: all gates pass", all(x["status"] == "PASS" for x in g),
          str([(x["gate"], x["value"]) for x in g if x["status"] != "PASS"]))

check("held-out phrasings are disjoint from the map",
      not ({normalise(c["input"]) for c in HELD["cases"]}
           & {normalise(e["input"]) for e in SMAP["entries"]}))
check("the sealed slice is disjoint from both",
      not ({normalise(c["input"]) for c in HELD["sealed_cases"]}
           & ({normalise(e["input"]) for e in SMAP["entries"]}
              | {normalise(c["input"]) for c in HELD["cases"]})))


print("\nMUTATION CHECK")


def build_with(tmp_src, tmp_unmapped=None):
    """Run the real builder against a doctored source and return its output."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    src = os.path.join(REPO_ROOT, "knowledge", "symptom_synonyms.source.tsv")
    un = os.path.join(REPO_ROOT, "knowledge", "symptom_unmapped.tsv")
    bak_src = open(src, encoding="utf-8").read()
    bak_un = open(un, encoding="utf-8").read()
    try:
        open(src, "w", encoding="utf-8").write(tmp_src)
        if tmp_unmapped is not None:
            open(un, "w", encoding="utf-8").write(tmp_unmapped)
        p = subprocess.run([sys.executable,
                            os.path.join(REPO_ROOT, "pipeline",
                                         "build_symptom_map.py")],
                           cwd=REPO_ROOT, capture_output=True, text=True, env=env)
        return p.returncode, p.stdout + p.stderr
    finally:
        open(src, "w", encoding="utf-8").write(bak_src)
        open(un, "w", encoding="utf-8").write(bak_un)


# 1. a synonym pointing at a tree that does not exist -----------------------
rc, out = build_with("boom slow hai\tHM99\troman_urdu\ttester\n")
check("a synonym targeting a nonexistent tree fails the build",
      rc != 0 and "HM99" in out, out[-200:])

# 2. a synonym that would cite nothing --------------------------------------
#    Every citation is rendered from golden/, so "cites nothing" can only
#    arise from an id with no tree -- which is mutation 1. The reachable form
#    of this defect is an entry naming NO target at all.
rc, out = build_with("boom slow hai\t\troman_urdu\ttester\n")
check("a synonym naming no target at all fails the build",
      rc != 0 and "no target" in out, out[-200:])

# 3. ambiguity resolved by picking the first candidate ----------------------
first_only = M.match("swing slow hai")["symptom_ids"][:1]
check("picking the first candidate would differ from what the matcher does",
      first_only == ["HM28"]
      and M.match("swing slow hai")["outcome"] == ASK
      and len(M.match("swing slow hai")["symptom_ids"]) == 2,
      "if these ever agree, the matcher has started guessing")

# 4. normalisation applied on ONE side only ---------------------------------
#    The quiet one: it produces confident misses, never an error.
raw_lookup = {M.trees[s]["symptom"]: s for s in M.trees}        # un-normalised
q = "  BOOM   speed or POWER is low!! "
check("one-sided normalisation misses what two-sided matching finds",
      q not in raw_lookup and normalise(q) in M.by_norm,
      "the defect is silent: a miss, not an exception")
check("the matcher itself normalises both sides",
      M.match(q)["symptom_ids"] == ["HM06"])

# 5. an unmapped input forced to the nearest tree ---------------------------
#    Also quiet: plausible output, no error.
for q in ("AC kaam nahi kar raha", "horn nahi baj raha",
          "wiper kaam nahi karta"):
    r = M.match(q)
    check(f"unmapped input {q[:22]!r} is not routed to a tree",
          r["outcome"] == UNMAPPED and not r["symptom_ids"],
          f"got {r['symptom_ids']} via {r['layer']}")

# 6. an input on both lists at once -----------------------------------------
rc, out = build_with("oil leak ho raha hai\tHM06\troman_urdu\ttester\n")
check("an input on the map and the unmapped list at once fails the build",
      rc != 0 and "cannot be both" in out, out[-200:])

# 7. a synonym shadowing a manual phrasing ----------------------------------
shadow = M.trees["HM06"]["symptom"] + "\tHM07\tenglish\ttester\n"
rc, out = build_with(shadow)
check("a synonym that shadows a manual phrasing fails the build",
      rc != 0 and "layer 2 would answer" in out, out[-200:])


print("\nNEGATIVE COVERAGE -- every gate can fail")
wrong = {"n": 10, "symptom_entry_accuracy": 1.0, "symptom_wrong_tree_rate": 0.1,
         "symptom_ask_without_answer_rate": 0.0}
check("symptom_wrong_tree_rate fails on a single wrong tree",
      [g for g in gates(wrong) if g["gate"] == "symptom_wrong_tree_rate"
       ][0]["status"] == "FAIL")
low = {"n": 10, "symptom_entry_accuracy": 0.5, "symptom_wrong_tree_rate": 0.0,
       "symptom_ask_without_answer_rate": 0.0}
check("symptom_entry_accuracy fails below its threshold",
      [g for g in gates(low) if g["gate"] == "symptom_entry_accuracy"
       ][0]["status"] == "FAIL")
noisy = {"n": 10, "symptom_entry_accuracy": 1.0, "symptom_wrong_tree_rate": 0.0,
         "symptom_ask_without_answer_rate": 0.5}
check("symptom_ask_without_answer_rate fails when asking blindly",
      [g for g in gates(noisy) if g["gate"] == "symptom_ask_without_answer_rate"
       ][0]["status"] == "FAIL")
clean_ = {"n": 10, "symptom_entry_accuracy": 1.0, "symptom_wrong_tree_rate": 0.0,
          "symptom_ask_without_answer_rate": 0.0}
check("all gates pass on a clean result",
      all(g["status"] == "PASS" for g in gates(clean_)))
# ...and asking must NOT be penalised, or the matcher will learn to guess
asky = score(M, [{"input": "swing slow hai", "expected": ["HM28"]}])
check("asking with the right tree among the candidates is not a failure",
      asky["symptom_wrong_tree_rate"] == 0.0
      and asky["symptom_entry_accuracy"] == 1.0
      and asky["symptom_ask_rate"] == 1.0, str(asky["counts"]))


print("\nINDEPENDENT DERIVATION -- over the parsed AST")
MODEL_HINTS = ("openai", "anthropic", "cohere", "transformers", "llama_cpp",
               "langchain", "litellm", "requests", "httpx", "urllib", "socket")
for mod in ("core/symptom_match.py", "pipeline/build_symptom_map.py",
            "pipeline/build_symptom_heldout.py"):
    tree = ast.parse(open(os.path.join(REPO_ROOT, mod), encoding="utf-8").read())
    imported = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            imported |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module:
            imported.add(n.module.split(".")[0])
    check(f"no model or network import in {mod}",
          not (imported & set(MODEL_HINTS)),
          str(sorted(imported & set(MODEL_HINTS))))

# the matcher must use the extractor's clean(), not its own copy
tree = ast.parse(open(os.path.join(REPO_ROOT, "core/symptom_match.py"),
                      encoding="utf-8").read())
froms = {n.module: {a.name for a in n.names}
         for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
check("normalisation imports clean() from the extractor rather than redefining it",
      "clean" in froms.get("extract_golden", set())
      and "clean" not in {f.name for f in ast.walk(tree)
                          if isinstance(f, ast.FunctionDef)},
      "one normalisation pipeline, not two")

check("the four layer constants are distinct",
      len({EXACT, NORMALISED, SYNONYM, RETRIEVAL}) == 4)
check("retrieval is the last layer consulted",
      [n for n in ("exact", "normalised", "synonym", "retrieval")].index(
          "retrieval") == 3)

print("\n" + "=" * 62)
print("E2E: DEFERRED -- a symptom session over HTTP needs the agent's symptom")
print("     path, which is still a stub. Not skipped silently.")
if FAILED:
    print(f"FAILED: {len(FAILED)} check(s)")
    for n in FAILED:
        print("  -", n)
    sys.exit(1)
print("OK: symptom map")
