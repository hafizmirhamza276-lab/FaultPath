#!/usr/bin/env python3
"""Deterministic symptom matching. The second entry point.

A technician with a failure code gets a dict lookup. A technician with a
symptom gets this, and it has to be just as unwilling to guess.

FOUR LAYERS, IN ORDER. Retrieval is the last resort, not the first move:

  1  EXACT        the manual's own phrasing, byte for byte
  2  NORMALISED   case, whitespace, punctuation, documented de-hyphenation
  3  SYNONYM      the curated map, every entry citing a manual phrasing
  4  RETRIEVAL    token overlap, and ONLY when 1-3 find nothing

The first three are lookups and cannot be wrong about what they matched. The
fourth is a guess with a score attached, so it never returns a decision -- it
returns candidates to be confirmed.

NO LLM. Not in normalisation, not in synonym lookup, not in retrieval. This
module imports no model client and no network client, asserted over its parsed
AST by tests/test_symptom_map.py.

THREE OUTCOMES, and the middle one is not a failure:

  MATCHED    exactly one tree, confidently
  ASK        more than one candidate, or one the matcher is not sure of
  UNMAPPED   nothing at any layer

ASK IS A FIRST-CLASS RESULT. A technician who is asked loses ten seconds; one
sent silently into the wrong tree loses an hour and stops trusting the tool.
Any scoring that treats a question as a miss will tune itself into guessing,
which is why symptom_wrong_tree_rate is reported apart from symptom_ask_rate
and the two are never averaged.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))

# clean() comes from the extractor itself, and this import is now a plain one.
#
# It used to be wrapped in an argv-blanking workaround, because extract_golden
# resolved the PDF path from sys.argv at MODULE scope and exited if it was not
# a file -- so importing a pure string function needed a manual, and any
# process with arguments of its own resolved the manual to its own first flag.
# That workaround treated the symptom. The guard has since moved into
# require_pdf(), called from main(), so there is nothing left to work around.
#
# Re-implementing clean() here was never the alternative: two normalisation
# pipelines is precisely the defect that sent three symptom titles to the wrong
# page one module ago. Both sides of every comparison go through the same
# function, and that function has exactly one definition.
from extract_golden import clean                                  # noqa: E402

GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
KNOWLEDGE = os.path.join(REPO_ROOT, "knowledge")
MAP_PATH = os.path.join(KNOWLEDGE, "symptom_synonyms.json")
UNMAPPED_PATH = os.path.join(KNOWLEDGE, "symptom_unmapped.tsv")

EXACT, NORMALISED, SYNONYM, RETRIEVAL = "exact", "normalised", "synonym", "retrieval"
MATCHED, ASK, UNMAPPED = "MATCHED", "ASK", "UNMAPPED"

# Retrieval only proposes. A candidate below this shares too little with the
# title to be worth a technician's attention; at or above it, the candidate is
# still only offered for confirmation, never entered.
RETRIEVAL_FLOOR = 0.34
MAX_CANDIDATES = 4

# Words that carry no diagnostic signal. Deliberately SHORT: an aggressive stop
# list is how "does not" and "is low" stop being distinguishable, and those are
# exactly the words that separate one tree from another.
STOP = {"hai", "ho", "raha", "rahi", "rha", "ka", "ki", "ke", "mein", "me",
        "se", "par", "is", "are", "the", "a", "of", "in", "to", "and", "or",
        "from", "it", "its", "when"}


def normalise(text: str) -> str:
    """Layer-2 normal form, used on BOTH sides of every comparison.

    clean() first, because that is what the extractor and the citation
    resolver both use -- the title-scan defect came from normalising one side
    with a different pipeline from the other, and this module does not get to
    repeat it one file over.
    """
    t = clean(text or "").lower()
    t = t.replace("“", '"').replace("”", '"')
    t = t.replace("‘", "'").replace("’", "'")
    t = re.sub(r"[^\w\s]+", " ", t, flags=re.UNICODE)
    return " ".join(t.split())


def tokens(text: str) -> List[str]:
    return [w for w in normalise(text).split() if w not in STOP]


# ------------------------------------------------------------- the corpus

def load_trees(gold: Optional[str] = None) -> Dict[str, dict]:
    import glob
    out = {}
    for p in sorted(glob.glob(os.path.join(gold or GOLD, "symptoms", "*.json"))):
        if os.path.basename(p) == "index.json":
            continue
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        out[r["symptom_id"]] = r
    return out


def load_map(path: Optional[str] = None) -> dict:
    p = path or MAP_PATH
    if not os.path.isfile(p):
        return {"entries": [], "unmapped": []}
    with open(p, encoding="utf-8") as f:
        return json.load(f)


class SymptomMatcher:
    def __init__(self, trees: Optional[dict] = None, synonyms: Optional[dict] = None):
        self.trees = trees if trees is not None else load_trees()
        data = synonyms if synonyms is not None else load_map()
        self.entries = data.get("entries", [])
        self.unmapped_inputs = {normalise(u["input"])
                                for u in data.get("unmapped", [])}

        self.by_exact = {r["symptom"]: sid for sid, r in self.trees.items()}
        self.by_norm: Dict[str, List[str]] = {}
        for sid, r in self.trees.items():
            self.by_norm.setdefault(normalise(r["symptom"]), []).append(sid)

        # Every word the manual uses in a symptom title. A query word outside
        # this set cannot match anything, so retrieval ignores it rather than
        # scoring a candidate down for it.
        self.vocabulary = {w for r in self.trees.values()
                           for w in tokens(r["symptom"])}

        self.by_synonym: Dict[str, List[str]] = {}
        for e in self.entries:
            self.by_synonym.setdefault(normalise(e["input"]), []).extend(
                e["target_symptom_ids"])

    # ------------------------------------------------------------ layers

    def _result(self, layer, ids, query, note=""):
        ids = [i for i in ids if i in self.trees]
        if not ids:
            return None
        outcome = MATCHED if len(ids) == 1 else ASK
        return {"query": query, "layer": layer, "outcome": outcome,
                "symptom_ids": sorted(ids),
                "candidates": [self.describe(i) for i in sorted(ids)],
                "note": note}

    def describe(self, sid: str) -> dict:
        r = self.trees[sid]
        tp = r.get("title_provenance") or {}
        return {"symptom_id": sid, "symptom": r["symptom"],
                "tree_kind": r["tree_kind"],
                "manual_page": tp.get("manual_page"),
                "pdf_page": tp.get("pdf_page")}

    def match(self, query: str) -> dict:
        q = query or ""

        # 1 -- exact, the manual's own words
        if q in self.by_exact:
            return self._result(EXACT, [self.by_exact[q]], q,
                                "the manual's phrasing, byte for byte")

        n = normalise(q)
        if not n:
            return {"query": q, "layer": None, "outcome": UNMAPPED,
                    "symptom_ids": [], "candidates": [],
                    "note": "empty query"}

        # 2 -- normalised, both sides through the same pipeline
        if n in self.by_norm:
            return self._result(NORMALISED, self.by_norm[n], q,
                                "same phrasing after case, punctuation and "
                                "de-hyphenation")

        # 3 -- the curated map
        if n in self.by_synonym:
            ids = self.by_synonym[n]
            return self._result(
                SYNONYM, ids, q,
                "curated synonym" if len(set(ids)) == 1 else
                "curated synonym covering more than one tree; the phrase does "
                "not pick one")

        # An input on the unmapped list is a KNOWN gap, not a retrieval
        # problem. Saying so beats offering the nearest tree, which is how a
        # question about the air conditioner becomes an engine diagnosis.
        if n in self.unmapped_inputs:
            return {"query": q, "layer": None, "outcome": UNMAPPED,
                    "symptom_ids": [], "candidates": [],
                    "note": "recorded as having no symptom tree in this manual"}

        # 4 -- retrieval, last resort, proposes only
        # Scored over the MANUAL'S OWN VOCABULARY only.
        #
        # Titles are English; queries are Roman Urdu or a mix. "nahi", "chalte"
        # and "kam" cannot match an English title no matter which tree is
        # right, so counting them inflates the denominator by a language
        # barrier and depresses every score equally. Jaccard scored
        # "all work nahi chalte" against "All Work Equipment, Swing and Travel
        # Do Not Work" at 0.20 -- below any usable floor -- for a query that
        # names its tree in its first two words.
        #
        # A query token that appears in NO title is unmatchable by
        # construction, so it is excluded from the denominator rather than
        # counted against the candidate. The exclusion set is derived from the
        # corpus, not hand-listed, so it cannot be tuned toward a test set.
        scored = []
        qt = set(tokens(q))
        matchable = qt & self.vocabulary
        if matchable:
            for sid, r in self.trees.items():
                tt = set(tokens(r["symptom"]))
                if not tt:
                    continue
                # what fraction of the query's matchable words this title has
                score = len(matchable & tt) / len(matchable)
                if score >= RETRIEVAL_FLOOR:
                    # tie-break toward the title that spends less of itself on
                    # words the query never mentioned
                    scored.append((score, -len(tt - matchable), sid))
        if not scored:
            return {"query": q, "layer": RETRIEVAL, "outcome": UNMAPPED,
                    "symptom_ids": [], "candidates": [],
                    "note": f"no title shares enough with the query "
                            f"(floor {RETRIEVAL_FLOOR})"}

        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        top = [sid for _, _, sid in scored[:MAX_CANDIDATES]]
        # RETRIEVAL NEVER RETURNS "MATCHED", even on a single candidate. Layers
        # 1-3 know what they matched; this one has a number and an opinion.
        return {"query": q, "layer": RETRIEVAL, "outcome": ASK,
                "symptom_ids": sorted(top),
                "candidates": [dict(self.describe(s), score=round(sc, 4))
                               for sc, _, s in scored[:MAX_CANDIDATES]],
                "note": "retrieval proposes; it does not decide"}


# ------------------------------------------------------------- metrics

def score(matcher: "SymptomMatcher", cases: List[dict]) -> dict:
    """Each case is {"input": str, "expected": [symptom_id, ...]}.

    `expected` is a LIST because some phrases genuinely do not pick one tree.
    For those the right behaviour is ASK with the right candidates, and
    scoring it as a miss would train the matcher to guess.

    wrong_tree and ask are counted separately and never combined. A run that
    asks too often is annoying; a run that enters the wrong tree is dangerous,
    and one number cannot say both.
    """
    n = len(cases)
    entered_right = ask_right = ask_wrong = wrong_tree = unmapped = 0
    rows = []
    for c in cases:
        exp = set(c["expected"])
        res = matcher.match(c["input"])
        got = set(res["symptom_ids"])
        if res["outcome"] == MATCHED:
            if got == exp:
                entered_right += 1
                verdict = "entered_right"
            else:
                wrong_tree += 1
                verdict = "WRONG_TREE"
        elif res["outcome"] == ASK:
            # Asking is right when the true tree is among the candidates.
            # Asking with the answer absent is a miss, not a courtesy.
            if exp & got:
                ask_right += 1
                verdict = "asked_right"
            else:
                ask_wrong += 1
                verdict = "asked_without_the_answer"
        else:
            unmapped += 1
            verdict = "unmapped"
        rows.append({"input": c["input"], "expected": sorted(exp),
                     "got": sorted(got), "layer": res["layer"],
                     "outcome": res["outcome"], "verdict": verdict})

    d = n or 1
    return {
        "n": n,
        # the right tree was reached, whether directly or via a question the
        # technician can answer
        "symptom_entry_accuracy": (entered_right + ask_right) / d,
        "symptom_direct_entry_rate": entered_right / d,
        "symptom_ask_rate": (ask_right + ask_wrong) / d,
        "symptom_wrong_tree_rate": wrong_tree / d,
        "symptom_ask_without_answer_rate": ask_wrong / d,
        "unmapped_rate": unmapped / d,
        "counts": {"entered_right": entered_right, "asked_right": ask_right,
                   "asked_without_answer": ask_wrong,
                   "wrong_tree": wrong_tree, "unmapped": unmapped},
        "rows": rows,
    }


# Direction, for run-to-run diffing. THREE VALUES, NOT TWO.
#
#   True   higher is better -- a fall is a regression
#   False  lower is better  -- a rise is a regression
#   None   NO BETTER DIRECTION. Movement is reported and never judged.
#
# The third value exists for ASK. symptom_ask_rate has no good direction: a
# matcher that asks less is not better, it is more willing to guess, and a
# matcher that asks more is not better either. Marking it higher-is-better
# would report every extra question as an improvement; marking it
# lower-is-better would report asking as a REGRESSION, which is precisely the
# pressure that turns a matcher into a guesser. So it is diffed and never
# classified, and symptom_direct_entry_rate is the same metric from the other
# side.
#
# unmapped_rate IS lower-is-better here, and the reason is set-specific: every
# case in the four evaluation sets has a correct tree, so an unmapped result on
# them is a miss. That would not hold for a set built from the unmapped list,
# and this direction should be revisited if such a set is ever scored.
METRIC_DIRECTION = {
    "symptom_entry_accuracy": True,
    "symptom_wrong_tree_rate": False,
    "symptom_ask_without_answer_rate": False,
    "unmapped_rate": False,
    "symptom_ask_rate": None,
    "symptom_direct_entry_rate": None,
}


def run_record_metrics(scored: dict) -> dict:
    """{set_name: score(...)} -> the {name: {value, n, ...}} shape compare.py
    already consumes, with the set in the metric NAME.

    THE SET IS IN THE NAME, NEVER BLENDED AWAY. symptom_entry_accuracy[sealed]
    and symptom_entry_accuracy[held_out] are different measurements over
    different populations, and sealed is the one that counts -- held_out is
    labelled contaminated because retrieval scoring was revised after seeing
    its failures. One averaged number over both would be an improvement in
    coverage and a regression in honesty, so no aggregate is emitted here.
    """
    out = {}
    for set_name, s in scored.items():
        for metric, direction in METRIC_DIRECTION.items():
            if metric not in s:
                continue
            out[f"{metric}[{set_name}]"] = {
                "value": s[metric],
                "n": s["n"],
                "higher_is_better": direction,
                "set": set_name,
                # Travels WITH the number. A contaminated figure quoted without
                # its label is the failure this flag exists to prevent.
                "contaminated": set_name == "held_out",
            }
    return out


def gates(s: dict) -> List[dict]:
    """Wrong tree is the serious one and is gated at zero.

    Entry accuracy is gated below 1.0 on purpose: some inputs SHOULD be
    unmapped, and a gate that demands every input reach a tree would be asking
    the matcher to force them.
    """
    return [
        {"gate": "symptom_wrong_tree_rate", "value": s["symptom_wrong_tree_rate"],
         "op": "<=", "threshold": 0.0,
         "status": "PASS" if s["symptom_wrong_tree_rate"] <= 0.0 else "FAIL",
         "n": s["n"]},
        {"gate": "symptom_entry_accuracy", "value": s["symptom_entry_accuracy"],
         "op": ">=", "threshold": 0.95,
         "status": "PASS" if s["symptom_entry_accuracy"] >= 0.95 else "FAIL",
         "n": s["n"]},
        {"gate": "symptom_ask_without_answer_rate",
         "value": s["symptom_ask_without_answer_rate"], "op": "<=",
         "threshold": 0.05,
         "status": "PASS" if s["symptom_ask_without_answer_rate"] <= 0.05 else "FAIL",
         "n": s["n"]},
    ]
