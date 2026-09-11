#!/usr/bin/env python3
"""Held-out symptom phrasings, generated independently of the synonym map.

    python pipeline/build_symptom_heldout.py  # -> knowledge/symptom_heldout.json

WHY THIS EXISTS. A synonym map tested on the phrasings it was built from proves
only that a dict can look up its own keys. The held-out set has to come from
somewhere else, and "somewhere else" has to mean something checkable.

HOW INDEPENDENCE IS OBTAINED, and exactly how far it goes:

  1. CONSTRUCTION.  Each phrasing is derived MECHANICALLY from a manual title:
     take its content words, drop the stop words, keep the first few, and
     append a fixed Roman Urdu predicate chosen by the fault word already
     present in the title. The synonym map is never read while generating.

  2. LABELS.  The expected tree is the tree whose title the phrasing was
     derived FROM. The label comes from the manual, not from anybody's
     judgement about which tree a phrase ought to reach. This is the part that
     matters most: a held-out set whose answers I chose would be testing my
     opinion twice.

  3. DISJOINTNESS.  Every generated phrasing is checked against the map, in
     both raw and normalised form, and any collision is DROPPED and reported.
     So no held-out item can be a map key by accident.

WHAT THIS IS NOT. The Roman Urdu predicate lexicon below is mine, and it shares
vocabulary with the map -- both use "kam", "nahi", "zyada", because those are
the words. Independence here is structural (different process, derived labels,
disjoint surface forms), not adversarial. A genuinely independent set would
come from real technician logs, and this repo has none. That limitation is
stated rather than papered over, because a held-out number carries exactly as
much weight as the process that produced it.

No LLM. Template expansion over a fixed lexicon; nothing generates text.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from core.symptom_match import load_trees, load_map, normalise, STOP  # noqa: E402

OUT = os.path.join(REPO_ROOT, "knowledge", "symptom_heldout.json")

# Fault word in the title -> Roman Urdu predicate a technician would type.
# Ordered: the first key found in the title wins, so "does not" beats "low"
# for "Does Not Change, or ... Too Slow or Fast".
PREDICATES = [
    ("does not work", "kaam nahi karta"),
    ("cannot be", "nahi hota"),
    ("does not", "nahi chalta"),
    ("do not", "nahi chalte"),
    ("overruns", "zyada aage nikal jata hai"),
    ("unsatisfactory", "theek nahi hai"),
    ("abnormal", "theek nahi hai"),
    ("excessive", "zyada hai"),
    ("too high", "zyada hai"),
    ("rises too high", "zyada garam hai"),
    ("is large", "zyada hai"),
    ("is low", "kam hai"),
    ("lack", "kam hai"),
    ("lacks", "kam hai"),
    ("drops", "gir raha hai"),
    ("falls", "gir jati hai"),
    ("is poor", "kharab hai"),
    ("is black", "kala hai"),
    ("stops", "band ho jata hai"),
    ("slower", "slow chalta hai"),
    ("is heard", "aa rahi hai"),
    ("is unstable", "unstable hai"),
    ("contaminated", "ganda ho jata hai"),
    ("is in", "mein aa raha hai"),
    ("mixes into", "mein mil raha hai"),
]
DEFAULT_PREDICATE = "ka masla hai"

# Words that carry no diagnostic signal in a TITLE specifically.
TITLE_NOISE = {"is", "are", "does", "do", "not", "too", "only", "when", "it",
               "its", "the", "a", "an", "of", "in", "on", "to", "and", "or",
               "from", "be", "been", "with", "while", "has", "have", "there",
               "large", "low", "high", "slow", "fast", "poor", "black",
               "excessive", "abnormal", "unsatisfactory", "cannot", "stops",
               "drops", "falls", "heard", "lacks", "lack", "slower", "rises",
               "overruns", "mixes", "into", "becomes", "quickly", "goes",
               "down", "back", "out", "comes", "but", "no", "does"}

KEEP_WORDS = 3


def predicate_for(title: str) -> str:
    low = " ".join(title.lower().split())
    for key, pred in PREDICATES:
        if key in low:
            return pred
    return DEFAULT_PREDICATE


def subject_of(title: str, n: int = KEEP_WORDS) -> str:
    words = [w for w in normalise(title).split()
             if w not in TITLE_NOISE and w not in STOP]
    return " ".join(words[:n])


def phrasings_for(title: str):
    """Two per title: a short subject and a slightly longer one.

    Two rather than one because a single phrasing per tree makes the whole
    held-out score hostage to one lucky or unlucky word choice.
    """
    pred = predicate_for(title)
    out = []
    for n in (2, KEEP_WORDS):
        subj = subject_of(title, n)
        if subj:
            out.append(f"{subj} {pred}")
    return out


def main() -> int:
    trees = load_trees()
    if not trees:
        sys.exit("no symptom trees -- run pipeline/extract_symptoms.py first")
    smap = load_map()
    map_keys = {normalise(e["input"]) for e in smap.get("entries", [])}
    map_raw = {e["input"] for e in smap.get("entries", [])}
    map_keys |= {normalise(u["input"]) for u in smap.get("unmapped", [])}

    cases, dropped, seen = [], [], set()
    for sid in sorted(trees):
        title = trees[sid]["symptom"]
        for ph in phrasings_for(title):
            n = normalise(ph)
            if n in map_keys or ph in map_raw:
                dropped.append({"phrasing": ph, "symptom_id": sid,
                                "reason": "collides with a synonym-map input"})
                continue
            if n in seen:
                dropped.append({"phrasing": ph, "symptom_id": sid,
                                "reason": "duplicate of an earlier phrasing"})
                continue
            seen.add(n)
            cases.append({"input": ph, "expected": [sid],
                          "derived_from": title,
                          "derivation": "content words of the manual title + "
                                        "a predicate keyed on its fault word"})

    # ---------------------------------------------------------- sealed slice
    #
    # The set above is CONTAMINATED and says so. Retrieval scoring was revised
    # after looking at its failures -- for a principled reason, but the number
    # it now produces is partly in-sample and cannot be quoted as held out.
    #
    # This slice is built by a DIFFERENT rule: the LAST content words of the
    # title rather than the first. It is generated here, run once, and reported
    # whatever it says. Nothing is tuned against it afterwards; if it ever is,
    # it stops being sealed and this comment is the record of that.
    sealed, sealed_dropped = [], []
    for sid in sorted(trees):
        title = trees[sid]["symptom"]
        words = [w for w in normalise(title).split()
                 if w not in TITLE_NOISE and w not in STOP]
        if len(words) < 2:
            continue
        ph = " ".join(words[-2:]) + " " + predicate_for(title)
        n = normalise(ph)
        if n in map_keys or n in seen:
            sealed_dropped.append({"phrasing": ph, "symptom_id": sid,
                                   "reason": "collides with the map or the "
                                             "contaminated slice"})
            continue
        sealed.append({"input": ph, "expected": [sid], "derived_from": title,
                       "derivation": "LAST content words of the title + the "
                                     "same fault-keyed predicate"})

    payload = {
        "schema_version": 1,
        "manual_id": "SEN06867-13",
        "sealed_n": len(sealed),
        "sealed_dropped": sealed_dropped,
        "sealed_cases": sealed,
        "independence": {
            "construction": "mechanical derivation from manual titles; the "
                            "synonym map is not read while generating",
            "labels": "the tree whose title the phrasing was derived from -- "
                      "taken from the manual, not chosen by a person",
            "disjointness": "every phrasing checked against the map raw and "
                            "normalised; collisions dropped and listed",
            "limitation": "the Roman Urdu predicate lexicon is hand-written "
                          "and shares vocabulary with the map. Independence "
                          "is structural, not adversarial. A truly "
                          "independent set would come from technician logs, "
                          "which this repo does not have.",
        },
        "n": len(cases), "dropped": dropped, "cases": cases,
    }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"held-out phrasings: {len(cases)} over {len(trees)} trees "
          f"({len(dropped)} dropped as colliding or duplicate)")
    for d in dropped:
        print(f"  dropped: {d['phrasing']!r} ({d['reason']})")
    print("->", os.path.relpath(OUT, REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
