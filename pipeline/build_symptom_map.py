#!/usr/bin/env python3
"""Build and validate the symptom synonym map.

    python pipeline/build_symptom_map.py     # -> knowledge/symptom_synonyms.json

Reads the hand-authored source tables and renders the citation for every entry
FROM golden/symptoms/. The author supplies input -> target id; the manual
phrasing and page are never typed.

That is the same rule the rest of this project runs on: the model names a
fact_id and the system renders the citation, so what nobody types nobody can
get wrong. Here the author names a symptom_id. A typo produces a build failure
rather than an entry that cites a page convincingly and wrongly.

VALIDATION IS THE POINT, not a formality. The build FAILS on:
  - a target id with no tree in golden/symptoms/
  - an entry whose rendered citation is empty
  - a synonym whose normal form collides with a manual phrasing
  - a duplicate input mapping to different targets
  - an input that is on both the synonym map and the unmapped list

No LLM. The map is authored by a person and validated against ground truth;
nothing here generates text.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)

from core.symptom_match import load_trees, normalise                # noqa: E402

KNOWLEDGE = os.path.join(REPO_ROOT, "knowledge")
SRC = os.path.join(KNOWLEDGE, "symptom_synonyms.source.tsv")
UNMAPPED_SRC = os.path.join(KNOWLEDGE, "symptom_unmapped.tsv")
OUT = os.path.join(KNOWLEDGE, "symptom_synonyms.json")


def read_tsv(path, arity):
    rows = []
    with open(path, encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) != arity:
                sys.exit(f"{os.path.basename(path)}:{ln}: expected {arity} "
                         f"tab-separated fields, got {len(parts)}: {line!r}")
            rows.append((ln, [p.strip() for p in parts]))
    return rows


def build():
    trees = load_trees()
    if not trees:
        sys.exit("no symptom trees in golden/symptoms/ -- run "
                 "pipeline/extract_symptoms.py first")

    errors = []
    entries, seen = [], {}
    title_norms = {normalise(r["symptom"]): sid for sid, r in trees.items()}

    for ln, (inp, ids_raw, lang, who) in read_tsv(SRC, 4):
        ids = [i for i in ids_raw.split("|") if i]
        n = normalise(inp)

        missing = [i for i in ids if i not in trees]
        if missing:
            errors.append(f"line {ln}: {inp!r} targets {missing}, which have "
                          f"no tree in golden/symptoms/")
            continue
        if not ids:
            errors.append(f"line {ln}: {inp!r} names no target at all")
            continue
        if not n:
            errors.append(f"line {ln}: input normalises to nothing")
            continue
        if n in seen and seen[n] != ids:
            errors.append(f"line {ln}: {inp!r} duplicates an earlier input "
                          f"with different targets {seen[n]} vs {ids}")
            continue
        if n in title_norms and ids != [title_norms[n]]:
            errors.append(f"line {ln}: {inp!r} is a manual phrasing "
                          f"({title_norms[n]}) but is mapped to {ids}; layer 2 "
                          f"would answer before the map is ever consulted")
            continue
        seen[n] = ids

        # THE CITATION, RENDERED -- never typed by the author.
        cites = []
        for sid in ids:
            r = trees[sid]
            tp = r.get("title_provenance") or {}
            cites.append({"symptom_id": sid,
                          "manual_phrasing": r["symptom"],
                          "manual_page": tp.get("manual_page"),
                          "pdf_page": tp.get("pdf_page"),
                          "tree_kind": r["tree_kind"]})
        if any(not c["manual_phrasing"] or not c["manual_page"] for c in cites):
            errors.append(f"line {ln}: {inp!r} rendered an empty citation; an "
                          f"entry citing nothing is a guess in a mapping's "
                          f"clothes")
            continue

        entries.append({"input": inp, "normalised": n, "lang": lang,
                        "target_symptom_ids": ids, "cites": cites,
                        "ambiguous": len(ids) > 1, "added_by": who,
                        "source_line": ln})

    unmapped = []
    for ln, (inp, why, who) in read_tsv(UNMAPPED_SRC, 3):
        n = normalise(inp)
        if n in seen:
            errors.append(f"unmapped line {ln}: {inp!r} is also on the synonym "
                          f"map; it cannot be both covered and uncovered")
            continue
        unmapped.append({"input": inp, "normalised": n, "reason": why,
                         "added_by": who, "source_line": ln})

    if errors:
        print("SYMPTOM MAP VALIDATION FAILED")
        for e in errors:
            print("  -", e)
        return None, errors

    payload = {
        "schema_version": 1,
        "manual_id": "SEN06867-13",
        "built_from": ["knowledge/symptom_synonyms.source.tsv",
                       "knowledge/symptom_unmapped.tsv"],
        "citation_policy": (
            "The author supplies input -> symptom_id. Every manual phrasing and "
            "page in this file is RENDERED from golden/symptoms/ by "
            "pipeline/build_symptom_map.py and is never typed by hand, so a "
            "wrong id fails the build instead of shipping a convincing wrong "
            "citation."),
        "trees_available": len(trees),
        "entries": entries,
        "unmapped": unmapped,
    }
    return payload, []


def main() -> int:
    payload, errors = build()
    if payload is None:
        return 1
    os.makedirs(KNOWLEDGE, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    amb = [e for e in payload["entries"] if e["ambiguous"]]
    covered = {i for e in payload["entries"] for i in e["target_symptom_ids"]}
    print(f"synonym entries : {len(payload['entries'])} "
          f"({len(amb)} ambiguous)")
    print(f"unmapped inputs : {len(payload['unmapped'])}")
    print(f"trees covered   : {len(covered)}/{payload['trees_available']}")
    uncovered = sorted(set(load_trees()) - covered)
    if uncovered:
        # Stated, not hidden. A tree with no synonym is only reachable by its
        # own phrasing or by retrieval, and that is worth knowing per tree.
        print(f"trees with no synonym ({len(uncovered)}): {', '.join(uncovered)}")
    print("->", os.path.relpath(OUT, REPO_ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
