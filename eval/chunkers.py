#!/usr/bin/env python3
"""
chunkers.py
Builds the retrieval corpus from golden/, three ways, over BOTH sections.

Same content, same retriever, three chunkings. Everything else is held still, so
the spread between the three runs is the cost of the chunking decision and
nothing else. That is a number worth having before anyone argues about
embeddings.

  structural   one chunk per record. Respects the document's own boundaries --
               a code's procedure is a unit and the manual already says so.
  fixed_2000   2000 characters, 200 overlap. The common default.
  fixed_512    512 characters, 50 overlap. Small chunks, precise but fragmented.

Every chunk carries `model` and `manual_id` so a filtered search can be
evaluated, `code` so per-record diagnostics are possible, and `section` so a
mixed corpus can be sliced after the fact. Metrics never key on `id`.

TWO SECTIONS, ONE CORPUS
------------------------
The corpus is 174 failure codes plus the 57 H-Mode and S-Mode symptom trees.
Section 40 alone is still reachable with --corpus section40, which is how the
older baseline regenerates.

WHAT THE 57 TREES COST, MEASURED BEFORE THEY WERE ADDED. Six runs, bm25,
--system good, over the 1,318 Section 40 retrieval cases, with and without the
symptom chunks:

  metric                structural        fixed_2000        fixed_512
  ceiling_recall        1.0000  +0.0000   0.9997  +0.0000   0.9914  +0.0000
  recall@5              0.9979  +0.0000   0.9665  +0.0018   0.7896  -0.0025
  answer_coverage@5     0.9970  +0.0000   0.9492  +0.0023   0.6844  -0.0068
  fragmentation_gap@10  0.0004  +0.0000   0.0138  -0.0052   0.1069  -0.0081
  mrr                   0.9917  +0.0000   0.9556  +0.0012   0.8104  +0.0028
  context_precision     0.8497  -0.0655   0.7020  +0.0004   0.6452  -0.0224
  gates                 7/7 -> 7/7        7/7 -> 7/7        6/7 -> 6/7

RANKING IS UNMOVED. On structural -- the chunking the system actually uses --
every ranking metric moves by exactly 0.0000. The correct chunk keeps its exact
rank; the extra records do not displace it.

THE ONE MATERIAL MOVEMENT IS context_precision ON structural, -0.0655, and it
is dilution rather than degradation. context_precision is TOKEN-weighted, and
symptom trees are bigger documents than failure codes:

  section40 chunks 174  avg 4,338 chars  max 17,455
  symptom   chunks  57  avg 6,855 chars  max 26,800
  symptom share of corpus tokens: 0.3469 -- from 24.7% of the chunks

25.2% of Section 40 cases now retrieve at least one symptom chunk inside the
top 10, contributing a mean 14.6% of the top-10 tokens. Those chunks land in
the TAIL and displace other irrelevant chunks, not the answer -- which is why
recall and mrr do not move while a token-weighted precision does. The cost is
real and it is a prompt-length cost, not an accuracy one.

PER-MEASUREMENT PROVENANCE, and what carrying it cost
-----------------------------------------------------
Each measurement line now ends with its OWN page -- see _meas_page. Measured,
both scopes, three chunkings, before and after:

  --corpus all          structural         fixed_2000         fixed_512
  ceiling_recall     1.0000  +0.0000    0.9987  +0.0003    0.9809  +0.0004
  recall@5           0.9974  +0.0000    0.9633  +0.0032    0.8080  +0.0245
  answer_coverage@5  0.9968  +0.0000    0.9493  -0.0032    0.7102  +0.0448
  fragmentation@10   0.0003  +0.0000    0.0075  +0.0075    0.0857  -0.0280
  context_precision  0.7146  +0.0019    0.6491  -0.0088    0.5920  +0.0124
  mrr                0.9872  -0.0003    0.9468  -0.0071    0.8239  +0.0269
  chunks              231 -> 231         729 -> 740       2573 -> 2607

CEILING_RECALL DID NOT FALL ANYWHERE, in either scope. Text was appended, not
displaced, and the check that would have caught displacement is the one that
mattered: a fall would have meant a measurement string was pushed out of every
chunk by the page suffix.

structural is flat to four decimals, because one chunk per record means a
longer line cannot cross a boundary. fixed_512 moves most and moves BETTER --
recall@5 +0.0245, answer_coverage@5 +0.0448, mrr +0.0269, fragmentation -0.0280
-- because a page number makes each measurement line more distinctive, and
small chunks live or die on distinctiveness. The extra 34 chunks are the cost.

context_precision is not gated. The seven gates are numeric_exactness,
fabricated_values, citation_accuracy, refusal_correctness, over_refusal,
hit_rate@5 and filter_correctness, and they are unchanged in both directions,
including which one fixed_512 fails.
"""
import json
import glob
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))

SCOPES = ("all", "section40", "symptoms")


def load_code_records():
    paths = sorted(glob.glob(os.path.join(GOLD, "failure_codes", "*.json")))
    if not paths:
        raise SystemExit(
            f"ERROR: no failure-code JSON under {GOLD}/failure_codes/\n"
            "  Run pipeline/extract_golden.py first."
        )
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def load_symptom_records():
    paths = [p for p in sorted(glob.glob(os.path.join(GOLD, "symptoms", "*.json")))
             if os.path.basename(p) != "index.json"]
    if not paths:
        raise SystemExit(
            f"ERROR: no symptom JSON under {GOLD}/symptoms/\n"
            "  Run pipeline/extract_symptoms.py first."
        )
    out = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            out.append(json.load(f))
    return out


def load_records(scope="all"):
    """Records for a scope. A MISSING DIRECTORY IS FATAL, never an empty list.

    Returning [] for an absent half would let a run report every retrieval
    metric over a corpus missing 57 records and call the result a baseline.
    The numbers would look fine -- they would simply be about a different
    corpus than the one named. Both loaders exit rather than shrink.
    """
    if scope not in SCOPES:
        raise SystemExit(f"unknown corpus scope {scope!r}; "
                         f"choose from {sorted(SCOPES)}")
    if scope == "section40":
        return load_code_records()
    if scope == "symptoms":
        return load_symptom_records()
    return load_code_records() + load_symptom_records()


def is_symptom(rec):
    """Dispatch on the RECORD, never on a flag passed alongside it.

    A flag has to survive every hop between the loader and the renderer, and
    the one place it gets dropped is the place that fails silently: a symptom
    tree rendered as a failure code produces text, not an error. The record
    already knows what it is.
    """
    return "symptom_id" in rec


def key(rec):
    """The record's identity, whichever corpus it came from."""
    return rec["symptom_id"] if is_symptom(rec) else rec["code"]


def section_of(rec):
    return rec.get("section") if is_symptom(rec) else "40"


def render(rec):
    """Flatten one record to text, dispatching on the record itself."""
    return _render_symptom(rec) if is_symptom(rec) else _render_code(rec)


def _meas_page(m):
    """The measurement's OWN page, as a suffix for its rendered line.

    THE THIRD INSTANCE of one pattern: provenance captured carefully and then
    dropped at a layer boundary.

      1. branch_provenance was captured by the S4 fix and consumed by
         fidelity, but eval/citations.py kept reading the STEP's page, so a
         branch straddling a page break was cited to the wrong one.
      2. title_provenance did not exist at all -- header_provenance located the
         detail page, and for 132 of 174 codes the title is not printed there.
      3. THIS. Every measurement carries {manual_page, pdf_page, table_index,
         row_index}, and the renderer emitted none of it, so the only page in a
         chunk was the record's first. 75.0% of Section 40 measurements and
         82.8% of symptom measurements are not on that page.

    Three occurrences is a property of this codebase, not a coincidence: the
    capture side is treated as the hard part and the carry side as plumbing.
    The measurable consequence here was a model citing at 23.7% against a
    ceiling of 23.1% -- at the ceiling, and the ceiling was the bug.

    The record-level "Manual page:" header line is NOT removed. It is still
    correct for header facts, which genuinely belong to the record rather than
    to any one row.
    """
    page = (m.get("provenance") or {}).get("manual_page")
    return f" (page {page})" if page else ""


def _render_code(rec):
    """Flatten one failure code to text.

    Every string a case can ask about must survive into this text, or
    ceiling_recall drops for reasons that have nothing to do with retrieval.
    That includes each measurement criterion verbatim and each branch outcome in
    full -- the two things the metrics check containment of.
    """
    L = []
    L.append(f"Failure Code [{rec['code']}] {rec.get('title') or ''}")
    L.append(f"Model: {rec['model']}  Manual: {rec['manual_id']}  "
             f"Serial: {rec['serial_range']}")
    L.append(f"Manual page: {', '.join(rec.get('manual_pages') or [])}")
    for label, key in (("Action level", "action_level"),
                       ("Details of failure", "detail_of_failure"),
                       ("Action of controller", "controller_action"),
                       ("Phenomenon on machine", "machine_effect"),
                       ("Related information", "related_information"),
                       ("Controller", "controller"),
                       ("System category", "system_category")):
        if rec.get(key):
            L.append(f"{label}: {rec[key]}")

    for m in rec.get("standalone_measurements", []):
        L.append(f"Measurement. {m['quantity']}. Measuring point: {m['point']}. "
                 f"Standard value: {m['criteria']}{_meas_page(m)}")

    for st in rec.get("steps", []):
        L.append(f"Step {st['step']}. Cause: {st.get('cause','')}. "
                 f"Procedure: {st.get('procedure','')}")
        for br, outcome in (st.get("branches") or {}).items():
            L.append(f"Step {st['step']} {br}: {outcome}")
        for m in st.get("measurements", []):
            L.append(f"Step {st['step']} measurement. {m['quantity']}. "
                     f"Measuring point: {m['point']}. "
                     f"Standard value: {m['criteria']}{_meas_page(m)}")
        if st.get("extraction_warning"):
            L.append(f"Step {st['step']} provenance: {st['extraction_warning']}")
    if rec.get("default_conclusion"):
        L.append(f"Default conclusion: {rec['default_conclusion']}")
    return "\n".join(L)


def _render_symptom(rec):
    """Flatten one symptom tree to text.

    THE TWO TREE KINDS ARE RENDERED DIFFERENTLY BECAUSE THEY INVERT.

      branching  a step stores YES and NO outcomes. YES means the check was
                 normal and the tree ADVANCES.
      flat       a row stores cause / point to check / remedy and NO branch
                 outcomes at all. Confirming the point to check means the fault
                 is FOUND and the walk STOPS.

    Rendering both into "Step n ... YES/NO" would flatten that distinction into
    the retrieval text, and a corpus that says a flat row has branch outcomes is
    asserting something the manual does not. The words differ -- "Check n" and
    "Point to check" and "remedy" -- so a retrieved chunk reads the way the
    manual's own page reads.

    point_to_check and remedy must both survive: they are the two strings a
    flat-tree case can ask about, and a renderer that dropped them would cost
    ceiling_recall for a reason unrelated to retrieval.
    """
    L = [f"Symptom [{rec['symptom_id']}] {rec.get('symptom') or ''}",
         f"Section: {rec.get('section')}  Tree kind: {rec.get('tree_kind')}",
         f"Model: {rec['model']}  Manual: {rec['manual_id']}  "
         f"Serial: {rec['serial_range']}",
         f"Manual page: {', '.join(rec.get('manual_pages') or [])}"]
    for label, k in (("Details of failure", "detail_of_failure"),
                     ("Related information", "related_information")):
        if rec.get(k):
            L.append(f"{label}: {rec[k]}")

    for m in rec.get("standalone_measurements") or []:
        L.append(f"Measurement. {m['quantity']}. Measuring point: {m['point']}. "
                 f"Standard value: {m['criteria']}{_meas_page(m)}")

    flat = rec.get("tree_kind") == "SymptomTreeFlat"
    for st in rec.get("steps", []):
        n = st["step"]
        if flat:
            L.append(f"Check {n}. Cause: {st.get('cause','')}. "
                     f"Point to check: {st.get('point_to_check','')}")
            if st.get("remedy"):
                L.append(f"Check {n} remedy: {st['remedy']}")
        else:
            L.append(f"Step {n}. Cause: {st.get('cause','')}. "
                     f"Procedure: {st.get('procedure','')}")
            for br, outcome in (st.get("branches") or {}).items():
                L.append(f"Step {n} {br}: {outcome}")
        for m in st.get("measurements") or []:
            L.append(f"Step {n} measurement. {m['quantity']}. "
                     f"Measuring point: {m['point']}. "
                     f"Standard value: {m['criteria']}{_meas_page(m)}")
        if st.get("extraction_warning"):
            L.append(f"Step {n} provenance: {st['extraction_warning']}")

    # The manual sending the reader elsewhere is part of what this page says.
    # Carried as the manual's own words, with no attempt to resolve a target.
    for p in rec.get("unresolved_pointers") or []:
        if p.get("text"):
            L.append(f"Refers elsewhere: {p['text']}")
    return "\n".join(L)


def _meta(rec, idx, total=None):
    m = {
        # KEEPS THE NAME "code" for a symptom id. Every metric and the retriever
        # already key on this field; renaming it would be a wide rename of
        # working code to gain a synonym. `section` says which corpus it is.
        "code": key(rec),
        "section": section_of(rec),
        "model": rec["model"],
        "manual_id": rec["manual_id"],
        "manual_page": (rec.get("manual_pages") or [None])[0],
        "part": idx,
    }
    if total is not None:
        m["parts"] = total
    return m


def chunk_structural(records):
    """One chunk per record. No procedure is ever split across a boundary."""
    return [dict(id=f"{key(r)}#0", text=render(r), **_meta(r, 0, 1))
            for r in records]


def _fixed(records, size, overlap):
    chunks = []
    for r in records:
        text = render(r)
        if len(text) <= size:
            spans = [(0, len(text))]
        else:
            spans, start = [], 0
            step = size - overlap
            while start < len(text):
                spans.append((start, min(start + size, len(text))))
                if start + size >= len(text):
                    break
                start += step
        for i, (a, b) in enumerate(spans):
            chunks.append(dict(id=f"{key(r)}#{i}", text=text[a:b],
                               **_meta(r, i, len(spans))))
    return chunks


def chunk_fixed_2000(records):
    return _fixed(records, 2000, 200)


def chunk_fixed_512(records):
    return _fixed(records, 512, 50)


CHUNKERS = {
    "structural": chunk_structural,
    "fixed_2000": chunk_fixed_2000,
    "fixed_512": chunk_fixed_512,
}


def build(name, records=None, scope="all"):
    if name not in CHUNKERS:
        raise SystemExit(f"unknown chunker {name!r}; choose from {sorted(CHUNKERS)}")
    return CHUNKERS[name](records if records is not None else load_records(scope))


if __name__ == "__main__":
    import sys
    scope = sys.argv[1] if len(sys.argv) > 1 else "all"
    recs = load_records(scope)
    n_sym = sum(1 for r in recs if is_symptom(r))
    print(f"scope={scope}  {len(recs) - n_sym} failure codes + {n_sym} symptom trees")
    print(f"{'chunker':14} {'chunks':>7} {'avg chars':>10} {'max chars':>10}")
    for n in CHUNKERS:
        cs = build(n, recs)
        lens = [len(c["text"]) for c in cs]
        print(f"{n:14} {len(cs):7} {sum(lens)//len(lens):10} {max(lens):10}")
