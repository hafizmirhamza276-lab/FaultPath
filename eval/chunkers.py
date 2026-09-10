#!/usr/bin/env python3
"""
chunkers.py
Builds the retrieval corpus from golden/failure_codes/*.json, three ways.

Same content, same retriever, three chunkings. Everything else is held still, so
the spread between the three runs is the cost of the chunking decision and
nothing else. That is a number worth having before anyone argues about
embeddings.

  structural   one chunk per failure code. Respects the document's own
               boundaries -- a code's procedure is a unit and the manual already
               says so.
  fixed_2000   2000 characters, 200 overlap. The common default.
  fixed_512    512 characters, 50 overlap. Small chunks, precise but fragmented.

Every chunk carries `model` and `manual_id` so a filtered search can be
evaluated, and `code` so per-code diagnostics are possible. Metrics never key on
`id`.
"""
import json
import glob
import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))


def load_records():
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


def render(rec):
    """Flatten one code to text.

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
                 f"Standard value: {m['criteria']}")

    for st in rec.get("steps", []):
        L.append(f"Step {st['step']}. Cause: {st.get('cause','')}. "
                 f"Procedure: {st.get('procedure','')}")
        for br, outcome in (st.get("branches") or {}).items():
            L.append(f"Step {st['step']} {br}: {outcome}")
        for m in st.get("measurements", []):
            L.append(f"Step {st['step']} measurement. {m['quantity']}. "
                     f"Measuring point: {m['point']}. "
                     f"Standard value: {m['criteria']}")
        if st.get("extraction_warning"):
            L.append(f"Step {st['step']} provenance: {st['extraction_warning']}")
    if rec.get("default_conclusion"):
        L.append(f"Default conclusion: {rec['default_conclusion']}")
    return "\n".join(L)


def _meta(rec, idx, total=None):
    m = {
        "code": rec["code"],
        "model": rec["model"],
        "manual_id": rec["manual_id"],
        "manual_page": (rec.get("manual_pages") or [None])[0],
        "part": idx,
    }
    if total is not None:
        m["parts"] = total
    return m


def chunk_structural(records):
    """One chunk per code. No procedure is ever split across a boundary."""
    return [dict(id=f"{r['code']}#0", text=render(r), **_meta(r, 0, 1))
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
            chunks.append(dict(id=f"{r['code']}#{i}", text=text[a:b],
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


def build(name, records=None):
    if name not in CHUNKERS:
        raise SystemExit(f"unknown chunker {name!r}; choose from {sorted(CHUNKERS)}")
    return CHUNKERS[name](records if records is not None else load_records())


if __name__ == "__main__":
    recs = load_records()
    print(f"{'chunker':14} {'chunks':>7} {'avg chars':>10} {'max chars':>10}")
    for n in CHUNKERS:
        cs = build(n, recs)
        lens = [len(c["text"]) for c in cs]
        print(f"{n:14} {len(cs):7} {sum(lens)//len(lens):10} {max(lens):10}")
