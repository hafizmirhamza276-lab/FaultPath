#!/usr/bin/env python3
"""
extract_symptoms.py
H-Mode and S-Mode symptom trees -> golden/symptoms/

Two structurally different things, modelled as two record kinds rather than one
bent shape:

  H-Mode (37 entries, pp 1289-1471)  SymptomTreeBranching
      5 columns, YES/NO on separate rows, "Details of failure" header,
      standalone "Item" measurement tables. Genuinely the Format A shape, so
      the Format A path is reused.

  S-Mode (20 entries, pp 1473-1498)  SymptomTreeFlat
      4 columns: No. | Cause | Point to check, remarks | Remedy.
      No branches, no measurements, and a Remedy column with no Section 40
      equivalent. Given its own header path and a first-class `remedy` field.
      Burying Remedy in procedure prose would be exactly the shape-bending this
      record kind exists to avoid.

The agent must be able to tell them apart: a flat tree has no YES/NO to ask
about, and asking anyway would invent an interaction the manual does not have.
`tree_kind` carries that.

Section 40 is untouched. Its parser paths are imported read-only and its
records are not written to.

Deterministic. No LLM.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import OrderedDict

import pdfplumber
import fitz

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

PDF = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KOMATSU_PDF", "")
OUT = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))

if not PDF or not os.path.isfile(PDF):
    sys.exit("ERROR: source manual PDF not found.\n"
             "  Pass the path as the first argument, or set KOMATSU_PDF.\n"
             f"  Tried: {PDF or '<unset>'}")

# Read-only reuse of the Section 40 helpers, where the layout is genuinely the
# same. Nothing here writes to Section 40 or changes its behaviour.
from pipeline.extract_golden import (                      # noqa: E402
    clean, norm_label, manual_page, table_kind, parse_causes,
    parse_measurement_table, parse_header_a, CRIT_VALUE, SCHEMA_VERSION,
    MANUAL_ID, REVISION, MODEL, SERIAL_RANGE)

H_MODE_RANGE = (1289, 1471)
S_MODE_RANGE = (1473, 1498)

BRANCHING = "SymptomTreeBranching"
FLAT = "SymptomTreeFlat"

# S-Mode's own header labels. Deliberately NOT added to HEADER_LABELS_A --
# that dict belongs to Section 40 and this section is not Section 40.
S_HEADER_LABELS = {
    "failure": "symptom",
    "related information": "related_information",
}

# A criterion that states a relationship rather than a value. Declared bucket,
# like Section 40's E5 continuity cases: these leave the numeric denominator by
# explicit declaration rather than by silence, and must still resolve against
# the page.
RELATIONAL_RE = re.compile(
    r"oil pressure ratio|pressure ratio|ratio of|compared with|same as|"
    r"higher than|lower than|difference between", re.I)


def is_relational(criteria: str) -> bool:
    c = (criteria or "").strip()
    if not c:
        return False
    return bool(RELATIONAL_RE.search(c)) and not CRIT_VALUE.search(c)


def s_mode_header_kind(tbl) -> bool:
    """True when this is an S-Mode symptom header.

    Its own classifier because S-Mode's header leads with 'Failure', which is
    not in HEADER_LABELS_A. Under the Section 40 classifier all 20 of these
    returned 'unknown' and the symptom text -- the one field that makes an entry
    findable -- was dropped with no error. That is the Format C failure again.
    """
    if not tbl or not tbl[0]:
        return False
    first = norm_label(tbl[0][0] or "")
    if first != "failure":
        return False
    labels = {norm_label(r[0] or "") for r in tbl if r}
    return bool(labels & set(S_HEADER_LABELS))


def parse_s_header(tbl) -> dict:
    out = {}
    for row in tbl:
        if len(row) < 2:
            continue
        key = S_HEADER_LABELS.get(norm_label(row[0] or ""))
        if key:
            out[key] = clean(row[1])
    return out


def parse_flat_causes(rows, prov_rows) -> list:
    """S-Mode: No. | Cause | Point to check, remarks | Remedy.

    A separate parser because the columns are different and `remedy` is a field
    in its own right. Reusing parse_causes would put the remedy into spill-over
    procedure text, which is precisely the shape-bending this avoids.
    """
    steps = []
    for (row, prov) in zip(rows[1:], prov_rows[1:]):
        cells = [clean(c) for c in row]
        if not cells or not cells[0]:
            # continuation of the previous step
            if steps and any(cells):
                extra = " ".join(x for x in cells[1:] if x)
                if extra:
                    steps[-1]["point_to_check"] = (
                        steps[-1]["point_to_check"] + " " + extra).strip()
            continue
        if not re.fullmatch(r"\d+", cells[0]):
            continue
        steps.append({
            "step": int(cells[0]),
            "cause": cells[1] if len(cells) > 1 else "",
            "point_to_check": cells[2] if len(cells) > 2 else "",
            "remedy": cells[3] if len(cells) > 3 else "",
            "provenance": dict(prov),
        })
    return steps


POINTER_RE = re.compile(
    r"(?:do|perform|carry out) the troubleshooting[^.]*\.", re.I)
BRACKET_CODE_RE = re.compile(r"\[([A-Z0-9@#]{4,7})\]")


def find_pointers(text: str, prov: dict, section40_codes: set) -> list:
    """Prose pointers, RECORDED not resolved.

    130 of these exist and none uses the bracket convention. A text-matching
    rule to resolve them would produce false positives, and a false positive
    here sends a technician into the wrong tree -- worse than no pointer. Only a
    pointer naming a Section 40 code unambiguously IN BRACKET FORM is resolved.
    """
    out = []
    for m in POINTER_RE.finditer(text or ""):
        span = m.group(0).strip()
        codes = [c for c in BRACKET_CODE_RE.findall(span) if c in section40_codes]
        out.append({
            "text": span[:300],
            "resolved_to": codes[0] if len(codes) == 1 else None,
            "resolution": ("bracketed Section 40 code, unambiguous"
                           if len(codes) == 1 else
                           "prose only; not resolved by design -- a guessed "
                           "target sends a technician into the wrong tree"),
            "provenance": dict(prov),
        })
    return out


def symptom_id(mode: str, ordinal: int) -> str:
    """HM01..HM37 / SM01..SM20. Fits the existing fact-id character class so
    the citation resolver needs no new id grammar."""
    return f"{mode}M{ordinal:02d}"


def extract_entry(pdf, sid, title, start, end, mode, section40_codes):
    rec = OrderedDict(
        schema_version=SCHEMA_VERSION,
        manual_id=MANUAL_ID, revision=REVISION, model=MODEL,
        serial_range=SERIAL_RANGE,
        section="H-Mode" if mode == "H" else "S-Mode",
        symptom_id=sid,
        tree_kind=BRANCHING if mode == "H" else FLAT,
        symptom=clean(title),
        symptom_source="bookmark title",
        detail_of_failure=None, related_information=None,
        pdf_pages=[start, end], manual_pages=[],
        steps=[], standalone_measurements=[],
        unresolved_pointers=[], refs_failure_codes=[],
        skipped_tables=[],
    )

    cause_rows, flat_rows = [], []
    for pno in range(start, end + 1):
        if pno - 1 >= len(pdf.pages):
            continue
        page = pdf.pages[pno - 1]
        mp = manual_page(page)
        if mp and mp not in rec["manual_pages"]:
            rec["manual_pages"].append(mp)

        for ti, tbl in enumerate(page.extract_tables()):
            prov = {"manual_page": mp, "pdf_page": pno, "table_index": ti}
            flat_text = " ".join(clean(c) for r in tbl for c in r if c)

            # Eight empty phantom tables live on pp 1319-1345. Skipped, but
            # LOGGED -- a silent skip is how the Format C collapse survived.
            if not flat_text.strip():
                rec["skipped_tables"].append(
                    {**prov, "reason": "table is entirely empty",
                     "shape": f"{len(tbl)}x{max((len(r) for r in tbl), default=0)}"})
                continue

            rec["unresolved_pointers"] += find_pointers(flat_text, prov,
                                                        section40_codes)

            if mode == "S" and s_mode_header_kind(tbl):
                rec.update({k: v for k, v in parse_s_header(tbl).items() if v})
                continue

            kind = table_kind(tbl)
            if kind == "header_a":
                rec.update({k: v for k, v in parse_header_a(tbl).items() if v})
            elif kind == "causes":
                body = list(enumerate(tbl))
                provs = [dict(prov, row_index=i) for i, _ in body]
                if mode == "H":
                    cause_rows.append(([r for _, r in body], provs))
                else:
                    flat_rows.append(([r for _, r in body], provs))
            elif kind == "measurement":
                rec["standalone_measurements"] += parse_measurement_table(tbl, prov)
            else:
                rec["skipped_tables"].append(
                    {**prov, "reason": f"table_kind={kind}",
                     "first_cell": clean((tbl[0] or [""])[0])[:40],
                     "shape": f"{len(tbl)}x{max((len(r) for r in tbl), default=0)}"})

    if mode == "H":
        merged, mprov = [], []
        for i, (rows, provs) in enumerate(cause_rows):
            sl = slice(0, None) if i == 0 else slice(1, None)
            merged += rows[sl]
            mprov += provs[sl]
        if merged:
            rec["steps"] = parse_causes(list(zip(merged, mprov)), "A")
    else:
        for rows, provs in flat_rows:
            rec["steps"] += parse_flat_causes(rows, provs)

    # ---- fact identity, same scheme as Section 40
    for i, m in enumerate(rec["standalone_measurements"]):
        m["fact_id"] = f"{sid}:0:meas:{i}"
        m["criteria_kind"] = ("relational" if is_relational(m.get("criteria"))
                              else "numeric")
    for st in rec["steps"]:
        st["fact_id"] = f"{sid}:{st['step']}:step:0"
        for i, m in enumerate(st.get("measurements", []) or []):
            m["fact_id"] = f"{sid}:{st['step']}:meas:{i}"
            m["criteria_kind"] = ("relational" if is_relational(m.get("criteria"))
                                  else "numeric")
        st["branch_fact_ids"] = {
            br: f"{sid}:{st['step']}:branch:{i}"
            for i, br in enumerate(sorted(st.get("branches") or {}))}
        if st.get("remedy"):
            st["remedy_fact_id"] = f"{sid}:{st['step']}:remedy:0"

    rec["refs_failure_codes"] = sorted(
        {p["resolved_to"] for p in rec["unresolved_pointers"] if p["resolved_to"]})
    return rec


def entries(doc):
    toc = doc.get_toc()
    out = []
    for lvl, title, pg in toc:
        if lvl != 3 or title.strip().startswith("Information Shown"):
            continue
        if H_MODE_RANGE[0] <= pg <= H_MODE_RANGE[1]:
            out.append(("H", title, pg))
        elif S_MODE_RANGE[0] <= pg <= S_MODE_RANGE[1]:
            out.append(("S", title, pg))
    spans = []
    for i, (mode, title, pg) in enumerate(out):
        end = out[i + 1][2] - 1 if i + 1 < len(out) else S_MODE_RANGE[1]
        spans.append((mode, title, pg, end))
    return spans


def self_test() -> None:
    """The S-Mode header classifier must prove it can tell the two apart.

    Every S-Mode header returned 'unknown' under the Section 40 classifier and
    the symptom text vanished with no error. A classifier that returns unknown
    silently is what this whole finding was about, so it is asserted against a
    real header of each kind before it runs.
    """
    s_header = [["Failure", "Engine does not crank when starting switch..."],
                ["Related infor-\nmation", "If any failure code is displayed..."]]
    assert s_mode_header_kind(s_header), \
        "S-Mode header classifier does not recognise a real S-Mode header"
    parsed = parse_s_header(s_header)
    assert parsed.get("symptom", "").startswith("Engine does not crank"), \
        "S-Mode header parser does not extract the symptom text"

    s40_header = [["Details of failure", "A high voltage occurs..."],
                  ["Action level", "L03"]]
    assert not s_mode_header_kind(s40_header), \
        "S-Mode header classifier accepts a Section 40 header"
    cause_tbl = [["No.", "Cause", "Point to check, remarks", "Remedy"]]
    assert not s_mode_header_kind(cause_tbl), \
        "S-Mode header classifier accepts a cause table"

    assert is_relational("Oil pressure ratio pump discharged pressure"), \
        "relational bucket does not catch a ratio criterion"
    assert not is_relational("0 MPa {0 kgf/cm2}"), \
        "relational bucket swallows a real dual-unit value"
    assert not is_relational("Max. 1 Ω"), "relational bucket swallows a bound"

    flat = parse_flat_causes(
        [["No.", "Cause", "Point to check", "Remedy"],
         ["1", "Defective starting circuit", "When switch is turned", "Replace it"]],
        [{}, {"manual_page": "40-x", "pdf_page": 1, "table_index": 0, "row_index": 1}])
    assert flat and flat[0]["remedy"] == "Replace it", \
        "flat parser does not give remedy a field of its own"


def main() -> int:
    self_test()
    out_dir = os.path.join(OUT, "symptoms")
    os.makedirs(out_dir, exist_ok=True)

    import glob
    s40 = set()
    for p in glob.glob(os.path.join(OUT, "failure_codes", "*.json")):
        with open(p, encoding="utf-8") as f:
            s40.add(json.load(f)["code"])

    doc = fitz.open(PDF)
    spans = entries(doc)
    doc.close()

    counters = {"H": 0, "S": 0}
    index, skipped, pointers = [], [], []
    with pdfplumber.open(PDF) as pdf:
        for mode, title, p0, p1 in spans:
            counters[mode] += 1
            sid = symptom_id(mode, counters[mode])
            rec = extract_entry(pdf, sid, title, p0, p1, mode, s40)
            with open(os.path.join(out_dir, f"{sid}.json"), "w",
                      encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            skipped += rec["skipped_tables"]
            pointers += rec["unresolved_pointers"]
            index.append({
                "symptom_id": sid, "section": rec["section"],
                "tree_kind": rec["tree_kind"], "symptom": rec["symptom"],
                "manual_page": (rec["manual_pages"] or [None])[0],
                "n_steps": len(rec["steps"]),
                "n_measurements": len(rec["standalone_measurements"])
                + sum(len(s.get("measurements") or []) for s in rec["steps"]),
                "n_pointers": len(rec["unresolved_pointers"]),
            })

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    h = [x for x in index if x["section"] == "H-Mode"]
    s = [x for x in index if x["section"] == "S-Mode"]
    print(f"H-Mode  {len(h):3} entries  {sum(x['n_steps'] for x in h):4} steps  "
          f"{sum(x['n_measurements'] for x in h):4} measurements")
    print(f"S-Mode  {len(s):3} entries  {sum(x['n_steps'] for x in s):4} steps  "
          f"{sum(x['n_measurements'] for x in s):4} measurements")
    print(f"\nunresolved pointers: {len(pointers)} "
          f"({sum(1 for p in pointers if p['resolved_to'])} resolved)")
    print(f"skipped tables     : {len(skipped)}")
    by_reason = {}
    for x in skipped:
        by_reason[x["reason"]] = by_reason.get(x["reason"], 0) + 1
    for r, n in sorted(by_reason.items()):
        print(f"   {n:3}  {r}")
    named = [x for x in skipped if not x["reason"].startswith("table is empty")]
    for x in named[:12]:
        print(f"     p{x['pdf_page']} t{x['table_index']}: {x['reason']}"
              f"  first_cell={x.get('first_cell','')!r} shape={x['shape']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
