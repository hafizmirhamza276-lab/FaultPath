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
    """A criterion that states a CONDITION rather than a VALUE.

    Three tests, in order, and the first one is the guard:

      1. it carries a criterion value  -> numeric, always. Nothing below can
         override this, so no criterion holding a bound can be swept out of
         the numeric bucket by a phrasing rule.
      2. it states a ratio             -> relational
         "Oil pressure ratio pump discharged pressure : PC valve discharged
         pressure 1:0.6" -- two quantities compared, no bound on either.
      3. it contains no digit at all   -> relational
         "Pressure for each flow setting" is what the manual prints where a
         value would go: the figure depends on a setting stated elsewhere.
         There is no number to reproduce, so scoring it by numeric comparison
         measures nothing.

    Test 3 is safe only because test 1 runs first and returns early. A
    criterion with a value always has a digit, so the two can never disagree --
    but the ordering is what makes that true, not the arithmetic.
    """
    c = (criteria or "").strip()
    if not c:
        return False
    if CRIT_VALUE.search(c):
        return False
    if RELATIONAL_RE.search(c):
        return True
    return not re.search(r"\d", c)


# ------------------------------------------------------ merged table cells
#
# A criteria cell merged with the row above and an empty criteria cell are the
# same thing in the extracted text -- "" either way -- and are not the same
# thing at all. The merged one is governed by a criterion the table draws once;
# the empty one has nothing in it. Telling them apart from the text is
# impossible, so this reads the geometry.
#
# pdfplumber reports a cell covered by a merge as None. That alone is not
# enough: a HORIZONTALLY merged header cell is also None, and inheriting
# downward from one would be wrong. The vertical case is identified by finding
# the cell that actually covers this position -- a cell in an earlier row, same
# column, whose bottom edge reaches into this row.

def merged_from_above(cells, ri, ci):
    """Row index of the cell vertically covering (ri, ci), or None.

    `cells` is the per-row cell-rect grid from find_tables(): cells[r][c] is a
    bbox (x0, top, x1, bottom) or None where a merge covers it.
    """
    if ri == 0 or ci >= len(cells[ri]) or cells[ri][ci] is not None:
        return None
    own = next((c for c in cells[ri] if c is not None), None)
    if own is None:
        return None
    row_bottom = own[3]
    for r in range(ri - 1, -1, -1):
        if ci >= len(cells[r]):
            continue
        above = cells[r][ci]
        if above is None:
            continue
        # covers this row only if its bottom edge reaches our bottom edge
        return r if above[3] >= row_bottom - 0.5 else None
    return None


def inherit_merged_cells(tbl, cells):
    """Fill cells covered by a vertical merge with the text that governs them.

    Returns (table, [(row, col, source_row)]). The table is copied, never
    mutated in place: the caller still needs the raw text to tell a genuinely
    empty cell from this one.
    """
    out = [list(r) for r in tbl]
    filled = []
    for ri in range(len(out)):
        for ci in range(len(out[ri])):
            if (out[ri][ci] or "").strip():
                continue
            src = merged_from_above(cells, ri, ci)
            if src is None or not (tbl[src][ci] or "").strip():
                continue
            out[ri][ci] = tbl[src][ci]
            filled.append((ri, ci, src))
    return out, filled


def _strip_leading_blank_rows(tbl):
    """(reduced_table, n_dropped). Leading all-blank rows only."""
    i = 0
    while i < len(tbl) and not any(clean(c) for c in tbl[i] if c):
        i += 1
    return tbl[i:], i


# Column boundaries of a continuation align with its predecessor's to within
# 0.1pt once the page margin is removed. 1.0pt is therefore generous against
# rendering noise while being 14x tighter than the recto/verso margin shift and
# orders of magnitude looser than nothing -- a genuinely different table differs
# by tens of points and usually in column count too. Chosen from the observed
# spread, not tuned to make the answer come out right.
GEOMETRY_TOLERANCE_PT = 1.0


def column_widths(boundaries):
    """Widths between successive boundaries.

    Compared instead of absolute x because odd and even pages carry mirrored
    gutter margins: every continuation in this document sits exactly 14.20pt
    from its predecessor. Absolute alignment would reject all of them, and that
    shift is a property of the page, not of the table. Widths are invariant --
    one table drawn across two pages keeps its columns.
    """
    return [round(b - a, 2) for a, b in zip(boundaries, boundaries[1:])]


def geometry_matches(frag_bounds, prev_bounds, tol=GEOMETRY_TOLERANCE_PT):
    """True when the fragment's rules sit where the predecessor's do.

    Alignment is tested as a SUBSET, not an equality: every boundary the
    predecessor draws must reappear in the fragment, and both outer edges must
    coincide. The fragment may carry extra interior boundaries the predecessor
    does not, because a merged cell draws fewer rules over the same span --
    three of these tables end a page on a header row whose "Measurement
    position" cell spans two columns (width 133.3 = 66.6 + 66.7), so the stub
    reports 4 boundaries where its own body has 5. Requiring equality would
    reject a table on the strength of one merged heading cell.

    The offset is measured from the left edge rather than assumed to be zero:
    odd and even pages carry mirrored gutter margins, and every continuation in
    this document sits exactly 14.20pt from its predecessor. That shift belongs
    to the page, not the table.
    """
    if len(prev_bounds) < 2 or len(frag_bounds) < 2:
        return False, None
    if len(frag_bounds) < len(prev_bounds):
        return False, None            # columns were lost, not merged
    shift = frag_bounds[0] - prev_bounds[0]
    worst = abs((prev_bounds[-1] + shift) - frag_bounds[-1])   # right edge
    for b in prev_bounds[1:-1]:
        worst = max(worst, min(abs((b + shift) - f) for f in frag_bounds))
    return worst <= tol, round(worst, 2)


def has_own_header(tbl):
    """A fragment carrying its own header row is a NEW table, not a
    continuation. Checked over every row, not just the first."""
    for row in tbl:
        labels = [norm_label(c or "") for c in row]
        if "item" in labels:
            return True
        for c in row:
            if c and "standard value" in clean(c).lower():
                return True
    return False


def symptom_table_kind(tbl, prev=None, is_first_on_page=False, bounds=None):
    """Symptom-local classifier. Returns (kind, admitting_rule).

    table_kind() is NOT modified: Section 40's 174 codes, 996 steps and 872
    measurements rest on it, and changing a shared classifier to serve new
    content would risk the mature half to help the new one. Section 40 also has
    no need of this -- the blank-leading-row and header-less-continuation shapes
    occur only in the symptom sections, and orphan_detection and fidelity are
    both clean over Section 40.

    Two shapes beyond the Section 40 set:

    a) BLANK LEADING ROW -- scan past leading all-blank rows and classify the
       real header. Low risk: the header is still present and still read.

    b) HEADER-LESS CONTINUATION -- PROVEN, never assumed. All four must hold:
         1. the previous page's last table was a measurement table
         2. it has no fewer columns than the predecessor (a merged heading
            cell draws fewer rules over the same span, so equality would
            reject a table over one merged cell -- see geometry_matches)
         3. the fragment carries NO header row of its own
         4. its column widths match the predecessor's within
            GEOMETRY_TOLERANCE_PT
       `prev` is {"kind", "cols", "bounds"} for the previous page's last table.

       Position was the original third condition and has been replaced. Cause
       tables also cross pages, so a measurement continuation always sits at t1
       and the condition could not be true anywhere in this corpus. Conditions
       3 and 4 answer the real question -- is this the same table -- instead of
       "is it nearby".

    The admitting rule travels with the result so a wrong admission is
    traceable rather than invisible.
    """
    base = table_kind(tbl)
    if base != "unknown":
        return base, "section40_classifier"

    reduced, dropped = _strip_leading_blank_rows(tbl)
    if dropped and reduced:
        k = table_kind(reduced)
        if k != "unknown":
            return k, f"blank_leading_row(+{dropped})"

    cols = max((len(r) for r in tbl), default=0)
    if reduced and prev and prev.get("kind") == "measurement" \
            and cols >= prev.get("cols", 0) and not has_own_header(tbl):
        ok, delta = geometry_matches(bounds or [], prev.get("bounds") or [])
        if ok:
            return "measurement", "proven_continuation(geom_delta=%.2fpt)" % delta

    return "unknown", "unclassified"


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
            # The remedy is the OUTCOME of a flat tree and is cited on its own,
            # so it gets its own page rather than the step's. Identical today
            # because both are read from this row -- and that is a property of
            # the current layout, not a guarantee. Captured here so a remedy
            # that ever moves to another row moves its citation with it.
            "remedy_provenance": dict(prov) if len(cells) > 3 and cells[3] else None,
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


def page_prints_title(page_text: str, title: str) -> bool:
    """Does this page actually print the symptom title?

    Both sides go through clean(), which is what the resolver's _norm() mirrors.
    A raw substring match moved three titles (HM18, HM34, HM35) to their SECOND
    page, because the first renders them hyphenated across a line break --
    "work equipment with heavi- er load moves slower". The title really was on
    the first page; only the comparison was wrong.

    Matching provenance with one text pipeline and verifying it with another is
    how a fix ends up disagreeing with its own check.
    """
    needle = " ".join(clean(title or "").lower().split())
    hay = " ".join(clean(page_text or "").lower().split())
    return bool(needle) and needle in hay


def assign_branch_provenance(steps, rows, provs):
    """Give each branch outcome the page it was actually read from.

    A step that straddles a page break has its YES outcome printed on one page
    and its NO outcome on the next. Both inherit the step's provenance, so one
    of them cites a page its text is not on -- and a citation naming the wrong
    page is worse than none, because it manufactures confidence.

    This is the manual_pages[0] defect one level down. CLAUDE.md already states
    the general form: "75% of measurements are not on their code's first page,
    so a per-code page is not a citation." A per-STEP page is not one either.

    Matching is EXACT against a cleaned cell, never fuzzy. A near-match would
    silently attach a branch to a neighbouring row, which is the failure this
    is meant to remove rather than relocate. Anything not matched exactly keeps
    the step's provenance and is counted, so the fallback can never be silent.
    """
    index = {}
    for row, prov in zip(rows, provs):
        for ci, cell in enumerate(row):
            t = clean(cell or "")
            if t:
                index.setdefault(t, []).append(dict(prov, column_index=ci))

    own, inherited = 0, []
    for st in steps:
        # parse_causes now captures this at parse time, which is strictly
        # better than matching after the fact: the row provenance is in hand
        # at the moment the outcome is read, so there is nothing to match and
        # nothing to get wrong. This pass only fills what parse time could not.
        at_parse = dict(st.get("branch_provenance") or {})
        st["branch_provenance"] = at_parse
        for br, txt in sorted((st.get("branches") or {}).items()):
            if at_parse.get(br):
                own += 1
                continue
            hits = index.get(txt or "")
            if hits and len(hits) == 1:
                st["branch_provenance"][br] = hits[0]
                own += 1
            elif hits:
                # Ambiguous: the same outcome text appears in more than one
                # row. Prefer a hit on the step's own page -- that is the row
                # the step was built from -- and fall back only if none is.
                same = [h for h in hits
                        if h.get("pdf_page") == st["provenance"].get("pdf_page")]
                st["branch_provenance"][br] = (same or hits)[0]
                own += 1
            else:
                st["branch_provenance"][br] = dict(st["provenance"])
                inherited.append(f"{st['step']}:{br}")
    return own, inherited


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
        merged_cells_inherited=[],
        title_provenance=None,
        # Every fact kind that could not capture its own page, counted by kind.
        # A fallback that is not counted is a fallback nobody knows happened.
        provenance_fallbacks={},
    )

    cause_rows, flat_rows = [], []
    prev_last = None                      # previous page's last table, for (b)
    for pno in range(start, end + 1):
        if pno - 1 >= len(pdf.pages):
            continue
        page = pdf.pages[pno - 1]
        mp = manual_page(page)
        if mp and mp not in rec["manual_pages"]:
            rec["manual_pages"].append(mp)

        # TITLE PROVENANCE. pdf_pages[0] is the manual_pages[0] pattern
        # verbatim -- the shape that sent 635 citations to the wrong page. The
        # title happens to sit on the entry's first page in this revision, and
        # "happens to" is exactly what this is meant to stop relying on.
        #
        # The bookmark supplies the title TEXT; this finds the page that
        # actually prints it. Matched exactly on the normalised page text, never
        # fuzzily, and the first page carrying it wins.
        #
        # Both sides go through clean(), which is what the resolver's _norm()
        # mirrors. A raw substring match moved three titles (HM18, HM34, HM35)
        # to their SECOND page, because the first renders them hyphenated
        # across a line break -- "work equipment with heavi- er load moves
        # slower". The title really is on the first page; only the comparison
        # was wrong. Matching provenance with one text pipeline and verifying
        # it with another is how a fix ends up disagreeing with its own check.
        if rec.get("title_provenance") is None:
            if page_prints_title(page.extract_text() or "", title):
                rec["title_provenance"] = {"manual_page": mp, "pdf_page": pno,
                                           "table_index": None, "row_index": None}

        tables = []
        for ft in page.find_tables():
            xs = sorted({round(c[0], 1) for r in ft.rows for c in r.cells if c}
                        | {round(ft.bbox[2], 1)})
            tables.append((ft.extract(), xs, [list(r.cells) for r in ft.rows]))
        page_last = None
        for ti, (tbl, tbounds, tcells) in enumerate(tables):
            prov = {"manual_page": mp, "pdf_page": pno, "table_index": ti}
            flat_text = " ".join(clean(c) for r in tbl for c in r if c)

            # Eight empty phantom tables live on pp 1319-1345. Skipped, but
            # LOGGED -- a silent skip is how the Format C collapse survived.
            if not flat_text.strip():
                rec["skipped_tables"].append(
                    {**prov, "reason": "table is entirely empty",
                     "shape": f"{len(tbl)}x{max((len(r) for r in tbl), default=0)}"})
                page_last = None
                continue

            rec["unresolved_pointers"] += find_pointers(flat_text, prov,
                                                        section40_codes)

            if mode == "S" and s_mode_header_kind(tbl):
                rec.update({k: v for k, v in parse_s_header(tbl).items() if v})
                continue

            kind, rule = symptom_table_kind(
                tbl, prev=prev_last, is_first_on_page=(ti == 0),
                bounds=tbounds)
            page_last = {"kind": kind,
                         "cols": max((len(r) for r in tbl), default=0),
                         "bounds": tbounds}
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
                # Inherit vertically merged cells BEFORE parsing. A criterion
                # the table draws once governs every row it spans; reading the
                # covered row as empty records a measurement with no value.
                src, filled = inherit_merged_cells(tbl, tcells)
                for ri, ci, sr in filled:
                    rec["merged_cells_inherited"].append(
                        {**prov, "row_index": ri, "column_index": ci,
                         "inherited_from_row": sr,
                         "reason": "criteria cell vertically merged with the "
                                   "row above; the covering cell's criterion "
                                   "governs both rows"})
                if rule.startswith("blank_leading_row"):
                    src, _ = _strip_leading_blank_rows(src)
                elif rule == "proven_continuation":
                    # No header row to skip: parse_measurement_table drops
                    # tbl[0], so a header is prepended to keep every data row.
                    src = [["Item", "", "", ""]] + [
                        r for r in tbl if any(clean(c) for c in r if c)]
                got = parse_measurement_table(src, prov)
                for m in got:
                    m["admitted_by"] = rule
                rec["standalone_measurements"] += got
            else:
                rec["skipped_tables"].append(
                    {**prov, "reason": f"table_kind={kind}", "rule": rule,
                     "first_cell": clean((tbl[0] or [""])[0])[:40],
                     "shape": f"{len(tbl)}x{max((len(r) for r in tbl), default=0)}"})

        prev_last = page_last
    if mode == "H":
        merged, mprov = [], []
        for i, (rows, provs) in enumerate(cause_rows):
            sl = slice(0, None) if i == 0 else slice(1, None)
            merged += rows[sl]
            mprov += provs[sl]
        if merged:
            rec["steps"] = parse_causes(list(zip(merged, mprov)), "A")
            own, inherited = assign_branch_provenance(rec["steps"], merged, mprov)
            rec["branch_provenance_own"] = own
            rec["branch_provenance_inherited"] = inherited
            if inherited:
                rec["provenance_fallbacks"]["branch"] = len(inherited)

    # Fallbacks, counted by kind. Each falls back to the parent's page -- the
    # old behaviour -- but never silently.
    if rec.get("title_provenance") is None:
        rec["title_provenance"] = {"manual_page": rec["manual_pages"][0]
                                   if rec["manual_pages"] else None,
                                   "pdf_page": start,
                                   "table_index": None, "row_index": None}
        rec["provenance_fallbacks"]["symptom_title"] = 1
    n = sum(1 for s in rec["steps"]
            if (s.get("remedy") or "").strip() and not s.get("remedy_provenance"))
    if n:
        rec["provenance_fallbacks"]["remedy"] = n
        for s in rec["steps"]:
            if (s.get("remedy") or "").strip() and not s.get("remedy_provenance"):
                s["remedy_provenance"] = dict(s["provenance"])
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
        # An entry may never run past its own section. Computing the end as
        # next_entry_page - 1 gave the last H-Mode entry pp 1471-1472, which
        # swallowed the S-Mode legend page and made its provenance wrong by one.
        limit = H_MODE_RANGE[1] if mode == "H" else S_MODE_RANGE[1]
        nxt = out[i + 1][2] - 1 if i + 1 < len(out) else limit
        spans.append((mode, title, pg, min(nxt, limit)))
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

    # C3 -- a criterion expressed as a condition rather than a value.
    assert is_relational("Pressure for each flow setting"), \
        "a criterion with no number in it was typed numeric"
    # ...and the guard that makes rule 3 safe: anything carrying a value stays
    # numeric no matter how it is phrased.
    for keeps_value in ("0.70 to 1.09 MPa {7.1 to 11.1 kgf/cm2}", "Min. 100kΩ",
                        "0 mA", "800 to 1000 mA", "2.84 to 3.43 MPa {29 to 35 kgf/cm2}",
                        "Pressure ratio 1 to 5Ω"):
        assert not is_relational(keeps_value), \
            "a criterion carrying a value was swept into the relational " \
            "bucket: %r" % keeps_value
    assert not is_relational(""), "an empty criterion was typed relational"

    # S1 -- merged versus empty, the same text from opposite sides.
    #   grid: row 0 has a cell in col 1 spanning down into row 1 (bottom 60.0);
    #         row 1 col 1 is None (covered). Col 2 is present-but-blank in both.
    # The real criterion, not a stand-in -- a toy string can be typed
    # differently from the text the fix actually has to carry.
    RATIO = ("Oil pressure ratio pump discharged pressure: PC valve "
             "discharged pressure 1:0.6 (approximately 3/5)")
    GRID = [[(0, 0, 10, 30), (10, 0, 20, 60), (20, 0, 30, 30)],
            [(0, 30, 10, 60), None, (20, 30, 30, 60)]]
    TXT = [["Pump supply", RATIO, ""],
           ["PC valve", "", ""]]
    assert merged_from_above(GRID, 1, 1) == 0, \
        "a vertically merged cell was not traced to the cell covering it"
    assert merged_from_above(GRID, 1, 2) is None, \
        "an EMPTY cell was treated as merged -- it has its own rect and is " \
        "simply blank"
    assert merged_from_above(GRID, 0, 1) is None, \
        "a cell in the first row was treated as inheriting from above"
    # a horizontally merged header cell is also None, and must NOT inherit
    HGRID = [[(0, 0, 20, 30), None, (20, 0, 30, 30)],
             [(0, 30, 10, 60), (10, 30, 20, 60), (20, 30, 30, 60)]]
    assert merged_from_above(HGRID, 0, 1) is None, \
        "a horizontally merged header cell was read as a vertical merge"

    filled_tbl, filled = inherit_merged_cells(TXT, GRID)
    assert filled_tbl[1][1] == RATIO, \
        "a merged cell did not inherit the criterion governing it"
    assert filled_tbl[1][2] == "", \
        "an empty cell was filled from above -- empty and merged are not the " \
        "same thing"
    assert filled == [(1, 1, 0)], \
        "the inheritance record does not name exactly the cell it filled: %r" % (filled,)
    assert TXT[1][1] == "", "inherit_merged_cells mutated the raw table"
    # the inherited text must carry its TYPE with it, not just its characters
    assert is_relational(filled_tbl[1][1]), \
        "an inherited relational criterion was retyped as numeric"

    # S2 -- a branch outcome takes the page its own row was read from.
    ROWS = [["5", "cause", "proc", "", "• Valve is normal."],
            ["", "", "", "", "• Valve is defective."]]
    PROVS = [{"manual_page": "40-857", "pdf_page": 1399, "table_index": 0,
              "row_index": 5},
             {"manual_page": "40-858", "pdf_page": 1400, "table_index": 0,
              "row_index": 2}]
    steps = [{"step": 5, "provenance": dict(PROVS[0]),
              "branches": {"YES": "• Valve is normal.",
                           "NO": "• Valve is defective."}}]
    own, inherited = assign_branch_provenance(steps, ROWS, PROVS)
    assert not inherited, "a branch outcome fell back to its step's page: %r" % (inherited,)
    assert steps[0]["branch_provenance"]["YES"]["pdf_page"] == 1399, \
        "YES did not take the page its own row was read from"
    assert steps[0]["branch_provenance"]["NO"]["pdf_page"] == 1400, \
        "NO inherited the step's page instead of its own -- the S2 defect"
    # and the fallback must be counted, never silent
    lost = [{"step": 6, "provenance": dict(PROVS[0]),
             "branches": {"YES": "text that is in no row"}}]
    own2, inherited2 = assign_branch_provenance(lost, ROWS, PROVS)
    assert inherited2 == ["6:YES"] and own2 == 0, \
        "an unmatched branch inherited the step's page without being counted"

    meas_hdr = [["Item", "Measurement position", "", "Standard value"],
                ["EPC Current", "Monitoring code: 08000", "", "0 mA"]]
    blank_lead = [["", "", "", ""]] + meas_hdr
    k, r = symptom_table_kind(blank_lead)
    assert (k, r) == ("measurement", "blank_leading_row(+1)"),         f"blank-leading-row measurement table not admitted: {(k, r)}"

    frag = [["EPC Current", "Monitoring code: 08000", "", "0 mA"]]
    PREV_B = [157.4, 197.4, 264.0, 330.7, 397.3]     # observed predecessor
    FRAG_B = [143.3, 183.2, 249.9, 316.5, 383.1]     # same table, 14.20pt over
    CAUSE_B = [42.5, 64.4, 137.4, 389.0, 414.6]      # a cause table's shape
    good_prev = {"kind": "measurement", "cols": 4, "bounds": PREV_B}

    k, r = symptom_table_kind(frag, prev=good_prev, bounds=FRAG_B)
    assert k == "measurement" and r.startswith("proven_continuation"), \
        "a genuine continuation was rejected: %s" % ((k, r),)

    # every rejection case: all four conditions are required
    assert symptom_table_kind(frag, prev={"kind": "causes", "cols": 4,
                                          "bounds": PREV_B},
                              bounds=FRAG_B)[0] == "unknown", \
        "continuation admitted whose predecessor was not a measurement table"
    assert symptom_table_kind(frag, prev={"kind": "measurement", "cols": 5,
                                          "bounds": PREV_B},
                              bounds=FRAG_B)[0] == "unknown", \
        "continuation admitted that had lost a column"

    # a predecessor that is a header-only stub: its "Measurement position"
    # heading is one merged cell, so it draws 4 rules where its body has 5.
    # Same table, and the outer edges and every surviving rule still line up.
    STUB_B = [143.3, 183.2, 316.5, 383.1]
    assert geometry_matches(FRAG_B, STUB_B)[0], \
        "a merged heading cell was mistaken for a different table"
    assert symptom_table_kind(frag, prev={"kind": "measurement", "cols": 3,
                                          "bounds": STUB_B},
                              bounds=FRAG_B)[0] == "measurement", \
        "continuation of a merged-header stub rejected on column count"
    # ... but the subset must be genuine: shifting one rule breaks it
    assert not geometry_matches(FRAG_B, [143.3, 183.2, 300.0, 383.1])[0], \
        "a boundary the predecessor does not draw was accepted anyway"
    assert symptom_table_kind(frag, prev=None, bounds=FRAG_B)[0] == "unknown", \
        "continuation admitted with no predecessor at all"

    # condition 3 -- a fragment carrying its own header is a NEW table
    own_hdr = [["Item", "Measurement position", "", "Standard value"],
               ["EPC Current", "Monitoring code: 08000", "", "0 mA"]]
    assert has_own_header(own_hdr), "own-header detector missed an Item row"
    assert not has_own_header(frag), "own-header detector fired on a fragment"
    assert not symptom_table_kind(
        own_hdr, prev=good_prev, bounds=FRAG_B)[1].startswith(
            "proven_continuation"), \
        "a fragment carrying its own header was admitted as a continuation"

    # condition 4 -- geometry, compared after the margin shift is removed
    ok, d = geometry_matches(FRAG_B, PREV_B)
    assert ok and d <= GEOMETRY_TOLERANCE_PT, \
        "observed continuation geometry delta %s exceeds the tolerance" % d
    assert not geometry_matches(CAUSE_B, PREV_B)[0], \
        "tolerance is wide enough to admit a genuinely different table"
    assert symptom_table_kind(frag, prev=good_prev,
                              bounds=CAUSE_B)[0] == "unknown", \
        "a fragment with mismatched column geometry was admitted"

    cause_t = [["No.", "Cause", "Point to check", "Remedy"]]
    assert symptom_table_kind(cause_t)[0] == "causes",         "symptom classifier mangles a cause table"
    assert symptom_table_kind([["", "", ""]] + cause_t)[0] == "causes",         "blank-led cause table misread as a measurement table"
    s40 = [["Details of failure", "A high voltage occurs..."]]
    assert symptom_table_kind(s40)[0] == "header_a",         "symptom classifier mangles a Section 40 header"

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
