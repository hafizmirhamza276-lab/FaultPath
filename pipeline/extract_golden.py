#!/usr/bin/env python3
"""
extract_golden.py
Builds the GOLDEN GROUND-TRUTH dataset of failure-code diagnostic trees
from the Komatsu PC200-10M0 Shop Manual (SEN06867-13).

Deterministic parsing only - NO LLM is used here. That is the point:
this file is the reference the LLM pipeline gets scored against.
"""
import json, re, sys, os, traceback, subprocess
from collections import OrderedDict

import pdfplumber
import fitz  # PyMuPDF, for bookmarks + manual page numbers

# Paths resolve from this file's own location (pipeline/ -> repo root), so the
# script runs correctly from any working directory. Env vars override the
# defaults; an explicit argv path overrides both.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# RESOLVED AT CALL TIME, NOT IMPORT TIME. Both halves of this, deliberately.
#
# These four lines used to run at module scope, and the cost was paid twice:
#
#   argv  `python core/orchestrator.py --force` made argv[1] "--force", so any
#         process with arguments of its own that imported this module resolved
#         the manual to "--force" and exited. That halted a full run once and
#         was WORKED AROUND in core/symptom_match.py by blanking sys.argv
#         across the import -- which is why it came back.
#   guard  importing clean() -- a pure string function that touches no PDF --
#         required the PDF to be present. The agent must run offline with no
#         network, no key and no manual, and the symptom matcher needs clean()
#         on both sides of every comparison.
#
# Nothing about WHAT is extracted changes; only WHEN the guard fires. Proven by
# diff: re-extraction after this change is byte-identical to the committed
# golden/ across all 174 files and index.json.
#
# A script still refuses to run without the manual -- require_pdf() is the
# first line of main(). The guard did not weaken, it moved.

def source_pdf() -> str:
    return sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KOMATSU_PDF", "")


def out_dir() -> str:
    return sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
        "GOLD_DIR", os.path.join(REPO_ROOT, "golden"))


def require_pdf() -> str:
    """The guard, unchanged, at the point of USE rather than the point of import."""
    p = source_pdf()
    if not p or not os.path.isfile(p):
        sys.exit(
            "ERROR: source manual PDF not found.\n"
            "  Pass the path as the first argument, or set KOMATSU_PDF.\n"
            f"  Tried: {p or '<unset>'}\n"
            "  The manual is Komatsu copyrighted material and is deliberately not\n"
            "  stored in this repo -- see CLAUDE.md."
        )
    return p

# Bumped when the record shape changes. v2 adds span-level provenance
# (manual_page, pdf_page, table_index, row_index) and fact_id to every step and
# measurement. Purely additive -- no extracted value moves.
SCHEMA_VERSION = 2

MANUAL_ID = "SEN06867-13"
REVISION = 13
MODEL = "PC200-10M0"
SERIAL_RANGE = "700001 and up"

# ----------------------------------------------------------------- helpers

HEADER_LABELS_A = {
    "details of failure": "detail_of_failure",
    "detail of failure": "detail_of_failure",
    "action level": "action_level",
    "action of controller": "controller_action",
    "phenomenon on machine": "machine_effect",
    "associated information": "related_information",
    "related information": "related_information",
}


def clean(s):
    """Normalise cell text: kill line-break hyphenation and stray whitespace."""
    if not s:
        return ""
    s = s.replace("\n", " ")
    # de-hyphenate words broken across lines: "addi- tional" -> "additional"
    s = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def norm_label(s):
    return re.sub(r"[^a-z ]", "", clean(s).lower()).strip()


def manual_page(page):
    """Return the printed page number (e.g. '40-181'), not the PDF page index."""
    txt = page.extract_text() or ""
    lines = [l.strip() for l in txt.split("\n") if l.strip()]
    for l in lines[-3:]:
        m = re.search(r"\b(\d{2}-\d{1,4})\b", l)
        if m:
            return m.group(1)
    return None


# ----------------------------------------------------------- table typing

def table_kind(tbl):
    if not tbl or not tbl[0]:
        return "unknown"
    first = norm_label(tbl[0][0] or "")
    row0 = [norm_label(c or "") for c in tbl[0]]
    if first == "no":
        return "causes"
    if first == "item":
        return "measurement"
    if first == "action level" and "failure code" in row0:
        return "header_b"
    if first in HEADER_LABELS_A:
        return "header_a"
    return "unknown"


# --------------------------------------------------------------- parsers

def parse_header_a(tbl):
    """Format A: 2-column label/value block."""
    out = {}
    for row in tbl:
        if len(row) < 2:
            continue
        key = HEADER_LABELS_A.get(norm_label(row[0] or ""))
        if key:
            out[key] = clean(row[1])
    return out


def parse_header_b(tbl):
    """Format B (Cummins CAxxx): 4-column block, code+title on the first two rows."""
    out = {}
    if len(tbl) >= 1 and len(tbl[0]) >= 4:
        out["title"] = clean(tbl[0][3])
    if len(tbl) >= 2 and len(tbl[1]) >= 2:
        out["action_level"] = clean(tbl[1][0])
    for row in tbl[2:]:
        key = HEADER_LABELS_A.get(norm_label(row[0] or ""))
        if key:
            out[key] = clean(row[1])
    return out


CRITERIA_RE = re.compile(
    r"(Max\.?\s*[\d.]+\s*\S*|Min\.?\s*[\d.]+\s*\S*|Approx\.?\s*[\d.]+\s*\S*|"
    r"[\d.]+\s*to\s*[\d.]+\s*\S*)",
    re.I,
)


# A cell holds a criterion if it states a bound, a range, a number with a unit,
# or a continuity verdict.
CRIT_VALUE = re.compile(
    r"(?:max|min|approx)\.?\s*\d"
    r"|\d+(?:\.\d+)?\s*to\s*\d"
    r"|\d+(?:\.\d+)?\s*(?:k|m|\u00b5)?(?:\u03a9|\u03c9|\u2126|v|a|kpa|mpa|rpm|ma|hz|%|min|s)\b"
    r"|^(?:no\s+)?continuity",
    re.I)

GLUED_STEP = re.compile(r"^(?=[^\d]*\d)(.*?)\b(\d{1,2})\b(.*)$")


def _fuse_split_decimals(cells):
    """Rejoin a decimal broken across a cell boundary, in place in the cell list:
    ['Sensor output 0', '.2 to 4.6V'] -> ['Sensor output 0.2 to 4.6V'].

    This must run BEFORE the criterion cell is chosen. '.2 to 4.6V' matches
    CRIT_VALUE on its own (the range regex sees '2 to 4'), so a right-to-left
    scan stops on the fragment and strands the integer part in the measuring
    point: 'Between ECM (25) and (47) / Sensor output 0'.
    """
    out = []
    for c in cells:
        if out and out[-1] and re.search(r"\d$", out[-1]) and re.match(r"^\.\d", c or ""):
            out[-1] += c
        else:
            out.append(c)
    return out


def _join_criteria(parts):
    """Rejoin criteria fragments. A fragment starting with '.' is a decimal that
    the table extractor split across a cell boundary: '0' + '.2 to 4.6V'."""
    out = ""
    for p in parts:
        if not p:
            continue
        if out and (p.startswith(".") or out.endswith(".")):
            out += p
        else:
            out = (out + " " + p).strip()
    return out


def parse_causes(rows, fmt):
    """
    Turn cause-table rows into ordered steps.
    Format A rows carry an explicit YES/NO branch; format B rows are sequential
    with measurement sub-rows embedded underneath the parent step.

    `rows` is [(cells, provenance)]. Provenance is captured from the page the
    row was actually read from and carried through untouched -- never inferred
    afterwards from the code's page range. A code spans up to 11 pages, so a
    citation to the code's first page is not a citation to the measurement.
    """
    steps = []
    cur = None
    for row, prov in rows[1:]:  # skip header row
        cells = [clean(c) for c in row]
        no = cells[0] if cells else ""

        step_no, glued_cause = None, None
        if re.fullmatch(r"\d+", no):
            step_no = int(no)
        elif no and cells[2:3] and cells[2]:
            # Column-split corruption: the step number is rendered inside the
            # cause column, e.g. 'Defectiv pressure 6 voltage controlle'.
            # Recover the number and keep the mangled text rather than dropping
            # the whole step.
            m = GLUED_STEP.match(no)
            if m and steps and int(m.group(2)) == steps[-1]["step"] + 1:
                step_no = int(m.group(2))
                glued_cause = clean(m.group(1) + " " + m.group(3))

        if step_no is not None:
            cause = cells[1] if len(cells) > 1 else ""
            if glued_cause:
                cause = clean(glued_cause + " " + cause)
            cur = {
                "step": step_no,
                "cause": cause,
                "procedure": cells[2] if len(cells) > 2 else "",
                "branches": {},
                # PER-BRANCH provenance, captured here rather than inferred
                # later. A step that straddles a page break has its YES outcome
                # printed on one page and its NO on the next; both taking the
                # step's page sends one citation to a page its text is not on.
                # This is the same defect as citing a code's first page for a
                # measurement eleven pages later, one level down.
                #
                # Nothing about WHAT is extracted changes -- `prov` is the row
                # provenance already in hand at the moment the branch is read,
                # so this records a value that was being discarded.
                "branch_provenance": {},
                "measurements": [],
                "provenance": dict(prov),
            }
            if glued_cause:
                cur["extraction_warning"] = "column_split_recovered"
            steps.append(cur)
            for i, c in enumerate(cells[3:], 3):
                if c in ("YES", "NO"):
                    nxt = next((cells[j] for j in range(i + 1, len(cells)) if cells[j]), "")
                    if nxt:
                        cur["branches"][c] = nxt
                        cur["branch_provenance"][c] = dict(prov)
                    break
            continue

        if cur is None:
            continue

        # ---- continuation row -------------------------------------------
        # Repair decimals split across a cell boundary before anything reads
        # the cells positionally, so the criterion is a single cell by the time
        # crit_idx is chosen.
        cells = _fuse_split_decimals(cells)

        # Section 40 uses three different cause-table layouts and the YES/NO
        # column sits in a different position in each. Locate it by content
        # rather than by index.
        used = set()
        for i, c in enumerate(cells):
            if c in ("YES", "NO"):
                nxt = next((cells[j] for j in range(i + 1, len(cells)) if cells[j]), "")
                if nxt:
                    cur["branches"][c] = nxt
                    # the CONTINUATION row's page, which is the whole point:
                    # this row may be on the far side of a page break from the
                    # step row that opened the step
                    cur["branch_provenance"][c] = dict(prov)
                    used |= {i, i + 1}
                break

        # A measurement sub-row is identified by containing a criterion value
        # (Max./Min./Approx./a range/a number with a unit), NOT by guessing the
        # quantity's name. Quantity labels vary widely - 'Resistance' and
        # 'Voltage' dominate, but fuel-system checks use labels like
        # 'Discharged volume from supply pump'.
        crit_idx = None
        for i in range(len(cells) - 1, -1, -1):
            if i in used or not cells[i]:
                continue
            if CRIT_VALUE.search(cells[i]):
                crit_idx = i
                break

        if crit_idx is not None:
            head = [x for j, x in enumerate(cells[:crit_idx])
                    if x and j not in used
                    and norm_label(x) not in ("item", "standard value", "yes", "no")]
            criteria = _join_criteria(cells[crit_idx:])
            if len(head) >= 2:
                qty, point = head[0], " / ".join(head[1:])
            elif len(head) == 1:
                # quantity carried over from the row above (merged cell)
                prev = cur["measurements"][-1]["quantity"] if cur["measurements"] else ""
                qty, point = prev, head[0]
            else:
                qty = point = ""
            if point:
                cur["measurements"].append(
                    {"quantity": qty, "point": point, "criteria": criteria,
                     "provenance": dict(prov)}
                )

        if not cur["branches"] and not cur["measurements"]:
            # spill-over text for the same step
            extra = " ".join(x for x in cells[1:] if x)
            if extra:
                cur["procedure"] = (cur["procedure"] + " " + extra).strip()

    # Pointer-only codes: the whole "cause table" is a single unnumbered row that
    # just redirects elsewhere, e.g. "Do the troubleshooting for failure code
    # [CA187]." These break naive RAG, so capture them explicitly.
    if not steps:
        for row, prov in rows[1:]:
            text = " ".join(clean(c) for c in row if clean(c)).strip()
            if text:
                steps.append({"step": 1, "cause": "Redirect", "procedure": text,
                              "branches": {}, "measurements": [],
                              "provenance": dict(prov), "redirect": True})
                break
    return steps


def parse_measurement_table(tbl, prov_base):
    """Standalone 'Item | Measuring point | Standard value' block.

    `prov_base` carries the page and table this block was read from; row_index
    is filled in per row so each value points at its own line.
    """
    out = []
    last_q = ""
    for ri, row in enumerate(tbl[1:], 1):
        cells = [clean(c) for c in row]
        if not any(cells):
            continue
        prov = dict(prov_base, row_index=ri)
        q = cells[0] or last_q
        last_q = q
        if len(cells) == 3:
            out.append({"quantity": q, "point": cells[1], "criteria": cells[2],
                        "provenance": prov})
        elif len(cells) >= 4:
            point = " / ".join(x for x in cells[1:-1] if x)
            out.append({"quantity": q, "point": point, "criteria": cells[-1],
                        "provenance": prov})
    return out


# ------------------------------------------------------------- extraction

def extract_code(pdf, code, start, end):
    rec = OrderedDict(
        schema_version=SCHEMA_VERSION,
        manual_id=MANUAL_ID,
        revision=REVISION,
        model=MODEL,
        serial_range=SERIAL_RANGE,
        section="40",
        code=code,
        title=None,
        action_level=None,
        detail_of_failure=None,
        controller_action=None,
        machine_effect=None,
        related_information=None,
        format=None,
        pdf_pages=[start, end],
        manual_pages=[],
        steps=[],
        standalone_measurements=[],
        refs_failure_codes=[],
        refs_monitoring_codes=[],
        connectors=[],
        default_conclusion=None,
        is_pointer_only=False,
    )

    cause_rows, fmt = [], None
    for pno in range(start, end + 1):
        page = pdf.pages[pno - 1]
        mp = manual_page(page)
        if mp:
            rec["manual_pages"].append(mp)

        for ti, tbl in enumerate(page.extract_tables()):
            # Captured here, from the page actually being read. Anything derived
            # later would be a guess dressed as a citation.
            prov = {"manual_page": mp, "pdf_page": pno, "table_index": ti}
            kind = table_kind(tbl)
            if kind == "header_a":
                fmt = fmt or "A"
                rec.update({k: v for k, v in parse_header_a(tbl).items() if v})
                rec.setdefault("_header_prov", dict(prov, row_index=0))
            elif kind == "header_b":
                fmt = fmt or "B"
                rec.update({k: v for k, v in parse_header_b(tbl).items() if v})
                rec.setdefault("_header_prov", dict(prov, row_index=0))
            elif kind == "causes":
                cause_rows.append((tbl, prov))
            elif kind == "measurement":
                rec["standalone_measurements"] += parse_measurement_table(tbl, prov)

    rec["format"] = fmt or "B"

    # merge multi-page cause tables (each repeats its header row). row_index
    # stays relative to the table the row came from, so a provenance record
    # points at a real row of a real table on a real page.
    merged = []
    for i, (t, prov) in enumerate(cause_rows):
        body = list(enumerate(t))
        if i:
            body = body[1:]
        for ri, row in body:
            merged.append((row, dict(prov, row_index=ri)))
    if merged:
        rec["steps"] = parse_causes(merged, rec["format"])

    # ---- title fallback for format A (comes from the bookmark heading)
    if not rec["title"]:
        txt = pdf.pages[start - 1].extract_text() or ""
        for line in txt.split("\n"):
            if f"[{code}]" in line and "Failure Code" in line:
                continue
        # detail_of_failure is initialised to None and rec.update filters falsy
        # values, so .get's default never applies -- coerce explicitly.
        rec["title"] = (rec.get("detail_of_failure") or "")[:120] or None

    # ---- fact identity
    # <code>:<step>:<kind>:<index>. Derived from position, so it is stable
    # across rebuilds unless the fact itself moves. Standalone measurements sit
    # at step 0 -- they belong to the code, not to any one check.
    rec["header_provenance"] = rec.pop("_header_prov", None)
    for i, m in enumerate(rec["standalone_measurements"]):
        m["fact_id"] = f"{code}:0:meas:{i}"
    for st in rec["steps"]:
        st["fact_id"] = f"{code}:{st['step']}:step:0"
        for i, m in enumerate(st["measurements"]):
            m["fact_id"] = f"{code}:{st['step']}:meas:{i}"
        st["branch_fact_ids"] = {
            br: f"{code}:{st['step']}:branch:{i}"
            for i, br in enumerate(sorted(st.get("branches") or {}))
        }

    # ---- cross-references, monitoring codes, connectors
    blob = json.dumps(rec)
    rec["refs_failure_codes"] = sorted(
        {c for c in re.findall(r"\[([A-Z0-9@#]{4,7})\]", blob) if c != code}
    )
    rec["refs_monitoring_codes"] = sorted(set(re.findall(r"[Cc]ode:?\s*(\d{5})", blob)))
    rec["connectors"] = sorted(
        set(re.findall(r"\b([A-Z]{2,6}\d{0,3})\s*\((?:male|female|\d{1,3})", blob))
    )

    # ---- pointer-only detection
    rec["is_pointer_only"] = bool(rec["steps"]) and all(
        st.get("redirect") for st in rec["steps"]
    ) or (
        len(rec["steps"]) <= 2
        and all(
            re.search(r"[Dd]o the troubleshooting for failure code", st["procedure"] + st["cause"])
            or "other failure codes" in (st["procedure"] + st["cause"])
            for st in rec["steps"]
        )
        and bool(rec["steps"])
    )

    # ---- default conclusion = last step when it is a "controller defective" catch-all
    if rec["steps"]:
        last = rec["steps"][-1]
        text = (last["cause"] + " " + last["procedure"]).lower()
        if "no failure is found" in text or "defective" in last["cause"].lower():
            rec["default_conclusion"] = last["cause"] or last["procedure"][:200]

    return rec


def parse_code_table(pdf, start=655, end=662):
    """The 'Failure Code Table' lists every code with its canonical screen name,
    controller, action level and system category. Used for enrichment AND as an
    independent cross-check of the per-code detail pages."""
    canon = {}
    for pno in range(start, end + 1):
        for tbl in pdf.pages[pno - 1].extract_tables():
            if not tbl or norm_label(tbl[0][0] or "") != "failure code":
                continue
            for row in tbl[1:]:
                c = [clean(x) for x in row]
                if not c or not re.fullmatch(r"[A-Z0-9@#]{4,7}", c[0]):
                    continue
                canon[c[0]] = {
                    "title": c[1] if len(c) > 1 else None,
                    "controller": c[2] if len(c) > 2 else None,
                    "action_level": (c[3] if len(c) > 3 and c[3] not in ("", "-") else None),
                    "system_category": c[4] if len(c) > 4 else None,
                }
    return canon


def main():
    PDF = require_pdf()
    OUT = out_dir()
    os.makedirs(f"{OUT}/failure_codes", exist_ok=True)

    doc = fitz.open(PDF)
    toc = doc.get_toc()
    entries = []
    fc = [t for t in toc if t[0] == 3 and re.search(r"Failure Code \[(.+?)\]", t[1])]
    for i, t in enumerate(fc):
        code = re.search(r"Failure Code \[(.+?)\]", t[1]).group(1)
        start = t[2]
        end = fc[i + 1][2] - 1 if i + 1 < len(fc) else 1152
        entries.append((code, start, end))
    doc.close()

    index, failures, conflicts = [], [], []
    with pdfplumber.open(PDF) as pdf:
        canon = parse_code_table(pdf)
        print(f"canonical code table: {len(canon)} entries")
        for n, (code, s, e) in enumerate(entries, 1):
            try:
                rec = extract_code(pdf, code, s, e)
                cn = canon.get(code)
                if cn:
                    rec["title"] = cn["title"] or rec["title"]
                    rec["controller"] = cn["controller"]
                    rec["system_category"] = cn["system_category"]
                    if cn["action_level"] and rec["action_level"] and \
                       cn["action_level"] != rec["action_level"]:
                        conflicts.append((code, cn["action_level"], rec["action_level"]))
                    rec["action_level"] = rec["action_level"] or cn["action_level"]
                    rec["in_code_table"] = True
                else:
                    rec["in_code_table"] = False
            except Exception as exc:  # keep going; report at the end
                # A swallowed exception here drops the code from the dataset
                # entirely. Print the traceback so a silent shrink is loud.
                failures.append((code, repr(exc)))
                print(f"\nFAIL {code}: extraction raised, code dropped", file=sys.stderr)
                traceback.print_exc()
                continue
            with open(f"{OUT}/failure_codes/{code}.json", "w", encoding="utf-8") as f:
                json.dump(rec, f, indent=2, ensure_ascii=False)
            index.append(
                {
                    "code": code,
                    "format": rec["format"],
                    "action_level": rec["action_level"],
                    "manual_page": rec["manual_pages"][0] if rec["manual_pages"] else None,
                    "n_steps": len(rec["steps"]),
                    "n_measurements": len(rec["standalone_measurements"])
                    + sum(len(st["measurements"]) for st in rec["steps"]),
                    "refs": rec["refs_failure_codes"],
                }
            )
            if n % 25 == 0:
                print(f"  ...{n}/{len(entries)}", flush=True)

    with open(f"{OUT}/index.json", "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)

    print(f"\nextracted {len(index)} codes, {len(failures)} failed")
    for c, e in failures[:10]:
        print("  FAIL", c, e)
    print(f"action-level conflicts (table vs detail page): {len(conflicts)}")
    for c in conflicts[:10]:
        print("  CONFLICT", c)

    run_regression_guard()


def run_regression_guard():
    """Verify the freshly written dataset before anyone treats it as ground truth.

    Regeneration is only valid together with this check. If a count moves, the
    parser changed behaviour -- that is a regression to investigate, not a new
    baseline to accept.
    """
    guard = os.path.join(REPO_ROOT, "tests", "test_extraction.py")
    if not os.path.isfile(guard):
        print(f"\nWARNING: regression guard not found at {guard}; dataset UNVERIFIED")
        return

    print("\n" + "=" * 60)
    print("running regression guard")
    rc = subprocess.call([sys.executable, guard],
                         env={**os.environ, "GOLD_DIR": out_dir()})
    if rc != 0:
        sys.exit(
            "\nREGRESSION GUARD FAILED -- the regenerated dataset is NOT a valid\n"
            "baseline. Do not commit it. See the failures above."
        )


if __name__ == "__main__":
    main()
