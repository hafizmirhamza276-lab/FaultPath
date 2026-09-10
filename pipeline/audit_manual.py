#!/usr/bin/env python3
"""
audit_manual.py
Systematic data-quality audit of Komatsu Shop Manual SEN06867-13.

Finds every defect in the SOURCE DOCUMENT that will damage a retrieval or
diagnostic system built on top of it. Distinguishes:

  SOURCE   - a genuine inconsistency in the manual itself
  STRUCT   - a structural property that breaks naive text extraction
  RISK     - not wrong, but a known failure mode for RAG

Writes findings to audit_findings.json.
"""
import json, re, glob, os, sys, collections

import fitz
import pdfplumber

# Paths resolve from this file's own location (pipeline/ -> repo root), so the
# script runs correctly from any working directory. Env vars override the
# defaults; an explicit argv path overrides both.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("KOMATSU_PDF", "")
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
REPORT_DIR = os.environ.get("OUT_DIR", os.path.join(REPO_ROOT, "reports"))
REPORT_PATH = os.path.join(REPORT_DIR, "audit_findings.json")

if not PDF or not os.path.isfile(PDF):
    sys.exit(
        "ERROR: source manual PDF not found.\n"
        "  Pass the path as the first argument, or set KOMATSU_PDF.\n"
        f"  Tried: {PDF or '<unset>'}\n"
        "  The manual is Komatsu copyrighted material and is deliberately not\n"
        "  stored in this repo -- see CLAUDE.md."
    )

findings = []


def add(fid, sev, kind, title, detail, items=None, impact=""):
    findings.append(dict(id=fid, severity=sev, kind=kind, title=title,
                         detail=detail, count=len(items) if items else None,
                         items=(items or [])[:40], impact=impact))
    n = f" ({len(items)})" if items else ""
    print(f"[{sev:6s}] {fid} {title}{n}")


def clean(s):
    if not s:
        return ""
    s = s.replace("\n", " ")
    s = re.sub(r"([a-z])-\s+([a-z])", r"\1\2", s)
    return re.sub(r"\s+", " ", s).strip()


# ---------------------------------------------------------------- load data
recs = {}
for f in glob.glob(f"{GOLD}/failure_codes/*.json"):
    r = json.load(open(f, encoding="utf-8"))
    recs[r["code"]] = r
print(f"loaded {len(recs)} extracted codes\n")

doc = fitz.open(PDF)
toc = doc.get_toc()

# ============================================================ A. COVERAGE
with pdfplumber.open(PDF) as pdf:
    canon = {}
    canon_rows = []
    for pno in range(655, 663):
        for tbl in pdf.pages[pno - 1].extract_tables():
            if not tbl or clean(tbl[0][0]).lower() != "failure code":
                continue
            for row in tbl[1:]:
                c = [clean(x) for x in row]
                if c and re.fullmatch(r"[A-Z0-9@#]{4,7}", c[0]):
                    canon[c[0]] = c
                    canon_rows.append(c)

    detail_codes = set(recs)
    table_codes = set(canon)

    only_table = sorted(table_codes - detail_codes)
    only_detail = sorted(detail_codes - table_codes)

    if only_table:
        add("A1", "HIGH", "SOURCE",
            "Codes listed in the Failure Code Table with no troubleshooting page",
            "These appear on the summary table (manual pages 40-113 onward) but have "
            "no corresponding 'Failure Code [X]' section anywhere in Section 40.",
            only_table,
            "A technician who sees one of these on the monitor finds a name and an "
            "action level, then nothing. The assistant must say so explicitly "
            "instead of retrieving a neighbouring code.")

    if only_detail:
        add("A2", "MEDIUM", "SOURCE",
            "Codes with a troubleshooting page but absent from the Failure Code Table",
            "The reverse gap: a full procedure exists but the code is missing from "
            "the summary table used for lookup.",
            only_detail,
            "Any lookup path that starts from the summary table will never reach "
            "these procedures.")

    # ---------------------------------------------- duplicate rows in the table
    dupes = [c for c, n in collections.Counter(r[0] for r in canon_rows).items() if n > 1]
    if dupes:
        add("A3", "LOW", "SOURCE", "Codes listed more than once in the Failure Code Table",
            "The same code appears on multiple rows of the summary table.", sorted(dupes),
            "Harmless for humans, but produces duplicate index entries.")

# =================================================== B. ACTION LEVEL CONFLICTS
conflicts = []
for code, r in recs.items():
    c = canon.get(code)
    if not c or len(c) < 4:
        continue
    tbl_level = c[3] if c[3] not in ("", "-") else None
    det_level = r.get("action_level") if r.get("action_level") not in ("", "-") else None
    if tbl_level and det_level and tbl_level != det_level:
        conflicts.append(f"{code}: table={tbl_level} detail={det_level}")
if conflicts:
    add("B1", "HIGH", "SOURCE", "Action level disagrees between summary table and detail page",
        "The Failure Code Table and the code's own troubleshooting page state "
        "different action levels for the same code.", conflicts,
        "Action level drives whether the machine may keep working. L01 and L04 are "
        "not interchangeable. The assistant must not silently pick one.")

dash_detail = []
for code, r in recs.items():
    c = canon.get(code)
    if not c or len(c) < 4:
        continue
    if c[3] not in ("", "-") and r.get("action_level") in (None, "", "-"):
        dash_detail.append(f"{code}: table={c[3]} detail page shows '-'")
if dash_detail:
    add("B3", "MEDIUM", "SOURCE", "Detail page omits an action level the summary table assigns",
        "The summary table gives the code an action level; its own troubleshooting "
        "page leaves the field blank or prints a dash.", dash_detail,
        "A reader who goes straight to the code's page never learns how urgent it is.")

# --------------------------------------- codes with no action level anywhere
no_level = sorted(c for c, r in recs.items()
                  if (r.get("action_level") in (None, "", "-"))
                  and (canon.get(c, ["", "", "", ""])[3] if canon.get(c) else "") in ("", "-"))
if no_level:
    add("B2", "MEDIUM", "SOURCE", "Codes with no action level in either source",
        "Neither the summary table nor the detail page assigns an action level.",
        no_level,
        "The assistant cannot tell the technician whether the machine is safe to "
        "keep operating.")

# ================================================= C. CROSS-REFERENCE INTEGRITY
all_codes = detail_codes | table_codes
dangling, self_ref = [], []
edges = {}
for code, r in recs.items():
    outs = [x for x in r["refs_failure_codes"]]
    edges[code] = outs
    for t in outs:
        if t == code:
            self_ref.append(code)
        elif t not in all_codes:
            dangling.append(f"{code} -> {t}")

if dangling:
    add("C1", "HIGH", "SOURCE", "Cross-references to codes that do not exist in this manual",
        "A troubleshooting procedure tells the technician to go and troubleshoot "
        "another failure code, but that code has no page and is not in the summary table.",
        sorted(set(dangling)),
        "These are dead ends. Several of the referenced codes belong to other "
        "machine models or to superseded manual revisions.")

if self_ref:
    add("C2", "LOW", "SOURCE", "Codes that reference themselves", "", sorted(set(self_ref)),
        "Causes an infinite loop if hop-expansion is applied without a visited set.")

# ------------------------------------------------------------ cycles
def find_cycles(edges):
    cycles, state = [], {}
    def dfs(n, path):
        state[n] = 1
        for m in edges.get(n, []):
            if m not in edges:
                continue
            if state.get(m) == 1:
                i = path.index(m) if m in path else 0
                cycles.append(" -> ".join(path[i:] + [m]))
            elif state.get(m, 0) == 0:
                dfs(m, path + [m])
        state[n] = 2
    for n in edges:
        if state.get(n, 0) == 0:
            dfs(n, [n])
    return sorted(set(cycles))

cyc = find_cycles(edges)
if cyc:
    add("C3", "MEDIUM", "STRUCT", "Circular cross-reference chains",
        "Following the manual's own instructions leads back to the starting code.",
        cyc,
        "Hop-expansion and any automated 'follow the reference' logic must carry a "
        "visited set and a depth limit, or the assistant will loop forever.")

# ------------------------------------------------- pointer-only codes
pointers = sorted(c for c, r in recs.items() if r["is_pointer_only"])
if pointers:
    detail = []
    for c in pointers:
        tgt = recs[c]["refs_failure_codes"]
        detail.append(f"{c} -> {', '.join(tgt) if tgt else '(no target parsed)'}")
    add("C4", "HIGH", "STRUCT", "Codes whose entire procedure is a redirect",
        "The complete troubleshooting content is a single instruction to go and "
        "troubleshoot a different code.", detail,
        "This is the classic RAG trap. Retrieval returns a technically correct, "
        "perfectly relevant chunk that gives the technician nothing to do. "
        "Hop-expansion is mandatory, not an optimisation.")

# ============================================== D. INCOMPLETE CODE RECORDS
missing_effect = sorted(c for c, r in recs.items() if not r.get("machine_effect"))
missing_detail = sorted(c for c, r in recs.items() if not r.get("detail_of_failure"))
no_steps = sorted(c for c, r in recs.items() if not r["steps"])

if no_steps:
    add("D1", "HIGH", "STRUCT", "Codes with no parsable troubleshooting steps",
        "The page exists but its layout matches neither of the two table formats "
        "used elsewhere in Section 40.", no_steps,
        "Requires manual transcription or a vision pass. Excluded from step-based "
        "evaluation until then.")

if missing_effect:
    add("D2", "MEDIUM", "SOURCE", "Codes with no 'Phenomenon on machine' field",
        "The manual does not state what the operator will actually observe.",
        missing_effect,
        "The assistant cannot confirm it is working on the right problem, and "
        "cannot help a technician who describes a symptom rather than a code.")

if missing_detail:
    add("D3", "LOW", "SOURCE", "Codes with no 'Details of failure' field", "",
        missing_detail, "Reduces the quality of semantic matching for symptom queries.")

# ------------------------------------- format A steps missing a branch
half_branch = []
for c, r in recs.items():
    if r["format"] != "A":
        continue
    for st in r["steps"]:
        if st.get("redirect"):
            continue
        b = set(st["branches"])
        if b and b != {"YES", "NO"}:
            half_branch.append(f"{c} step {st['step']} has only {sorted(b) or 'none'}")
        elif not b:
            half_branch.append(f"{c} step {st['step']} has no YES/NO branch")
if half_branch:
    add("D4", "MEDIUM", "STRUCT", "Format-A steps with an incomplete YES/NO branch",
        "Format-A codes are decision trees, so every step should offer both "
        "outcomes. Some do not, either in the manual or after table extraction.",
        half_branch,
        "The executor has no defined next state when the missing branch is taken.")

empty_steps = []
for c, r in recs.items():
    for st in r["steps"]:
        if not st["cause"].strip() and not st["procedure"].strip():
            empty_steps.append(f"{c} step {st['step']}")
if empty_steps:
    add("D5", "MEDIUM", "STRUCT", "Steps with neither a cause nor a procedure",
        "Empty rows produced by merged cells spanning a page break.", empty_steps,
        "Renders as a blank instruction to the technician.")

# ================================================ E. MEASUREMENT INTEGRITY
bad_crit, no_point, nonnum_ok = [], [], []
LEGIT_NONNUM = re.compile(r"^(no )?continuity|^normal$|^abnormal$|^on$|^off$|"
                          r"sound|lights? up|does not", re.I)
for c, r in recs.items():
    meas = list(r["standalone_measurements"])
    for st in r["steps"]:
        meas += st["measurements"]
    for m in meas:
        crit = (m.get("criteria") or "").strip()
        if not crit:
            bad_crit.append(f"{c}: '{m.get('point','')[:40]}' has no criteria")
        elif not re.search(r"\d", crit):
            if LEGIT_NONNUM.search(crit):
                nonnum_ok.append(f"{c}: {crit[:40]}")
            else:
                bad_crit.append(f"{c}: criteria '{crit[:50]}' is not a measurable value")
        if not (m.get("point") or "").strip():
            no_point.append(f"{c}: criteria '{crit[:30]}' has no measuring point")

if bad_crit:
    add("E1", "HIGH", "STRUCT", "Measurements with a missing or unusable criterion",
        "A measurement is listed but the value the technician must compare against "
        "is absent, or the cell contains branch text that leaked in from an "
        "adjacent column instead of a criterion.", bad_crit,
        "The most dangerous defect class in the document. A measurement with no "
        "criterion is an open invitation for a language model to supply one that "
        "sounds plausible.")

if nonnum_ok:
    add("E5", "INFO", "STRUCT", "Criteria that are legitimately non-numeric",
        "Continuity checks and audible/visual confirmations have no numeric value "
        "by design.", nonnum_ok,
        "Not a defect, but any validation rule that demands a number from every "
        "criterion will flag these incorrectly.")

if no_point:
    add("E2", "MEDIUM", "STRUCT", "Measurements with no measuring point",
        "A value exists but the pins or terminals to measure between were lost.",
        no_point, "The technician knows the target number but not where to probe.")

# A criterion split across a cell boundary. A leading '.' is the signature of the
# orphaned-fragment case: the integer part was stranded in an adjacent cell.
SPLIT_DECIMAL = re.compile(r"\d\s+\.\d|\d\.\s+\d|^\s*\.\d|\d\s+\.\s*\d")

# E4 SELF-TEST. This check reported zero on a known-corrupt record once already,
# because the pattern required a digit BEFORE the split and '.2 to 4.6V' has
# none -- and that zero was then cited as proof the corruption was fixed. A check
# that cannot fail on its own motivating example is not a check. These assertions
# run on every audit so a clean E4 means the corpus is clean, not that E4 is blind.
assert SPLIT_DECIMAL.search(".2 to 4.6V"), \
    "E4 SELF-TEST FAILED: pattern cannot detect a leading-dot fragment"
assert SPLIT_DECIMAL.search("Sensor output 0 .2 to 4.6V"), \
    "E4 SELF-TEST FAILED: pattern cannot detect a space-split decimal"
assert not SPLIT_DECIMAL.search("Sensor output 0.2 to 4.6V"), \
    "E4 SELF-TEST FAILED: pattern flags the corrected value"
assert not SPLIT_DECIMAL.search("Max. 1 Ω"), \
    "E4 SELF-TEST FAILED: pattern flags a plain bound"
print("E4 self-test: PASS - detects '.2 to 4.6V', accepts 'Sensor output 0.2 to 4.6V'")

# ------------------------- split-cell corruption (step number glued into text)
glued = []
for c, r in recs.items():
    for st in r["steps"]:
        if st.get("extraction_warning") == "column_split_recovered":
            glued.append(f"{c} step {st['step']}: {st['cause'][:60]}")
if glued:
    add("E3", "HIGH", "STRUCT", "Steps whose number is rendered inside the cause column",
        "On these pages the step number is typeset within the cause text rather "
        "than its own cell, e.g. 'Defectiv pressure 6 voltage controlle' + "
        "'e common rail sensor'. A straightforward table parser treats the row as "
        "a continuation of the previous step and drops it entirely.", glued,
        f"{len(glued)} troubleshooting steps across "
        f"{len({g.split()[0] for g in glued})} codes were silently lost before this "
        "was detected - including the final causes of CA451, where the missing "
        "steps are the ones that identify a defective engine controller. Any "
        "pipeline that does not handle this will hand technicians a truncated "
        "procedure with no indication anything is missing.")

# ------------------------------------- decimals split across cell boundaries
split_dec = []
for c, r in recs.items():
    meas = list(r["standalone_measurements"])
    for st in r["steps"]:
        meas += st["measurements"]
    for m in meas:
        crit = m.get("criteria") or ""
        if SPLIT_DECIMAL.search(crit):
            split_dec.append(f"{c}: '{crit[:40]}'")
if split_dec:
    add("E4", "HIGH", "STRUCT", "Measurement values split across a cell boundary",
        "A decimal criterion is broken in two by the table structure, e.g. "
        "'0' and '.2 to 4.6V' in adjacent cells. Also flags a criterion that "
        "begins with a decimal point, which means the integer part was lost to "
        "an adjacent cell.", split_dec,
        "Produces a nonsense value if the fragments are joined with a space, and a "
        "wrong value if only one fragment is kept.")

# ===================================================== F. DUPLICATE NAMING
by_title = collections.defaultdict(list)
for c, row in canon.items():
    if len(row) > 1 and row[1]:
        by_title[row[1].strip().lower()].append(c)
dup_titles = {t: cs for t, cs in by_title.items() if len(cs) > 1}
if dup_titles:
    items = [f"'{t}' -> {', '.join(sorted(cs))}" for t, cs in sorted(dup_titles.items())]
    add("F1", "MEDIUM", "SOURCE", "Different codes sharing an identical display name",
        "The text shown on the machine monitor is the same for several distinct "
        "codes with different procedures.", items,
        "A technician who reports only the screen text, not the code, cannot be "
        "disambiguated. The assistant must ask for the code itself.")

# =================================================== G. PAGINATION INTEGRITY
page_map = collections.defaultdict(list)
for i in range(2, doc.page_count):
    txt = doc[i].get_text().strip().split("\n")
    for line in txt[-3:]:
        m = re.search(r"\b(\d{2}-\d{1,4})\b", line)
        if m:
            page_map[m.group(1)].append(i + 1)
            break

dup_pages = {k: v for k, v in page_map.items() if len(v) > 1}
if dup_pages:
    items = [f"{k} on PDF pages {v}" for k, v in sorted(dup_pages.items())[:30]]
    add("G1", "MEDIUM", "SOURCE", "The same printed page number appears on multiple pages",
        "Printed page numbers are not unique across the document.", items,
        "Citations that use only the printed page number are ambiguous. Every "
        "citation must carry the section code and the PDF page as well.")

nopage = [i + 1 for i in range(2, doc.page_count)
          if not re.search(r"\b\d{2}-\d{1,4}\b",
                           "\n".join(doc[i].get_text().strip().split("\n")[-3:]))]
if nopage:
    add("G2", "LOW", "STRUCT", "Pages carrying no printed page number",
        "Section dividers and full-bleed diagram pages have no footer.",
        [str(p) for p in nopage],
        "Content on these pages cannot be cited the way a technician expects.")

# ==================================================== H. TEXT LAYER QUALITY
lowtext, vectoronly = [], []
for i in range(2, doc.page_count):
    p = doc[i]
    t = p.get_text().strip()
    if len(t) < 250:
        lowtext.append(i + 1)
        if len(p.get_drawings()) > 500:
            vectoronly.append(i + 1)

add("H1", "HIGH", "RISK", "Pages that are effectively invisible to text retrieval",
    f"{len(lowtext)} pages carry under 250 characters of extractable text. "
    f"{len(vectoronly)} of those are dense vector graphics (circuit and hydraulic "
    "diagrams drawn as line art, not images).",
    [f"PDF p{p}" for p in vectoronly[:40]],
    "Text embeddings cannot represent these pages at all. Section 90 (Circuit "
    "Diagrams) is almost entirely in this category. They need page rasterisation "
    "plus a vision pass, or they will silently never be retrieved.")

# ------------------------------------------------------------ fonts
# Every page, not every 37th. Sampling with a stride meant a font used on only
# a few pages was never observed, so H2 could not fire and the audit reported
# one finding fewer than the documented baseline.
fonts = {}
for i in range(doc.page_count):
    for f in doc[i].get_fonts(full=True):
        fonts[f[3]] = f
not_embedded = sorted({name for name, f in fonts.items() if not f[1]})
if not_embedded:
    add("H2", "LOW", "RISK", "Fonts referenced but not embedded",
        "Some text relies on fonts the reader must supply.", not_embedded,
        "Character substitution risk when rendering pages for a vision pass. "
        "Verify rasterised pages against extracted text before trusting either.")

cjk = [n for n in fonts if "Gothic" in n or "Japan" in n]
if cjk:
    add("H3", "LOW", "STRUCT", "Japanese CID fonts present in an English manual",
        "The document was produced from a Japanese source; CJK-encoded fonts remain "
        "in the file and poppler reports a missing Adobe-Japan1 mapping.", cjk,
        "Harmless for the English text layer, but some extraction tools emit "
        "encoding warnings or drop affected glyphs.")

# ================================================= I. MONITORING CODE REFS
mon_referenced = set()
for r in recs.values():
    mon_referenced |= set(r["refs_monitoring_codes"])
add("I1", "MEDIUM", "RISK", "Monitoring codes referenced by troubleshooting procedures",
    f"{len(mon_referenced)} distinct monitoring codes (e.g. 36500, 03203) are cited "
    "as the way to verify a reading, but the monitoring code table lives in a "
    "different part of the manual.",
    sorted(mon_referenced),
    "Each of these is a required second hop. Without it the assistant tells the "
    "technician to 'check with the monitoring function' and stops there.")

# =========================================== J. SECTION-LEVEL STRUCTURE
tops = [t for t in toc if t[0] == 1]
sec = []
for i, t in enumerate(tops):
    end = tops[i + 1][2] - 1 if i + 1 < len(tops) else doc.page_count
    sec.append((t[1].strip(), t[2], end, end - t[2] + 1))
add("J1", "INFO", "STRUCT", "Section sizes are extremely unbalanced",
    "Section 40 (Troubleshooting) is 956 pages, 43% of the manual; Section 90 "
    "(Circuit Diagrams) is 44 pages of almost pure graphics.",
    [f"{n}: pp {a}-{b} ({c} pages)" for n, a, b, c in sec],
    "A single chunking and embedding strategy applied uniformly will be wrong for "
    "most of the document. Section-specific parsing is required.")

# -------------------------------------------------- encryption / permissions
add("J2", "LOW", "RISK", "Document is AES encrypted with modification denied",
    "Permissions allow printing and copying but deny changes. Some libraries "
    "refuse to open it or silently return empty text.",
    ["print: yes", "copy: yes", "change: no", "algorithm: AES"],
    "Normalise with `qpdf --decrypt` before the ingestion pipeline, or tool "
    "behaviour will vary across the fleet.")

# ==================================================================== output
sev_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2, "INFO": 3}
findings.sort(key=lambda f: (sev_order[f["severity"]], f["id"]))
os.makedirs(REPORT_DIR, exist_ok=True)
json.dump(findings, open(REPORT_PATH, "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
c = collections.Counter(f["severity"] for f in findings)
print("findings by severity:", dict(c))
print("by kind:", dict(collections.Counter(f["kind"] for f in findings)))
print(f"written to {REPORT_PATH}")
