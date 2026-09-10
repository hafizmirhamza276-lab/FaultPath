# Komatsu Diagnostic Assistant — Golden Dataset & Evaluation Harness

Ground truth and measurement scaffolding for a guided diagnostic assistant built
over Komatsu Shop Manual **SEN06867-13** (PC200-10M0 Hydraulic Excavator,
S/N 700001 and up, revision 13, 2,226 pages).

Built without workshop access: the manual is the ground truth.

---

## Contents

```
extract_golden.py      PDF -> structured diagnostic trees (deterministic, no LLM)
build_qa_set.py        trees -> golden Q&A test cases with expected answers
evaluate.py            scores a system run; module + pipeline metrics, release gates
make_mock_runs.py      synthetic good/weak runs to validate the harness itself
audit_manual.py        source-document data-quality audit
METRICS.md             metric definitions and interpretation guide

golden/
  failure_codes/*.json  174 failure codes, fully structured
  index.json            summary index
  qa_set.json           1,330 golden test cases
audit_findings.json     machine-readable audit output
eval_out/               reports, per-case CSV, metrics JSON, run history
runs/                   system outputs to be scored (JSONL)
```

## Quick start

```bash
pip install pymupdf pdfplumber

python extract_golden.py /path/to/SEN06867-13.pdf
python build_qa_set.py
python audit_manual.py
python make_mock_runs.py
python evaluate.py runs/run_good.jsonl --run-id good   # 7/7 gates pass
python evaluate.py runs/run_weak.jsonl --run-id weak   # 0/7 gates pass
```

## Wiring in a real system

Emit one JSON object per test case to `runs/<name>.jsonl`:

```json
{"id": "<case id from qa_set.json>",
 "answer": "<final text shown to the technician>",
 "citations": ["40-181"],
 "retrieved": ["CA122", "CA227"],
 "refused": false,
 "module_trace": {"filters_applied": {"model": "PC200-10M0"},
                  "resolved_refs": ["CA441"],
                  "selected_step": 1}}
```

`module_trace` is optional but unlocks the module-level metrics — without it you
can see that something broke, not where.

## What the dataset contains

| | |
|---|---|
| Failure codes | 174 (117 format A, 57 format B) |
| Diagnostic steps | 996 |
| YES/NO branch outcomes | 1,437 |
| Measurement criteria | 872 |
| Cross-reference edges | 94 |
| Pointer-only codes | 9 |
| Golden test cases | 1,330 |

---

# Source Document Audit

Run `python audit_manual.py` to reproduce. Findings are classified as:

- **SOURCE** — a genuine inconsistency in the manual itself
- **STRUCT** — a structural property that breaks straightforward extraction
- **RISK** — not wrong, but a known failure mode for retrieval systems

**20 findings: 6 high, 8 medium, 3 low, 3 informational.**

Two things worth saying up front. First, this manual is unusually well built —
2,226 pages, 954 bookmarks with accurate page targets, a clean text layer
throughout, and no duplicate printed page numbers anywhere in the document. Most
technical PDFs this size are far worse. Second, the defects that do exist are
concentrated in exactly the places that matter most for a diagnostic assistant,
and several of them fail silently. That combination is what makes an audit worth
running rather than assuming.

---

## HIGH severity

### E3 — Steps whose number is typeset inside the cause column
**27 steps across 14 codes. STRUCT.**

On certain pages the step number is rendered within the cause text rather than
in its own cell. Extraction produces rows like:

```
['Defectiv pressure 6 voltage controlle', 'e common rail sensor (input error…', …]
```

A table parser sees no step number, treats the row as a continuation of the
previous step, and drops it. **These steps disappear with no error and no gap in
the output.**

The worst instance is `CA451` (Common Rail Pressure Sensor High Error), where
the manual lists seven causes and a naive parser recovers five. The two lost
steps are the final ones — the checks that identify a defective engine
controller. A technician following the truncated procedure runs out of things to
check and is told nothing further.

Affected codes: `CA451`, `DWK0KA`, `DWK0KY`, `DWK8KA`, `DWK8KY`, `DWK2KA`,
`DWK2KY`, `DWA2KA`, `DWA2KY`, `DW91KA`, `DW91KY`, `DY20KA`, and two others.

*Handled:* `extract_golden.py` recovers the step number from the mangled text
and flags the record with `extraction_warning: column_split_recovered`. Verify
these 27 steps by eye before trusting them.

### C4 — Codes whose entire procedure is a redirect
**9 codes. STRUCT.**

The complete troubleshooting content is a single instruction to go and
troubleshoot a different code:

| Code | Redirects to |
|---|---|
| `CA111` | `CA441` |
| `CA227` | `CA187` |
| `CA342` | `CA441` |
| `CA351` | `CA441` |
| `CA386` | `CA352` |
| `CA442` | `CA441` |
| `CA757` | `CA441` |
| `CA2249` | `CA559` |
| `CA2311` | `CA271`, `CA272` |

This is the classic retrieval trap. The chunk is correct, relevant, and ranks
first — and gives the technician nothing to do. No amount of embedding quality
fixes it. Hop-expansion is mandatory, not an optimisation.

### C1 — Cross-reference to a code that does not exist in this manual
**1 instance. SOURCE.**

`D8ARKR` instructs the technician to troubleshoot `CA445`. There is no `CA445`
page in Section 40 and no `CA445` row in the Failure Code Table. The reference
is a dead end — most likely a code from a different engine variant or a
superseded revision.

The assistant must recognise this and say so, rather than retrieving the
nearest-matching code.

### B1 — Action level disagrees between the two sources
**1 instance. SOURCE.**

`F@BBZL`: the Failure Code Table assigns **L01**; the code's own detail page
prints **L00a**.

Action level governs whether the machine may keep operating. These are not
interchangeable and the assistant must not silently pick one. Escalate to
Komatsu rather than guessing.

### D1 — Code with no parsable troubleshooting steps
**1 code. STRUCT.**

`DR31KX` uses a page layout matching none of the three table formats found
elsewhere in Section 40. It needs manual transcription or a vision pass, and is
excluded from step-based evaluation until then.

### H1 — Pages effectively invisible to text retrieval
**218 pages, of which 103 are dense vector graphics. RISK.**

218 pages carry under 250 characters of extractable text. 103 of those contain
over 500 vector drawing operations each — circuit diagrams and hydraulic
schematics drawn as line art, not as embedded images.

`pdfimages` will not extract them because they are not image objects. Text
embeddings cannot represent them at all. Section 90 (Circuit Diagrams) is almost
entirely in this category: 44 pages averaging 4,668 vector objects and 313
characters of text.

**These pages will never be retrieved by a text pipeline and will never produce
an error.** They need page rasterisation plus a vision pass, or they are simply
absent from the system.

---

## MEDIUM severity

### B2 — Codes with no action level in either source
**44 codes. SOURCE.**

Neither the summary table nor the detail page assigns an action level; both show
a dash. Includes `602KNX`, the entire `879*` air-conditioner family, `989*`, and
others.

The assistant cannot tell the technician whether the machine is safe to keep
operating. For these codes it should say the manual does not specify, rather
than inferring one from severity language elsewhere on the page.

### B3 — Detail page omits an action level the summary table assigns
**3 codes. SOURCE.**

`CA234`, `DKR2MA`, `DR31KX` — the summary table gives a level; the code's own
page prints a dash. A reader who navigates straight to the code page never
learns how urgent it is. Merge from the table, and record which source the value
came from.

### C3 — Circular cross-reference chains
**7 cycles. STRUCT.**

Following the manual's own instructions leads back to the starting code:

```
D8AQKR -> DA2QKR -> D8AQKR
D8AQKR -> DA2QKR -> DAZQKR -> D8AQKR
DB2QKR -> D8AQKR -> DA2QKR -> DAZQKR -> DB2QKR
B@BAZG -> CA435 -> B@BAZG
```

These are legitimate for a human — the codes are related CAN-bus faults and a
technician reading laterally understands the context. They are not legitimate
for automated hop-expansion. **Any "follow the reference" logic needs a visited
set and a depth limit**, or the assistant loops until it runs out of tokens.

### D4 — Format-A step with an incomplete YES/NO branch
**1 step. STRUCT.**

`DY20KA` step 7 carries a NO outcome with no corresponding YES. The executor has
no defined next state when the missing branch is taken.

*(This was 20 steps before the third table format was handled — see the note on
table formats below.)*

### A2 — Code with a troubleshooting page but absent from the Failure Code Table
**1 code. SOURCE.**

`DAF8KB` has a full procedure but no row in the summary table. Any lookup path
that starts from the summary table will never reach it.

### D2 — Code with no "Phenomenon on machine" field
**1 code. SOURCE.**

`DAFGMC`. The manual does not state what the operator will actually observe, so
the assistant cannot confirm it is working on the right problem and cannot help
a technician who describes a symptom rather than a code.

### F1 — Different codes sharing an identical display name
**1 pair. SOURCE.**

`B@BCQA` and `B@BCZK` both display **"Eng Water Level Low"** on the machine
monitor. They have different action levels (L02 vs L01) and different
procedures.

A technician who reports only the screen text cannot be disambiguated. The
assistant must ask for the code itself, not accept the display name.

### I1 — Monitoring codes referenced from a different part of the manual
**54 distinct codes. RISK.**

Troubleshooting procedures routinely say to verify a reading "with the
monitoring function (Code: 36501)". The monitoring code table lives elsewhere in
the manual, so each of these is a required second hop.

Without it the assistant tells the technician to check the monitoring function
and stops — technically faithful to the retrieved chunk, and useless.

---

## LOW severity

### G2 — Pages carrying no printed page number
**19 pages. STRUCT.** Section dividers and full-bleed diagram pages have no
footer, so content on them cannot be cited the way a technician expects.

### H3 — Japanese CID fonts in an English manual
**4 fonts. STRUCT.** The document was produced from a Japanese source
(`MS-Gothic-90ms-RKSJ-H`). Poppler reports a missing Adobe-Japan1 mapping.
Harmless for the English text layer, but some tools emit warnings or drop
affected glyphs.

### H2 / J2 — Non-embedded fonts and AES encryption
Arial and Arial-Bold are referenced but not embedded, which is a character
substitution risk when rasterising pages for a vision pass. The file is AES
encrypted with modification denied — printing and copying are allowed, but some
libraries refuse to open it or silently return empty text.

Normalise with `qpdf --decrypt` before ingestion, or tool behaviour will vary
across machines.

---

## Informational

### J1 — Section sizes are extremely unbalanced

| Section | Pages | Character |
|---|---|---|
| 00 Index and Foreword | 86 | Text |
| 01 Specifications | 14 | Tables + drawings |
| 10 Structure and Function | 226 | Text + heavy vector diagrams |
| 20 Standard Value Table | 22 | Pure tables |
| 30 Testing and Adjusting | 192 | Mixed |
| **40 Troubleshooting** | **956** | Structured decision tables |
| 50 Disassembly and Assembly | 482 | Text + 1,394 photos |
| 60 Maintenance Standard | 74 | Tables + dimension drawings |
| 80 Others | 112 | Mixed |
| 90 Circuit Diagrams | 44 | Vector schematics, almost no text |

Section 40 alone is 43% of the manual. A single chunking and embedding strategy
applied uniformly will be wrong for most of the document.

### E5 — Criteria that are legitimately non-numeric
**22 measurements.** Continuity checks and audible confirmations ("No continuity
(there is no sound)") have no numeric value by design. Not a defect — but any
validation rule demanding a number from every criterion will flag them
incorrectly.

---

## Three table formats, not two

Worth calling out separately because it was found only after the first two were
handled, and the third was silently losing data the whole time.

Section 40 uses **three distinct cause-table layouts**:

| Format | Codes | Layout |
|---|---|---|
| **A** | Komatsu machine codes (`602KNX`, `B@BAZK`, `D8AQKR`) | 5 columns, YES/NO in column 4 |
| **B** | Cummins engine codes (`CA111`, `CA451`) | 3–6 columns, sequential causes, measurements inline |
| **C** | KOMTRAX / communication codes (`F313KA`, `F313KB`) | 7 columns, YES/NO in **column 6**, headed "Check item" and "Judgment and remedy" |

A parser hard-coded to find YES/NO at a fixed column index handles A and B
correctly and returns format C with **no branches at all** — the decision tree
collapses into a flat list. This accounted for 19 of the original 20 D4
findings.

Detect the branch column by content, never by index. The same applies to the
measurement sub-rows, whose position also shifts between formats.

---

## Also found and repaired

### Measurement values split across a cell boundary

`CA451` step 6 stores its criterion as two cells: `'Sensor output 0'` and
`'.2 to 4.6V'`. Joined with a space this reads **"0 .2 to 4.6V"**; keep only one
fragment and it reads **"0"**.

The correct value is `0.2 to 4.6V` — a common rail pressure sensor output range.
Either corruption is the kind of number a technician would act on.

`extract_golden.py` rejoins fragments beginning with a decimal point. The audit
check for this pattern (`E4`) now reports zero, which is the intended end state
rather than evidence the problem never existed.

### Quantity labels vary far more than the common cases suggest

97% of measurements are labelled `Resistance`, `Voltage` or `Continuity`. The
remaining 3% are not: the fuel-system checks use labels like
`Discharged volume from supply pump`, `Fuel return rate from injector` and
`Leakage of pressure limiter`.

An extractor that identifies measurement rows by matching known quantity names
silently drops these — and because they are rare, nothing in the aggregate
counts looks wrong. The reliable signal is the **criterion value** (`Max.`,
`Min.`, a range, a number with a unit, a continuity verdict), not the label.

Identifying rows by value rather than by name raised the captured measurement
count from 775 to 872, and simultaneously removed 9 junk rows where a table
header (`Item`) had been captured as a quantity.

---

## What this means for the build

Ranked by how much damage each causes if ignored:

1. **Hop-expansion is not optional.** 9 pointer-only codes plus 94
   cross-reference edges plus 54 monitoring-code references. A pipeline that
   returns one chunk and stops is wrong for a large fraction of Section 40.

2. **Cycle protection is required** before any hop-expansion ships. Seven known
   cycles exist.

3. **Silent extraction loss is the real enemy.** 27 dropped steps, an entire
   collapsed table format, and a mangled sensor voltage range — none of which
   produced an error, a warning, or output that looked wrong. The only reason
   they were caught is that the extracted trees were checked against the source
   page by page. Build that check into the pipeline, not into a one-off review.

4. **Diagrams need a separate path.** 103 vector-graphic pages are invisible to
   text retrieval and will stay invisible until a vision pass is added.

5. **Where the manual contradicts itself, surface it.** `F@BBZL`'s action level
   and `D8ARKR`'s dead-end reference are not problems to paper over. Show the
   conflict and cite both pages — a technician who sees the disagreement can
   escalate; one who gets a confidently wrong single value cannot.

---

## Known gaps in this dataset

1. **Section 40 failure codes only** — 174 codes. H-Mode (hydraulic/mechanical
   symptoms, pp. 1288–1471) and S-Mode (engine symptoms, pp. 1472–1498) trees
   are not extracted, so symptom-entry queries are unmeasured.

2. **`DR31KX` has no parsable steps** — excluded from step-based metrics.

3. **Diagrams are not evaluated.** Any answer that depends on reading a
   schematic is outside what this harness measures.

4. **Single model, single manual.** Every number here describes PC200-10M0
   behaviour. Adding a second manual requires regenerating the golden set and
   re-baselining, because cross-model confusion cannot appear in a single-model
   test set.

5. **The 27 column-split recoveries are machine-repaired, not verified.** They
   carry `extraction_warning` and should be checked by eye before being treated
   as ground truth.
