# Komatsu Diagnostic Assistant — Golden Dataset & Evaluation Harness

Ground truth and measurement scaffolding for a guided diagnostic assistant built
over Komatsu Shop Manual **SEN06867-13** (PC200-10M0 Hydraulic Excavator,
S/N 700001 and up, revision 13, 2,226 pages).

Built without workshop access: the manual is the ground truth.

---

> **`golden/` is current and fully regenerable.** `failure_codes/`,
> `index.json` and `qa_set.json` are all rebuilt from the source PDF, with the
> eight regression counts and the qa_set/ground-truth agreement enforced by
> `tests/test_extraction.py`.
>
> `qa_set.json` previously had no producer, which is how it went stale: the
> ground truth was corrected and the test set kept requiring the old value.
> `eval/build_qa_set.py` closes that. Do not hand-edit either — a hand-corrected
> value is indistinguishable from a parser bug on the next regeneration. See
> `CLAUDE.md` Rule 2.
>
> Still missing: `evaluate.py` and `make_mock_runs.py`. Nothing can be *scored*
> yet.

---

## Contents

```
pipeline/
  extract_golden.py    PDF -> structured diagnostic trees (deterministic, no LLM)
  audit_manual.py      source-document data-quality audit
eval/
  build_qa_set.py      trees -> golden Q&A test cases (deterministic, no LLM)
  adapters.py          Retriever + Generator seams; LocalBM25Retriever ships
  chunkers.py          corpus builder: structural / fixed_2000 / fixed_512
  metrics/             retrieval, generation, conversation, safety
  synthetic.py         good/weak systems that verify the harness itself
  run_eval.py          Tier-1 CLI -> eval_out/runs/<timestamp>_<label>.json
  compare.py           run-to-run regression diff; exits non-zero on regression
tests/
  test_extraction.py   ground-truth regression guard; runs after extraction
  test_eval_harness.py verifies the harness: good 7/7, weak 0/7
METRICS.md             metric definitions and interpretation guide

golden/
  failure_codes/*.json  174 failure codes, fully structured
  index.json            summary index
  qa_set.json           1,330 golden test cases
reports/
  audit_findings.json   machine-readable audit output
eval_out/               reports, per-case CSV, metrics JSON, run history
runs/                   system outputs to be scored (JSONL)
```

## Quick start

```bash
pip install -r requirements.txt
export KOMATSU_PDF="../komatsu-manuals/SEN06867-13.pdf"

python pipeline/extract_golden.py     # -> golden/failure_codes/, index.json
python eval/build_qa_set.py           # -> golden/qa_set.json
python pipeline/audit_manual.py       # -> reports/audit_findings.json
python tests/test_extraction.py       # regression guard (also runs automatically)
```

Tier-1 evaluation:

```bash
python eval/run_eval.py --chunker structural --system good   # 7/7 gates
python eval/run_eval.py --chunker structural --system weak   # 0/7 gates
python eval/compare.py eval_out/runs/<a>.json eval_out/runs/<b>.json
python tests/test_eval_harness.py            # verifies the harness itself
```

`--system` takes a synthetic system; a real one plugs into the `Generator` seam
in `eval/adapters.py`. Tier 2 (judge model) is not wired — metrics register
through the same interface with `TIER = 2` and are aggregated separately, so a
judge score can never be quoted as a Tier-1 figure.

### Local baseline

Same corpus, same retriever (BM25), three chunkings. Everything else held still,
so the spread is the cost of the chunking decision alone.

| | structural | fixed_2000 | fixed_512 |
|---|---:|---:|---:|
| chunks | 174 | 491 | 1,708 |
| **ceiling_recall** | **1.0000** | **0.9997** | **0.9914** |
| recall@5 | 0.9979 | 0.9665 | 0.7896 |
| answer_coverage@5 | 0.9970 | 0.9492 | 0.6844 |
| fragmentation_gap@10 | 0.0004 | 0.0138 | 0.1069 |
| context_precision | 0.8497 | 0.7020 | 0.6452 |
| mrr | 0.9917 | 0.9556 | 0.8104 |
| gates | 7/7 | 7/7 | **6/7** |

Read `ceiling_recall` first: it is set by ingestion and chunking, and every
other retrieval number lives under it. `fixed_512` cannot reach 1.0 at any k
because some facts — full step procedures — are longer than 512 characters and
exist in no single chunk. No amount of ranking work recovers them.

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

Run `python pipeline/audit_manual.py` to reproduce. Findings are classified as:

- **SOURCE** — a genuine inconsistency in the manual itself
- **STRUCT** — a structural property that breaks straightforward extraction
- **RISK** — not wrong, but a known failure mode for retrieval systems

**21 findings: 6 high, 8 medium, 5 low, 2 informational.**

Every count in this document is verified against `reports/audit_findings.json`,
which was regenerated over the full 2,226-page range. Corrections to what this
section previously claimed:

- The original split (**6/8/3/3**) was wrong in two columns.
- Checks `G1`, `G2` and `H1` iterated a hardcoded `range(2, 2210)` that stopped
  exactly at the end of Section 90, excluding the 16-page Index. All three now
  use `doc.page_count`.
- Three checks were reporting zero because they were **incapable of reporting
  anything else**: `E4` (split decimals), `H2` (non-embedded fonts) and `D3`
  (missing "Details of failure"). Each now carries a self-test that asserts it
  can fail on a known-bad input, and prints `<id> self-test: PASS` on every run.
  See *Checks that could not fail* below.

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
**27 steps across 12 codes. STRUCT.**

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

Affected codes, in full: `CA451`, `DW91KA`, `DW91KY`, `DWA2KA`, `DWA2KY`,
`DWK0KA`, `DWK0KY`, `DWK2KA`, `DWK2KY`, `DWK8KA`, `DWK8KY`, `DY20KA`.

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
**220 pages, of which 103 are dense vector graphics. RISK.**

220 pages carry under 250 characters of extractable text. 103 of those contain
over 500 vector drawing operations each — circuit diagrams and hydraulic
schematics drawn as line art, not as embedded images.

`pdfimages` will not extract them because they are not image objects. Text
embeddings cannot represent them at all. Section 90 (Circuit Diagrams) is almost
entirely in this category: 44 pages averaging 4,668 vector objects and 313
characters of text.

**These pages will never be retrieved by a text pipeline and will never produce
an error.** They need page rasterisation plus a vision pass, or they are simply
absent from the system.

*Was 218/103 before the audit range was extended to the full document; the two
additional low-text pages are in the Index.*

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
**35 pages. STRUCT.** Section dividers and full-bleed diagram pages have no
footer, so content on them cannot be cited the way a technician expects.

*Was 19 when the audit stopped at page 2210, then 21 once the range was extended.
It is 35 now that the footer rule is correct: 19 body pages plus all 16 Index
pages, which are footed `Index` and a plain sequence number rather than a
`NN-NNN` page number. The rise is the check becoming accurate, not the document
getting worse.*

### D3 — Codes with no "Details of failure" field
**1 code. SOURCE.** `DR31KX` — the same code whose table layout matches no known
format (`D1`). This check previously reported zero; see *Checks that could not
fail*.

### H3 — Japanese CID fonts in an English manual
**94 font instances. STRUCT.** The document was produced from a Japanese source
(`MS-Gothic-90ms-RKSJ-H`). Poppler reports a missing Adobe-Japan1 mapping.
Harmless for the English text layer, but some tools emit warnings or drop
affected glyphs.

*Was 4 under the old every-37th-page sampling. The jump to 94 is not new
content: PDF subsetting gives each subset a unique six-letter prefix
(`WTHOJK+MS-Gothic-…`, `GYTDCU+MS-Gothic-…`), so 87 of the 94 are subsets of the
same MS-Gothic face. Count families, not font objects, before reading anything
into this number.*

### H2 — Fonts referenced but not embedded
**10 faces. RISK.**

Ten faces are referenced by page content but carry no embedded font programme,
so the reader must substitute one from the host system. All ten are Arial
family:

```
Arial          Arial,Bold      Arial-BoldMT    ArialMT      ArialNarrow
JVVZIU+ArialMT SLLUPO+ArialMT  TRNOVV+ArialMT  VMTNYV+ArialMT  VBITAO+ArialNarrow
```

**Why this matters specifically here.** The audit's own recommendation for the
220 low-text pages (`H1`) is to rasterise them and run a vision pass. Rasterising
a page whose text depends on a font the machine does not have substitutes glyphs
silently — metrics shift, characters are replaced, and the rendered image stops
agreeing with the text layer it is supposed to complement. Because these are
Arial faces carrying body text rather than decorative elements, the exposure is
spread across the document. Install the fonts, or verify rasterised output
against extracted text before trusting either.

*Type3 fonts are excluded from this count. Their glyphs are content streams
inside the PDF, so they report `ext='n/a'` by construction and carry no
substitution risk. One is present in this document.*

### J2 — AES encryption
The file is AES encrypted with modification denied — printing and copying are
allowed, but some libraries refuse to open it or silently return empty text.

Normalise with `qpdf --decrypt` before ingestion, or tool behaviour will vary
across machines.

*This finding is hardcoded in `audit_manual.py` — its permission strings are
literals, never read from `doc.permissions`. The copy used for the current
baseline reports `is_encrypted: False`, i.e. it has already been normalised.
Treat the finding as a standing note about the distributed file, not a
measurement of the file the audit ran against.*

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
| Index | 16 | Text |

Section 40 alone is 43% of the manual. A single chunking and embedding strategy
applied uniformly will be wrong for most of the document.

The trailing 16-page Index was previously omitted from this table and, more
consequentially, from the audit itself: checks `G1`, `G2` and `H1` iterated a
hardcoded `range(2, 2210)`, which stops exactly at the end of Section 90. All
three now use `doc.page_count`. The effect: `H1` 218 → 220 low-text pages
(103 vector-only, unchanged) and `G2` 19 → 35. `G1` reports zero either way —
the manual's printed page numbers are genuinely unique.

### E5 — Criteria that are legitimately non-numeric
**26 measurements.** Continuity checks and audible confirmations ("No continuity
(there is no sound)") have no numeric value by design. Not a defect — but any
validation rule demanding a number from every criterion will flag them
incorrectly.

---

## Checks that could not fail

Four checks in `audit_manual.py` were reporting a result they were structurally
incapable of changing. Three reported zero and one reported noise. In every case
the empty or wrong result had been read as a statement about the document.

| Check | Defect | Was | Now |
|---|---|---|---|
| `E4` split decimals | pattern `\d\s+\.\d` needs a digit *before* the split; `.2 to 4.6V` has none | 0 | 0, self-tested |
| `H2` non-embedded fonts | `not f[1]` where `f[1]` is `'n/a'`; `not "n/a"` is always `False` | 0 | **10** |
| `D3` missing detail field | `not value` where the manual prints `-`; `-` is truthy | 0 | **1** |
| `G1` duplicate page numbers | substring match picks up index entries' trailing references | 14 false | **0** |

`H2` is the one worth dwelling on. Fixing the page stride made it *capable* of
seeing all 123 fonts instead of 18 — but the predicate was still broken, so it
evaluated every font as embedded and returned nothing. "Finds nothing" was then
written into this document as "nothing to find", and a true finding was deleted
on the strength of a broken check. That is the same mistake `E4` had already
demonstrated, made a second time while documenting the first.

Each of these now asserts against known-good and known-bad inputs before it runs,
and prints `<id> self-test: PASS`. The audit aborts if an assertion fails, so a
clean result means a clean document rather than a blind check.

**`D2` was under-reporting for the same reason as `D3`** (1 code, now 2) — the
shared root cause is applying truthiness to a field whose "absent" state is the
non-empty string `-`. Both now use an explicit `missing()` predicate.

The remaining truthiness-on-string predicates in `audit_manual.py` — `E1`
criteria, `E2` measuring point, `D5` cause/procedure, `F1` canonical title —
were checked against the corpus. None currently carries a placeholder value in
its absent state, so all four are correct on this data. They are, however, the
same shape, and would break the same way if a future revision printed `-` in
those columns.

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

## Also found

### Measurement values split across a cell boundary — found, regressed, re-fixed

`CA451` step 6 stores its criterion as two cells: `'Sensor output 0'` and
`'.2 to 4.6V'`. Joined with a space this reads **"0 .2 to 4.6V"**; keep only one
fragment and it reads **"0"**.

The correct value is `Sensor output 0.2 to 4.6V` — a common rail pressure sensor
output range. Every corruption of it is the kind of number a technician acts on.

This was fixed once, then **silently reintroduced** when measurement detection
was rewritten to handle the third cause-table layout. `_join_criteria` still
rejoins decimal fragments correctly, but the rewrite chose the criterion cell
*before* calling it: the right-to-left scan for `CRIT_VALUE` matches `.2 to 4.6V`
on its own, because the range pattern sees `2 to 4`. The scan therefore stopped
on the fragment and stranded the integer part in the measuring point:

```
point:    "Between ECM (25) and (47) / Sensor output 0"
criteria: ".2 to 4.6V"
```

`extract_golden.py` now fuses split decimals on the raw cell list, before any
positional read (`_fuse_split_decimals`). A cell ending in a digit followed by
one beginning `.<digit>` is a decimal broken by the table structure and is
rejoined into a single cell.

**The audit could not have caught this.** Check `E4` matched `\d\s+\.\d`, which
requires a digit *before* the split point — and `.2 to 4.6V` has none. `E4`
reported zero on a known-corrupt record, and this document previously cited that
zero as evidence the problem was fixed.

> **`E4` currently reports zero. That is now meaningful.** A check that cannot
> fail on its own motivating example is not a check, so `audit_manual.py` runs
> four assertions on the `E4` pattern before using it — it must match
> `".2 to 4.6V"` and `"Sensor output 0 .2 to 4.6V"`, and must not match
> `"Sensor output 0.2 to 4.6V"` or `"Max. 1 Ω"`. The audit prints
> `E4 self-test: PASS` on every run and aborts if the pattern ever goes blind
> again. Against the pre-fix corpus the widened check found `CA451`; against the
> regenerated corpus it finds nothing, because the corruption is gone.

The corpus was swept for the same shape — every criterion starting with `.` and
every measuring point ending in an orphaned digit. `CA451` step 6 was the only
genuine instance across all 872 measurements, and instrumenting the repair
during a full re-extraction confirms it fired exactly once, on that one row.
`tests/test_extraction.py` pins the corrected value so this cannot regress a
third time.

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

---

## Human verification: Round 1 decision rule

**Pre-committed before the result is known.** A threshold chosen after seeing
the number is not a threshold.

Round 1 is all **12 column-split codes** — `CA451`, `DW91KA`, `DW91KY`,
`DWA2KA`, `DWA2KY`, `DWK0KA`, `DWK0KY`, `DWK2KA`, `DWK2KY`, `DWK8KA`, `DWK8KY`,
`DY20KA` — covering **all 27** machine-repaired steps. Text a parser reassembled
from a mangled table, flagged `NEEDS_HUMAN_VERIFICATION` from the beginning and
never read by a person. About 2.4–4.3 hours.

```
python pipeline/human_verify.py --progress    # what is done, what is pending
python pipeline/human_verify.py               # the per-step table
```

### The rule

| outcome over the 27 steps | decision |
|---|---|
| **all 27 agree** | Extractor is sound on its highest-risk output. Proceed; Round 2 runs in the background alongside other work. |
| **1–2 disagree** | Investigate each. Fix and re-verify before proceeding. Finding and fix stay in separate commits. |
| **3 or more disagree** | Column-split recovery is not trustworthy. Stop feature work and revisit the parser. |

**Absolute counts, deliberately not scaled to the larger denominator.** The
thresholds were set at 1–2 and 3+ against 17 steps and are unchanged against 27.
Scaling them (to ~2–3 and 5+) would mean deciding, after the denominator grew,
that more broken steps are now acceptable. The question these thresholds answer
is "how many reconstructed steps may be wrong before the reconstruction is
untrustworthy", and that answer does not depend on how many we managed to check.
One wrong step among 27 matters; a 96.3% agreement rate hides it, which is why
`column_split_accuracy` is reported per step and never aggregated.

### No unreachable remainder

Earlier, four of these codes sat in the metric holdout and 10 steps were out of
reach. The holdout has been reselected to exclude column-split codes entirely,
so Round 1 now covers 27 of 27. See *The metric holdout* below for what that
reselection costs.

---

## The metric holdout

20% of failure codes, reserved so nothing tunes to them. Rebuilt (v2) under one
added constraint: **no column-split code may be in the holdout.**

Those 12 codes have steps the parser *reconstructed* from a mangled table.
Asking "did we tune to our fixtures?" of content the parser partly invented is
circular on its own terms — a disagreement there could be overfitting or could
be the reconstruction, and the holdout cannot tell you which. They belong in the
human-transcription pool, where a person reading the page settles it.

Selection rule, implemented in `agent/holdout.py`:

1. Eligible = all codes minus the 12 column-split codes.
2. Stratify on `(format, action_level, page-span band, pointer-only)`.
3. Largest-remainder allocation of 20% across strata — a plain floor sends every
   stratum smaller than 5 to zero, and with a four-axis key most strata are small.
4. Axis-coverage guarantee: any band with ≥5 members must appear in the holdout,
   never taken from a stratum whose last train member it would be.

Result: **32 of 174 codes (18.4%)**, all 26 strata eligible, every band with ≥5
members represented. `L00` (1 code), `L00a` (1) and `L02` (3) have no holdout
member by design — taking them would empty the stratum from train.

### The sequencing bias, stated plainly

The transcription set is defined by a property of the data, and the holdout is
drawn from what remains. **That ordering is a bias and it has a direction:** the
holdout now systematically excludes the pages the parser found hardest, so it
measures generalisation on cleaner-than-average material and cannot detect
tuning that only appears on difficult layouts.

This is not fixable by reordering — the two constraints genuinely conflict. What
closes it is the other half of the design: all 12 column-split codes go to human
transcription. The holdout answers *"did we tune to our fixtures on
cleanly-extracted codes"*; the human round answers *"did the parser read the hard
pages correctly"*. Neither answers both, and quoting either as if it covered the
whole corpus would be wrong.

### What reselection did to the numbers

| metric | old train | old holdout | old gap | new train | new holdout | new gap |
|---|---:|---:|---:|---:|---:|---:|
| fidelity measurement | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| fidelity branch | 1.0000 | 1.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 |
| fidelity cause | 0.9639 | 0.9518 | 0.0120 | 0.9538 | **0.9942** | **0.0404** |
| fidelity ALL | 0.9889 | 0.9861 | 0.0028 | 0.9861 | 0.9983 | 0.0122 |
| every agent metric | — | — | 0.0000 | — | — | 0.0000 |

The `cause` gap moved materially **and flipped direction** — the holdout now
scores *better* than train. The cause is mechanical, not a sign the old split was
measuring noise: the 39 unresolvable facts are concentrated in column-split
codes, and moving all 12 into train necessarily lowers train's cause rate and
raises the holdout's.

The consequence is worth being explicit about: **`cause` fidelity gap is no
longer a usable generalisation signal under this split.** It is dominated by
where the known-unresolvable facts sit. The gated kinds — `measurement` and
`branch`, both 0.0000 on both splits — and the agent metrics remain meaningful.

---

## Human verification: NOT PERFORMED

**`human_verified` is 0 across all 3,305 facts.** The transcription round was
designed and the kit was built — 27 machine-repaired steps, 54 crop boxes,
roughly 1–2 hours of work — and it was never executed. Recording that here
rather than letting it disappear: a known gap that stops being visible becomes
an unknown gap.

### What is still covered

**3,266 of 3,305 facts are resolver-verified against the PDF.** All **872**
measurement criteria and all **1,437** branch outcomes resolve exactly — every
safety-critical value has external confirmation from the source document, found
on the page it is cited to.

### What remains unverified

Reassembled cause text on **27 steps**. The extractor recovered a step number
typeset inside the cause column and kept the mangled fragments; no person has
read those pages. These are `cause` strings only — no measurement, no branch,
no criterion.

### Supporting evidence, and its limit

The crop work produced independent support for the extractor's behaviour on
these tables. Glyphs overflow their declared column by up to **36pt** — measured
on page 784, where the cause text runs `x0=70.4..164.4` against a declared cell
of `106.3..181.3` — and **pdfplumber and PyMuPDF agree on what lies inside the
rectangle**. That is consistent with the extractor having read the tables
correctly.

It is **not a substitute for a human reading.** Both libraries share the same
notion of a cell, so a misreading rooted in that shared notion is invisible to
both. That is precisely the class of error the human round existed to catch.

### To close it

`reports/transcription/crops/index.html` — 54 boxes, 0.9–1.8 hours.
`pipeline/human_verify.py` compares the result and sets `human_verified` only on
full agreement.

Every orchestrator run prints `human_verified: 0` in its summary.
