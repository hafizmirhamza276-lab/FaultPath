# H-Mode / S-Mode extraction — findings

Status: **extraction built, two defects found, neither fixed.**
Per the standing rule, finding and fix stay in separate commits.

## Extraction counts

| section | entries | steps | measurements | pointers |
|---|---:|---:|---:|---:|
| H-Mode | 37 | 333 | 250 | — |
| S-Mode | 20 | 209 | 0 | — |
| total | **57** | **542** | **250** | **182** |

S-Mode having zero measurements matches the survey: its layout is
`No. | Cause | Point to check, remarks | Remedy` with no measurement column.

## DEFECT 1 — 24 measurement rows dropped (blank leading row)

Eight tables classify as `unknown` and are skipped. They are **not** the empty
phantom tables the survey suggested — they carry real criteria.

```
p1319 t1 (4x4)
  r0: ['', '', '', '']                                   <- blank leading row
  r1: ['Item', 'Measurement position, conditio', '', 'Standard value']
  r2: ['EPC Current', 'Monitoring code: 08000', ..., '0 mA']
  r3: ['', '', '• Work equipment control lever', '800 to 1000 mA']
```

**Root cause.** `table_kind()` reads `tbl[0][0]`. These tables have a blank
leading row, so the real `Item` header sits in row 1 and the table classifies as
`unknown`. Some (p1332, p1401, p1344) are continuation fragments whose header
was on the previous page and have no `Item` row at all.

**Cost:** 24 measurement rows carrying a criterion, across pp 1319, 1320, 1332,
1333, 1344, 1345, 1401, 1402. Values include `0 mA`, `800 to 1000 mA`,
`0 to 0.49 MPa {0 to 5 kgf/cm2}`, `2.84 to 3.43 MPa {29 to 35 kgf/cm2}`.

**Why not fixed here.** The fix belongs in classification, and `table_kind()` is
a Section 40 parser path which this task must not modify. It needs either a
symptom-local classifier that tolerates a leading blank row and header-less
continuation fragments, or an agreed change to the shared classifier. That is a
decision, not a detail — a classifier that accepts header-less fragments will
also accept things that are not measurement tables.

This is the same class as the Format C collapse and the 27 dropped steps:
content-bearing tables that classify as nothing and vanish without an error.

## DEFECT 2 — H-Mode's last entry swallows the S-Mode legend page

Entry spans are computed as `next_entry_page - 1`. The last H-Mode entry
(HM37, p1471) is followed by the first S-Mode entry (p1473), so HM37 is given
pp 1471–1472 and absorbs the S-Mode legend page 1472. Its two tables appear in
HM37's `skipped_tables` as `first_cell='Failure'` and `first_cell='Cause'`.

No data is lost — the legend is not an entry — but HM37's page range is wrong by
one page, which makes its provenance wrong for any fact on 1472. H-Mode should
terminate at 1471.

## Unknown tables after the changes — all 10 named

| page | table | first cell | shape | what it is |
|---|---|---|---|---|
| 1319 | 1 | `''` | 4x4 | measurement table, blank leading row — DEFECT 1 |
| 1320 | 1 | `''` | 6x4 | measurement table, blank leading row — DEFECT 1 |
| 1332 | 1 | `''` | 3x4 | measurement continuation, no header — DEFECT 1 |
| 1333 | 1 | `''` | 5x4 | measurement continuation, no header — DEFECT 1 |
| 1344 | 1 | `''` | 3x4 | measurement continuation, no header — DEFECT 1 |
| 1345 | 1 | `''` | 5x4 | measurement continuation, no header — DEFECT 1 |
| 1401 | 1 | `''` | 3x4 | measurement continuation, no header — DEFECT 1 |
| 1402 | 1 | `''` | 5x4 | measurement continuation, no header — DEFECT 1 |
| 1472 | 0 | `Failure` | 2x2 | S-Mode legend header — DEFECT 2 |
| 1472 | 1 | `Cause` | 6x4 | S-Mode legend table — DEFECT 2 |

Zero unknown tables remain that are not accounted for by one of the two defects.

## Prose pointers — 182 found, 0 resolved

No pointer uses the bracket form. The single `CA343` in these sections does not
occur inside a pointer sentence, so nothing met the "unambiguous bracketed
Section 40 code" bar. All 182 are recorded as `unresolved_pointer` with verbatim
text and provenance, per Decision 2. Nothing is gated on them.

## Relational bucket

`criteria_kind` is set per measurement: `relational` where the criterion states
a relationship rather than a value, `numeric` otherwise. Relational criteria
leave the numeric denominator by declaration and must still resolve against the
page.

---

# Update — both defects addressed, one blocker surfaced

## DEFECT 2 — FIXED

Entry spans now stop at the section boundary. `HM37` is `pdf_pages: [1471, 1471]`,
manual page `40-929`, `skipped_tables: 0`. Nothing on 1472 is attributed to it.

## DEFECT 1 — HALF FIXED (6 of 24 rows), blocked on a rule conflict

`symptom_table_kind()` is a symptom-local classifier; `table_kind()` is
unmodified. Its self-test runs before extraction and covers all four rejection
cases.

**Rule (a) blank leading row — works.** 6 rows recovered, all in HM06:

| page | criteria | rule |
|---|---|---|
| 1319 | `0 mA` | `blank_leading_row(+1)` |
| 1319 | `800 to 1000 mA` | `blank_leading_row(+1)` |
| 1320 | `0 mA` | `blank_leading_row(+1)` |
| 1320 | `0 to 0.49 MPa {0 to 5 kgf/ cm2}` | `blank_leading_row(+1)` |
| 1320 | `2.84 to 3.43 MPa {29 to 35 kgf/cm2}` | `blank_leading_row(+1)` |
| 1320 | `800 to 1000 mA` | `blank_leading_row(+1)` |

Measurements 250 -> 256. Skipped tables 10 -> 6.

**Rule (b) proven continuation — admits nothing. 18 rows still dropped.**

The third condition, "the fragment is the first table on its page", does not
hold for any real continuation in this document. Observed layout:

```
page 1331   t0 causes(5 cols)   t1 measurement(4 cols)   <- runs off the page
page 1332   t0 causes(5 cols)   t1 UNKNOWN(4 cols)       t2 measurement(3 cols)
```

The measurement fragment is `t1`, because a **cause-table continuation occupies
`t0`**. Conditions one and two are satisfied — the previous page's last table is
a measurement table and the column counts match at 4 — but the third fails.

Identical on 1344/1345 and 1400/1401.

**Not relaxed, because the choice is yours.** Two readings of the intent:

- *literal* — keep "first table on its page"; 18 rows stay dropped in
  HM13 (1332/1333), HM15 (1344/1345) and HM24 (1401/1402).
- *positional intent* — "nothing of substance intervened", i.e. first table
  that is not itself a continuation, or no measurement table precedes it on the
  page. Admits these 6 tables and recovers the 18 rows.

The second is weaker evidence: it admits a fragment with one more thing standing
between it and its predecessor. Given a classifier that accepts header-less
fragments is exactly what we agreed to be careful about, this is a decision
rather than a detail.

**Fidelity has NOT been run.** A measurement gate at 100% over a denominator
still missing 18 rows is the same failure as before, one third the size.
