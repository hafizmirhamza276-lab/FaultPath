# H-Mode and S-Mode — fidelity and first audit

Run against SEN06867-13 with the complete denominator: 274 H-Mode
measurements, 0 unclassified tables. No LLM anywhere in extraction, fidelity or
audit.

**Three HIGH findings, all needing a change to `extract_symptoms.py`. Reported,
not worked around. Two of them fail a gate, and the gate was not lowered.**

---

## 1. Fidelity by fact kind and section

`fact_resolution_rate` — the stored verbatim text is re-found on the PDF page
the fact claims to be on. `+NR` is the relational count, excluded from the
scored denominator by declaration (see §3).

| kind | section40 | hmode | smode |
|---|---|---|---|
| `measurement` | **1.0000** 872/872 | **0.9213** 234/254 `+20R` | n/a |
| `branch` | **1.0000** 1437/1437 | **0.9985** 665/666 | n/a |
| `cause` | 0.9608 957/996 | 1.0000 333/333 | 1.0000 209/209 |
| `step_procedure` | 0.6509 647/994 | 0.5646 188/333 | n/a |
| `remedy` | n/a | n/a | **1.0000** 209/209 |
| `symptom_title` | n/a | 1.0000 37/37 | 1.0000 20/20 |

Corpus: 5,808 / 6,360 = 0.9132, with 20 relational facts excluded.
`page_containment` 1.0000 over all 6,360.

### Gates

Each gated kind is now gated **per section as well as corpus-wide**. Section 40
has 872 measurements against H-Mode's 254; a combined rate moves by roughly a
quarter of what the new section's own failures should move it, which is exactly
how a weak new section rides a mature one.

| gate | value | n | |
|---|---|---|---|
| `fact_resolution_rate[section40/measurement]` | 1.0000 | 872 | PASS |
| `fact_resolution_rate[hmode/measurement]` | 0.9213 | 254 | **FAIL** |
| `fact_resolution_rate[section40/branch]` | 1.0000 | 1437 | PASS |
| `fact_resolution_rate[hmode/branch]` | 0.9985 | 666 | **FAIL** |
| `page_containment` | 1.0000 | 6360 | PASS |
| `known_overhang_stable` | — | 9 | PASS |

Both failures are accounted for exactly — 20 facts and 1 fact — and both are
findings below, not tolerances to be widened.

### The 552 unresolved facts, accounted for

| classification | n | what |
|---|---:|---|
| `KNOWN_LIMITATION` | 504 | 492 reassembled step procedures (S3) + 9 cell-overhang + 3 synthetic `Redirect` |
| `NEEDS_HUMAN_VERIFICATION` | 27 | the Section 40 machine-repaired column splits, unchanged |
| `DEFECT` | **21** | **20 from S1 + 1 from S2** |

The DEFECT count equals the two gate failures exactly. Nothing is unclassified:
`classify()` still defaults to DEFECT, so a shape nobody has explained cannot
hide in the KNOWN bucket.

Two arithmetic faults of my own, found and fixed while producing this table.
`verbatim_integrity` read **1.0028** — above 1.0, which is the arithmetic
announcing that its numerator counted every row while its denominator excluded
relational ones. And `step_procedure`, newly enumerated, fell through to
DEFECT and reported **488** of them; it is a declared shape (S3) and now
classifies as a known limitation.

### `step_procedure` is low in *both* sections and is not a regression

0.65 in Section 40 and 0.56 in H-Mode. A procedure is reassembled from several
cells of a step's row, including embedded measurement sub-tables, so the stored
string is accurate but exists nowhere on the page as one contiguous run. The
resolver refuses it correctly, for the same reason it refuses the 27
machine-repaired Section 40 causes and the 3 synthetic `Redirect` labels.

This kind was **not enumerated before this change** in either section, so the
figure is new information rather than a drop. It is ungated and no numeric
metric is scored on it. Recorded as audit **S3 (LOW)**: procedures may be shown
to a technician as guidance but must not be presented as verbatim citations.

---

## 2. Dual-unit criteria — confirmed

**174 dual-unit criteria across both sections; 174 intact; 0 truncated.**

Checked by audit **C1**, which requires a digit on *both* sides of the bracket —
`0 to 0.49 MPa {0 to 5 kgf/cm2}` passes, `0 to 0.49 MPa`, `0 to 0.49 MPa {}` and
`{0 to 5 kgf/cm2}` all fail. All four assertions run before the check does, so
the zero is a statement about the document rather than about the check.

The positive result is recorded as its own finding (**C4, INFO**) rather than
left as C1's silence. A check that found nothing and a check that never ran read
identically in a report, and that is the mistake H2 made.

Both numbers also enter the `fabricated_values` allowed set, because the allowed
set is built from **the whole criteria string** — `{0 to 5 kgf/cm2}` included —
not from a parsed SI value. A technician reading a gauge graduated in kgf/cm²
needs the bracketed figure; losing it would be silent, since the SI value alone
still looks like a complete answer.

---

## 3. The relational bucket — excluded by declaration, count shown

**20 relational criteria**, all H-Mode, all of the form:

> `Oil pressure ratio pump discharged pressure : PC valve discharged pressure 1:0.6 (approximately 3/5)`

These state a **ratio between two measured quantities** rather than a bound on
one, so there is no number for the row on its own and the numeric comparison
that backs `numeric_exactness` cannot score them.

They are carried verbatim, counted, reported as `+20R` beside the figure they are
excluded from, and kept out of the scored denominator. Dropping them silently
would inflate the numeric rate by shrinking its denominator; folding them in
would fail a row for lacking a number it was never meant to have. Both are
wrong, and only one of them is visible.

`fidelity.self_test()` asserts both halves: a relational fact must leave the
denominator **and** must still appear in `relational_excluded`.

---

## 4. Audit — H-Mode and S-Mode

`pipeline/audit_symptoms.py`, same check classes (SOURCE / STRUCT / RISK),
severities and report shape as `audit_manual.py`.
→ `reports/audit_symptoms.json`

**8 findings — 3 HIGH / 1 MEDIUM / 1 LOW / 3 INFO.**

**12 self-tests, all PASS.** Every check that reports zero — T1 T2 T3 T4 T5 D1
D3 S2 — carries one, and the audit aborts if any fails. These sections have
never been audited before; a clean result on a first pass would be evidence of a
blind check rather than a clean document, which is the lesson E4 and H2 already
taught here once each.

| id | sev | kind | finding | n |
|---|---|---|---|---|
| **S1** | HIGH | STRUCT | Empty criteria typed as numeric | 20 |
| **S2** | HIGH | STRUCT | Branch outcome cites its step's page, not its own | 1 |
| **C3** | HIGH | STRUCT | Criterion typed numeric contains no digit | 2 |
| P1 | MEDIUM | RISK | Prose cross-references recorded, not resolved | **182** |
| S3 | LOW | STRUCT | Step procedures are reassembled, not verbatim-citable | 333 |
| C2 | INFO | STRUCT | Relational criteria state a ratio, not a value | 20 |
| C4 | INFO | STRUCT | Dual-unit criteria carry both values | 174 |
| T6 | INFO | STRUCT | Tables skipped during extraction | 0 |

`n` is the quantity the finding is about. S3, C4 and T6 carry it in their item
text rather than as a list length, because a list of 333 identical entries is
not evidence of anything.

### S1 — 20 empty criteria in the numeric bucket (HIGH)

The criteria cell of these rows is **vertically merged** with the row above.
One relational criterion — an oil-pressure ratio — governs both rows, and the
table draws it once:

```
HM02:0:meas:13  Pump supply pressure      "Oil pressure ratio ... 1:0.6"   relational
HM02:0:meas:14  PC valve output pressure  ""                               numeric   <-- merged cell
```

pdfplumber assigns the merged text to the first row and leaves the second
empty, which is the correct reading of the page. The extractor then types the
second row `numeric` with an empty criteria.

**20 of 20 are the second row of a merged relational pair**, in HM02, HM03,
HM05 and HM21 — verified, not assumed.

Two compounding effects. They enter the numeric denominator carrying no value;
and **an empty verbatim resolves against any page in the manual**, so without an
explicit refusal these would have reported as PASS and raised the rate while
measuring nothing. `check_facts()` now refuses empty verbatim text outright, and
`self_test()` asserts it — which is why the gate reads 0.9213 rather than 1.0000.

Fix needs a parser change: the second row must inherit the merged cell's
criterion and its `relational` type. **Not applied.**

### S2 — branch cites its step's page (HIGH, 1 case)

`HM22:5:branch:0`. Step 5 straddles pages 1399/1400: the YES outcome is printed
on 1399 and the NO outcome on 1400. Both branch facts inherit the **step's**
provenance, so YES resolves and NO does not. The recorded text is correct and
verbatim — what is wrong is the page it claims to be on.

This is the defect CLAUDE.md already names one level up: *"75% of measurements
are not on their code's first page — so a per-code page is not a citation."* A
per-**step** page is not a citation for a branch on the next page either.

Fix needs a parser change: branch facts need their own provenance rather than
the step's. **Not applied.** 1 of 666 H-Mode branches; the check's self-test
proves it would find a second.

### C3 — criterion typed numeric with no digit (HIGH, 2 cases)

`HM17:0:meas:1` and `HM17:0:meas:4`, both `"Pressure for each flow setting"` —
what the manual actually prints in the standard-value column, because the value
depends on a setting stated elsewhere. The text extracts correctly and resolves
on its page; the **type** is wrong.

Disjoint from S1 by construction, so the two counts add rather than overlap.

A technician asking for this pressure needs to be told the value is
setting-dependent, not handed a sentence where a number belongs. Fix needs a
parser change to the criteria classifier. **Not applied.**

### P1 — 182 prose pointers (MEDIUM, named gap)

Symptom trees hand off in running prose — *see Testing and Adjusting, "Examine
PPC Valve Outlet Pressure"* — rather than through a bracketed code. All 182 are
recorded verbatim as unresolved pointers, by the decision taken in the survey.

Kept as a named finding with its count so it stays visible. A symptom session
can reach a step whose next action lies outside the tree; the agent must surface
the pointer text and stop, not appear to continue. Closing it needs a resolver
for Testing-and-Adjusting titles, which does not exist.

### Checks that found nothing, and can

T1 branching step missing YES or NO — **0**. T2 flat tree carrying branches —
**0**. T3 flat step with no remedy — **0**. T4 non-contiguous step numbering —
**0**. T5 entry with no steps — **0**. D1 duplicate symptom phrasing — **0**.
D2 duplicate fact id — **0**. D3 overlapping page spans — **0**. P2 dangling
failure-code reference — **0**. T6 skipped tables — **0**.

Each has a self-test that fires on a hand-built bad record.

---

## Stopping here

Three HIGH findings, all requiring changes to `extract_symptoms.py`, and two
failing gates that the findings explain exactly. The standing instruction is to
report a parser defect rather than work around it — that call has been right
three times now, and S1 in particular would be tempting to patch in the scorer,
where it would silently become a tolerance.

Not started: symptom map, agent integration, evaluation split, the three test
levels and the overfitting guard. The symptom map depends on S1 and C3 being
settled, because it would otherwise be built over criteria whose types are known
to be wrong.
