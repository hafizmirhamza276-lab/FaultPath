# Komatsu Diagnostic Assistant — Project State & Plan

Handover document. Written so work can resume cold, without the conversation
that produced it.

Repo: `D:\Hackathon_project\komatsu-rag`
Source manual: `SEN06867-13` (Komatsu PC200-10M0 Hydraulic Excavator,
S/N 700001 and up, revision 13, 2,226 pages), kept **outside** the repo at
`../komatsu-manuals/SEN06867-13.pdf`.

---

## 1. What this project is

A **guided diagnostic assistant** for excavator technicians. A technician
reports a failure code or a symptom; the system walks them through the
manual's own troubleshooting procedure one step at a time, collects readings,
and reaches a diagnosis.

**It is not a RAG chatbot, and that distinction is the whole design.**

Komatsu already wrote the diagnosis — 174 failure codes, 996 ordered steps,
1,437 YES/NO branches. The agent *executes* that tree. It does not reason about
it. The LLM does exactly two things: understand what the technician typed
(Roman Urdu / English / mixed), and phrase the next step in natural language.

The LLM never selects a step, never compares a measurement, never decides a
diagnosis, never emits a number or a page. Those are code.

**Why this matters:** if the model is free to reason, it will answer "engine
code" with "check the oil and note the temperature" — plausible, confident, and
wrong. The actual CA122 procedure is six electrical checks with a multimeter.
A technician who follows the invented version wastes forty minutes and stops
trusting the system.

---

## 2. Current state

### Extraction — Section 40 (failure codes)

These eight numbers are **pinned regression counts**. If any moves, something
broke:

```
174 failure codes        996 steps           872 measurements
1,437 branch outcomes    9 pointer-only      27 column-split recovered
94 cross-reference edges 117 format A / 57 format B
```

### Extraction — symptom trees (H-Mode / S-Mode)

```
57 symptom trees   37 H-Mode (branching)  20 S-Mode (flat)
542 steps          274 H-Mode measurements (232 numeric + 42 relational)
209 S-Mode remedies
```

### Verification

Every fact is checked against the PDF by a **citation resolver** that re-opens
the page and confirms the verbatim text is there.

```
DEFECT: 0
KNOWN_LIMITATION: 504    NEEDS_HUMAN_VERIFICATION: 27
human_verified: 0        (transcription round built, not executed — see §5)
```

All measurements and all branches resolve at 1.0000 in every section, and are
gated there.

### Everything else

| Layer | State |
|---|---|
| Audit | Both sections, self-tests on every check |
| Evaluation | Tier 1 deterministic; 1,330 golden Q&A cases |
| Agent | LangGraph, 12 nodes, replay determinism over 45 code + 6 symptom sessions |
| API | FastAPI + Pydantic; HTTP output proven identical to in-process |
| Orchestrator | 13 stages, 42 gates, gating, resume, run records |
| Symptom map | 82 synonym entries, 57/57 trees covered, 14 unmapped |
| Symptom agent | `tree_kind` dispatch; ASK and UNMAPPED are first-class states |
| Holdout | 32 of 174 codes (18.4%), stratified, no column-split codes |

Two figures here were wrong and are corrected above: the agent suite is **45**
code sessions, not 49, and it has been 45 throughout — 49 was a miscount that
propagated into several summaries. The orchestrator is **13** stages, not 11,
since every `run_all.py` level became a stage.

**No longer in flight:** the `require_pdf` move, the level/stage parity check
and `tree_kind` driving agent behaviour are all landed. See
`reports/symptom_agent_integration.md` and `reports/false_green_findings.md`.

---

## 3. Architecture decisions, and why

These were argued through and settled. Do not relitigate them without new
evidence.

### The LLM's role is deliberately tiny

Reasoning lives in structured data, not in the model. Consequence: the runtime
model can be small and fast (`gpt-5-mini` / Claude Haiku 4.5), because it is
only doing language, not diagnosis. That is the payoff of the whole design.

### The model never types a citation

The model refers to facts by `fact_id`; **code renders the citation** from the
retrieved record. What the model does not generate, it cannot get wrong. This
turns `citation_accuracy` from something measured into something structurally
true.

### Provenance is per-fact, never inherited

Every fact carries its own `manual_page`, `pdf_page`, `table_index`,
`row_index`, captured at parse time from the page the row was actually read
from.

This was learned expensively — see §4.

### Hop-expansion is mandatory, with cycle protection

Nine codes contain nothing but a redirect (`CA111` → `CA441`). Retrieval
returns a correct, relevant, first-ranked chunk that helps nobody. Seven
reference cycles exist (`D8AQKR` → `DA2QKR` → `D8AQKR`), so any
follow-the-reference logic needs a visited set and a depth limit.

### Source defects are recorded, never patched around

Precedent, applied consistently:

- `F@BBZL` — action level is L01 in the summary table, L00a on its own page. Both cited, contested.
- `D8ARKR` → `CA445` — target does not exist in this manual. Recorded as a dead end.
- `DY20KA` step 7 — incomplete branch. Reported, not gated.
- **Glyph overhang** — 9 cause strings read short because the document places the final glyph *outside* its own ruled cell border (0.79pt to 7.10pt). Both pdfplumber and PyMuPDF agree on what lies inside the rectangle; the parser is correct. Padding the clip recovers 9 strings and corrupts ~570 others. Recorded as `KNOWN_LIMITATION`.

### Ambiguity is surfaced, never resolved silently

`swing slow hai` matches HM28 and HM29. The agent presents both with their
pages and asks. A technician who is asked loses ten seconds; one sent down the
wrong tree loses an hour and their trust.

`ASK` and `UNMAPPED` are first-class outcomes, not error paths. A system tuned
to never ask will guess.

### Flat and branching trees invert — they do not merely differ

On an H-Mode branching step, YES means *normal, advance*. On an S-Mode flat
row, confirming the point-to-check means *fault found, stop*. Sharing one
verdict pair walks a flat tree backwards, producing plausible output and no
error.

### Prose cross-references are recorded, not resolved

182 prose pointers, none in bracket form. A text-matching rule over 182
occurrences produces false positives, and a false positive sends a technician
into the wrong tree. Recorded with verbatim text and page; nothing gated.

---

## 4. The failure pattern that kept recurring

**Read this section first when resuming.** It is the most expensive lesson in
the project, and it recurred at least eight times.

> **Something checked itself, found nothing, and reported success.**

Every instance produced a green result. None produced an error.

| What | How it hid |
|---|---|
| `citation_accuracy` read 1.0000 | Test set and system both used `manual_pages[0]`. 635 of 846 citations pointed at the wrong page. The metric agreed with itself. |
| `orphan_detection` never fired | Its search window was `min(recorded)..max(recorded)` — so dropped *trailing* steps fell outside a window defined by the thing it was auditing. |
| `E4` (split decimals) reported zero | Regex needed a digit before the split; `.2 to 4.6V` has none. A clean result read as proof of a fix that never landed. |
| `H2` (non-embedded fonts) never fired | `if not f[1]` where `f[1]` is the string `"n/a"`. `not "n/a"` is `False`, always. |
| `D2`/`D3` reported zero | Same truthiness-on-string bug. |
| Overfitting guard passed | It sampled step 1, which carries no measurement — checked 0 facts, reported success. |
| Orchestrator reported 36/36 gates | A test level had been red for four commits. It had no orchestrator stage, so nothing was watching it. |
| `SKIPPED == "SKIPPED"` | A check that cannot fail. Caught before commit. |
| `verbatim_integrity` read 1.0028 | Above 1.0 — the arithmetic announcing that numerator and denominator disagreed. |

### The discipline that catches it

Applied to every check in the repo:

1. **Self-test on known-bad input, before the check runs.** Assert it *can*
   fail. A check that cannot fail is not a check.
2. **Independent derivation.** A verifier must not share helpers with the thing
   it verifies. Assert this over the parsed AST — a grep test once failed on
   its own docstring prose.
3. **Mutation testing.** Plant the exact fault and confirm something catches it.
   Report which check caught which. Name anything nothing catches.
4. **Negative coverage.** Every gate needs a case that fails it.
5. **Good/bad harness separation.** A deliberately bad implementation must fail
   *every* gate. If it passes one, that gate has a hole.

### Corollaries learned the hard way

- **A fix verified once, then surrounded by rewritten code, is unverified.** The
  `.2 to 4.6V` bug was fixed, verified, then reintroduced by a later refactor
  that was never re-checked against the same case.
- **Matching with one text pipeline and verifying with another is how a fix
  disagrees with its own check.** A raw substring match missed a hyphenated
  title that the de-hyphenating normaliser had always found.
- **A number above 1.0 is arithmetic telling you the denominator is wrong.**
- **Discard impossible measurements.** A "27% faster with logging" result was a
  cold-cache artifact, not a finding.

---

## 5. Known gaps — declared, not hidden

Both appear in every orchestrator run summary. A gap that stops being visible
becomes an unknown gap.

### `human_verified: 0`

27 machine-repaired steps have never been read by a person. The transcription
kit is **built and committed** — 12 codes, 54 crop images, an `index.html` that
needs no JSON editing, roughly 1–2 hours of work.

It was not executed. That is a recorded decision, not an oversight.

**What is still covered:** every safety-critical value has external
confirmation — all 872 measurements and all 1,437 branches resolve against the
PDF. What remains unverified is *reassembled cause text*.

**Supporting evidence, with its limit:** the crop work independently found that
glyphs overflow their declared column by up to 36pt, and both libraries agree
on cell boundaries. That confirms the extractor behaved reasonably — but both
libraries sharing a notion of a cell tells you they share it, not that it is
right. It is not a substitute for a human reading.

**To resume:** open `reports/transcription/crops/index.html`, type 54 boxes,
download, run `python pipeline/human_verify.py`. The decision rule is
pre-committed in the README: all 27 agree → proceed; 1–2 disagree →
investigate; 3+ → column-split recovery is untrustworthy, revisit the parser.

### `step_procedure` has no single page

994 Section 40 + 333 H-Mode procedures are reassembled from several cells and
exist nowhere on the page as one contiguous run. Not verbatim-citable, ungated,
no numeric metric scored on it. Structural, not fixable.

---

## 6. What is left

### Immediate — completes the foundation

The in-flight commit: `require_pdf()` move, level/stage parity check, agent
symptom integration with `tree_kind` polarity, six new scripted sessions,
symptom E2E over HTTP.

### Then — feature work, in order

**1. Real LLM integration.** Everything currently runs on `MockLLM`. Wire the
actual deployments (see §7). The graph and its tests must stay offline-capable.

**2. Tier 2 evaluation (DeepEval).** Faithfulness, answer relevancy,
hallucination, toxicity, bias, PII leakage. Two rules that must hold:
- The judge model must be a **different family** from the generator under test.
- Tier 2 numbers are reported in a **separate block**, never mixed into Tier 1
  headlines. Tier 1 is deterministic and defensible line by line; a judge is
  itself a model with its own error rate.

Tier 2 needs a noise floor — derive it by running the judge three times on
fixed inputs and using the observed spread. Tier 1 has no noise floor by
design: any movement is real.

**3. Azure AI Search.** This was the original starting point and is still
unaddressed. The existing index is already chunked and embedded by someone
else's pipeline.

**Run `probe_index.py` first.** It answers three questions in order:
- What is in the index (schema, filterable fields, chunk sizes)
- **Ceiling recall** — does each required fact exist *anywhere* in the index
- **Fragmentation** — is an answer intact in one chunk, or scattered

Read ceiling first: it is set by ingestion and bounds everything else. The gap
between `recall@k` and `answer_coverage@k` is fragmentation, and **no reranker
closes it** — reranking reorders what retrieval returned; it cannot assemble
one answer from two partial chunks.

Local baseline already exists for comparison: BM25 over three chunkings. If
Azure AI Search cannot beat plain BM25 on this corpus, that is a finding.

**4. Adapter boundary.** The runtime is already source-agnostic — the agent,
metrics, citations and API never touch a PDF, only the IR. Formalise that:
`SourceAdapter` with `probe()`, `extract()`, `audit()`, `provenance_for()`.
Every adapter must implement `audit()` and a citation resolver; one whose
citations cannot be resolved is not finished.

Leverage is **per format, not per document**. The 100+ Komatsu shop manuals
reuse the PDF adapter. A different OEM or an Excel source is days of work
again, because it has its own defects.

**5. Cross-source precedence.** Deterministic, no model:
`service_bulletin > shop_manual(rev N) > shop_manual(rev N-1) > field_data`.
Same-authority disagreement is surfaced as contested with both citations —
the `F@BBZL` policy applied across sources.

**6. Remaining manual sections.** Section 20 (standard values), 50
(disassembly, 1,394 photos), 60 (maintenance standards), 90 (circuit diagrams).
Section 90 is 44 pages averaging 4,668 vector objects and 313 characters of
text — invisible to text retrieval, needs rasterisation plus a vision pass or
multimodal embeddings.

---

## 7. Reference — model and platform notes

Verified against Microsoft Foundry documentation. **Prices were not
obtainable** — the Azure pricing page renders every figure via JavaScript and
search was failing. Use the Azure Pricing Calculator with your own agreement
and region.

### Deployment plan

| Role | Model | Notes |
|---|---|---|
| Document parsing | `mistral-document-ai-2512` or `Cohere-parse-v5` | Purpose-built, Markdown output, cheaper than a frontier model |
| Hard extraction cases | `gpt-5.4` (Batch) or `claude-sonnet-5` | Batch = 50% discount; **not available on the 5.6 family** |
| Runtime conversation | `gpt-5.4-mini` or `claude-haiku-4-5` | ~95% of token volume; latency matters |
| Value parsing | `gpt-5.4-nano` / `gpt-5-nano` | **GPT only** — see gotcha below |
| Escalation notes | `gpt-5.4` or `claude-sonnet-5` | Low frequency |
| Eval judge | Different family from the generator | Non-negotiable |
| Text embeddings | `text-embedding-3-large`, `dimensions=1024` | |
| Image embeddings | `embed-v-4-0` (Cohere) | Text **and** images — the Section 90 answer |
| Reranker | `Cohere-rerank-v4.0-fast` | Try Azure AI Search's semantic ranker first |

### Gotchas

- **Claude on Foundry does not support `temperature` or `top_k`**, and `top_p`
  must be 0.99. Deterministic value parsing therefore goes on GPT.
- **Claude uses the native Messages API** (`/anthropic/v1/messages`), not the
  OpenAI-compatible surface. Separate client code.
- **Batch API** covers GPT-5.4, 5.4-mini, 5, 4.1 series, o3, o4-mini — *not*
  the 5.6 family.
- **Long-context pricing trap:** on GPT-5.6 and GPT-6, prompts over 272,000
  tokens are charged at the long-context rate for the **entire request**.
- **Data residency:** Claude Data Zone Standard exists only for `opus-4-8`,
  `opus-5`, `sonnet-5` — **not** Haiku 4.5.
- **Cohere rerank v4.0 supports Hindi, not Urdu.** Roman Urdu is lexically
  close, but test it rather than assuming.
- **Dedupe extraction across manuals.** Cummins codes (`CA111`, `CA441`)
  repeat across dozens of Komatsu manuals. A `code + controller family`
  registry cuts extraction cost 60–70%.

---

## 8. How to work on this

Every change — feature or fix — gets tested at **three levels plus an
overfitting guard**:

- **MODULE** — the new logic in isolation
- **PIPELINE** — full orchestrator run; the eight regression counts must not
  move; 49 agent sessions must stay good 7/7, bad 0/7
- **E2E** — over HTTP, asserting transcripts identical to in-process
- **OVERFITTING GUARD** — held-out split reported separately, independent
  derivation asserted over the parsed AST, mutation table naming which check
  catches which fault, negative coverage on every gate

### Standing rules

- **No LLM** in extraction, audit, fidelity, symptom matching, or Tier 1
  evaluation. Determinism is the product.
- **Finding and fix in separate commits.** When a defect surfaces mid-task,
  **report and stop**. This judgement has been correct every single time it was
  exercised.
- **Do not edit a test to accommodate new code** unless the old assertion was
  itself wrong — and say which.
- **Stop at a clean boundary** rather than claiming completion. Four honest
  commits beat one that overstates.
- **The PDF stays outside the repo.** Komatsu copyrighted material. `.gitignore`
  is the safety net; keeping it out is the policy.

### Environment

```
KOMATSU_PDF=../komatsu-manuals/SEN06867-13.pdf
python -m pip    # pip and python point at different interpreters on this machine
```

**Watch for silent file reverts.** Edits to some source files were being undone
mid-session — likely OneDrive/Dropbox sync or an editor auto-format. If the
project directory sits inside a sync folder, move it out.

---

## 9. Resuming in one paragraph

The foundation is essentially complete: 174 failure codes and 57 symptom trees
extracted deterministically from the manual, every fact carrying its own
page-level provenance and verified against the PDF, zero outstanding defects,
a LangGraph agent that executes the manual's own decision trees with proven
replay determinism, a FastAPI layer proven identical to in-process, and an
11-stage orchestrator that gates the whole thing and halts rather than
producing a false green. Two gaps are declared and visible in every run:
27 machine-repaired steps have no human verification, and `step_procedure`
facts have no single page. What remains is feature work — real LLM
integration, Tier 2 evaluation with a cross-family judge, and the Azure AI
Search index that started all of this and has still not been probed.
