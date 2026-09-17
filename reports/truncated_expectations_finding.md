# Finding — qa_set expectations truncated mid-word

**Date:** 2026-09-17 · **Component:** `eval/build_qa_set.py:309,548` · **Status:** reported, pinned, not fixed

## The question that was asked

Fragments like `'Repair o'`, `'Rep'`, `'Re'`, `'Inst'` surfaced while checking
short bullet units in `736c209`. Two possibilities:

- **(a)** an artefact of splitting facts on bullets — `golden/` fine, splitter
  needs a note
- **(b)** a real truncation — `golden/` carries truncated facts and everything
  downstream is built on them

## The answer: (b), but narrower than (b) as feared

Determined against the **raw JSON**, with no splitting involved: the strings are
genuinely truncated. `'• The wiring harness or a connector is defective. • Repair o'`
ends mid-word in the file.

**But the truncation is in `golden/qa_set.json`, not in the ground truth.**
`golden/failure_codes/*.json` and `golden/symptoms/*.json` hold the complete
string:

```
qa_set: '• A wiring harness or connector is defective. • Repair or re'
golden: '• A wiring harness or connector is defective. • Repair or replace the
         defective wiring harness or connector. • Go to “Confirmation of repair”.'
```

## Cause

`eval/build_qa_set.py`, twice — line 309 (Section 40) and line 548 (symptoms):

```python
"must_contain": [outcome[:60]],
```

A bare `[:60]` with no comment recording why. `expected.outcome` on the same
case keeps the **full** string; only `must_contain` is cut.

## Count and scope

| | |
|---|---:|
| `branch_following` cases with a truncated `must_contain` | **153** |
| of those, ending **mid-word** | **134** |
| ending at a word boundary by luck | 19 |
| all exactly 60 characters | yes |
| `must_contain_verbatim` affected | **0** |
| other case types affected | **0** |
| `golden/` records affected | **0** |

Split by section: 116 Section 40, 18 symptoms.

## Is the truncated text what the PDF shows?

No — and the ground truth already proves it. The full strings live in
`golden/`, and `branch` is one of the two `GATED_KINDS` in `pipeline/fidelity.py`,
gated at 100%: every branch fact in `golden/` resolves against the page it cites
in the source PDF. The fidelity level passes. So the manual prints the complete
outcome, `golden/` records it completely, and `qa_set` cuts it afterwards.

## Consequence

**Permissive, not incorrect.** `must_contain` is a containment check, so an
answer carrying the full correct outcome passes, and an answer that stops after
60 characters also passes. Nothing is marked wrong for being right.

What it does mean: for `branch_following`, "the whole expected answer" is not
what the harness requires. Any metric reading `must_contain` — `content_recall`
and the relevance judgement behind `golden_facts` — under-requires on those 153
cases. `expected.outcome` is intact, so a fix has somewhere correct to read from.

What it does **not** touch: the eight regression counts (they count records,
steps, measurements and branches, not expectation strings), `citations`,
`fidelity`, or `golden/` in any form.

## The actual gap

Not the slice. The slice is one line and a decision someone could defend. The
gap is that **nothing was looking** — the same shape as the README reproduction
claim. A qa_set expectation that is a strict prefix of its own golden source,
cut mid-word, is mechanically detectable, and no check detected it.

`tests/test_extraction.py` now does, over **both** sections:

- an expectation is truncated when it is a **strict prefix** of a string in its
  own golden record and the characters either side of the cut are both
  alphanumeric — derived, with no heuristic about what a word is
- the count is **pinned at 134**, not gated at zero, because the fix moves
  qa_set expectations and lands separately. Pinning stops it growing, which is
  exactly the property missing when the slice was introduced: the check would
  have gone 0 → 153 at that commit
- `must_contain_verbatim` is asserted to contain **no** fragment, at zero —
  a verbatim expectation that is a fragment would be a correctness bug, not a
  permissive one
- a self-test plants a mid-word expectation and requires it to be seen, with the
  cut position derived rather than a fixed offset that might land on a space

## Not fixed here

Widening or removing the cut changes 153 qa_set expectations. That moves
`content_recall` and the relevance judgement for every `branch_following` case,
so it wants its own commit with before/after per system — and it should happen
before the sealed set is spent, since 189 sealed cases read the same builder.
