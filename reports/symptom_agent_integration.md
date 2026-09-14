# Symptom entry, wired into the agent

The second entry point. A technician with a failure code gets a dict lookup; a
technician with a symptom gets `core/symptom_match.py`, and this is what the
graph does with the answer.

## tree_kind drives behaviour, and the two kinds INVERT

Not "differ" — invert. This is the whole design, and the reason for a separate
verdict pair rather than a reused one.

| | branching (37 H-Mode) | flat (20 S-Mode) |
|---|---|---|
| stored per step | `branches{YES, NO}` | `cause`, `point_to_check`, `remedy` |
| question asked | "Is it normal? Answer yes or no." | "Is that what you are seeing?" |
| affirmative means | the check was normal | **you are looking at the fault** |
| affirmative does | ADVANCE | **STOP, and give the remedy** |
| verdict recorded | `YES` / `NO` | `OBSERVED` / `NOT_OBSERVED` |
| `Awaiting` state | `READING` | `OBSERVATION` |

Asking YES/NO on a flat tree invents an interaction the manual does not have —
there are no branch outcomes to read back. Worse, a flat tree walked with
branching polarity runs every row, matches nothing and hands over, or matches
the wrong one: **plausible output, no error**. That is why the verdicts are
different enum members and not two spellings of one, and why
`symptom_flat_polarity` is gated at 1.0 with a bad agent aimed at it.

Asserted structurally, not just behaviourally:

```
no flat tree carries a single branch outcome        PASS
every flat row carries a remedy of its own          PASS
OBSERVED/NOT_OBSERVED are distinct from YES/NO      PASS
```

## ASK is a state, not an error path

`resolve_entry` gets one of three things back and never collapses them.

- **MATCHED** → enter the tree.
- **ASK** → emit every candidate with its manual page, set
  `Awaiting.SYMPTOM_CHOICE`, and pause. The agent does not pick. The candidate
  list is kept on the state so the next turn resolves *the question that was
  asked* — re-running retrieval against a differently worded reply could
  silently change what the technician is answering about.
- **UNMAPPED** → say so, offer a failure code or the section where the topic
  actually lives, and stop. Never the nearest tree.

`swing slow hai` returns HM28 and HM29:

```
I need one more thing to be sure. The manual has more than one troubleshooting
tree that could fit what you described:
1. Swing Acceleration or Swing Speed is Low in Two Directions of Right and Left (page 40-900)
2. Swing Acceleration Performance is Unsatisfactory or Swing Speed is Slow in Only One Direction (page 40-906)
Which one matches? Give me the number.
```

The pick is resolved by `tools.resolve_symptom_choice`, deterministically and
narrowly: the symptom id, its position in the list as offered, a Roman Urdu
ordinal, or a word distinctive to exactly one candidate. Anything else asks
again. A matcher that refuses to guess followed by a picker that guesses is a
matcher that guesses.

Over HTTP, `awaiting` is its own literal (`symptom_choice`) and the candidates
cross the boundary as structured data, so a UI never has to parse the prose to
know it is being asked a question.

## Remedy is a first-class field

The manual gives it its own column and its own cell, so it gets its own
`fact_id`, its own provenance, its own state field and its own API field — not
a sentence appended to the diagnosis that a caller would have to split back out.

```
Diagnosis: Breakage of flywheel ring gear
Remedy:    Replace if the item is broken
```

`remedy_correct` is gated apart from `diagnosis_correct` because they fail
differently: the wrong row gets both wrong, but the **right** row with the
remedy replaced by an invented instruction gets the diagnosis right and the
remedy wrong. One number could not tell those apart.

## Prose pointers — surfaced, never followed

Of the 182:

| where | n | handling |
|---|---|---|
| step `procedure` | 145 | supplementary detail; the step is still executable, so the pointer is surfaced alongside it |
| `related_information` | 35 | the pre-troubleshooting note, already handled by preflight |
| step `remedy` | **2** | the tree *ends* in a pointer |

Only the last kind is "a symptom tree ending in a prose pointer", and **both
instances are SM01 rows 1 and 2**. The fixture is built from SM01 because it is
the only real instance in the corpus — one passing test here is not broad
coverage of pointer handling, and should not be read as such.

```
Diagnosis: Defective starting circuit wiring system
Remedy: Perform the troubleshooting for "ENGINE DOES NOT START (ENGINE DOES NOT
CRANK)" in E mode, and take corrective action.
Note: the manual does not carry the procedure here -- it refers you to another
section (page 40-931). I am not going to guess which tree that is.
```

## A failure code produced mid-symptom takes over

This is the manual's instruction, not a preference. Every H-Mode tree's
`related_information` opens *"Pre-troubleshooting: If a failure code is shown,
do the troubleshooting for that code first."* — 35 of the 182 pointers **are**
that sentence.

In a *code* session the same reply means something different: those are
concurrent codes for `pending_codes`, to be worked afterwards. Only the symptom
session hands over.

## Graph changes

`resolve_entry`'s symptom path, `tree_kind` dispatch in execute/parse/evaluate,
`remedy` and `prose_pointer` on conclude. Plus four beyond that, each listed
because "expect no other graph changes" was the instruction:

1. **`e_receive`** routes `SYMPTOM_CHOICE` to `intake`. No new edge — the pick
   is technician text becoming entry state, which is what intake is for.
2. **`n_identify_machine`** no longer clears `awaiting` unconditionally. It was
   wiping an unanswered `SYMPTOM_CHOICE` on the way past, and `resolve_entry`
   then asked the same question twice in one turn.
3. **`n_intake`** falls through to symptom when candidates were proposed, none
   is real, and none is code-shaped. The manual's own S-Mode title contains
   `START` — five upper-case characters — so the extractor offered it as a code
   candidate and the whole title stopped being a symptom. Disposed of in the
   graph, not by tightening the model's regex, which would drop `DAFQKR` and
   `B@BAZG`.
4. **`n_preflight`** implements the code takeover described above.

## Two defects found while wiring, both reported

### `eval/citations.py` ignored `branch_provenance`

The S4 fix captured per-branch provenance at parse time and `pipeline/fidelity.py`
consumes it — but the **citation renderer, the thing that puts a page in front
of a technician, did not**. It read the step's provenance for branch facts.

That is the mutation "provenance captured but not used by the resolver", live.

Measured rather than assumed: 2,103 branch and remedy facts carry their own
provenance; **one** is on a different page from its step.

```
HM22 step 5, NO branch:  step is on 40-857, the branch prints on 40-858

resolve() against the actual PDF:
  NEW (branch provenance): 40-858 -> True
  OLD (step provenance)   : 40-857 -> False  verbatim text NOT on the cited page
```

The old behaviour shipped a citation whose text is not on the page it names.
Fixed in the same edit that taught `_fact` about symptom records, since both
required changing that function. **This changes one existing Section 40
citation** and is called out here rather than buried in a diff.

### The graph guarded on evidence it did not record

Caught by the E2E transcript comparison, which is what that check is for. `emit()`
passed the symptom candidates' pages to `enforce()` but stored only the caller's
`grounded_text` on the `Emission`. The API boundary guard re-runs the guard on
what was *recorded*, so it blocked an ASK the graph had allowed, and HTTP
diverged from in-process.

Fixed by building the evidence list once and both using and recording it.
Guarding on more than is recorded is the same defect as recording more than is
guarded.

## Results

Six sessions, one per behaviour. Not a sample of trees — running fifty branching
symptom trees would grow the number and test one thing.

| session | tree | outcome |
|---|---|---|
| `symptom_branching` | HM01 branching | concluded on the NO branch |
| `symptom_flat_remedy` | SM01 flat, row 3 | concluded, real remedy |
| `symptom_ask` | HM28\|HM29 → HM29 | asked, then concluded |
| `symptom_unmapped` | none | escalated without entering a tree |
| `symptom_pointer` | SM01 flat, row 1 | concluded, remedy is a pointer |
| `symptom_to_code` | HM01 → 602KNX | switched, concluded on the code |

Symptom gates are scored **apart** from the 45 code sessions. Folding six into
45 would let the majority carry a broken symptom path over every threshold while
the averages kept looking healthy.

| gate | n | good | bad | separation |
|---|---|---|---|---|
| `symptom_right_tree` | 5 | 1.0000 | 0.6000 | yes |
| `symptom_asks_on_ambiguity` | 1 | 1.0000 | 0.0000 | yes |
| `symptom_unmapped_not_routed` | 1 | 1.0000 | 0.0000 | yes |
| `symptom_flat_polarity` | 2 | 1.0000 | 0.0000 | yes |
| `symptom_pointer_surfaced` | 1 | 1.0000 | 0.0000 | yes |
| `symptom_code_takes_over` | 1 | 1.0000 | 0.0000 | yes |
| `remedy_correct` | 2 | 1.0000 | 0.0000 | yes |

**There is deliberately no gate on how often the agent asks.** A ceiling on the
ask rate is how a system gets tuned into guessing, and `symptom_wrong_tree_rate`
already carries the cost of guessing wrong. `symptom_right_tree` and
`symptom_asks_on_ambiguity` are never averaged: an agent that never asks and is
usually right would score well on a mean of the two while being exactly the
guessing machine the ASK outcome exists to prevent.

A gate scored on zero sessions is also a failure, not a pass — that is the
E4/H2 shape, and `every symptom gate is scored on at least one session` fails
loudly rather than printing a dash.
