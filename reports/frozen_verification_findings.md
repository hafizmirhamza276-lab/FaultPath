# Finding — fact verification is a frozen snapshot, not a derived value

Recorded before the fix, as with `false_green_findings.md` and
`resolver_dead_branch.md`.

## What is wrong

`reports/fact_verification.json` has not been rebuilt since `e907cf4`.
`core/loader.py:build_verification()` — the function that would rebuild it —
**has no caller anywhere in the repo**. The artefact is read as a given.

```
git log --oneline -- reports/fact_verification.json
  e907cf4 Add extraction fidelity: measure whether golden/ says what the PDF says
```

## The consequence is worse than a stale count

`load_ground_truth()` stamps every fact with
`by_fact.get(fact_id, UNVERIFIED)`. A fact absent from the artefact is not
reported as *unknown*; it is reported as **unverified**, which is a claim.
`api/contracts.py` then surfaces that stamp to the caller.

Measured against a current fidelity run:

| | |
|---|---:|
| facts in the frozen artefact | 3,305 |
| facts in the live fidelity run | 6,554 |
| live facts **absent** from the artefact | **3,249** |
| ghosts (in the artefact, no longer live) | 0 |
| **of the 3,249, how many the resolver actually verifies** | **2,757** |

**2,757 facts are reported `unverified` to an API caller when the resolver
verifies them against the PDF.** The agent understates its own grounding, and
it does so silently, because a missing key and a genuinely unverified fact are
the same value.

The understated facts, by section and kind:

| section | kind | n |
|---|---|---:|
| hmode/smode | branch | 666 |
| hmode/smode | cause | 542 |
| hmode/smode | measurement | 274 |
| hmode/smode | remedy | 209 |
| hmode/smode | step_procedure | 333 |
| hmode/smode | symptom_title | 57 |
| section40 | step_procedure | 994 |
| section40 | title | 174 |

Every symptom fact is in that list — the artefact predates symptom trees
entirely. So does every `step_procedure`, and the `title` kind added in
`856fd54`.

## Why it stayed invisible

`resolver_verified: 3266` is pinned in **two** places:

- `METRICS.md:292`
- `tests/test_orchestrator.py:373`

Both read the same frozen file, so they agree with each other, and their
agreement reads as corroboration. It is the same shape as the shared
`NOT_CEILING` reason in `ceiling.py`, which fitted `citation_resolvability`
and never fitted `uncited_claim_rate`: two things pointing at one source and
the agreement mistaken for evidence.

What the artefact would say if derived from the current run:

```
resolver_verified   3266  ->  6023
unverified            39  ->   531
total               3305  ->  6554
human_verified         0  ->     0
```

`unverified` 531 is exactly the unresolved-facts total, as it should be.

## A second declared-but-empty bucket

`pipeline/fidelity.py:FACT_KINDS` has **no consumer anywhere**. It lists
`action_level`, which nothing enumerates, and listed `title`, which nothing
enumerated until `856fd54`.

Same shape as the dead `header` branch in `resolver_dead_branch.md`: a
declaration that reads as coverage and provides none. Not fixed here.

## What the fix has to be

Regenerating once does not stop it going stale again. The artefact must be
derived from the fidelity run inside the pipeline, and something must fail when
what is on disk disagrees with what the current run produces — both sides
derived, as with LEVELS/STAGES and CEILING/NOT_CEILING.

`human_verified` must stay 0. Nobody has read the pages, and the docstring is
right that claiming otherwise is the self-agreement this exercise exists to
avoid.

Fix follows separately.
