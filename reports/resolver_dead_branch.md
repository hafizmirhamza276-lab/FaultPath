# Finding — `kind == "header"` is unreachable in the citation resolver

Recorded before the fix and separately from it, as with
`reports/false_green_findings.md`.

## What is wrong

`eval/citations.py:_fact()` dispatches on the fact kind. It handles one kind
before looking up a step, and every other kind after:

```python
if kind == "meas" and step == 0:
    ...standalone measurements, which legitimately sit at step 0...

st = next((s for s in rec["steps"] if s["step"] == step), None)
if st is None:
    raise UnknownFact(f"no step {step} on {code}")

if kind == "step":   ...
if kind == "meas":   ...
if kind == "branch": ...
if kind == "remedy": ...
if kind == "header": ...   # <-- can never be reached
```

A header fact is a property of the RECORD, not of a step, so its fact id sits
at step 0 — the same shape `standalone_measurements` uses. But the step lookup
runs first and raises for step 0, because no record has a step numbered 0.

The branch is dead. It reads as support for citing a code's header fields and
provides none.

```
CA131:0:header:0       UnknownFact: 'no step 0 on CA131'
CA131:0:header:1       UnknownFact: 'no step 0 on CA131'
CA131:0:header:4       UnknownFact: 'no step 0 on CA131'
602KNX:0:header:4      UnknownFact: 'no step 0 on 602KNX'
```

Reaching it requires a non-zero step — `CA131:1:header:0` — which would find
step 1 and then return a record-level field as though it belonged to that step.
So the only ids that reach the branch are ids that should not exist.

## Why it stayed invisible

Nothing mints header fact ids. `build_qa_set.py` never emitted one, so no case
carried one, so no citation was ever rendered from one and no test exercised
the path. `UnknownFact` is caught and turned into "no citation" by
`agent/tools.py:render_citations`, which is correct behaviour for an invented
id and indistinguishable from this.

The same shape as audit checks E4 and H2: a branch that cannot fire, sitting
quietly among branches that can, and looking like coverage.

## Scope

This is a defect in the resolver only. It is independent of whether any case
ought to cite a header field — that question is separate and is answered
separately. `pipeline/extract_golden.py` is not involved: `header_provenance`
is already captured, complete on all 174 records.

Fix and its proof follow in the next commit.
