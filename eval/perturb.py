#!/usr/bin/env python3
"""
perturb.py
Deterministic rewrites of a question, to measure whether selection survives a
technician who does not type the template.

WHY THIS EXISTS
---------------
`eval/build_qa_set.py` templates the measuring point into the question. Over
the 1,183 dev cases with an expected point, the point appears VERBATIM in the
question **701 of 701 times -- 100%**. So any question-anchored selector is
being measured under conditions far friendlier than reality, and a clean
reachability number on this corpus is close to meaningless on its own.

These rewrites are the correction. They change how the question is written
while keeping what it asks, so a selector that only works on template wording
shows a gap, and that gap is the honest estimate.

ENGLISH ONLY, AND NOT A PHRASE LIST
-----------------------------------
Technicians using this manual write English technical terms inside romanised
Hinglish scaffolding, which is what the corpus already contains. Nothing here
adds a language; every rewrite is a STRUCTURAL operation on the text that is
there -- case, punctuation, number formatting, dropping a qualifier, one
abbreviation table, one character. None of them is a guess about how a model
phrases anything, so Rule 4 does not apply: this is our text being rewritten,
not a model's output being matched.

Deterministic: every choice is keyed on a sha256 of the question, never on an
RNG. Running twice gives the same set, so a perturbed measurement is
reproducible and can be diffed.

WHAT THESE DO NOT COVER, stated rather than discovered later
------------------------------------------------------------
  - A technician who names the connector something the manual never calls it.
    There is no synonym source here that is not a model.
  - Word order changes and free-form sentences. Every rewrite keeps the
    question's structure; only its surface changes.
  - Misread pin numbers. A wrong number is a different question, not a
    perturbation of this one.
"""
from __future__ import annotations

import hashlib
import re

# One table, and it is ours: the abbreviations a technician types for the
# manual's own quantity words. Not a list of ways a model might phrase itself.
ABBREV = {"resistance": "res", "voltage": "volts", "continuity": "cont",
          "pressure": "press", "temperature": "temp"}

_PIN_PAIR = re.compile(r"\((\d{1,3})\)\s+and\s+\((\d{1,3})\)")
_QUALIFIER = re.compile(r"\s*\((?:female|male)\)")
_PUNCT = re.compile(r"[^\w\s()]")
_WORD = re.compile(r"[A-Za-z]{6,}")


def _pick(q: str, n: int) -> int:
    return int(hashlib.sha256(q.encode("utf-8")).hexdigest()[:8], 16) % n


def lower(q: str) -> str:
    """Typed without the shift key."""
    return q.lower()


def strip_punct(q: str) -> str:
    """No quotes, commas or question marks. Parentheses kept -- removing them
    is the loose_pins rewrite's job, and doing both here would make one
    rewrite two."""
    return _PUNCT.sub("", q)


def loose_pins(q: str) -> str:
    """"(37) and (44)" the way it actually gets typed. One of three forms,
    chosen by hash so the set is reproducible and not all one shape."""
    form = _pick(q, 3)
    def rep(m):
        a, b = m.group(1), m.group(2)
        return (f"pins {a} and {b}", f"{a}-{b}", f"{a} & {b}")[form]
    return _PIN_PAIR.sub(rep, q)


def short_names(q: str) -> str:
    """"ECM (female)" -> "ECM". The qualifier is in the manual; nobody types
    it."""
    return _QUALIFIER.sub("", q)


def abbreviate(q: str) -> str:
    """res, volts, cont. Whole words only, case-insensitive."""
    out = q
    for full, short in ABBREV.items():
        out = re.sub(rf"\b{full}\b", short, out, flags=re.I)
    return out


def typo(q: str) -> str:
    """One character wrong, in one non-numeric word of 6+ characters.

    Numbers are left alone deliberately: a mistyped pin number is a different
    question, and scoring a selector for failing to find the answer to a
    question that was not asked would be measuring the wrong thing.
    """
    words = [m for m in _WORD.finditer(q)]
    if not words:
        return q
    m = words[_pick(q, len(words))]
    w, k = m.group(0), len(m.group(0)) // 2
    ch = w[k]
    nxt = chr((ord(ch.lower()) - 97 + 1) % 26 + 97)
    return q[:m.start()] + w[:k] + (nxt.upper() if ch.isupper() else nxt) + \
        w[k + 1:] + q[m.end():]


def combined(q: str) -> str:
    """All of them at once -- the worst realistic case, not an average one."""
    for f in (lower, strip_punct, loose_pins, short_names, abbreviate, typo):
        q = f(q)
    return q


PERTURBATIONS = {"lower": lower, "strip_punct": strip_punct,
                 "loose_pins": loose_pins, "short_names": short_names,
                 "abbreviate": abbreviate, "typo": typo,
                 "combined": combined}


def apply(name: str, q: str) -> str:
    return PERTURBATIONS[name](q)


def self_test() -> None:
    """Each rewrite must actually CHANGE a real question, or a perturbed
    measurement is the unperturbed one under another name."""
    q = ('CA122 ke liye Between ECM (female) (37) and (44) par resistance '
         'ki standard value kya honi chahiye?')
    changed = {n: f(q) for n, f in PERTURBATIONS.items()}
    for n, out in changed.items():
        assert out != q, f"perturbation {n} is a no-op on a template question"
    assert changed["lower"] == q.lower()
    assert "?" not in changed["strip_punct"]
    assert "(37) and (44)" not in changed["loose_pins"]
    assert "(female)" not in changed["short_names"]
    assert "res " in changed["abbreviate"] and "resistance" not in changed["abbreviate"]
    # The typo must be one character, in a letter word, and not in a number.
    t = changed["typo"]
    assert len(t) == len(q) and sum(a != b for a, b in zip(t, q)) == 1
    assert re.findall(r"\d+", t) == re.findall(r"\d+", q), \
        "the typo landed in a number, which makes it a different question"
    # Deterministic.
    assert combined(q) == combined(q)
    for n, f in PERTURBATIONS.items():
        assert f(q) == f(q), f"{n} is not deterministic"


if __name__ == "__main__":
    self_test()
    q = ('CA122 ke liye Between ECM (female) (37) and (44) par resistance '
         'ki standard value kya honi chahiye?')
    print(f"{'original':14} {q}")
    for n, f in PERTURBATIONS.items():
        print(f"{n:14} {f(q)}")
    print("\nperturb self-test passed")
