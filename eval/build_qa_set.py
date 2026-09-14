#!/usr/bin/env python3
"""
build_qa_set.py
Derives golden/qa_set.json from golden/failure_codes/*.json.

    python eval/build_qa_set.py

Deterministic. No LLM, no network, no randomness -- not even a seeded RNG,
because nothing here needs one: every case is a pure function of the ground
truth, and every expected answer is copied verbatim out of it. Nothing is
paraphrased and nothing is invented. That is what makes numeric_exactness a
defensible number: if the assistant returns a string this file did not read out
of the manual, it made it up.

Questions are written in Roman Urdu / Hinglish, which is how technicians
actually type. The expected answers stay in the manual's English, verbatim.

WHY THIS FILE EXISTS
--------------------
qa_set.json was committed with no producer. It could not be regenerated, so when
CA451's criterion was corrected in the ground truth the test set kept requiring
the old corrupt value -- '.2 to 4.6V' -- as a verbatim answer, penalising a
correct assistant on a common-rail sensor voltage. A dataset that cannot be
rebuilt cannot be corrected. This ends that.
"""
import json
import glob
import os
import re
import sys
import hashlib
import collections

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
OUT_PATH = os.path.join(GOLD, "qa_set.json")

MODEL = "PC200-10M0"
MANUAL_ID = "SEN06867-13"
WRONG_MODEL = "PC490LC-11"

FILTERS = {"model": MODEL, "manual_id": MANUAL_ID}

# Fabricated codes: shaped like real ones, absent from the manual. The only
# correct behaviour is to say so.
FAKE_CODES = ["CA999", "CA1234", "B@BZZZ", "D9XQKR", "CA118", "X0X0X0"]

# Real codes asked about the wrong machine. Retrieval will happily return the
# PC200 page; pin numbers and resistances differ per model.
WRONG_MODEL_CODES = ["CA131", "CA441", "AB00KE", "CA2186", "F@BBZL", "DA2QKR"]

TRAP_UNKNOWN = ("A code that looks plausible but is not in the manual. "
                "Any procedure returned here is a pure hallucination.")
TRAP_MODEL = ("{code} exists in the PC200-10M0 manual, so retrieval will happily "
              "return it. Pin numbers and resistances differ per model.")
TRAP_HOP = "Answering from this code's own page alone gives the technician nothing."
TRAP_PRECOND = "Starting with the wrong code wastes the whole session."
NOTE_HOP = "This code has no standalone procedure; it points elsewhere."

# A step that tells the technician to go and solve a different code first.
# solve_first is the FIRST code named in that sentence, in textual order -- not
# the first entry of refs_failure_codes. CA451 says "[CA227] or [CA187]" and
# CA123 says "[CA187] or [CA227]"; they resolve differently, and only reading
# the sentence gets that right.
PRECONDITION_RE = re.compile(
    r"If (?:the )?failure code.*?do the troubleshooting", re.I | re.S)

CODE_REF_RE = re.compile(r"\[([A-Z0-9@#]{4,7})\]")


SECTION40 = "section40"
SYMPTOMS = "symptoms"


def load_codes():
    paths = sorted(glob.glob(os.path.join(GOLD, "failure_codes", "*.json")))
    if not paths:
        sys.exit(f"ERROR: no failure-code JSON found under {GOLD}/failure_codes/\n"
                 "  Run pipeline/extract_golden.py first.")
    recs = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["code"]] = r
    return recs


def load_symptoms():
    paths = [p for p in sorted(glob.glob(os.path.join(GOLD, "symptoms", "*.json")))
             if os.path.basename(p) != "index.json"]
    if not paths:
        sys.exit(f"ERROR: no symptom JSON found under {GOLD}/symptoms/\n"
                 "  Run pipeline/extract_symptoms.py first.")
    recs = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["symptom_id"]] = r
    return recs


def page_of(rec):
    """Fallback page for a code-level fact, e.g. '40-181'."""
    return rec["manual_pages"][0] if rec["manual_pages"] else None


def page_of_fact(obj, rec):
    """The page the FACT is on, not the page its code starts on.

    146 of 174 codes span more than one page and one runs to 11, so 75% of
    measurements do not sit on their code's first page. Citing the first page
    was wrong for three quarters of numeric cases -- and citation_accuracy
    still scored 1.0000, because it was comparing the answer against the same
    wrong page the test set had recorded.
    """
    prov = (obj or {}).get("provenance") or {}
    return prov.get("manual_page") or page_of(rec)


def is_numeric_criterion(criteria):
    """Continuity checks and audible confirmations have no numeric value by
    design (audit finding E5). They are real measurements but they cannot test
    numeric exactness, so they are not turned into cases."""
    return bool(re.search(r"\d", criteria or ""))


def measurements_of(rec):
    """(measurement, step_number_or_None) for every measurement on a record."""
    out = [(m, None) for m in (rec.get("standalone_measurements") or [])]
    for st in rec.get("steps", []):
        out += [(m, st["step"]) for m in (st.get("measurements") or [])]
    return out


def is_numeric_symptom_criterion(m):
    """Symptom measurements carry criteria_kind. USE IT, not the has-a-digit rule.

    The two disagree on 40 of the 42 relational criteria, and the disagreements
    are the whole point:

        "Oil pressure ratio pump discharged pressure: PC valve discharged
         pressure = 1 : 0.6 (approximately 3/5)"

    has digits, so Section 40's is_numeric_criterion calls it numeric -- but it
    is a RATIO between two readings, not a value read off an instrument. Pinning
    it as must_contain_verbatim would test whether an assistant can quote a
    sentence, dressed up as testing whether it can report a measurement.

    Section 40 has no criteria_kind field, so it keeps the digit rule; the
    symptom extractor classifies at parse time and that classification is
    better evidence than a regex over the result of it.
    """
    return m.get("criteria_kind") == "numeric"


# ------------------------------------------------------------------ case types

def case_direct_lookup(rec):
    """code -> title, action level, what the operator sees."""
    if not rec.get("machine_effect"):
        return None
    return {
        "type": "direct_lookup",
        "section": SECTION40,
        "difficulty": "easy",
        "question": f"Failure code {rec['code']} aa raha hai {MODEL} par. Ye kya hai?",
        "filters": dict(FILTERS),
        "expected": {
            "title": rec.get("title"),
            "action_level": rec.get("action_level"),
            "machine_effect": rec.get("machine_effect"),
        },
        "must_contain": [rec.get("title")] if rec.get("title") else [],
        "must_cite_page": page_of(rec),
        "must_not_refuse": True,
        "source_code": rec["code"],
    }


def cases_numeric_exactness(rec):
    """One case per measurement carrying a numeric criterion.

    must_contain_verbatim holds the criterion EXACTLY as the ground truth stores
    it -- the whole string, never a fragment of it. If the manual says
    'Sensor output 0.2 to 4.6V' then that is the expected value; trimming it to
    '0.2 to 4.6V' would bake a different bug into the test set.
    """
    out = []
    for m, step in measurements_of(rec):
        criteria = m.get("criteria") or ""
        if not is_numeric_criterion(criteria):
            continue
        point = m.get("point") or ""
        quantity = m.get("quantity") or ""
        out.append({
            "type": "numeric_exactness",
            "section": SECTION40,
            "difficulty": "hard",
            "question": (f"{rec['code']} ke liye {point} par {quantity.lower()} "
                         "ki standard value kya honi chahiye?"),
            "filters": dict(FILTERS),
            "expected": {
                "quantity": quantity,
                "point": point,
                "criteria": criteria,
                "step": step,
            },
            "must_contain_verbatim": [criteria],
            "must_cite_page": page_of_fact(m, rec),
            "must_not_refuse": True,
            "fact_ids": [m["fact_id"]] if m.get("fact_id") else [],
            "source_code": rec["code"],
        })
    return out


def case_step_ordering(rec):
    """Which check comes first.

    Pointer-only codes are excluded: their whole procedure is 'go and
    troubleshoot X', so there is no first check to name. Asking one for its
    opening step would have no correct answer -- they are covered by
    cross_ref_hop instead. Redirect steps are skipped for the same reason.
    """
    if rec["is_pointer_only"]:
        return None
    steps = [s for s in rec["steps"] if not s.get("redirect")]
    if not steps:
        return None
    first = steps[0]
    return {
        "type": "step_ordering",
        "section": SECTION40,
        "difficulty": "medium",
        "question": f"{rec['code']} ke liye sab se pehla check kya karna chahiye?",
        "filters": dict(FILTERS),
        "expected": {
            "step": first["step"],
            "cause": first.get("cause"),
            "procedure": first.get("procedure"),
        },
        "must_contain": [first["cause"]] if first.get("cause") else [],
        "must_cite_page": page_of_fact(first, rec),
        "must_not_refuse": True,
        "fact_ids": [first["fact_id"]] if first.get("fact_id") else [],
        "source_code": rec["code"],
    }


def case_branch_following(rec):
    """Format-A codes are decision trees. Take the first step offering a NO
    outcome and check the assistant walks to the right next state."""
    if rec["format"] != "A":
        return None
    step = next((s for s in rec["steps"] if "NO" in s.get("branches", {})), None)
    if step is None:
        return None
    outcome = step["branches"]["NO"]
    return {
        "type": "branch_following",
        "section": SECTION40,
        "difficulty": "medium",
        "question": (f"{rec['code']} ka troubleshooting kar raha hun. "
                     f"Step {step['step']} ({step.get('cause','')}) ka result NO aaya. "
                     "Ab kya karun?"),
        "filters": dict(FILTERS),
        "expected": {"step": step["step"], "branch": "NO", "outcome": outcome},
        "must_contain": [outcome[:60]],
        "must_cite_page": page_of_fact(step, rec),
        "must_not_refuse": True,
        "fact_ids": [(step.get("branch_fact_ids") or {}).get("NO")]
                    if (step.get("branch_fact_ids") or {}).get("NO") else [],
        "source_code": rec["code"],
    }


def derive_precondition(rec, known_codes):
    """The code that must be solved first, read out of the step text."""
    for st in rec["steps"]:
        m = PRECONDITION_RE.search(st.get("procedure") or "")
        if not m:
            continue
        refs = [c for c in CODE_REF_RE.findall(m.group(0))
                if c != rec["code"] and c in known_codes]
        if refs:
            return refs[0]
    return None


def case_precondition(rec, known_codes):
    first = derive_precondition(rec, known_codes)
    if not first:
        return None
    return {
        "type": "precondition",
        "section": SECTION40,
        "difficulty": "hard",
        "question": (f"Monitor par {rec['code']} aur {first} dono show ho rahe hain. "
                     "Pehle kaunsa solve karun?"),
        "filters": dict(FILTERS),
        "expected": {"solve_first": first, "then": rec["code"]},
        "must_contain": [first],
        "must_cite_page": page_of(rec),
        "must_not_refuse": True,
        "trap": TRAP_PRECOND,
        "source_code": rec["code"],
    }


def case_cross_ref_hop(rec):
    """Pointer-only codes: the whole procedure is 'go and troubleshoot X'.
    Retrieval returns a correct, relevant chunk that helps nobody."""
    if not rec["is_pointer_only"]:
        return None
    refs = rec["refs_failure_codes"]
    if not refs:
        return None
    return {
        "type": "cross_ref_hop",
        "section": SECTION40,
        "difficulty": "hard",
        "question": f"{rec['code']} aa raha hai. Mujhe step by step batao kya check karun.",
        "filters": dict(FILTERS),
        "expected": {"redirect_to": refs[0], "note": NOTE_HOP},
        "must_contain": [refs[0]],
        "must_cite_page": page_of(rec),
        "must_not_refuse": True,
        "trap": TRAP_HOP,
        "source_code": rec["code"],
    }


def cases_adversarial(recs):
    """Refusal tests. These carry no correct chunk: no must_contain, no
    must_cite_page, and must_not_contain_numbers, because a refusal that still
    leaks a measurement value is not a refusal."""
    out = []
    for code in FAKE_CODES:
        if code in recs:
            sys.exit(f"ERROR: adversarial code {code} actually exists in the manual. "
                     "It cannot be used as a fabricated code.")
        out.append({
            "type": "adversarial_unknown",
            "section": SECTION40,
            "difficulty": "hard",
            "question": (f"Failure code {code} aaya hai {MODEL} par, "
                         "iska troubleshooting batao."),
            "filters": dict(FILTERS),
            "expected": {"behaviour": "refuse_not_found"},
            "must_refuse": True,
            "must_not_contain_numbers": True,
            "trap": TRAP_UNKNOWN,
            "source_code": None,
        })
    for code in WRONG_MODEL_CODES:
        if code not in recs:
            sys.exit(f"ERROR: adversarial_model code {code} is not in the manual. "
                     "The trap only works if the code is real for PC200-10M0.")
        out.append({
            "type": "adversarial_model",
            "section": SECTION40,
            "difficulty": "hard",
            "question": f"{WRONG_MODEL} par failure code {code} aaya hai, values batao.",
            "filters": {"model": WRONG_MODEL},
            "expected": {"behaviour": "refuse_wrong_model"},
            "must_refuse": True,
            "must_not_contain_numbers": True,
            "trap": TRAP_MODEL.format(code=code),
            "source_code": code,
        })
    return out


# ------------------------------------------------------------ symptom cases
#
# THESE REUSE THE EXISTING TYPE NAMES and carry a "section" field instead.
#
# generation.py's NON_ADVERSARIAL, retrieval.py's RETRIEVAL_TYPES and several
# APPLIES_TO tuples are keyed on the type STRING. A new name like
# symptom_numeric_exactness would make every one of those skip the new cases
# silently -- each metric would report a clean number over a set it never
# looked at, which is the defect class this project keeps removing. Section is
# a reporting axis; the question being asked is the same question.
#
# symptom_remedy is the one genuinely new type, because a flat S-Mode row has
# no Section 40 analogue. It is registered in NON_ADVERSARIAL, RETRIEVAL_TYPES
# and both synthetic systems in the same commit.
#
# NO direct_lookup: symptom records carry no action_level and no
# machine_effect, and a technician does not type "HM01" -- they describe the
# machine. Matching a description to a tree is entry matching, which is
# core/symptom_match.py's job and a separate piece of work.


def symptom_question_subject(rec):
    """How a technician would name this tree. The manual's own title."""
    return (rec.get("symptom") or "").rstrip(". ")


def cases_symptom_numeric(rec):
    """One case per measurement whose criteria_kind is numeric."""
    out = []
    for m, step in measurements_of(rec):
        if not is_numeric_symptom_criterion(m):
            continue
        criteria = m.get("criteria") or ""
        point = m.get("point") or ""
        quantity = m.get("quantity") or ""
        out.append({
            "type": "numeric_exactness",
            "section": SYMPTOMS,
            "difficulty": "hard",
            "question": (f"\"{symptom_question_subject(rec)}\" ke liye "
                         f"{point} par {quantity.lower()} ki standard value "
                         "kya honi chahiye?"),
            "filters": dict(FILTERS),
            "expected": {
                "quantity": quantity,
                "point": point,
                "criteria": criteria,
                "step": step,
            },
            # The WHOLE criterion string, as in Section 40. Trimming it to what
            # looks like "just the number" is the same bug from the other side.
            "must_contain_verbatim": [criteria],
            "must_cite_page": page_of_fact(m, rec),
            "must_not_refuse": True,
            "fact_ids": [m["fact_id"]] if m.get("fact_id") else [],
            "source_code": rec["symptom_id"],
        })
    return out


def case_symptom_step_ordering(rec):
    """Which check the tree opens with. Both tree kinds have a first row."""
    steps = rec.get("steps") or []
    if not steps:
        return None
    first = steps[0]
    flat = rec.get("tree_kind") == "SymptomTreeFlat"
    return {
        "type": "step_ordering",
        "section": SYMPTOMS,
        "difficulty": "medium",
        "question": (f"\"{symptom_question_subject(rec)}\" -- sab se pehle "
                     "kya check karna chahiye?"),
        "filters": dict(FILTERS),
        "expected": {
            "step": first["step"],
            "cause": first.get("cause"),
            # A flat row has a point to check where a branching step has a
            # procedure. Carrying the wrong one would make the expectation
            # empty for half the trees.
            "procedure": (first.get("point_to_check") if flat
                          else first.get("procedure")),
        },
        "must_contain": [first["cause"]] if first.get("cause") else [],
        "must_cite_page": page_of_fact(first, rec),
        "must_not_refuse": True,
        "fact_ids": [first["fact_id"]] if first.get("fact_id") else [],
        "source_code": rec["symptom_id"],
    }


def case_symptom_branch_following(rec):
    """Branching trees only. A flat tree has no branch outcomes at all, and
    inventing a YES/NO for one would assert something the manual does not."""
    if rec.get("tree_kind") != "SymptomTreeBranching":
        return None
    step = next((s for s in rec.get("steps", [])
                 if "NO" in (s.get("branches") or {})), None)
    if step is None:
        return None
    outcome = step["branches"]["NO"]
    bprov = (step.get("branch_provenance") or {}).get("NO") or {}
    return {
        "type": "branch_following",
        "section": SYMPTOMS,
        "difficulty": "medium",
        "question": (f"\"{symptom_question_subject(rec)}\" ka troubleshooting "
                     f"kar raha hun. Step {step['step']} "
                     f"({step.get('cause','')}) ka result NO aaya. Ab kya karun?"),
        "filters": dict(FILTERS),
        "expected": {"step": step["step"], "branch": "NO", "outcome": outcome},
        "must_contain": [outcome[:60]],
        # The BRANCH's own page, not the step's. HM22 step 5's NO prints on
        # 40-858 while its step is on 40-857.
        "must_cite_page": bprov.get("manual_page") or page_of_fact(step, rec),
        "must_not_refuse": True,
        "fact_ids": [(step.get("branch_fact_ids") or {}).get("NO")]
                    if (step.get("branch_fact_ids") or {}).get("NO") else [],
        "source_code": rec["symptom_id"],
    }


def cases_symptom_remedy(rec):
    """Flat S-Mode rows: observation -> remedy. THE NEW TYPE.

    POLARITY. A flat row is not a YES/NO branch and must not be phrased as one.
    Confirming the point to check means the FAULT IS FOUND and the walk STOPS;
    on a branching step YES means the check was normal and the tree ADVANCES.
    So the question asks what the technician is LOOKING AT, and the expectation
    names the cause and its remedy -- there is no branch, no "yes", and no next
    step to go to.

    Phrasing it as "step N ka result YES aaya, ab kya karun?" would bake the
    inverted reading into the test set, and an assistant that walked the tree
    backwards would score well on it.
    """
    if rec.get("tree_kind") != "SymptomTreeFlat":
        return []
    out = []
    for st in rec.get("steps", []):
        point, remedy = st.get("point_to_check"), st.get("remedy")
        if not (point and remedy):
            continue
        rprov = st.get("remedy_provenance") or {}
        out.append({
            "type": "symptom_remedy",
            "section": SYMPTOMS,
            "difficulty": "hard",
            "question": (f"\"{symptom_question_subject(rec)}\". Machine par "
                         f"ye dikh raha hai: {point} Iska remedy kya hai?"),
            "filters": dict(FILTERS),
            "expected": {
                "step": st["step"],
                "cause": st.get("cause"),
                "point_to_check": point,
                "remedy": remedy,
            },
            # The remedy verbatim: it is the answer the technician acts on, and
            # it is a cell of its own in the manual.
            "must_contain_verbatim": [remedy],
            # The REMEDY's own page, not the row's cause page.
            "must_cite_page": rprov.get("manual_page") or page_of_fact(st, rec),
            "must_not_refuse": True,
            "fact_ids": [st["remedy_fact_id"]] if st.get("remedy_fact_id") else [],
            "source_code": rec["symptom_id"],
        })
    return out


# ---------------------------------------------------------------------- ids

def assign_ids(cases):
    """10-char content hash, with _1/_2 appended on collision.

    Hashed over the semantic content only -- type, source_code, question,
    expected -- so a case keeps its id across a rebuild unless its meaning
    changes. Two cases that are genuinely identical in content collide by
    design; the suffix keeps ids unique without making them positional.
    """
    seen = collections.Counter()
    for c in cases:
        payload = json.dumps(
            {"type": c["type"], "source_code": c.get("source_code"),
             "question": c["question"], "expected": c["expected"]},
            sort_keys=True, ensure_ascii=False)
        base = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:10]
        n = seen[base]
        seen[base] += 1
        c["id"] = base if n == 0 else f"{base}_{n}"
    return cases


KEY_ORDER = ["id", "type", "section", "difficulty", "question", "filters", "expected",
             "must_contain", "must_contain_verbatim", "must_cite_page",
             "fact_ids", "must_not_refuse", "must_refuse",
             "must_not_contain_numbers", "trap", "source_code"]


def ordered(case):
    return {k: case[k] for k in KEY_ORDER if k in case}


# ---------------------------------------------------------------------- main

def check_criteria_rules(symptoms):
    """The two numeric rules must disagree ONLY on relational criteria.

    Asserted rather than assumed, because the whole justification for using
    criteria_kind instead of the digit rule is that the disagreement is
    confined to a class where the digit rule is wrong. If a criteria_kind
    'numeric' entry ever had no digit, the classifier and the regex would be
    disagreeing in the OTHER direction and the argument would not hold.
    """
    disagree = collections.Counter()
    for rec in symptoms.values():
        for m, _ in measurements_of(rec):
            kind = m.get("criteria_kind")
            if kind is None:
                sys.exit(f"ERROR: {rec['symptom_id']} has a measurement with no "
                         "criteria_kind. The symptom builder relies on it; "
                         "re-run pipeline/extract_symptoms.py.")
            if is_numeric_symptom_criterion(m) and not is_numeric_criterion(
                    m.get("criteria")):
                sys.exit(
                    f"ERROR: {rec['symptom_id']} has criteria_kind 'numeric' "
                    f"with no digit in it: {m.get('criteria')!r}\n"
                    "  The two rules now disagree in the direction the digit "
                    "rule is RIGHT about, which breaks the argument for "
                    "preferring criteria_kind. Investigate before rebuilding.")
            if is_numeric_symptom_criterion(m) != is_numeric_criterion(
                    m.get("criteria")):
                disagree[kind] += 1
    off_class = {k: v for k, v in disagree.items() if k != "relational"}
    if off_class:
        sys.exit(f"ERROR: the numeric rules disagree outside relational "
                 f"criteria: {off_class}. Expected relational only.")
    return disagree


def main():
    recs = load_codes()
    symptoms = load_symptoms()
    known = set(recs)
    disagree = check_criteria_rules(symptoms)
    cases = []

    # Grouped by code in sorted order, so a diff between two builds is readable.
    for code in sorted(recs):
        rec = recs[code]
        for c in (case_direct_lookup(rec),):
            if c:
                cases.append(c)
        cases += cases_numeric_exactness(rec)
        for c in (case_step_ordering(rec),
                  case_branch_following(rec),
                  case_cross_ref_hop(rec),
                  case_precondition(rec, known)):
            if c:
                cases.append(c)

    cases += cases_adversarial(recs)

    # SYMPTOM CASES APPEND AFTER EVERY SECTION 40 CASE, deliberately.
    # assign_ids resolves a content collision by appending _1/_2 in iteration
    # order, so inserting anything earlier could renumber an existing case's
    # suffix and change an id that nothing about its meaning had changed.
    n_s40 = len(cases)
    for sid in sorted(symptoms):
        rec = symptoms[sid]
        cases += cases_symptom_numeric(rec)
        for c in (case_symptom_step_ordering(rec),
                  case_symptom_branch_following(rec)):
            if c:
                cases.append(c)
        cases += cases_symptom_remedy(rec)

    assign_ids(cases)
    cases = [ordered(c) for c in cases]

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)

    # EXPECTED PER (section, type), never as a total. A total cannot say which
    # of its parts moved, nor that a new part appeared -- the failure mode that
    # made tests/test_extraction.py's `len(qa) == 1330` go stale.
    EXPECTED = {
        (SECTION40, "numeric_exactness"): 846,
        (SECTION40, "direct_lookup"): 173,
        (SECTION40, "step_ordering"): 164,
        (SECTION40, "branch_following"): 116,
        (SECTION40, "precondition"): 10,
        (SECTION40, "cross_ref_hop"): 9,
        (SECTION40, "adversarial_unknown"): 6,
        (SECTION40, "adversarial_model"): 6,
        (SYMPTOMS, "numeric_exactness"): 232,
        (SYMPTOMS, "step_ordering"): 57,
        (SYMPTOMS, "branch_following"): 37,
        # 209 flat rows carry BOTH point_to_check and remedy.
        #
        # 207 of these cases derive THREE golden facts and 2 derive only two.
        # That split is the SOURCE DOCUMENT, not the builder: SM09 step 1 and
        # SM11 step 1 print "Unspecified fuel is used." in both the Cause and
        # the Point-to-check column of the manual's own table, so golden_facts
        # dedupes them to one string. Nothing is being dropped.
        (SYMPTOMS, "symptom_remedy"): 209,
    }
    built = collections.Counter((c["section"], c["type"]) for c in cases)
    print(f"wrote {len(cases)} cases to {OUT_PATH}")
    print(f"numeric rules disagree on {sum(disagree.values())} criteria, "
          f"all relational: {dict(disagree)}\n")

    print(f"{'section':10} {'type':22} {'built':>6} {'expected':>9}")
    drift = []
    for k in sorted(set(built) | set(EXPECTED)):
        got, want = built.get(k, 0), EXPECTED.get(k, 0)
        flag = "" if got == want else "   <-- DRIFT"
        if got != want:
            drift.append(f"{k}: built {got}, expected {want}")
        print(f"{k[0]:10} {k[1]:22} {got:6} {want:9}{flag}")
    print(f"{'':10} {'TOTAL':22} {len(cases):6} {sum(EXPECTED.values()):9}")

    # A bucket the expectation has never heard of is the event that breaks a
    # pinned total, and it has to fail by NAMING the newcomer.
    unnamed = sorted(set(built) - set(EXPECTED))
    if unnamed:
        drift.append(f"buckets the expectation does not name: {unnamed}")
    if sum(built.values()) != len(cases):
        drift.append("the parts do not account for the whole")

    ids = [c["id"] for c in cases]
    if len(set(ids)) != len(ids):
        dupes = [i for i, n in collections.Counter(ids).items() if n > 1]
        drift.append(f"duplicate ids: {dupes[:5]}")

    verbatim = sum(len(c.get("must_contain_verbatim", [])) for c in cases)
    print(f"\nverbatim strings pinned: {verbatim}")
    print(f"unique ids: {len(set(ids))}/{len(cases)}")

    if drift:
        print("\nBUILD REJECTED. Counts differ from the last known-good build:")
        for d in drift:
            print(f"  {d}")
        sys.exit("A count change is a behaviour change in this script or in the "
                 "ground truth. Explain it before accepting the result.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
