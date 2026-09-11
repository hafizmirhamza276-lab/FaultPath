#!/usr/bin/env python3
"""
human_kit.py
Build the transcription kit: page images, blank templates, instructions.

The one rule that matters: THE TEMPLATE CONTAINS NO EXTRACTED VALUES. A form
showing what the machine read is not a transcription, it is a review, and a
person will agree with what is already on the page in front of them. That
agreement would be worth nothing and would look exactly like verification --
which is the failure this whole exercise exists to remove.

Templates carry structure only: how many steps, how many measurements, which
fields. Even the counts are a compromise, and the instruction sheet says so --
if the page disagrees with the count, the page wins and the transcriber is told
to say so.

No LLM. Nothing pre-fills, nothing suggests.
"""
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from pipeline import human_select                     # noqa: E402
from agent import tools                               # noqa: E402

OUT_DIR = os.path.join(os.environ.get("OUT_DIR",
                                      os.path.join(REPO_ROOT, "reports")),
                       "transcription")
DEFAULT_PDF = os.environ.get(
    "KOMATSU_PDF", os.path.join(os.path.dirname(REPO_ROOT),
                                "komatsu-manuals", "SEN06867-13.pdf"))
DPI = 200          # readable for 8pt table text without absurd file sizes

TEMPLATE_FIELDS = ("cause", "procedure", "branch_yes", "branch_no")
MEASUREMENT_FIELDS = ("quantity", "point", "criteria")


def blank_template(code: str, rec: dict) -> dict:
    """Structure only. No value the extractor produced appears here."""
    steps = tools.real_steps(rec)
    return {
        "code": code,
        "manual_id": rec["manual_id"],
        "manual_pages": sorted(set(rec["manual_pages"])),
        "transcriber": "",
        "date": "",
        "minutes_taken": None,
        "_instructions": "Type what the PAGE says. Do not consult any other "
                         "file. Leave a field empty if the page does not show "
                         "it. Use the notes field for anything unclear.",
        "steps": [
            {"step": i + 1,
             "cause": "",
             "procedure": "",
             "branch_yes": "",
             "branch_no": "",
             "measurements": [{f: "" for f in MEASUREMENT_FIELDS}
                              for _ in (s.get("measurements") or [])] or [],
             "notes": ""}
            for i, s in enumerate(steps)],
        "standalone_measurements": [{f: "" for f in MEASUREMENT_FIELDS}
                                    for _ in rec.get("standalone_measurements", [])],
        "page_disagrees_with_template": False,
        "notes": "",
    }


def assert_blank(template: dict) -> None:
    """A template with a value in it is a rubber stamp. Refuse to ship one."""
    def walk(node, path="root"):
        if isinstance(node, dict):
            for k, v in node.items():
                if k.startswith("_") or k in ("code", "manual_id", "manual_pages",
                                              "step", "transcriber", "date",
                                              "minutes_taken", "notes",
                                              "page_disagrees_with_template"):
                    continue
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")
        elif isinstance(node, str) and node.strip():
            raise AssertionError(
                f"template is pre-filled at {path}: {node[:40]!r}. A form that "
                "shows the answer produces agreement, not verification.")
    walk(template)


def render_pages(code: str, rec: dict, out_dir: str,
                 pdf_path: Optional[str] = None) -> List[str]:
    import fitz
    doc = fitz.open(pdf_path or DEFAULT_PDF)
    lo, hi = rec["pdf_pages"]
    written = []
    zoom = DPI / 72.0
    for p in range(lo, hi + 1):
        pix = doc[p - 1].get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        name = f"{code}_p{p}.png"
        pix.save(os.path.join(out_dir, name))
        written.append(name)
    doc.close()
    return written


INSTRUCTIONS = """\
TRANSCRIBING A KOMATSU FAILURE CODE
===================================

You are creating an independent record of what the manual page says. The machine
has already read these pages. You are NOT checking its work -- you are not shown
its work, on purpose. Two independent readings that agree are evidence. One
reading confirmed by a person who could see it is not.

WHAT TO DO

1. Open the PNG pages for the code. They are named <CODE>_p<pdf page>.png.
2. Open <CODE>.json. It has empty fields and the right number of slots.
3. Type what you see, field by field:

   cause        the text in the "Cause" column for that step
   procedure    the text in the procedure / "Diagnosis and treatment" column
   branch_yes   the whole YES outcome, including the bullet marks
   branch_no    the whole NO outcome
   measurements quantity (e.g. Resistance), point (e.g. Between CK06 (1) and
                (3)), criteria (e.g. Max. 1 ohm) -- exactly as printed

4. Copy CHARACTERS, not meaning. If the page reads "Max. 1 kOhm" type that, not
   "1000 ohm". Never convert a unit. Never tidy a number.

UNCLEAR TEXT

   If you cannot read a character, type the rest and put the word in the step's
   "notes" field with a question mark:  notes: "har?ess - last letters unclear"
   Do NOT guess. An honest gap is useful; a guess is indistinguishable from a
   reading and quietly becomes ground truth.

   Some words run past the ruled cell border in this manual. If a letter is
   clearly printed but sits outside the box, INCLUDE it and note it. That is a
   real property of the document and we want it recorded.

GENUINE SOURCE AMBIGUITY

   If the page itself is contradictory -- two different values for the same
   thing, a branch with no outcome -- record what you see and describe the
   conflict in "notes". Do not pick one. The manual does contain conflicts
   (F@BBZL states two different action levels) and finding more is a result,
   not a problem.

IF THE TEMPLATE IS WRONG

   The template's step and measurement counts came from the machine's reading.
   If the page shows more steps than the template has slots, or fewer, set
   "page_disagrees_with_template": true and describe it in "notes". THE PAGE
   WINS. A count mismatch is one of the most valuable things you can find --
   27 steps were once lost exactly this way and nothing noticed for months.

WHAT NOT TO DO

   Do not open golden/, the repo, or any extracted file.
   Do not ask a language model to read the page or fill a field.
   Do not copy from another transcriber.

WHEN FINISHED

   Fill transcriber, date and minutes_taken, and save the file in place.
   minutes_taken is used only to plan future rounds; accuracy matters more
   than speed and nobody is judged on it.

A worked example of a completed transcription is in example/.
"""


def build(codes: Optional[List[dict]] = None, out_dir: Optional[str] = None,
          pdf_path: Optional[str] = None, render: bool = True) -> dict:
    sel = codes or human_select.select()
    out = out_dir or OUT_DIR
    os.makedirs(out, exist_ok=True)
    recs = tools.records()

    manifest = {"codes": [], "dpi": DPI, "total_codes": len(sel)}
    for d in sel:
        code = d["code"]
        cdir = os.path.join(out, code)
        os.makedirs(cdir, exist_ok=True)
        tmpl = blank_template(code, recs[code])
        assert_blank(tmpl)
        with open(os.path.join(cdir, f"{code}.json"), "w", encoding="utf-8") as f:
            json.dump(tmpl, f, indent=2, ensure_ascii=False)
        imgs = render_pages(code, recs[code], cdir, pdf_path) if render else []
        manifest["codes"].append({**d, "template": f"{code}/{code}.json",
                                  "images": imgs})

    with open(os.path.join(out, "INSTRUCTIONS.txt"), "w", encoding="utf-8") as f:
        f.write(INSTRUCTIONS)

    # Worked example, on a code deliberately outside the 25.
    ex = human_select.example_code()
    exdir = os.path.join(out, "example")
    os.makedirs(exdir, exist_ok=True)
    ex_t = blank_template(ex, recs[ex])
    assert_blank(ex_t)
    with open(os.path.join(exdir, f"{ex}_blank.json"), "w", encoding="utf-8") as f:
        json.dump(ex_t, f, indent=2, ensure_ascii=False)
    filled = json.loads(json.dumps(ex_t))
    filled["transcriber"] = "worked example"
    filled["date"] = "2026-09-11"
    filled["minutes_taken"] = 12
    if filled["steps"]:
        filled["steps"][0]["cause"] = "<type the Cause column text here>"
        filled["steps"][0]["procedure"] = "<type the procedure column text here>"
        filled["steps"][0]["notes"] = ("if a letter is unreadable, write the word "
                                       "with a ? and leave the rest typed")
    with open(os.path.join(exdir, f"{ex}_worked.json"), "w", encoding="utf-8") as f:
        json.dump(filled, f, indent=2, ensure_ascii=False)
    if render:
        render_pages(ex, recs[ex], exdir, pdf_path)
    manifest["example_code"] = ex

    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def estimate_effort(sel: Optional[List[dict]] = None) -> dict:
    """Rough planning figure, stated as a range with its assumptions visible."""
    sel = sel or human_select.select()
    steps = sum(d["n_steps"] for d in sel)
    meas = sum(d["n_measurements"] for d in sel)
    pages = sum(len(d["pages"]) for d in sel)
    # 90s per step (cause + procedure + two branches), 45s per measurement,
    # 3 min per page for orientation and cross-checking.
    low = (steps * 75 + meas * 35 + pages * 120) / 60.0
    high = (steps * 120 + meas * 60 + pages * 240) / 60.0
    return {"steps": steps, "measurements": meas, "pages": pages,
            "minutes_low": round(low), "minutes_high": round(high),
            "hours_low": round(low / 60, 1), "hours_high": round(high / 60, 1),
            "assumptions": "75-120s per step, 35-60s per measurement, 2-4min "
                           "per page for orientation. Excludes breaks and any "
                           "second-pass adjudication."}


def self_test() -> None:
    recs = tools.records()
    code = human_select.select()[0]["code"]
    t = blank_template(code, recs[code])
    assert_blank(t)                       # must not raise on a clean template
    t["steps"][0]["cause"] = "Defective wiring harness connector"
    try:
        assert_blank(t)
        raise AssertionError("assert_blank cannot detect a pre-filled template")
    except AssertionError as exc:
        assert "pre-filled" in str(exc), \
            "assert_blank cannot detect a pre-filled template"


if __name__ == "__main__":
    self_test()
    m = build()
    e = estimate_effort()
    print(f"kit built for {m['total_codes']} codes in {os.path.relpath(OUT_DIR, REPO_ROOT)}")
    print(f"worked example: {m['example_code']} (not among the 25)")
    print(f"\neffort estimate: {e['hours_low']}-{e['hours_high']} hours "
          f"({e['steps']} steps, {e['measurements']} measurements, {e['pages']} pages)")
    print(f"  {e['assumptions']}")
