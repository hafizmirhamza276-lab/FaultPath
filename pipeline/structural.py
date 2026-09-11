#!/usr/bin/env python3
"""
structural.py
Does the extracted tree have the right SHAPE.

Fact-level resolution proves each fact exists on its page. It does not prove the
tree is complete. The 27 column-split steps each resolved fine on their own;
what was wrong was that steps 6 and 7 were missing entirely, and a fact that is
absent cannot fail to resolve.

INDEPENDENT BY CONSTRUCTION. This module imports NOTHING from
pipeline/extract_golden.py -- not parse_causes, not table_kind, not clean, not
the CRIT_VALUE or GLUED_STEP patterns. It does not use pdfplumber at all. It
reads the page with PyMuPDF's positional text and finds the "No." column by
geometry. If both sides shared a helper they would share its bugs, which is
exactly how citation_accuracy read 1.0000 over 635 wrong pages.

orphan_detection is the one that would have caught the original bug: a step
number printed on the page with no corresponding step in golden/.

Measures the parser. Never alters it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

GOLD = os.environ.get("GOLD_DIR", os.path.join(REPO_ROOT, "golden"))
DEFAULT_PDF = os.environ.get(
    "KOMATSU_PDF", os.path.join(os.path.dirname(REPO_ROOT),
                                "komatsu-manuals", "SEN06867-13.pdf"))

# A bare step number in the "No." column. Deliberately its own pattern, not
# imported from the extractor.
BARE_INT = re.compile(r"^\s*(\d{1,2})\s*$")

# The "No." column sits at the left edge of the cause table. Spans further right
# are cause text, pin numbers and criteria, none of which are step markers.
NO_COLUMN_X_LIMIT = 0.18        # fraction of page width

# The longest cause table in Section 40 runs to 11 steps. Anything above this in
# the No. column is not a step number.
MAX_PLAUSIBLE_STEP = 20


class PageReader:
    """PyMuPDF positional text. Opened once, cached per page."""

    def __init__(self, pdf_path: Optional[str] = None):
        self.path = pdf_path or DEFAULT_PDF
        self._doc = None
        self._cache: Dict[int, dict] = {}

    def available(self) -> bool:
        return bool(self.path) and os.path.isfile(self.path)

    def _open(self):
        if self._doc is None:
            import fitz
            self._doc = fitz.open(self.path)
        return self._doc

    def spans(self, pdf_page: int) -> List[dict]:
        if pdf_page in self._cache:
            return self._cache[pdf_page]
        page = self._open()[pdf_page - 1]
        width = page.rect.width or 1.0
        out = []
        for block in page.get_text("dict").get("blocks", []):
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    out.append({"text": span.get("text", ""),
                                "x0": span["bbox"][0] / width,
                                "y0": span["bbox"][1]})
        out.sort(key=lambda s: (s["y0"], s["x0"]))
        self._cache[pdf_page] = out
        return out

    def step_markers(self, pdf_page: int) -> List[int]:
        """Bare integers in the leftmost column, in reading order.

        This is the independent count. The extractor finds step numbers by
        matching cell[0] of a pdfplumber table row; this finds them by their
        position on the page. Two different methods, so a bug in one does not
        hide in the other.
        """
        out = []
        for s in self.spans(pdf_page):
            if s["x0"] > NO_COLUMN_X_LIMIT:
                continue
            m = BARE_INT.match(s["text"])
            if m:
                out.append(int(m.group(1)))
        return out

    def close(self):
        if self._doc is not None:
            self._doc.close()
            self._doc = None


def load_records(gold: Optional[str] = None) -> Dict[str, dict]:
    import glob
    recs = {}
    for p in sorted(glob.glob(os.path.join(gold or GOLD, "failure_codes", "*.json"))):
        with open(p, encoding="utf-8") as f:
            r = json.load(f)
        recs[r["code"]] = r
    return recs


# ------------------------------------------------------------- the checks

def orphan_detection(rec: dict, reader: PageReader) -> List[dict]:
    """Step numbers printed on the page that landed in no step.

    THE CHECK THAT WOULD HAVE CAUGHT THE ORIGINAL BUG. A dropped step leaves its
    number on the page and nothing in golden/, and no fact-level check can see
    an absence.
    """
    recorded = {s["step"] for s in rec.get("steps", []) if not s.get("redirect")}
    if not recorded:
        return []
    pages = sorted({s["provenance"]["pdf_page"] for s in rec.get("steps", [])
                    if s.get("provenance")})
    orphans = []
    for p in pages:
        markers = reader.step_markers(p)
        # NOT bounded above by the recorded maximum. An earlier version clipped
        # markers to min(recorded)..max(recorded), which made this check blind
        # to trailing dropped steps -- and trailing steps are exactly what the
        # original bug lost (CA451 dropped 6 and 7, the final two). A check
        # whose window is defined by the thing it is auditing cannot see what
        # that thing omitted.
        for n in sorted(set(markers)):
            if 1 <= n <= MAX_PLAUSIBLE_STEP and n not in recorded:
                orphans.append({"code": rec["code"], "pdf_page": p, "step": n,
                                "reason": "step number printed on the page with "
                                          "no corresponding step in golden/"})
    return orphans


def step_count_agreement(rec: dict, reader: PageReader) -> dict:
    """Count step markers on the page vs steps recorded, by the other method."""
    recorded = sorted({s["step"] for s in rec.get("steps", [])
                       if not s.get("redirect")})
    if not recorded:
        return {"code": rec["code"], "recorded": 0, "on_page": 0, "agree": True}
    pages = sorted({s["provenance"]["pdf_page"] for s in rec.get("steps", [])
                    if s.get("provenance")})
    seen = set()
    for p in pages:
        for n in reader.step_markers(p):
            if min(recorded) <= n <= max(recorded):
                seen.add(n)
    return {"code": rec["code"], "recorded": len(recorded),
            "on_page": len(seen), "agree": seen.issubset(set(recorded))
            and len(seen) <= len(recorded),
            "missing_from_golden": sorted(seen - set(recorded))}


def step_contiguity(rec: dict) -> dict:
    """Step numbers must run 1..N with no gaps."""
    nums = sorted(s["step"] for s in rec.get("steps", []) if not s.get("redirect"))
    if not nums:
        return {"code": rec["code"], "ok": True, "steps": [], "gaps": []}
    expected = list(range(1, len(nums) + 1))
    gaps = [n for n in range(1, max(nums) + 1) if n not in nums]
    return {"code": rec["code"], "ok": nums == expected, "steps": nums,
            "gaps": gaps}


def branch_completeness(rec: dict) -> dict:
    """Format-A steps are decision trees: both outcomes or neither."""
    if rec.get("format") != "A":
        return {"code": rec["code"], "applicable": False, "incomplete": []}
    bad = []
    for s in rec.get("steps", []):
        if s.get("redirect"):
            continue
        b = set(s.get("branches") or {})
        if b and b != {"YES", "NO"}:
            bad.append({"step": s["step"], "has": sorted(b)})
    return {"code": rec["code"], "applicable": True, "incomplete": bad}


def run(recs=None, reader=None, codes=None) -> dict:
    recs = recs or load_records()
    reader = reader or PageReader()
    if codes:
        recs = {c: r for c, r in recs.items() if c in set(codes)}

    orphans, counts, contig, branches = [], [], [], []
    for code in sorted(recs):
        r = recs[code]
        if reader.available():
            orphans += orphan_detection(r, reader)
            counts.append(step_count_agreement(r, reader))
        contig.append(step_contiguity(r))
        branches.append(branch_completeness(r))

    n = len(recs) or 1
    count_ok = sum(1 for c in counts if c["agree"])
    contig_ok = sum(1 for c in contig if c["ok"])
    branch_bad = [b for b in branches if b["applicable"] and b["incomplete"]]
    return {
        "available": reader.available(),
        "codes": len(recs),
        "orphan_detection": {"orphans": orphans, "rate": 1.0 - (
            len({o["code"] for o in orphans}) / n)},
        "step_count_agreement": {"rate": (count_ok / len(counts)) if counts else 0.0,
                                 "disagreements": [c for c in counts if not c["agree"]]},
        "step_contiguity": {"rate": contig_ok / n,
                            "gaps": [c for c in contig if not c["ok"]]},
        "branch_completeness": {"rate": 1.0 - (len(branch_bad) / n),
                                "incomplete": branch_bad},
    }


def gates(result: dict) -> List[dict]:
    out = []
    for name, key in (("orphan_detection", "orphan_detection"),
                      ("step_count_agreement", "step_count_agreement"),
                      ("step_contiguity", "step_contiguity")):
        v = result[key]["rate"]
        out.append({"gate": name, "value": v, "op": ">=", "threshold": 1.0,
                    "status": "PASS" if v >= 1.0 else "FAIL"})
    return out


def self_test() -> None:
    """Every structural check proves it can fail before it is trusted."""
    # contiguity: 1,2,4 must fail
    bad = {"code": "X", "format": "A",
           "steps": [{"step": 1}, {"step": 2}, {"step": 4}]}
    assert not step_contiguity(bad)["ok"], \
        "step_contiguity cannot fail on a 1,2,4 sequence"
    assert step_contiguity({"code": "X", "steps": [{"step": 1}, {"step": 2}]})["ok"]

    # branch completeness: a YES with no NO must fail
    one = {"code": "X", "format": "A",
           "steps": [{"step": 1, "branches": {"YES": "go on"}}]}
    assert branch_completeness(one)["incomplete"], \
        "branch_completeness cannot fail on a half branch"
    both = {"code": "X", "format": "A",
            "steps": [{"step": 1, "branches": {"YES": "a", "NO": "b"}}]}
    assert not branch_completeness(both)["incomplete"]

    # gates must be able to fail
    assert any(g["status"] == "FAIL" for g in gates({
        "orphan_detection": {"rate": 0.5}, "step_count_agreement": {"rate": 1.0},
        "step_contiguity": {"rate": 1.0}})), "structural gates cannot fail"


class _FakeReader(PageReader):
    """Reader over a fixed marker map, for proving orphan_detection fires."""

    def __init__(self, markers):
        super().__init__(pdf_path=None)
        self._markers = markers

    def available(self):
        return True

    def step_markers(self, pdf_page):
        return self._markers.get(pdf_page, [])


def prove_orphan_fires() -> dict:
    """orphan_detection must fire on a deliberately truncated tree.

    Built first and verified here, because it is the check that would have
    caught the original silent-loss bug and a check nobody has watched fail is
    not a check.
    """
    full = {"code": "TRUNC", "steps": [
        {"step": i, "provenance": {"pdf_page": 100}} for i in range(1, 8)]}
    reader = _FakeReader({100: [1, 2, 3, 4, 5, 6, 7]})
    clean = orphan_detection(full, reader)

    truncated = {"code": "TRUNC", "steps": [
        {"step": i, "provenance": {"pdf_page": 100}} for i in range(1, 6)]}
    fired = orphan_detection(truncated, reader)
    return {"clean_tree_orphans": clean,
            "truncated_tree_orphans": fired,
            "fires": len(fired) == 2 and {o["step"] for o in fired} == {6, 7},
            "silent_before": len(clean) == 0}


if __name__ == "__main__":
    self_test()
    proof = prove_orphan_fires()
    print("orphan_detection proof on a truncated tree:")
    print(f"  complete tree  -> {len(proof['clean_tree_orphans'])} orphans")
    print(f"  steps 6,7 cut  -> {len(proof['truncated_tree_orphans'])} orphans "
          f"{sorted(o['step'] for o in proof['truncated_tree_orphans'])}")
    print(f"  FIRES: {proof['fires']}\n")

    reader = PageReader()
    if not reader.available():
        sys.exit(f"ERROR: source PDF not found at {reader.path}")
    res = run(reader=reader)
    print(f"{'check':24}{'rate':>10}  detail")
    print(f"{'orphan_detection':24}{res['orphan_detection']['rate']:>10.4f}  "
          f"{len(res['orphan_detection']['orphans'])} orphan step numbers")
    print(f"{'step_count_agreement':24}{res['step_count_agreement']['rate']:>10.4f}  "
          f"{len(res['step_count_agreement']['disagreements'])} codes disagree")
    print(f"{'step_contiguity':24}{res['step_contiguity']['rate']:>10.4f}  "
          f"{len(res['step_contiguity']['gaps'])} codes with gaps")
    print(f"{'branch_completeness':24}{res['branch_completeness']['rate']:>10.4f}  "
          f"{len(res['branch_completeness']['incomplete'])} codes incomplete")
    for o in res["orphan_detection"]["orphans"][:10]:
        print(f"    ORPHAN {o['code']} step {o['step']} on pdf p{o['pdf_page']}")
    for g in res["step_contiguity"]["gaps"][:10]:
        print(f"    GAP    {g['code']} steps={g['steps']} missing={g['gaps']}")
    reader.close()
