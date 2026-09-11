#!/usr/bin/env python3
"""
human_crops.py
Reduce transcription to reading one box at a time.

The transcriber has no mechanical background. The 27 machine-repaired steps are
table cells, not diagrams, but a person unsure of what to look at will hesitate
-- and a hesitant transcriber agreeing with a template is the failure this whole
design exists to prevent.

So: one crop per FIELD, not per row. The reader never decides which column is
which, never interprets table structure, never needs to know what a common rail
pressure sensor is. The instruction is "type what is written in this box, or
UNREADABLE".

WHAT A CROP MAY CONTAIN
  Only pixels rendered from the PDF page. The machine's stored text is never
  drawn, never used as a caption, never placed in the HTML. The crop rectangle
  comes from the table geometry; golden/ supplies only WHICH cell to crop, not
  what it says. A crop showing the extracted text would make this a review.

FALLBACK
  If a cell rectangle cannot be derived, the entry falls back to the full page
  and says so. A bad crop -- one cutting a word in half -- is worse than no
  crop, because the reader would faithfully transcribe the half.

No LLM. Nothing reads, suggests or pre-fills a crop.
"""
from __future__ import annotations

import html
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from agent import tools                                   # noqa: E402

OUT_DIR = os.path.join(os.environ.get("OUT_DIR",
                                      os.path.join(REPO_ROOT, "reports")),
                       "transcription", "crops")
DEFAULT_PDF = os.environ.get(
    "KOMATSU_PDF", os.path.join(os.path.dirname(REPO_ROOT),
                                "komatsu-manuals", "SEN06867-13.pdf"))

DPI = 220
MARGIN = 4.0          # pt of breathing room so no glyph is clipped
HEADER_PT = 26.0      # strip of the header row above, for orientation

FIELDS = ("cause", "procedure")


def safe_name(fact_id: str, field: str) -> str:
    return fact_id.replace(":", "_").replace("@", "AT").replace("#", "H") \
        + f"__{field}"


def repaired_steps(recs: Optional[Dict[str, dict]] = None) -> List[dict]:
    """The 27 machine-repaired steps, in fact_id order."""
    recs = recs or tools.records()
    out = []
    for code in sorted(recs):
        for st in recs[code].get("steps", []):
            if st.get("extraction_warning") == "column_split_recovered":
                out.append({"code": code, "step": st["step"],
                            "fact_id": st["fact_id"],
                            "prov": st.get("provenance") or {}})
    return out


def locate_cells(code: str, prov: dict, pdf_path: Optional[str] = None):
    """(cells_by_field, table_bbox) in PDF points, or (None, None).

    Uses pdfplumber's table geometry to find the row, then takes the cause and
    procedure columns by position within that row. Returns None when the row
    cannot be located, so the caller can fall back rather than guess.
    """
    import pdfplumber
    page_no, t_idx, r_idx = (prov.get("pdf_page"), prov.get("table_index"),
                             prov.get("row_index"))
    if not page_no or t_idx is None or r_idx is None:
        return None, None
    try:
        with pdfplumber.open(pdf_path or DEFAULT_PDF) as pdf:
            page = pdf.pages[page_no - 1]
            tables = page.find_tables()
            if t_idx >= len(tables):
                return None, None
            tbl = tables[t_idx]
            if r_idx >= len(tbl.rows):
                return None, None
            row = tbl.rows[r_idx]
            # Column bands come from the TABLE's own geometry, not from this
            # row's cell list. Filtering None cells out of a row shifts the
            # index -- a row whose "No." cell is merged away then hands back the
            # procedure column as if it were the cause, and the crop lands
            # mid-word. Bands are stable across rows; cell lists are not.
            edges = sorted({round(c[0], 1) for r in tbl.rows for c in r.cells if c}
                           | {round(tbl.bbox[2], 1)})
            if len(edges) < 3:
                return None, None
            y0, y1 = row.bbox[1], row.bbox[3]
            bands = [(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
            # No. | Cause | Procedure...  Band 1 is the cause column; band 2
            # onward is the procedure column, which may be sub-divided by the
            # measurement sub-table, so it runs to the table's right edge.
            if len(bands) < 2:
                return None, None
            proc_x0 = bands[2][0] if len(bands) > 2 else bands[1][1]
            cause = (bands[1][0], y0, bands[1][1], y1)
            proc = (proc_x0, y0, tbl.bbox[2], y1)
            if cause[2] - cause[0] < 30 or proc[2] - proc[0] < 30:
                return None, None          # implausible band; fall back

            # The declared cell rectangle is NOT trustworthy as a crop bound.
            # On page 784 the cause text runs x0=70.4..164.4 while pdfplumber
            # declares the cause cell as 106.3..181.3 -- the glyphs overflow
            # their own column by 36pt, the same typesetting habit that puts a
            # trailing 'f' outside the right border. Cropping to the declared
            # rect sliced "Defective common rail" down to "ve common rail", and
            # a reader would have faithfully transcribed the fragment.
            #
            # So the rect is widened to the real glyph extents of the content
            # that belongs to each column, and the split between them is made
            # at the procedure column's left edge.
            cause = _glyph_extent(pdf_path, page_no, cause,
                                  x_limit=(tbl.bbox[0], proc_x0)) or cause
            proc = _glyph_extent(pdf_path, page_no, proc,
                                 x_limit=(proc_x0, tbl.bbox[2])) or proc
            return {"cause": cause, "procedure": proc}, tbl.bbox
    except Exception:
        return None, None


def _glyph_extent(pdf_path, page_no, rect, x_limit):
    """Widen a rect to cover every glyph that belongs in it.

    A glyph belongs if its horizontal midpoint sits inside x_limit and its
    vertical midpoint inside the row. Returns None when nothing is found, so
    the caller keeps the declared rect rather than cropping to nothing.
    """
    import fitz
    doc = fitz.open(pdf_path or DEFAULT_PDF)
    page = doc[page_no - 1]
    lo_x, hi_x = x_limit
    _, y0, _, y1 = rect
    xs0, xs1, ys0, ys1 = [], [], [], []
    for b in page.get_text("rawdict")["blocks"]:
        for l in b.get("lines", []):
            for s in l.get("spans", []):
                for ch in s["chars"]:
                    cx0, cy0, cx1, cy1 = ch["bbox"]
                    if not ch["c"].strip():
                        continue
                    if not (y0 <= (cy0 + cy1) / 2 <= y1):
                        continue
                    if not (lo_x <= (cx0 + cx1) / 2 < hi_x):
                        continue
                    xs0.append(cx0); xs1.append(cx1)
                    ys0.append(cy0); ys1.append(cy1)
    doc.close()
    if not xs0:
        return None
    return (min(min(xs0), rect[0]), min(min(ys0), y0),
            max(max(xs1), rect[2]), max(max(ys1), y1))


def render_crop(rect: Tuple[float, float, float, float], page_no: int,
                out_path: str, pdf_path: Optional[str] = None,
                header_top: Optional[float] = None) -> None:
    import fitz
    doc = fitz.open(pdf_path or DEFAULT_PDF)
    page = doc[page_no - 1]
    x0, y0, x1, y1 = rect
    top = y0 - MARGIN
    if header_top is not None:
        top = min(top, header_top)
    clip = fitz.Rect(max(0, x0 - MARGIN), max(0, top),
                     min(page.rect.width, x1 + MARGIN),
                     min(page.rect.height, y1 + MARGIN))
    pix = page.get_pixmap(matrix=fitz.Matrix(DPI / 72.0, DPI / 72.0), clip=clip)
    pix.save(out_path)
    doc.close()


def build(out_dir: Optional[str] = None, pdf_path: Optional[str] = None,
          render: bool = True) -> dict:
    out = out_dir or OUT_DIR
    os.makedirs(out, exist_ok=True)
    recs = tools.records()
    entries: List[dict] = []
    fallbacks: List[str] = []

    for item in repaired_steps(recs):
        cells, tbbox = locate_cells(item["code"], item["prov"], pdf_path)
        page_no = item["prov"].get("pdf_page")
        header_top = (tbbox[1] if tbbox else None)
        for field in FIELDS:
            name = safe_name(item["fact_id"], field) + ".png"
            entry = {"fact_id": item["fact_id"], "field": field,
                     "code": item["code"], "step": item["step"],
                     "manual_page": item["prov"].get("manual_page"),
                     "pdf_page": page_no, "image": name, "fallback": False}
            if cells and cells.get(field) and render:
                render_crop(cells[field], page_no, os.path.join(out, name),
                            pdf_path,
                            header_top=(header_top + HEADER_PT
                                        if header_top is not None else None))
            elif render:
                # A bad crop is worse than none: the reader would faithfully
                # transcribe a half-cut word. Fall back to the whole page and
                # say so on the card.
                import fitz
                doc = fitz.open(pdf_path or DEFAULT_PDF)
                pix = doc[page_no - 1].get_pixmap(
                    matrix=fitz.Matrix(DPI / 144.0, DPI / 144.0))
                pix.save(os.path.join(out, name))
                doc.close()
                entry["fallback"] = True
                fallbacks.append(f"{item['fact_id']}/{field}")
            entries.append(entry)

    entries.sort(key=lambda e: (e["fact_id"], e["field"]))
    manifest = {"entries": entries, "dpi": DPI,
                "steps": len(repaired_steps(recs)),
                "fallbacks": fallbacks}
    with open(os.path.join(out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(os.path.join(out, "index.html"), "w", encoding="utf-8") as f:
        f.write(render_index(entries))
    return manifest


INDEX_CSS = """
body{font:15px/1.5 system-ui,sans-serif;margin:0 auto;max-width:1100px;padding:24px;color:#111}
h1{font-size:22px} .lead{background:#f6f6f6;padding:14px 18px;border-left:4px solid #444}
.card{border:1px solid #ccc;border-radius:6px;padding:14px;margin:22px 0}
.fid{font:13px ui-monospace,monospace;color:#555}
img{max-width:100%;border:1px solid #ddd;display:block;margin:10px 0}
textarea{width:100%;min-height:64px;font:15px ui-monospace,monospace;padding:8px}
.warn{background:#fff4e5;border-left:4px solid #d08000;padding:8px 12px;font-size:14px}
button{font-size:15px;padding:8px 16px;margin:16px 0}
"""

INDEX_JS = """
function dump(){
  const out={};
  document.querySelectorAll('textarea').forEach(t=>{
    const v=t.value.trim(); if(v) out[t.dataset.key]=v;
  });
  const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  const a=document.createElement('a');
  a.href=URL.createObjectURL(blob); a.download='crops_filled.json'; a.click();
}
"""


def render_index(entries: List[dict]) -> str:
    parts = [f"<!doctype html><meta charset='utf-8'><title>Transcription crops</title>"
             f"<style>{INDEX_CSS}</style><h1>Transcription &mdash; one box at a time</h1>",
             "<div class='lead'><b>Type what is written in the box, exactly as "
             "printed.</b> If you cannot read it, type <code>UNREADABLE</code>. "
             "Leave it empty only if the box is genuinely blank.<br><br>"
             "Copy characters, not meaning. Do not convert units, do not tidy "
             "numbers, do not expand abbreviations. If a word runs past the "
             "ruled border, include it.<br><br>"
             "You do not need to know what any of it means.</div>",
             "<button onclick='dump()'>Download crops_filled.json</button>"]
    for e in entries:
        key = f"{e['fact_id']}|{e['field']}"
        parts.append("<div class='card'>")
        parts.append(f"<div class='fid'>{html.escape(e['fact_id'])} "
                     f"&middot; {html.escape(e['field'])} &middot; page "
                     f"{html.escape(str(e['manual_page']))}</div>")
        if e["fallback"]:
            parts.append("<div class='warn'>This box could not be cropped "
                         "reliably, so the whole page is shown. Find the "
                         f"<b>{html.escape(e['field'])}</b> cell for step "
                         f"{e['step']} yourself.</div>")
        parts.append(f"<img src='{html.escape(e['image'])}' alt=''>")
        parts.append(f"<textarea data-key='{html.escape(key)}' "
                     f"placeholder='type what the box says, or UNREADABLE'>"
                     f"</textarea>")
        parts.append("</div>")
    parts.append(f"<button onclick='dump()'>Download crops_filled.json</button>")
    parts.append(f"<script>{INDEX_JS}</script>")
    return "\n".join(parts)


# ------------------------------------------------------------- the writer

def write_into_templates(filled: Dict[str, str], kit_dir: Optional[str] = None,
                         transcriber: str = "", date: str = "",
                         minutes: Optional[int] = None) -> dict:
    """Write crop answers into the per-code JSON templates.

    The transcriber never edits JSON by hand.

    REFUSES TO WRITE A FIELD IT WAS NOT GIVEN. A missing key leaves the field
    empty and the step reports PARTIAL -- it never invents a blank and calls it
    transcribed, which is the same self-crediting the fact_id bug produced.
    """
    kit = kit_dir or os.path.join(os.path.dirname(OUT_DIR))
    recs = tools.records()
    touched: Dict[str, dict] = {}
    written, skipped = 0, []

    for key, value in sorted(filled.items()):
        if "|" not in key:
            skipped.append(f"{key}: malformed key")
            continue
        fact_id, field = key.split("|", 1)
        if field not in FIELDS:
            skipped.append(f"{key}: unknown field")
            continue
        code = fact_id.split(":")[0]
        step_no = int(fact_id.split(":")[1])
        path = _template_path(kit, code)
        if path is None:
            skipped.append(f"{key}: no template for {code}")
            continue
        if path not in touched:
            with open(path, encoding="utf-8") as f:
                touched[path] = json.load(f)
        doc = touched[path]
        slot = next((s for s in doc["steps"] if s["step"] == step_no), None)
        if slot is None:
            skipped.append(f"{key}: no step {step_no} slot")
            continue
        slot[field] = value
        written += 1

    for path, doc in touched.items():
        if transcriber:
            doc["transcriber"] = transcriber
        if date:
            doc["date"] = date
        if minutes is not None:
            doc["minutes_taken"] = minutes
        with open(path, "w", encoding="utf-8") as f:
            json.dump(doc, f, indent=2, ensure_ascii=False)

    return {"fields_written": written, "files_touched": sorted(touched),
            "skipped": skipped}


def _template_path(kit: str, code: str) -> Optional[str]:
    for cand in (os.path.join(kit, code, f"{code}.json"),
                 os.path.join(kit, "round2", code, f"{code}.json")):
        if os.path.isfile(cand):
            return cand
    return None


def self_test() -> None:
    """Crops must never carry the machine's text; the writer must never invent."""
    recs = tools.records()
    steps = repaired_steps(recs)
    assert len(steps) == 27, f"expected 27 repaired steps, got {len(steps)}"

    # The rendered HTML must contain no extracted string. Checked on the
    # derivation path rather than the pixels: the only text that reaches the
    # page is the fact id, the field name and the page label.
    entries = [{"fact_id": s["fact_id"], "field": "cause", "code": s["code"],
                "step": s["step"], "manual_page": s["prov"].get("manual_page"),
                "pdf_page": s["prov"].get("pdf_page"),
                "image": "x.png", "fallback": False} for s in steps]
    page = render_index(entries)
    for s in steps[:6]:
        stored = next(x for x in recs[s["code"]]["steps"]
                      if x["step"] == s["step"]).get("cause") or ""
        if len(stored) > 12:
            assert stored not in page, \
                f"crop index leaks the stored text for {s['fact_id']}"

    # The writer must not fabricate a field it was not handed.
    import tempfile
    d = tempfile.mkdtemp()
    try:
        code = steps[0]["code"]
        from pipeline import human_kit
        os.makedirs(os.path.join(d, code))
        t = human_kit.blank_template(code, recs[code])
        with open(os.path.join(d, code, f"{code}.json"), "w",
                  encoding="utf-8") as f:
            json.dump(t, f)
        fid = steps[0]["fact_id"]
        res = write_into_templates({f"{fid}|cause": "typed by hand"}, kit_dir=d)
        assert res["fields_written"] == 1
        with open(os.path.join(d, code, f"{code}.json"), encoding="utf-8") as f:
            back = json.load(f)
        slot = next(s for s in back["steps"] if s["step"] == steps[0]["step"])
        assert slot["cause"] == "typed by hand"
        assert slot["procedure"] == "", \
            "writer fabricated a field it was not given"
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    self_test()
    m = build()
    print(f"crops for {m['steps']} repaired steps -> "
          f"{len(m['entries'])} boxes in {os.path.relpath(OUT_DIR, REPO_ROOT)}")
    print(f"fallback to full page: {len(m['fallbacks'])} "
          f"{m['fallbacks'] if m['fallbacks'] else ''}")
    print(f"open {os.path.relpath(os.path.join(OUT_DIR, 'index.html'), REPO_ROOT)}")


def estimate_effort(n_steps: int = 27) -> dict:
    """Crop-based effort: read one box, type it, move on.

    Much cheaper than the page-based task because there is no navigation, no
    column identification and no JSON editing. A cause box is a few words; a
    procedure box is a short paragraph plus any measurement sub-row.
    """
    cause_lo, cause_hi = 30, 60          # seconds
    proc_lo, proc_hi = 90, 180
    lo = n_steps * (cause_lo + proc_lo) / 60.0
    hi = n_steps * (cause_hi + proc_hi) / 60.0
    return {"boxes": n_steps * 2, "minutes_low": round(lo),
            "minutes_high": round(hi), "hours_low": round(lo / 60, 1),
            "hours_high": round(hi / 60, 1),
            "assumptions": "30-60s per cause box, 90-180s per procedure box. "
                           "No navigation, no column identification, no JSON "
                           "editing. Excludes breaks."}
