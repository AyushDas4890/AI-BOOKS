#!/usr/bin/env python3
"""Build AI/ML Interview Book PDFs. Usage:
  python3 build_pdf.py --part 1     # single part PDF -> output/parts/Part_I.pdf
  python3 build_pdf.py --full       # full book       -> output/AI_ML_Interview_Book.pdf
  python3 build_pdf.py --file manuscript/part_01_foundations/ch01_mathematics.md  # one chapter
"""
import argparse, base64, functools, io, re, sys
from pathlib import Path

import markdown
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pygments.formatters import HtmlFormatter
from weasyprint import HTML

ROOT = Path(__file__).resolve().parent.parent
MS, OUT, CSS = ROOT / "manuscript", ROOT / "output", ROOT / "build" / "style.css"

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII"]
PARTS = [
    (1, "part_01_foundations", "Part I — Foundations"),
    (2, "part_02_classical_ml", "Part II — Classical Machine Learning"),
    (3, "part_03_deep_learning", "Part III — Deep Learning"),
    (4, "part_04_nlp_llms", "Part IV — NLP & Large Language Models"),
    (5, "part_05_ml_systems", "Part V — ML Systems & Production"),
    (6, "part_06_specialized", "Part VI — Specialized Topics"),
    (7, "part_07_coding_rounds", "Part VII — Coding & Practical Rounds"),
    (8, "part_08_the_interview", "Part VIII — The Interview Itself"),
]

FENCE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
BLOCK = re.compile(r"\$\$(.+?)\$\$", re.S)
INLINE = re.compile(r"(?<![\\$])\$([^$\n]+?)\$(?!\$)")

@functools.lru_cache(maxsize=4096)
def tex_to_img(tex: str, display: bool) -> str:
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${tex}$", fontsize=12 if display else 11)
    fig.canvas.draw()
    bb = t.get_window_extent()
    fig.set_size_inches(bb.width / 72, bb.height / 72)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    style = "display:block;margin:8px auto" if display else "vertical-align:middle"
    return f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'

def _convert_math_segment(text: str) -> str:
    def blk(m):
        try:
            return tex_to_img(m.group(1).strip().replace("\n", " "), True)
        except Exception as e:
            sys.exit(f"FATAL: bad display math: {m.group(1)[:80]!r} -> {e}")
    def inl(m):
        try:
            return tex_to_img(m.group(1).strip(), False)
        except Exception as e:
            sys.exit(f"FATAL: bad inline math: {m.group(1)[:80]!r} -> {e}")
    return INLINE.sub(inl, BLOCK.sub(blk, text))

def convert_math(md_text: str) -> str:
    parts = FENCE.split(md_text)
    return "".join(p if p.startswith("`") else _convert_math_segment(p) for p in parts)

def md_to_html(md_text: str) -> str:
    return markdown.markdown(
        convert_math(md_text),
        extensions=["extra", "codehilite", "tables", "sane_lists", "attr_list", "toc"],
        extension_configs={"codehilite": {"guess_lang": False, "noclasses": False}},
    )

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")

def chapter_files(part_dir: str):
    return sorted((MS / part_dir).glob("*.md"))

def render(files_by_part, title, out_path, with_front=True):
    body, toc = [], []
    body.append('<div class="cover"><h1>The AI/ML Interview Book</h1>'
                '<div class="subtitle">From Scratch to Pro — Theory, Code, and Interview Q&amp;A</div>'
                f'<div class="subtitle" style="font-size:11pt">{title}</div>'
                '<div class="author">AYuSh</div></div>')
    if with_front:
        fm = MS / "00_front_matter" / "front_matter.md"
        if fm.exists():
            body.append(md_to_html(fm.read_text(encoding="utf-8")))
    for part_title, files in files_by_part:
        if part_title:
            pid = slug(part_title)
            toc.append(f'<a class="toc-part" href="#{pid}">{part_title}</a>')
            body.append(f'<div class="part-divider" id="{pid}"><h1>{part_title}</h1></div>')
        for f in files:
            text = f.read_text(encoding="utf-8")
            m = re.match(r"#\s+(.+)", text)
            chap = m.group(1).strip() if m else f.stem
            cid = slug(chap)
            toc.append(f'<a class="toc-chap" href="#{cid}">{chap}</a>')
            html = md_to_html(text)
            html = html.replace("<h1", f'<h1 class="chapter" id="{cid}"', 1)
            body.append(html)
    toc_html = '<div class="toc-page"><h1>Contents</h1>' + "\n".join(toc) + "</div>"
    pyg = HtmlFormatter().get_style_defs(".codehilite")
    doc = (f"<html><head><meta charset='utf-8'><style>{pyg}</style></head><body>"
           + body[0] + toc_html + "".join(body[1:]) + "</body></html>")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    HTML(string=doc, base_url=str(MS)).write_pdf(str(out_path), stylesheets=[str(CSS)])
    print(f"WROTE {out_path} ({out_path.stat().st_size/1e6:.1f} MB)")

def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--part", type=int)
    g.add_argument("--full", action="store_true")
    g.add_argument("--file", type=Path)
    a = ap.parse_args()
    if a.file:
        render([("", [a.file if a.file.is_absolute() else ROOT / a.file])],
               a.file.stem, OUT / "parts" / f"{a.file.stem}.pdf", with_front=False)
    elif a.part:
        n, d, t = PARTS[a.part - 1]
        render([(t, chapter_files(d))], t, OUT / "parts" / f"Part_{ROMAN[n-1]}.pdf")
    else:
        fbp = [(t, chapter_files(d)) for _, d, t in PARTS]
        app = chapter_files("appendices")
        if app:
            fbp.append(("Appendices", app))
        render(fbp, "Complete Edition", OUT / "AI_ML_Interview_Book.pdf")

if __name__ == "__main__":
    main()
