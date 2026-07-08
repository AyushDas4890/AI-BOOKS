# AI/ML Interview Book Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a ~1,200-page AI/ML interview prep book (39 chapters + 5 appendices) as Markdown, compiled to a final `AI_ML_Interview_Book.pdf` via an automated pipeline.

**Architecture:** Markdown manuscript (one file per chapter) → Python build script (python-markdown + Pygments + latex2mathml → HTML → WeasyPrint) → per-part PDFs and final full-book PDF. Content produced sequentially, Part I → VIII → appendices, 1–3 chapters per session.

**Tech Stack:** Python 3, `markdown`, `pygments`, `latex2mathml`, `weasyprint`, `pypdf` (merge fallback), NumPy/scikit-learn/PyTorch for chapter code verification.

**IMPORTANT — no git:** Git does not work on this mounted folder (sandbox permission limits). Wherever a normal plan would say "commit", instead update `manuscript/PROGRESS.md`. Never run `git` commands.

**IMPORTANT — flaky mount:** This folder once dropped freshly written files mid-session. After every significant Write, verify with Glob that the file exists. If a file vanishes, recreate it from context immediately and note it in PROGRESS.md.

**Paths:** File tools (Read/Write/Edit) use `C:\ADVANCE RAG\AI_ML_Interview_Book\...`. Bash uses `/sessions/<session>/mnt/ADVANCE RAG/AI_ML_Interview_Book/...` (check your session's mount name; verify the mount actually shows the folder before relying on bash — if it doesn't, do file operations with host tools and only run Python via temp copies under `/tmp`). `$BOOK` below means the bash-side book directory.

**Authoritative outline:** `docs/outline.md` (copied in Task 1). Every bullet under a chapter in the outline is a mandatory topic for that chapter.

---

### Task 1: Scaffold project structure

**Files:**
- Create: `docs/outline.md` (copy of the user's uploaded outline `AI_ML BOOK.md`)
- Create: `manuscript/PROGRESS.md`

- [ ] **Step 1: Copy the outline into the project**

Read the uploaded `AI_ML BOOK.md` (session uploads folder; if gone, reconstruct from the chapter list in this plan + spec) and Write it to `docs/outline.md`. Verify with Glob.

- [ ] **Step 2: Write PROGRESS.md**

Create `manuscript/PROGRESS.md` (directories are created implicitly by Write):

```markdown
# Book Progress

Status values: `not-started` | `drafting (last section: <name>)` | `drafted` | `verified`

| # | Chapter | File | Status |
|---|---------|------|--------|
| FM | Front matter | 00_front_matter/front_matter.md | not-started |
| 1 | Mathematics for ML | part_01_foundations/ch01_mathematics.md | not-started |
| 2 | Python for ML Interviews | part_01_foundations/ch02_python.md | not-started |
| 3 | Data Structures & Algorithms | part_01_foundations/ch03_dsa.md | not-started |
| 4 | ML Fundamentals | part_02_classical_ml/ch04_ml_fundamentals.md | not-started |
| 5 | Regression | part_02_classical_ml/ch05_regression.md | not-started |
| 6 | Classification Algorithms | part_02_classical_ml/ch06_classification.md | not-started |
| 7 | Ensemble Methods | part_02_classical_ml/ch07_ensembles.md | not-started |
| 8 | Unsupervised Learning | part_02_classical_ml/ch08_unsupervised.md | not-started |
| 9 | Feature Engineering | part_02_classical_ml/ch09_feature_engineering.md | not-started |
| 10 | Model Evaluation & Metrics | part_02_classical_ml/ch10_evaluation.md | not-started |
| 11 | Interpretability | part_02_classical_ml/ch11_interpretability.md | not-started |
| 12 | NN Fundamentals | part_03_deep_learning/ch12_nn_fundamentals.md | not-started |
| 13 | Training Deep Networks | part_03_deep_learning/ch13_training.md | not-started |
| 14 | CNNs | part_03_deep_learning/ch14_cnns.md | not-started |
| 15 | RNNs & Sequence Models | part_03_deep_learning/ch15_rnns.md | not-started |
| 16 | Transformers | part_03_deep_learning/ch16_transformers.md | not-started |
| 17 | Computer Vision | part_03_deep_learning/ch17_vision.md | not-started |
| 18 | Generative Models | part_03_deep_learning/ch18_generative.md | not-started |
| 19 | Classical & Neural NLP | part_04_nlp_llms/ch19_nlp.md | not-started |
| 20 | Pretrained LMs | part_04_nlp_llms/ch20_pretrained_lms.md | not-started |
| 21 | LLM Training & Alignment | part_04_nlp_llms/ch21_llm_training.md | not-started |
| 22 | LLM Inference & Decoding | part_04_nlp_llms/ch22_llm_inference.md | not-started |
| 23 | Prompt Engineering | part_04_nlp_llms/ch23_prompting.md | not-started |
| 24 | RAG | part_04_nlp_llms/ch24_rag.md | not-started |
| 25 | AI Agents | part_04_nlp_llms/ch25_agents.md | not-started |
| 26 | ML System Design | part_05_ml_systems/ch26_system_design.md | not-started |
| 27 | MLOps & Deployment | part_05_ml_systems/ch27_mlops.md | not-started |
| 28 | Data Engineering for ML | part_05_ml_systems/ch28_data_engineering.md | not-started |
| 29 | Distributed Training | part_05_ml_systems/ch29_distributed.md | not-started |
| 30 | Recommender Systems | part_06_specialized/ch30_recsys.md | not-started |
| 31 | Time Series | part_06_specialized/ch31_time_series.md | not-started |
| 32 | Reinforcement Learning | part_06_specialized/ch32_rl.md | not-started |
| 33 | Responsible AI | part_06_specialized/ch33_responsible_ai.md | not-started |
| 34 | Implement From Scratch | part_07_coding_rounds/ch34_from_scratch.md | not-started |
| 35 | PyTorch/TensorFlow Qs | part_07_coding_rounds/ch35_frameworks.md | not-started |
| 36 | ML Debugging Scenarios | part_07_coding_rounds/ch36_debugging.md | not-started |
| 37 | Interview Formats & Strategy | part_08_the_interview/ch37_formats.md | not-started |
| 38 | Behavioral & Projects | part_08_the_interview/ch38_behavioral.md | not-started |
| 39 | Case Studies & Mocks | part_08_the_interview/ch39_case_studies.md | not-started |
| A | Cheat sheets | appendices/appendix_a_cheatsheets.md | not-started |
| B | Glossary | appendices/appendix_b_glossary.md | not-started |
| C | 100 rapid-fire questions | appendices/appendix_c_rapidfire.md | not-started |
| D | Datasets & resources | appendices/appendix_d_resources.md | not-started |
| E | SQL & Python reference | appendices/appendix_e_reference.md | not-started |

## Build log
(append: date — what was built/verified)
```

- [ ] **Step 3: Verify both files with Glob**, then append to build log: `2026-07-08 — project scaffolded`.

---

### Task 2: Install and verify build dependencies

- [ ] **Step 1: Install**

Run:
```bash
pip install --break-system-packages --quiet markdown pygments latex2mathml weasyprint pypdf && python3 -c "import markdown, pygments, latex2mathml, weasyprint, pypdf; print('deps OK')"
```
Expected: `deps OK`. WeasyPrint needs system libs (pango, cairo) — preinstalled on Ubuntu 22 sandbox; if import fails run `apt list --installed 2>/dev/null | grep -E 'pango|cairo'` and report before proceeding.

- [ ] **Step 2: Smoke-test WeasyPrint**

Run:
```bash
python3 -c "
from weasyprint import HTML
HTML(string='<h1>test</h1><p>hello</p>').write_pdf('/tmp/wp_smoke.pdf')
import os; print('smoke OK', os.path.getsize('/tmp/wp_smoke.pdf'))"
```
Expected: `smoke OK <nonzero size>`.

---

### Task 3: Write the book stylesheet

**Files:**
- Create: `build/style.css`

- [ ] **Step 1: Write `build/style.css`**

```css
@page {
  size: A4;
  margin: 22mm 18mm 20mm 18mm;
  @bottom-center { content: counter(page); font-size: 9pt; color: #666; }
  @top-left { content: string(chaptitle); font-size: 8.5pt; color: #888; }
}
@page cover { margin: 0; @bottom-center { content: none; } @top-left { content: none; } }
@page divider { @top-left { content: none; } }

body { font-family: "DejaVu Serif", Georgia, serif; font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; }
.cover { page: cover; height: 297mm; display: flex; flex-direction: column;
  justify-content: center; text-align: center; background: #10243e; color: #fff; }
.cover h1 { font-size: 34pt; margin: 0 20mm; border: none; }
.cover .subtitle { font-size: 15pt; color: #9fc2e8; margin-top: 8mm; }
.cover .author { font-size: 12pt; color: #ccc; margin-top: 30mm; }

.part-divider { page: divider; page-break-before: always; height: 250mm;
  display: flex; align-items: center; justify-content: center; }
.part-divider h1 { font-size: 26pt; color: #10243e; border: none; text-align: center; }

h1.chapter { string-set: chaptitle content(); page-break-before: always;
  font-size: 21pt; color: #10243e; border-bottom: 2.5pt solid #10243e; padding-bottom: 3mm; }
h2 { font-size: 14.5pt; color: #16365c; margin-top: 8mm; }
h3 { font-size: 12pt; color: #1f4e79; }
h4 { font-size: 10.5pt; }

pre { background: #f6f8fa; border: 0.5pt solid #d0d7de; border-radius: 3pt;
  padding: 3mm; font-size: 8.3pt; line-height: 1.4; white-space: pre-wrap;
  font-family: "DejaVu Sans Mono", monospace; page-break-inside: auto; }
code { font-family: "DejaVu Sans Mono", monospace; font-size: 8.8pt; background: #f2f2f2; padding: 0 1.5pt; }
pre code { background: none; padding: 0; }

table { border-collapse: collapse; width: 100%; font-size: 9pt; margin: 4mm 0; }
th, td { border: 0.5pt solid #999; padding: 1.6mm 2.2mm; text-align: left; }
th { background: #e8eef5; }
tr { page-break-inside: avoid; }

blockquote { border-left: 2.5pt solid #1f4e79; margin-left: 0; padding: 1mm 4mm;
  background: #f4f8fc; color: #333; }

.qa { border: 0.7pt solid #b8cce4; border-radius: 3pt; padding: 3mm 4mm; margin: 4mm 0; page-break-inside: avoid; }
.qa .q { font-weight: bold; color: #10243e; }

.toc-page { page-break-after: always; }
.toc-page a { text-decoration: none; color: #1a1a1a; display: block; margin: 1mm 0; }
.toc-page a.toc-part { font-weight: bold; color: #10243e; margin-top: 4mm; }
.toc-page a.toc-chap { margin-left: 6mm; }
.toc-page a::after { content: leader('.') " " target-counter(attr(href), page); }

math { font-size: 10.5pt; }
img { max-width: 100%; }
figure { text-align: center; page-break-inside: avoid; }
figcaption { font-size: 8.5pt; color: #555; }
```

- [ ] **Step 2: Verify** — Glob for `build/style.css`; bash `test -s` if the mount is healthy.

---

### Task 4: Write the build script

**Files:**
- Create: `build/build_pdf.py`

- [ ] **Step 1: Write `build/build_pdf.py`**

```python
#!/usr/bin/env python3
"""Build AI/ML Interview Book PDFs. Usage:
  python3 build_pdf.py --part 1     # single part PDF -> output/parts/Part_I.pdf
  python3 build_pdf.py --full       # full book       -> output/AI_ML_Interview_Book.pdf
  python3 build_pdf.py --file manuscript/part_01_foundations/ch01_mathematics.md  # one chapter
"""
import argparse, re, sys
from pathlib import Path

import markdown
from latex2mathml.converter import convert as tex2mml
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

def _convert_math_segment(text: str) -> str:
    def blk(m):
        try:
            return f'<div style="text-align:center">{tex2mml(m.group(1).strip(), display="block")}</div>'
        except Exception as e:
            sys.exit(f"FATAL: bad display math: {m.group(1)[:80]!r} -> {e}")
    def inl(m):
        try:
            return tex2mml(m.group(1).strip())
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
```

- [ ] **Step 2: Syntax check**

Run: `python3 -m py_compile "$BOOK/build/build_pdf.py" && echo COMPILE-OK`
Expected: `COMPILE-OK`

---

### Task 5: Pipeline pilot test (fixture chapter)

TDD for the pipeline: a fixture exercising every feature — headings, highlighted code, display + inline math, tables, blockquotes, Q&A blocks.

**Files:**
- Create: `build/fixture_test.md` (temporary)

- [ ] **Step 1: Write the fixture**

````markdown
# Chapter 0: Pipeline Fixture

## Math
Inline math $\hat{y} = \sigma(w^T x + b)$ in a sentence.

$$\mathcal{L} = -\frac{1}{N}\sum_{i=1}^{N} \left[ y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i) \right]$$

## Code
```python
import numpy as np
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))
print(sigmoid(np.array([0.0, 2.0])))
```

## Table
| Metric | Formula | Use |
|---|---|---|
| Precision | $\frac{TP}{TP+FP}$ | Costly false positives |

> **Interview tip:** always state assumptions first.

<div class="qa"><p class="q">Q: Why sigmoid for logistic regression?</p>
<p>Because it maps log-odds to (0,1)...</p></div>
````

- [ ] **Step 2: Build the fixture**

Run: `cd "$BOOK" && python3 build/build_pdf.py --file build/fixture_test.md`
Expected: `WROTE .../output/parts/fixture_test.pdf`

- [ ] **Step 3: Verify rendering — MANDATORY VISUAL CHECK (math decision gate)**

Read the PDF with the Read tool (renders pages as images). Check: (a) equations render as real math, not raw LaTeX or tofu; (b) code syntax-highlighted, no page overflow; (c) table borders present; (d) footer page number.

**Decision gate from spec:** if MathML renders badly in WeasyPrint, switch `_convert_math_segment` to matplotlib mathtext → inline SVG:

```python
import base64, io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def tex_to_img(tex, display):
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, f"${tex}$", fontsize=11)
    fig.canvas.draw()
    bb = t.get_window_extent()
    fig.set_size_inches(bb.width / 72, bb.height / 72)
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", bbox_inches="tight", pad_inches=0.02, transparent=True)
    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    style = "display:block;margin:6px auto" if display else "vertical-align:middle"
    return f'<img src="data:image/svg+xml;base64,{b64}" style="{style}"/>'
```
Replace the `tex2mml` calls with `tex_to_img(...)` (note: matplotlib mathtext accepts a subset of LaTeX — no `\text{}`; adjust macros accordingly). Rebuild fixture, re-verify. Record the chosen renderer in PROGRESS.md. The choice is then FIXED for the whole book.

- [ ] **Step 4: Clean up and log**

Delete `build/fixture_test.md` and `output/parts/fixture_test.pdf`. Append to PROGRESS.md build log: `pipeline verified; math renderer = <mathml|svg>`.

---

### Task 6: Front matter

**Files:**
- Create: `manuscript/00_front_matter/front_matter.md`

- [ ] **Step 1: Write front matter** containing, in order: title-page text block; preface (who this book is for: absolute beginner → advanced; structured around real interview loops); "How to use this book" (reading paths: full sequential for beginners; Parts II–IV + Ch. 34 for interview-in-2-weeks); and the **notation table** — one table defining every recurring symbol: $x$ (feature vector), $X$ (design matrix, $n \times d$), $y$/$\hat{y}$ (label/prediction), $w, b$ (weights, bias), $\theta$ (all parameters), $\eta$ (learning rate), $\mathcal{L}$ (loss), $\nabla$ (gradient), $\sigma(\cdot)$ (sigmoid), $\mathbb{E}$ (expectation), $\mathcal{N}(\mu,\sigma^2)$ (Gaussian), $D_{KL}$ (KL divergence). ~4–6 pages.

- [ ] **Step 2: Build and verify**

Run: `cd "$BOOK" && python3 build/build_pdf.py --file manuscript/00_front_matter/front_matter.md`
Read the PDF; verify the notation table renders. PROGRESS.md: front matter → `drafted`.

---

### Task 7: Chapter 1 — Mathematics for ML (content pilot)

First real chapter; validates the chapter template at exhaustive depth. **All later chapters follow exactly this procedure.**

**Files:**
- Create: `manuscript/part_01_foundations/ch01_mathematics.md`

**Coverage checklist (from `docs/outline.md`, Chapter 1 — ALL mandatory):** linear algebra (vectors, matrices, eigenvalues/eigenvectors, SVD, matrix decomposition); calculus (derivatives, partials, chain rule, gradients, Jacobians, Hessians); probability (Bayes, conditional probability, Gaussian/Bernoulli/Binomial/Poisson); statistics (mean/median/mode, variance, std, covariance, correlation); hypothesis testing (p-values, t-tests, chi-square, A/B testing, confidence intervals); CLT & LLN; MLE vs MAP; information theory (entropy, cross-entropy, KL divergence, mutual information).

**Chapter template (fixed for all 39 chapters):**
1. `# Chapter N: <Title>` opener — what's covered, why interviewers ask it (½–1 page)
2. One `##` section per outline bullet-group; within each: plain-language intuition first (assume zero background) → formal definition → derivation with every symbol defined → small worked numeric example → connection to ML practice
3. `## Code implementations` — complete runnable Python per major concept, line-by-line commentary
4. `## Pitfalls, comparisons and practical tips` — confusion tables, when-to-use-which
5. `## Interview questions and answers` — 15–30 questions as `<div class="qa">` blocks, mixed across: conceptual, mathematical/derivation, coding, scenario. Model answers at interview depth + "what the interviewer looks for" notes on hard ones.

- [ ] **Step 1: Write section-by-section, saving to disk after EACH `##` section** (resumability). After each save: Glob-verify + update PROGRESS.md to `drafting (last section: <name>)`.

- [ ] **Step 2: Execute every code block before it goes in the chapter.** Copy to `/tmp` and run:

```bash
cd /tmp && python3 ch1_snippet_eigen.py
```
Every listing must run cleanly; paste real output into the chapter where illustrative. A failing snippet blocks the chapter.

- [ ] **Step 3: Q&A section** — 25–30 questions (dense chapter). Verify: `grep -c 'class="qa"' <file>` ≥ 25.

- [ ] **Step 4: Build and visually verify**

Run: `cd "$BOOK" && python3 build/build_pdf.py --file manuscript/part_01_foundations/ch01_mathematics.md`
Read the PDF. Check ≥ 30 pages, math renders, code fits. If under 30 pages, deepen the thinnest sections (more worked examples/derivations) — never pad with filler.

- [ ] **Step 5: PROGRESS.md** — Chapter 1 → `drafted`; append build-log line.

---

### Task 8: Chapters 2–39 — repeatable production loop

**One iteration = one chapter,** strictly in PROGRESS.md order.

- [ ] **Step 1:** Extract chapter N's bullet list from `docs/outline.md` = mandatory coverage checklist.
- [ ] **Step 2:** Write the chapter at its `manuscript/...` path, exact Task 7 template (opener → concept sections → code → pitfalls → Q&A), section-by-section saves + PROGRESS.md `drafting` updates.
- [ ] **Step 3:** Execute all code (Task 7 Step 2 procedure). Framework-heavy chapters (13–18, 20–25, 35): if a library/model is unavailable (no GPU/API keys), import-check and run with mocked/small inputs where feasible; mark anything genuinely unrunnable `> Verified: syntax + logic review only` — sparingly.
- [ ] **Step 4:** Q&A: 15–30 questions (`grep -c 'class="qa"'`).
- [ ] **Step 5:** Build single-chapter PDF, Read, verify (≥ ~30 pages, clean rendering).
- [ ] **Step 6:** PROGRESS.md → `drafted`.
- [ ] **Step 7 (end of each part only) — part verification pass:**
  - Coverage: every outline bullet for every chapter in the part has a matching section; fix misses.
  - Re-run all code files for the part's chapters.
  - Notation spot-check against front-matter table.
  - `python3 build/build_pdf.py --part <n>`; Read and spot-check 5+ pages.
  - PROGRESS.md: part chapters → `verified`; build-log line.

**Session cadence:** 1–3 chapters per session. At session start, read PROGRESS.md, resume the first `not-started`/`drafting` chapter from its last saved section.

---

### Task 9: Appendices A–E

**Files:** the five `manuscript/appendices/appendix_*.md` files listed in PROGRESS.md.

- [ ] **Step 1: Appendix A — cheat sheets.** Formula tables per domain (metrics, distributions, gradients of common losses, attention shapes, RAG eval metrics); algorithm comparison tables (trees vs boosting vs bagging; optimizers; PEFT methods). Tables only, minimal prose.
- [ ] **Step 2: Appendix B — glossary.** Every bolded first-use term from chapters 1–39, alphabetized, 1–2 sentence definitions. Build by grepping `\*\*[A-Z]` across manuscript, dedup.
- [ ] **Step 3: Appendix C — 100 rapid-fire one-liners.** Numbered 1–100, question + ≤2-sentence answer, drawn proportionally from all parts. Verify count = 100 via grep `'^[0-9]\+\.'`.
- [ ] **Step 4: Appendix D — datasets & practice resources.** Table: dataset/platform, what to practice, difficulty. Classics (Titanic, MNIST, MovieLens, SQuAD) + practice platforms.
- [ ] **Step 5: Appendix E — SQL & Python quick reference.** Window functions, joins, CTE patterns with runnable examples (execute via sqlite3 in-sandbox); NumPy/Pandas idiom tables.
- [ ] **Step 6:** Execute all code, build each appendix via `--file`, Read-verify, PROGRESS.md → `drafted` then `verified` after coverage skim.

---

### Task 10: Final compile and book-level verification

- [ ] **Step 1: Full build**

Run: `cd "$BOOK" && python3 build/build_pdf.py --full`
Expected: `WROTE .../output/AI_ML_Interview_Book.pdf`. If killed (memory) or > ~10 min, fallback — build all 8 parts, build each appendix via `--file`, then merge:

```python
from pathlib import Path
from pypdf import PdfWriter
w = PdfWriter()
order = [f"output/parts/Part_{r}.pdf" for r in ["I","II","III","IV","V","VI","VII","VIII"]] \
      + sorted(str(p) for p in Path("output/parts").glob("appendix_*.pdf"))
for p in order:
    w.append(p)
w.write("output/AI_ML_Interview_Book.pdf")
print("merged")
```
(Caveat: merge fallback loses global TOC page-number continuity — note the compromise in PROGRESS.md if used.)

- [ ] **Step 2: Verify the final PDF**
  - Page count ≥ 1000: `python3 -c "from pypdf import PdfReader; print(len(PdfReader('output/AI_ML_Interview_Book.pdf').pages))"`
  - Read tool: cover, TOC (links + page numbers), one spot-check page per part (8 pages), one appendix page.
  - TOC entries = 39 chapters + 8 parts + appendices.

- [ ] **Step 3: Deliver.** Present `output/AI_ML_Interview_Book.pdf` via present_files. PROGRESS.md → all `verified`; final build-log entry.
