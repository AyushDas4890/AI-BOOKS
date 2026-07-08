# AI/ML Interview Book — Design Spec

Date: 2026-07-08
Status: Approved by user

## Goal

Produce a publishable, exhaustively detailed AI/ML interview preparation book (~1,200+ pages) that takes a reader from complete beginner to advanced. Final deliverable is a single compiled PDF: `AI_ML_Interview_Book.pdf`. Per-part PDFs are produced along the way for incremental review.

## Source of truth

The user's outline (`AI_ML BOOK.md`, copied to `docs/outline.md`) defines the full table of contents: 8 parts, 39 chapters, 5 appendices (A: cheat sheets, B: glossary, C: 100 rapid-fire questions, D: datasets & practice resources, E: SQL & Python quick reference). The outline's topic lists per chapter are the authoritative coverage checklist — every listed topic must appear in its chapter.

## User decisions

- Output format: PDF (compiled book; Markdown is internal working format only)
- Build order: sequential, Part I → Part VIII → appendices
- Depth: exhaustive, ~30+ pages per chapter
- Interview Q&A: at the end of each chapter
- Location: connected folder `C:\ADVANCE RAG\AI_ML_Interview_Book\`

## Architecture

```
AI_ML_Interview_Book/
├── docs/superpowers/specs/        # this spec + implementation plan
├── manuscript/
│   ├── 00_front_matter/           # cover text, preface, how-to-use-this-book
│   ├── part_01_foundations/       # ch01_mathematics.md ... ch03_dsa.md
│   ├── part_02_classical_ml/      # ch04 ... ch11
│   ├── part_03_deep_learning/     # ch12 ... ch18
│   ├── part_04_nlp_llms/          # ch19 ... ch25
│   ├── part_05_ml_systems/        # ch26 ... ch29
│   ├── part_06_specialized/       # ch30 ... ch33
│   ├── part_07_coding_rounds/     # ch34 ... ch36
│   ├── part_08_the_interview/     # ch37 ... ch39
│   └── appendices/                # appendix_a.md ... appendix_e.md
├── build/
│   ├── build_pdf.py               # md → HTML → WeasyPrint PDF
│   └── style.css                  # book typography, code blocks, page layout
└── output/
    ├── parts/                     # Part_I.pdf ... Part_VIII.pdf
    └── AI_ML_Interview_Book.pdf   # final compiled book
```

### Build pipeline (`build_pdf.py`)

- Python: `markdown` (with extensions: tables, fenced code, toc, attr_list) → HTML → WeasyPrint → PDF
- Code highlighting via Pygments; math rendered from LaTeX notation via `latex2mathml` (MathML). The Chapter 1 pilot build validates rendering quality; if WeasyPrint's MathML output is inadequate, the pipeline switches to matplotlib-mathtext-rendered SVG images for equations — decided once at the pilot, then fixed for the whole book
- Generates: cover page, title page, full table of contents with page numbers, part divider pages, running headers, page numbers
- Two modes: `--part N` (single part PDF) and `--full` (entire book)
- One command rebuilds everything; safe to run after every chapter

## Chapter template

Every chapter follows this fixed structure:

1. **Chapter opener** — what this chapter covers, why interviewers ask about it
2. **Concepts from scratch** — each topic explained in plain language first (no assumed background), then building to advanced depth: intuition → formal definition → math derivations where relevant → worked numeric examples
3. **Code implementation** — complete, runnable Python (NumPy / scikit-learn / PyTorch as appropriate), with line-by-line explanation. Implement-from-scratch versions where the outline demands it
4. **Pitfalls, comparisons & practical tips** — common confusions, when-to-use-which tables
5. **Interview Q&A** — 15–30 questions per chapter spanning four types: conceptual, mathematical/derivation, coding, and scenario/debugging. Each with a model answer at the depth an interviewer expects, plus "what interviewers look for" notes where useful

## Writing standards

- Subjective, teaching-style prose — explain *why*, not just *what*; analogies for hard concepts
- All math in LaTeX notation; every symbol defined at first use
- All code executed in the sandbox before inclusion — no untested snippets. Outputs shown where illustrative
- Consistent notation across chapters (notation table in front matter)
- Reference PDFs in the parent folder (Chip Huyen, Transformers guide, RAG/GenAI interview PDFs) may inform content for Parts IV–V, but all text is original

## Execution model

- One session produces 1–3 complete chapters at full depth, then rebuilds the PDF
- Estimated 15–20 sessions total for all 39 chapters + appendices
- After each part completes: verification pass — coverage check against the outline's topic list, notation consistency, code re-execution, Q&A count per chapter
- Progress tracked in `manuscript/PROGRESS.md` (chapter status: not started / drafted / verified). Git is unavailable on this mounted folder (sandbox permission limits), so PROGRESS.md is the sole progress record

## Error handling & risks

- **Session limits mid-chapter**: chapters are written section-by-section to disk, so a partial chapter is resumable; PROGRESS.md records last completed section
- **Folder-sync instability**: this mounted folder dropped previously-written files once during design (recovered from context). Mitigation: verify file existence (Glob) after every significant write; keep PROGRESS.md accurate so any loss is detectable and re-creatable
- **WeasyPrint scaling at 1,200 pages**: per-part PDFs are the review artifacts; full-book compile is run less frequently. If full compile becomes too slow/memory-heavy, fall back to compiling parts separately and merging with pypdf
- **Math rendering failures**: build script fails loudly on unparseable LaTeX so errors are caught at build time, not in the final PDF

## Testing

- Build script tested on a fixture chapter before mass production (fonts, code blocks, math, TOC, page numbers all verified by rendering and visually inspecting the PDF)
- Every code listing executed; failing code blocks block chapter completion
- Per-part verification pass as described above

## Out of scope

- Publishing/ISBN/typesetting for a specific publisher's template
- Original diagrams beyond what Markdown/SVG/matplotlib can produce in-pipeline
- Interactive or web versions of the book
