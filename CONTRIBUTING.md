# Contributing to The AI/ML Interview Book

Thank you for your interest in contributing! This guide will help you get started.

---

## 📋 Table of Contents

- [Reporting Issues](#-reporting-issues)
- [Suggesting Improvements](#-suggesting-improvements)
- [Setting Up the Project](#-setting-up-the-project)
- [Code Style & Conventions](#-code-style--conventions)
- [Pull Request Process](#-pull-request-process)

---

## 🐛 Reporting Issues

If you find an error in a code listing, a mathematical derivation, or a Q&A answer:

1. **Open a GitHub Issue** with a clear, descriptive title
2. Include the following details:
   - **Chapter and listing/Q&A number** (e.g., "Ch 7, Listing 4" or "Ch 5, Q12")
   - **What you expected** vs. **what you observed**
   - **Your environment** — Python version, OS, and relevant library versions
3. If possible, include the minimal code to reproduce the problem

### Example Issue Title
```
Ch 5, Listing 3: OLS normal equation produces different result on NumPy 2.x
```

---

## 💡 Suggesting Improvements

We welcome suggestions for:
- **Content additions** — New Q&A scenarios, alternative implementations, edge cases
- **Clarity improvements** — Reworded explanations, better diagrams, additional context
- **Build pipeline enhancements** — PDF styling, compilation speed, cross-platform fixes

Please open an issue to discuss the idea before submitting a large pull request.

---

## 🛠️ Setting Up the Project

### Prerequisites

- Python 3.8+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/AyushDas4890/AI-Interview-Prep.git
cd AI-Interview-Prep

# Create a virtual environment
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

# Install build dependencies
pip install -r requirements.txt
```

### Building a Chapter

```bash
# Build a single chapter PDF
python AI_ML_Interview_Book/build/build_pdf.py --file manuscript/part_02_classical_ml/ch05_regression.md

# Build an entire part
python AI_ML_Interview_Book/build/build_pdf.py --part 2

# Output goes to AI_ML_Interview_Book/output/parts/
```

---

## 📝 Code Style & Conventions

### Manuscript Files (Markdown)

- Each chapter is a self-contained `.md` file in `manuscript/part_XX_<topic>/`
- Use `<div class="qa">` blocks for Q&A boxes — **not** markdown blockquotes
- Use `<code>` tags inside Q&A HTML blocks — **not** backticks (they render literally)
- Use `<em>` for emphasis inside Q&A divs — **not** `*asterisks*`
- Escape `<` as `&lt;` in Q&A prose to prevent HTML tag swallowing

### Code Listings

- All implementations should be in **pure Python/NumPy** unless comparing against a library
- Include a docstring at the top of each listing explaining what it demonstrates
- Every listing must be **executable** and produce the **exact output** referenced in the text
- Gradient-checked implementations should report relative error (target: < 1e-6)

### Mathematical Notation

The build pipeline uses matplotlib's `mathtext` for rendering. The following LaTeX commands are **not supported** and will crash the build:
- `\underbrace`, `\implies`, `\le`, `\ge`, `\big`
- Use `\Rightarrow` instead of `\implies`
- Use `\leq` / `\geq` instead of `\le` / `\ge`

---

## 🔄 Pull Request Process

1. **Fork the repository** and create a feature branch:
   ```bash
   git checkout -b fix/chapter-7-listing-4
   ```

2. **Make your changes** following the code style conventions above

3. **Test your changes:**
   - If you modified a code listing, run it and verify the output matches
   - If you modified manuscript text, build the chapter PDF and check formatting:
     ```bash
     python AI_ML_Interview_Book/build/build_pdf.py --file manuscript/path/to/chapter.md
     ```

4. **Commit with a clear message:**
   ```bash
   git commit -m "fix(ch07): correct AdaBoost weight update formula in Listing 4"
   ```

5. **Push and open a Pull Request** against `main`

6. **In the PR description**, reference any related issues (e.g., `Closes #42`)

### Commit Message Format

Use conventional commit style:
```
type(scope): description

Types: fix, feat, docs, style, refactor, test, build
Scope: ch01-ch39, build, readme, etc.
```

Examples:
```
fix(ch05): correct Ridge regression gradient in Listing 2
feat(ch12): add dying-ReLU demonstration experiment
docs(readme): update chapter progress table
build: add requirements.txt for pip dependencies
```

---

## 📜 License

By contributing, you agree that your contributions will be licensed under the same terms as the project.

---

*Thank you for helping make this book better!*
