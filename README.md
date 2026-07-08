# 📘⚜️ AI-BOOKS: The Ultimate AI & ML Interview Guide

Welcome to **AI-BOOKS**—a comprehensive, production-grade guide designed to help you master AI, Machine Learning, Deep Learning, and System Design interviews. This repository scaffolds and builds a book of high-quality chapters, each complete with mathematical rigor, hands-on Python implementations from scratch, and real-world Q&A.

---

## 🌟 Features & Highlights

- **From-Scratch Implementations:** All classical ML algorithms implemented in pure Python/NumPy to match scikit-learn behavior.
- **Mathematical Rigor:** Deep dives into proofs, formulations, and scaling traps.
- **87+ Core Q&A & Code Scenarios:** Structured Q&A boxes, system design mockcases, and coding challenges designed to challenge your thinking.
- **Complete Scaffold:** Automatic PDF generation pipelines using Python and custom CSS/HTML styling.

---

## 🗺️ Book Roadmap & Draft Progress

| Part | Chapter | Focus & Highlights | Status |
| :--- | :--- | :--- | :--- |
| **I. Foundations** | Ch 1: Mathematics for ML | Linear Algebra, Calculus, Probability, Optimization | 🟢 Drafted |
| | Ch 2: Python for ML | Performance, Memory, Vectorization, Libraries | 🟢 Drafted |
| | Ch 3: Data Structures & Algorithms | Core structures, ML-specific algorithms (KNN, k-means from scratch) | 🟢 Drafted |
| **II. Classical ML** | Ch 4: ML Fundamentals | Bias-variance, leakage demos, cross-validation | 🟢 Drafted |
| | Ch 5: Regression | OLS 3 ways, Lasso, Ridge, Logistic Regression, GLMs | 🟢 Drafted |
| | Ch 6: Classification | KNN, Naive Bayes, SVMs, Decision Trees, Pruning | 🟢 Drafted |
| | Ch 7: Ensemble Methods | Bagging, Boosting (AdaBoost, GBM from scratch), XGBoost, Stacking | 🟢 Drafted |
| | Ch 8: Unsupervised Learning | K-Means, DBSCAN, GMMs (EM), PCA, t-SNE, UMAP | 🟢 Drafted |
| | Ch 9-11: Feature Eng., Evaluation, Interpretability | Feature engineering, evaluation metrics, model interpretability | ⏳ Up Next |
| **III. Deep Learning** | Ch 12-18: NNs, Vision, Transformers | Feedforward, CNNs, RNNs, Transformers, Generative Models | ⏳ Planned |
| **IV. NLP & LLMs** | Ch 19-25: LMs, RAG, Agents | Tokenizers, pretraining, LLM alignment, RAG pipelines, Agents | ⏳ Planned |
| **V. ML Systems** | Ch 26-29: System Design & MLOps | System design, deployment, distributed training | ⏳ Planned |
| **VI. Specialized** | Ch 30-33: RecSys, Time Series, RL, Ethics | Recommenders, Forecasting, Reinforcement Learning, Responsible AI | ⏳ Planned |
| **VII. Coding & Prep** | Ch 34-39: Frameworks & Mock Interviews | Frameworks (PyTorch/TF), debugging, case studies | ⏳ Planned |

---

## 🛠️ Build Pipeline & Compilation

The book is compiled using a custom Python script that converts Markdown source files to PDF:

```bash
# To compile the book parts
python AI_ML_Interview_Book/build/build_pdf.py
```

### Build Requirements
- Python 3.8+
- WeasyPrint (for HTML/CSS-to-PDF compilation)
- Matplotlib (for mathematical SVG rendering)

---

*Crafted with care. Designed to help you succeed.*
