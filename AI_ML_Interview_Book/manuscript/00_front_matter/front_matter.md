# Preface

If you are preparing for a machine learning interview in the current market, you are facing a strange paradox. The field has never been more accessible — free courses, open models, and endless tutorials — and yet interviews have never been harder to prepare for. The bar has moved. A decade ago you could pass an ML interview by explaining the bias-variance tradeoff and implementing k-means. Today you may be asked to derive backpropagation on a whiteboard at 10 a.m., design a retrieval-augmented generation system at 1 p.m., and debug a training loss curve at 4 p.m. — all for the same job.

This book exists to carry you across that entire range. It is written for three kinds of readers at once. The **complete beginner** who has decided to enter the field and wants a single, sequential path from "what is a matrix?" to "how does speculative decoding work?". The **practitioner** who uses scikit-learn or PyTorch daily but has never had to justify *why* Adam differs from RMSprop out loud, under pressure. And the **interview-ready candidate** who needs dense, honest model answers rather than hand-waving.

Three principles govern every chapter:

**Nothing is assumed.** Every topic begins in plain language, as if you have never met it before, and then climbs — intuition first, then formal definitions, then derivations with every symbol defined, then a small worked numeric example you can check with a pencil.

**All code runs.** Every code listing in this book was executed before it was printed. Where output is shown, it is real output. You can type any listing into a Python file and run it.

**Interviews are the destination.** Every chapter ends with real interview-style questions and model answers written at the depth an interviewer actually expects — including notes on what the interviewer is listening for.

# How to use this book

The book has eight parts and is designed to be read two different ways.

**The scratch-to-pro path (8–16 weeks).** Read sequentially, Part I through Part VIII. Part I builds the mathematical and programming foundation; nothing later will assume more than it teaches. Do the worked examples by hand. Type out and run the code. This path turns a motivated beginner into a credible candidate.

**The interview-in-two-weeks path.** If you already work in the field, go straight to where roughly 80% of interview questions concentrate: Part II (classical ML), Part III (deep learning), and Part IV (NLP and LLMs), plus Chapter 34 (implement-from-scratch — the single most requested interview practice topic). Skim the Q&A sections of every other chapter, and read Part VIII the weekend before your loop.

Whichever path you take, treat the end-of-chapter Q&A as active practice, not reading material: cover the answer, say your own answer out loud, then compare. Speaking an answer is a different skill from recognizing one.

A note on the appendices: Appendix A (formula cheat sheets) and Appendix C (100 rapid-fire one-liners) are designed for the final 48 hours before an interview. Appendix E (SQL and Python quick reference) covers the data-manipulation screens that increasingly precede ML rounds.

# Notation

The same symbols mean the same things in every chapter of this book. When a chapter introduces a symbol not listed here, it defines it at first use.

| Symbol | Meaning |
|---|---|
| $x$ | a single input (feature vector), $x \in \mathbb{R}^d$ |
| $X$ | the design matrix of all inputs, shape $n \times d$ (rows are examples) |
| $y$ | the true label or target |
| $\hat{y}$ | the model's prediction |
| $n$ | number of examples; $d$ — number of features |
| $w$, $b$ | weight vector and bias of a single model or layer |
| $\theta$ | the set of all learnable parameters of a model |
| $\eta$ | learning rate |
| $\mathcal{L}$ | loss function (to be minimized) |
| $\nabla_\theta \mathcal{L}$ | gradient of the loss with respect to parameters |
| $\sigma(\cdot)$ | the sigmoid function $\sigma(z) = 1/(1+e^{-z})$ |
| $\mathbb{E}[\cdot]$ | expectation; $\mathrm{Var}(\cdot)$ — variance |
| $P(A \mid B)$ | conditional probability of $A$ given $B$ |
| $\mathcal{N}(\mu, \sigma^2)$ | Gaussian distribution with mean $\mu$ and variance $\sigma^2$ |
| $D_{KL}(P \| Q)$ | Kullback–Leibler divergence from $Q$ to $P$ |
| $H(P)$ | entropy of distribution $P$ |
| $\|v\|$ | Euclidean (L2) norm of vector $v$ |
| $A^T$ | transpose of matrix $A$; $A^{-1}$ — inverse |
| $\lambda$ | regularization strength (or an eigenvalue — context makes it clear) |

Vectors are column vectors unless stated otherwise. Code follows Python conventions: 0-indexed, row-major NumPy arrays where `X[i]` is the $i$-th example.
