# Book Progress

Status values: `not-started` | `drafting (last section: <name>)` | `drafted` | `verified`

| # | Chapter | File | Status |
|---|---------|------|--------|
| FM | Front matter | 00_front_matter/front_matter.md | drafted |
| 1 | Mathematics for ML | part_01_foundations/ch01_mathematics.md | drafted |
| 2 | Python for ML Interviews | part_01_foundations/ch02_python.md | drafted |
| 3 | Data Structures & Algorithms | part_01_foundations/ch03_dsa.md | drafted |
| 4 | ML Fundamentals | part_02_classical_ml/ch04_ml_fundamentals.md | drafted |
| 5 | Regression | part_02_classical_ml/ch05_regression.md | drafted |
| 6 | Classification Algorithms | part_02_classical_ml/ch06_classification.md | drafted |
| 7 | Ensemble Methods | part_02_classical_ml/ch07_ensembles.md | drafted |
| 8 | Unsupervised Learning | part_02_classical_ml/ch08_unsupervised.md | drafted |
| 9 | Feature Engineering | part_02_classical_ml/ch09_feature_engineering.md | drafted |
| 10 | Model Evaluation & Metrics | part_02_classical_ml/ch10_evaluation.md | drafted |
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

- 2026-07-08 — project scaffolded
- 2026-07-08 — pipeline verified (cover, TOC+page numbers, code highlighting, tables, Q&A boxes OK); math renderer = matplotlib SVG (MathML rejected at decision gate)
- 2026-07-08 — ch01 drafted: 25-page PDF, 28 Q&A, 6 code listings all executed. NOTE: host-side Edit tool corrupts the sandbox mount view of a file (stale truncated reads) — after any Edit, rewrite the file whole via bash cp before building. Fixture files in build/ and output/parts/ are undeletable (no delete perm) but inert.
- 2026-07-08 — ch02 drafted: 28-page PDF, 28 Q&A, 8 code listings all executed. NOTE: markdown code spans (backticks) inside qa divs render literally — use <code> tags inside Q&A HTML blocks (fixed here via regex; apply to all future chapters).
- 2026-07-08 — ch03 drafted: 32-page PDF, 30 Q&A, 12 code listings all executed (incl. KNN, k-means, Fisher-Yates/reservoir, weighted/stratified sampling from scratch). Written via bash to avoid host-Edit mount corruption.
- 2026-07-08 — ch04 drafted: 26-page PDF, 28 Q&A, 8 code listings all executed (bias-variance decomposition measured empirically, leakage demo 0.50->0.90 on noise, walk-forward vs shuffled 131x optimism, distance concentration). NOTE: matplotlib mathtext rejects \underbrace — avoid in display math.
- 2026-07-08 — ch05 drafted: 28-page PDF, 28 Q&A, 8 code listings all executed (OLS 3 ways agree to 8.9e-16, condition number 8605->1.0 scaling demo, poly val-MSE hits noise floor at true degree, Lasso zeros 11/15, logistic scratch==sklearn, Poisson GLM one-line diff from logistic, collinearity coef-std 1.81 at rho=0.999, hetero CI coverage 83% vs nominal 95%). NOTE: matplotlib mathtext rejects \implies (use \Rightarrow) and \le/\ge (use \leq/\geq).
- 2026-07-08 — ch06 drafted: 26-page PDF, 28 Q&A, 8 code listings all executed (KNN k-sweep + x1000 scaling trap 0.887->0.792, scratch GNB==sklearn, scratch smoothed MNB spam filter==sklearn, SVM margin 3.88->1.04 & SVs 48->7 as C rises, kernel identity (a.b)^2==phi(a).phi(b) verified, tree split by hand entropy 0.655/gini 0.357 same split, ccp pruning 90->11 leaves test 0.798->0.842, gini-vs-entropy 40-dataset tie mean diff 0.019).
- 2026-07-08 — ch07 drafted: 26-page PDF, 28 Q&A, 8 code listings all executed (bagging disagreement 0.252->0.091 & acc 0.779->0.852, OOB 0.9035 vs test 0.9030, max_features U-shape, scratch AdaBoost==sklearn 0.860, scratch GBM MSE 23.6->1.89==sklearn, lr race 1.0-vs-0.1 with early-stop story, XGB/LGBM/HistGB head-to-head within noise, voting drags below best member 0.837<0.864, stacking wins 0.868 with learned weights incl. negative NB). NOTE: xgboost pip install times out; download wheel then extract via python3 -m zipfile, run with PYTHONPATH=/tmp/xgbpkg. lightgbm installs fine.
- 2026-07-08 — ch08 drafted: 29-page PDF, 28 Q&A, 10 code listings all executed (scratch Lloyd's + init lottery 1.24x, elbow+silhouette both point at true k=6, 3 staged k-means failures w/ DBSCAN/GMM cures, linkage double dissociation single-vs-ward on moons/blobs, DBSCAN eps cliff + HDBSCAN 0.967 no tuning, scratch EM monotone ll assert + recovers mixture, scratch PCA==sklearn + reconstruction, PCA/LDA/tSNE/UMAP 2-D bake-off 0.64->0.98, scratch apriori lift lesson beer->milk conf .80 lift 1.0, IsoForest/OCSVM/LOF injection study). umap-learn installs fine. NOTE: make_blobs defaults overlap — for clean pedagogy use center_box=(-15,15) or explicit centers.
- 2026-07-09 — ch09 drafted: 29-page PDF, 28 Q&A, 8 code listings all re-executed and outputs verified against manuscript (missing-mechanism study: HGB native NaN wins 0.877 MCAR/0.923 MNAR, indicator +MNAR 0.862; naive target-encoding leak +8pt fiction 0.688->0.780 vs OOF 0.695 on pure-noise cats; unscaled KNN 500x-feature collapse 0.579->~0.88, robust holds IQR=1.0 under 2% outliers; filter/RFE/RF all 8/8 real features, L1 6/10 by redundancy-pruning; imbalance PR-AUC baseline 0.768 beats all resampling, SMOTE-before-split fantasy 0.999; leverage outliers z-score 1/40 & IQR 2/40 vs IsoForest 32/40, Huber recovers slope 1.759; log R2 0.603->0.973, bins U-shape -0.001->0.889, XOR interaction 0.493->0.998; 900-cat smoothed target enc 0.734 ~ one-hot 0.727 > hashing 0.625 > frequency 0.619). Q&A section was absent from prior draft — added 28 boxes. Fixed scrambled in-code Listing docstring numbers (6/7/8). Deps this session need pip --trusted-host (self-signed cert in chain).
- 2026-07-09 — ch10 drafted: 25-page PDF, 28 Q&A, 9 code listings all executed (accuracy 0.953 vs always-neg 0.938 @6% prev, recall only 0.367; ROC-AUC 0.951->0.756 vs PR-AUC 0.867->0.266 as prev 30%->1.5%; threshold 0.5 P0.974/R0.664 vs max-F1 @0.360 P0.893/R0.749 vs precision>=0.95 @0.510; SVM calibration Brier 0.061->0.043 & logloss 0.249->0.162 via Platt/isotonic, S-curve reliability; proper-scoring: timid acc1.000/logloss0.511 vs bold acc0.902/logloss0.340, one ~0 prob = logloss 34.5; regression RMSE 5.11->6.75 vs MAE 4.08->4.40 on 3 outliers, MAPE 8.5%->497% near zero, R2 -24.6 for const, adjR2 0.914->0.912 on +40 noise; ranking MRR/MAP/NDCG good vs bad 1.000/0.917/0.967 vs 0.333/0.411/0.583, graded NDCG 1.000 vs 0.614; McNemar 184v54 chi2 58.3 p<1e-4 & bootstrap AUC gap +0.047 CI[+0.039,+0.055], seed-only gap +0.0022 CI[-0.0004,+0.0048] spans 0; split conformal coverage 90%->0.904, 95%->0.951). statsmodels needed for McNemar; deps via pip --trusted-host (self-signed cert).
