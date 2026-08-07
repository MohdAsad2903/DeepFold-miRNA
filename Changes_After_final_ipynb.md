# DeepFold — What Changed After `final.ipynb`

---

## Base: `final.ipynb` (Starting Point)

The original notebook contained the complete working pipeline ending at **AUC 0.720**:

| Component | What it was |
|---|---|
| Data pipeline | COSMIC + gnomAD → `final_dataset.csv` (2,372 samples, 1,186 per class) |
| UFold contact maps | 4-channel tensor: C1 healthy, C2 mutant, C3 diff, C4 mask |
| CNN v3 | Split-path CNN, single AdaptiveAvgPool(4), no position attention → AUC ~0.625 |
| Siamese GCN | Shared-weight graph encoder → AUC ~0.640 |
| k-mer XGBoost v1 | `GradientBoostingClassifier`, k=3 and k=4 features only (~971 dims), fixed hyperparameters → AUC ~0.704 |
| MFE XGBoost | ViennaRNA thermodynamic features, 12 dims → AUC ~0.633 |
| Ensemble v1 | 4-model logistic regression meta-learner → **AUC 0.720** |

---

## All Changes Made After `final.ipynb`

---

### `cells_group_A.py` — CNN v4 (Replaced CNN v3)

| Change | Reason |
|---|---|
| Added **CoordAttention** after the split-path merge | CNN v3 was spatially blind — a structural change at position 5 and position 64 were treated identically. CoordAttention adds horizontal+vertical positional weighting so the network learns which rows/columns of the contact map matter most for pathogenicity. |
| Replaced `AdaptiveAvgPool(4)` with **MultiScalePool** | Single pooling scale collapsed all spatial information into one scale. MultiScalePool = global avg (256d) + global max (256d) + 2×2 spatial avg (1024d) concatenated = 1,536 dims. Captures global magnitude + peak disruption + spatial quadrant of disruption simultaneously. |
| Added **Mixup augmentation** (α=0.2) | Contact maps are continuous tensors — interpolating between two contact maps is biologically valid (it represents a structure between two real structures). Smooths the decision boundary and improves generalisation. |
| Added **label smoothing** (ε=0.10) | COSMIC labels contain passenger mutation noise. Training toward hard 0/1 targets causes overconfident overfitting on noisy labels. Smoothed targets (0.9/0.1) regularise the output distribution. |
| Added **per-group channel normalisation** | C3 has ~10x smaller magnitude than C1/C2. Without separate normalisation, the shared encoder ignores C3 because the gradient from C1/C2 dominates. Normalising the main path (C3+C4) and ctx path (C1+C2) separately ensures balanced gradient contribution. |

**Result:** CNN v3 AUC ~0.625 → CNN v4 slightly improved, and critically the architecture is now correct.

---

### `cells_group_B.py` — k-mer XGBoost v2 (Replaced v1)

| Change | Reason |
|---|---|
| Switched from `GradientBoostingClassifier` to **XGBClassifier** | XGBoost is significantly faster, parallelised natively, and handles the high-dimensional (~2,000d) sparse feature space better than sklearn's GBM. |
| Added **5-mer difference features** (+1,024 dims) | k=5 captures longer structural RNA motifs. Only the difference vector (healthy − mutant) is used because each SNP changes at most 5 five-mers — the healthy and mutant vectors are ~99.5% identical at k=5, so encoding both separately adds 1,024 dims of near-identical noise. |
| Added **local context window features** (+5 dims) | GC content, AU/GU ratio, GU wobble ratio, context length, and Shannon entropy of the ±5nt window around the SNP position. Captures the immediate sequence neighbourhood of the mutation. |
| Added **Optuna hyperparameter tuning** (50 trials per fold) | v1 used fixed hyperparameters. Bayesian optimisation searches: `n_estimators` (200–800), `max_depth` (3–6), `learning_rate`, `subsample`, `colsample_bytree`, `min_child_weight`. Tuned on inner 3-fold CV of the training fold only — test fold never seen by Optuna. |

**Total feature vector:** ~2,000 dims (was ~971 in v1).

---

### `cells_group_C.py` — RNA-FM XGBoost (Entirely New 5th Model)

This model did not exist in `final.ipynb` at all. It is a completely new addition.

| What it adds | Why |
|---|---|
| **RNA-FM foundation model** (pre-trained on 23M non-coding RNA sequences) | Provides evolutionary context — which RNA sequences are biologically plausible, what structural patterns are conserved across evolution. No sequence feature or thermodynamic calculation can replicate this signal. |
| **Frozen 640-dim per-sequence embeddings** via mean-pooling | Frozen weights prevent catastrophic forgetting at 2,372 samples. Mean-pooling over all token positions gives a single 640d vector per sequence. |
| **Difference embedding** (healthy − mutant) as the feature | The difference embedding captures specifically how the mutation shifts the learned RNA representation, not general miRNA family properties (which are identical for healthy and mutant and thus useless for classification). |
| Combined with **k-mer v2 features** before XGBoost | k-mer captures local compositional context. RNA-FM captures global evolutionary context. Together they give the model both perspectives simultaneously. |
| **Optuna tuning** (50 trials, nested inner 3-fold) | Same tuning protocol as k-mer v2 — inner CV on training fold only. |

---

### `cells_group_D.py` — Ensemble v2 (Replaced v1 Stacking)

| Change | Reason |
|---|---|
| **5 models** instead of 4 (added RNA-FM XGB) | RNA-FM provides orthogonal evolutionary signal that CNN, GCN, k-mer, and MFE cannot replicate. Adding it gives the meta-learner genuinely new information. |
| **Rank normalisation** before stacking (`scipy.stats.rankdata`) | Each model outputs probabilities calibrated to its own decision boundary. A CNN probability of 0.65 and an MFE probability of 0.65 do not mean the same thing. Rank normalisation maps all 5 models to a common uniform [0,1] scale based on relative ordering, not absolute values. |
| **XGBoost meta-learner** instead of logistic regression | Logistic regression can only learn linear combinations. Non-linear interactions between models are important — e.g., "trust k-mer when RNA-FM agrees, downweight k-mer when MFE strongly disagrees." XGBoost tree structure captures these interactions; logistic regression cannot. |
| Full **OOF (Out-of-Fold) prediction discipline** | Base model predictions on their own training data are overfitted and useless for meta-learner training. OOF predictions are collected from each fold's held-out test set, so every sample has exactly one honest prediction from a model that never trained on it. |

**Result: AUC improved from 0.720 → 0.7338.**

---

### `cells_group_E.py` — Analytics and Scientific Metrics (New)

All of this is new. None existed in `final.ipynb`.

| Output file | What it contains |
|---|---|
| `shap_importance.pkl` | Global SHAP feature importance — top 50 features by mean absolute SHAP value across all 5 folds |
| `shap_values_per_sample.pkl` | Per-sample SHAP values for the test fold samples |
| `calibration_data.pkl` | Calibration curve data (predicted probability vs observed frequency) for reliability analysis |
| `pr_curve_data.pkl` | Precision-recall curve data for all models |
| `clinvar_validation.pkl` | Independent ClinVar benchmark results — performance on variants never seen during training |
| `snp_position_analysis.pkl` | Statistical distribution of SNP positions within hairpins, split by class |

---

### `generate_metrics.py` — Standalone Analytics Script (New)

A standalone Python script that regenerates all `.pkl` analytics files from the saved model checkpoints without re-running the full training pipeline. Used to refresh analytics data after model updates without GPU retraining.

---

### `deepfold-app/` — Complete Web Application (Entirely New)

The notebook produced model files. The web application is what makes them queryable and usable. None of this existed in `final.ipynb`.

```
deepfold-app/
├── backend/
│   ├── main.py              FastAPI server with all endpoints
│   ├── model_loader.py      ModelRegistry class — loads all 30 model checkpoints at startup
│   ├── predictor.py         Full ensemble inference + uncertainty quantification
│   ├── feature_extraction.py  SHAP computation, k-mer/MFE/RNA-FM pipelines
│   ├── contact_map.py       UFold wrapper, 4-channel tensor builder for inference
│   └── schemas.py           Pydantic request/response data models
│
└── frontend/
    ├── app/predict/         3-step prediction flow (select miRNA → review → run)
    ├── app/batch/           CSV batch upload → up to 500 variants
    ├── app/dashboard/       Calibration + PR + AUC performance charts
    ├── app/research/        ClinVar validation hub + SNP spatial hotspot chart
    ├── components/prediction/ResultCard  SHAP bar chart + UQ indicators
    ├── components/scene/    Three.js 3D RNA hairpin visualizer
    └── components/dashboard/ AUCChart, StatsCards, CalibrationChart
```

**Specific backend features added (all new):**

| Feature | Location | Detail |
|---|---|---|
| **Rate limiting** | `main.py` | `slowapi` — 15 req/min on `/predict`, 3 req/min on `/predict/batch` |
| **Input validation** | `main.py` | Sequence length 15–300nt, charset AUCTG only, exactly 1 mutation, equal lengths |
| **Batch endpoint** | `main.py` | `POST /predict/batch` — async CSV upload, max 500 rows, 4-concurrent throttle via `asyncio_throttle` |
| **Analytics endpoint** | `main.py` | `GET /analytics` — serves calibration + PR curve pkl data to frontend |
| **Validation endpoint** | `main.py` | `GET /validation` — serves ClinVar benchmark results to Research page |
| **SHAP explanation** | `feature_extraction.py` | `shap.TreeExplainer` on kmer fold-1 model, feature names mapped to biological labels |
| **Disagreement score** | `predictor.py` | `std([p_cnn, p_gcn, p_kmer, p_mfe, p_rnafm])` — real-time ensemble variance |
| **OOD detection** | `predictor.py` | Checks `mirna_id` against full training set index loaded at startup |
| **Prediction history** | `main.py` | In-memory log of last 50 predictions with timestamp, probability, label |

**Frontend features (all new):**

| Feature | Detail |
|---|---|
| **3D RNA hairpin** | Three.js helix visualizer that parses the sequence and renders a stem-loop structure with SNP site marked |
| **SHAP bar chart** | Recharts horizontal bar chart showing top biological drivers of each prediction |
| **Model Agreement** | Visual indicator (HIGH/MODERATE/LOW) based on ensemble disagreement score |
| **OOD warning** | Warning banner when the queried miRNA family was not in the training set |
| **Calibration curves** | Reliability diagram showing model confidence vs observed frequency |
| **ClinVar truth table** | Independent validation benchmark showing predicted vs ground-truth clinical labels |
| **SNP hotspot chart** | Area chart of pathogenic vs benign SNP position distribution across hairpin |
| **Batch CSV engine** | File upload → async processing → downloadable results table |

---

## Summary: Before and After

```
BEFORE (final.ipynb)              AFTER (all additions)
─────────────────────             ──────────────────────────────────
4 models                    →     5 models (+ RNA-FM XGB)
Logistic regression stack   →     XGBoost meta-learner
Raw probability stacking    →     Rank-normalised stacking
CNN v3 (no position, 1-pool)→     CNN v4 (CoordAttention, MultiScalePool, Mixup)
k-mer v1 (GBM, 971 dims)    →     k-mer v2 (XGB + Optuna, ~2000 dims)
No explainability           →     SHAP per prediction
No uncertainty              →     Disagreement score + OOD flag
No web interface            →     Full FastAPI + Next.js production app
No analytics                →     Calibration, PR, ClinVar, SNP position .pkl files
AUC 0.720                   →     AUC 0.7338
```

---

*All trained model checkpoints are saved in `DeepFold_models/`. The web application loads them at startup via `ModelRegistry`.*
