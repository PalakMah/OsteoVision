# Multimodal, Explainable Osteoporosis Screening from Knee X-Rays

A research pipeline that classifies knee X-rays into **normal / osteopenia / osteoporosis**, fuses the image with structured clinical risk factors, and — unusually for a "just train a CNN" notebook — ships a full explainability (XAI) layer that turns every prediction into both an *expert-facing* and a *patient-facing* explanation.

> ⚠️ **Not a diagnostic tool.** This is a research/educational pipeline trained on a small (n=239 images / 243 patient records) public dataset. It is explicitly designed as clinical **decision support**, not a replacement for DEXA scans or a radiologist's judgment. See [Limitations](#limitations--known-weaknesses).

---

## Table of Contents

- [Dataset](#dataset)
- [Pipeline Architecture](#pipeline-architecture)
- [Features](#features)
- [Results](#results)
- [Explainability Outputs](#explainability-outputs-example)
- [Limitations & Known Weaknesses](#limitations--known-weaknesses)
- [Repository / Output Artifacts](#repository--output-artifacts)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

---

## Dataset

- **Source:** [Knee X-ray Osteoporosis Database](https://www.kaggle.com/datasets/orvile/knee-x-ray-osteoporosis-database) (Kaggle), loaded from `/kaggle/input/.../Osteoporosis Knee X-ray`.
- **Images:** 239 knee X-rays across 3 folders (`normal`, `osteopenia`, `osteoporosis`).
- **Class distribution (heavily imbalanced, ~4.3x):**

| Class | Count | % |
|---|---|---|
| Osteopenia | 154 | 64.4% |
| Osteoporosis | 49 | 20.5% |
| Normal | 36 | 15.1% |

- **Clinical sheet:** `patient details.xlsx`, 243 rows × 28 columns of demographic and risk-factor data (age, gender, menopause age, height/weight, BMI, smoking/alcohol/diabetic status, pregnancy count, fracture history, family history of osteoporosis, T-score/Z-score, occupation, diet, medical history, etc.), with meaningful missingness in several columns (e.g. `Menopause Age` missing for 157 rows, `Number of Pregnancies` missing for 119 rows).
- **Image–record linkage:** filenames follow a `<BMDlevel>-<SubjectID>` convention (e.g. `N-12.jpg`, `OP-7.jpg`, `OS-3.jpg`). Images are matched to clinical rows by extracting the numeric subject ID and joining against the sheet — **100% match rate** was achieved in this run, but only **49.5%** agreement was found between the folder label and the sheet's own `Diagnosis` column (flagged in-notebook as a data-quality caveat, not silently resolved).

---

## Pipeline Architecture

The notebook is organized as a linear, stage-based pipeline. Each stage is a separate cell/section:

```
┌─────────────────────────────────────────────────────────────────────┐
│ 0. SETUP                                                             │
│    Installs deps, sets seeds, device, global CONFIG dict              │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 1. EXPLORATORY DATA ANALYSIS                                         │
│    - Folder/image census + class imbalance plot                      │
│    - Clinical sheet load, dtype/missingness audit                    │
│    - Auto-detection of id / T-score / diagnosis columns               │
│    - Per-class distributions of numeric risk factors                  │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 2. DATA FUSION & PREPROCESSING                                       │
│    - Regex-based image ↔ patient-ID matching                          │
│    - Folder-label vs. sheet-diagnosis agreement check                 │
│    - Numeric: median-impute → StandardScaler                          │
│    - Categorical: impute "missing" → OneHotEncoder                    │
│    - T-score explicitly EXCLUDED from features (would leak the label) │
│    - Stratified 5-fold split, saved to matched_dataset.csv            │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 3. DATASET / AUGMENTATION / SAMPLING                                 │
│    - MultimodalKneeDataset: returns (image, tabular_vector, label)    │
│    - Train-time augmentation: resize+crop, h-flip, rotation, jitter   │
│    - WeightedRandomSampler to counter class imbalance                 │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 4. MODELS & LOSS                                                     │
│                                                                       │
│   ImageOnlyModel (baseline)         MultimodalModel (main)           │
│   ┌───────────────┐                 ┌───────────────┐ ┌────────────┐│
│   │ EfficientNet-B0│                │ EfficientNet-B0│ │  Tabular   ││
│   │  (ImageNet wts)│                │  (ImageNet wts)│ │  MLP       ││
│   │  → 128-d embed │                │  → 128-d embed │ │ (in→64→128)││
│   └───────┬───────┘                 └───────┬───────┘ └─────┬──────┘│
│           │                                  └────concat────┘        │
│           ▼                                          ▼               │
│      Linear → 3 classes                    FC(256→128)→ReLU→Dropout  │
│                                             →Linear→ 3 classes        │
│                                                                       │
│   Loss: multi-class Focal Loss (γ=2.0), with per-class α computed    │
│         from each training fold's inverse class frequency            │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 5. TRAINING & CROSS-VALIDATION                                       │
│    - 5-fold stratified CV, both models trained fold-by-fold           │
│    - AdamW + cosine LR annealing, early stopping (patience=7)         │
│    - Per-fold checkpoints saved; out-of-fold predictions aggregated   │
│    - 5-fold logit-averaging ensemble built from the multimodal model  │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 6. EVALUATION                                                        │
│    - Out-of-fold classification report + confusion matrix             │
│    - Baseline (image-only) vs. multimodal macro-F1 comparison         │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 7. EXPLAINABILITY — BRANCH A (per-instance, local)                    │
│    - Grad-CAM on the last EfficientNet conv block (image branch only, │
│      tabular features held fixed) → visual heatmap                    │
│    - Occlusion sensitivity: sliding grey-patch perturbation, no       │
│      gradients, ground-truth check on Grad-CAM                        │
│    - LIME-style tabular surrogate: perturb clinical features by       │
│      resampling from the training marginal, refit a local linear      │
│      model on predicted-class probability → signed feature            │
│      contributions                                                    │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 8. EXPLAINABILITY — BRANCH B (global surrogate)                      │
│    - Extract fused 256-d embeddings for the whole dataset             │
│    - Fit a depth-4 Decision Tree to MIMIC the deep model's own        │
│      predictions (fidelity, not accuracy, is the target)              │
│    - Report fidelity-to-model and accuracy-vs-ground-truth separately │
│    - Permutation importance over raw tabular columns (macro-F1 drop   │
│      when each column is shuffled) as a model-agnostic global ranking │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 9. EXPLANATION BUNDLING & AUDIENCE-AWARE RENDERING                    │
│    - Combines Grad-CAM region + occlusion region + top LIME features  │
│      + surrogate-tree agreement into one "explanation bundle" per     │
│      patient                                                          │
│    - expert_view(): numeric, signed contributions, surrogate fidelity,│
│      caveats                                                          │
│    - layperson_view(): one plain-language sentence, qualitative       │
│      High/Medium/Low confidence, no raw numbers                       │
└───────────────────────────────┬───────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│ 10. LLM-NARRATED EXPLANATIONS                                        │
│    - Structured bundle → prompt → Claude (claude-sonnet-4-6) rewrites │
│      it into fluent prose, separately for:                            │
│        • Expert prompt: preserves numbers, states surrogate           │
│          disagreement if present, names model limitations             │
│        • Layperson prompt: 2-3 warm, jargon-free sentences, always    │
│          ends by pointing the patient to a clinician                  │
│    - Deterministic offline template fallback when no API key /        │
│      internet is available (keeps the notebook runnable end-to-end    │
│      on Kaggle's no-internet sessions)                                │
│    - All outputs logged to explanation_review_log.csv with empty      │
│      rating columns for a downstream human inter-rater review step    │
└─────────────────────────────────────────────────────────────────────┘
```

### Key design decisions worth noting

- **T-score is deliberately excluded** from the tabular feature set even though it's in the clinical sheet — including it would leak the diagnostic label (T-score is literally how osteoporosis/osteopenia are defined clinically).
- **Focal Loss with per-fold, per-class α** (rather than plain cross-entropy) is used specifically to fight the 4.3x class imbalance, on top of a `WeightedRandomSampler` at the batch level — i.e. imbalance is attacked at both the sampling and loss level.
- **Two independent XAI branches** are built and then cross-checked against each other: a *local* branch (Grad-CAM + occlusion + LIME-style tabular surrogate, one explanation per patient) and a *global* branch (a shallow decision tree trained to imitate the deep model's own predictions, plus permutation importance). The surrogate tree's agreement/disagreement with each individual prediction is reported as a trust signal.
- **Dual-audience communication layer**: every explanation exists in both an *expert* form (full numeric detail) and a *layperson* form (plain language, qualitative confidence, explicit "talk to your doctor" framing) — then both are optionally rewritten by an LLM into natural prose, with a fully offline deterministic fallback so the pipeline never breaks without API access.

---

## Features

- ✅ End-to-end EDA: class-balance visualization, clinical-sheet audit, automatic column-role detection (id/T-score/diagnosis) with an explicit "verify me" print statement instead of silently trusting heuristics.
- ✅ Robust image ↔ patient-record linkage via regex ID extraction, with an explicit data-quality check (folder label vs. sheet diagnosis agreement) surfaced rather than hidden.
- ✅ Leakage-aware feature engineering (T-score excluded on purpose).
- ✅ Stratified 5-fold cross-validation (not a single train/test split).
- ✅ Class-imbalance handling at two levels: `WeightedRandomSampler` + multi-class Focal Loss with data-driven per-class α.
- ✅ Image augmentation pipeline (resize+random-crop, flips, rotation, color jitter) with ImageNet normalization.
- ✅ Two comparable model architectures trained under identical CV splits for a fair ablation: image-only EfficientNet-B0 vs. image+tabular fusion model.
- ✅ Early stopping + cosine LR annealing + AdamW, per-fold checkpointing.
- ✅ 5-fold checkpoint ensembling utility for final inference.
- ✅ Multi-method explainability: Grad-CAM (gradient-based), occlusion sensitivity (perturbation-based, gradient-free — used as a sanity check against Grad-CAM), and a LIME-style local linear surrogate over tabular features.
- ✅ Global surrogate-model fidelity analysis (decision tree trained to mimic the deep net, with both fidelity-to-model and accuracy-vs-truth reported separately) plus permutation feature importance.
- ✅ Heatmap-to-anatomical-region translation (3×3 grid: upper/mid/lower × left/central/right, with the center cell labeled "joint space") for human-readable localization.
- ✅ Audience-aware explanation rendering (expert vs. layperson) at both the structured-bundle level and the LLM-narrated prose level.
- ✅ Optional LLM narration via the Anthropic API with a deterministic offline fallback, so the notebook remains fully reproducible without network/API access.
- ✅ Review-log export (`explanation_review_log.csv`) with blank rating columns, designed to plug into a downstream human inter-rater validation step.

---

## Results

### Image-only baseline (EfficientNet-B0, no clinical features) — out-of-fold

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.82 | 0.99 | 0.90 | 108 |
| Osteopenia | 0.94 | 0.75 | 0.84 | 240 |
| Osteoporosis | 0.77 | 0.93 | 0.84 | 135 |
| **Accuracy** | | | **0.85** | 483 |
| **Macro avg** | 0.85 | 0.89 | **0.86** | 483 |

### Multimodal model (image + clinical risk factors + focal loss + augmentation) — out-of-fold

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| Normal | 0.76 | 0.96 | 0.85 | 108 |
| Osteopenia | 0.93 | 0.71 | 0.81 | 240 |
| Osteoporosis | 0.76 | 0.93 | 0.83 | 135 |
| **Accuracy** | | | **0.83** | 483 |
| **Macro avg** | 0.82 | 0.87 | **0.83** | 483 |

> **Note:** support totals (483) exceed the 239 raw images because the notebook merges each image with clinical-sheet rows before splitting, and the join produces some duplicated image rows. Treat the absolute counts as pipeline-internal bookkeeping rather than "483 unique X-rays."

**Headline result:** in this run, the multimodal fusion model did **not** outperform the image-only baseline — macro-F1 moved from **0.859 → 0.830 (−0.028)**. The notebook reports this honestly rather than cherry-picking, and lists concrete next levers in order of effort:
1. Increase `FOCAL_GAMMA` or hand-tune per-class α further toward the `normal` class.
2. Collect/synthesize more `normal` images (only 36 exist).
3. Use the 5-fold ensemble (already implemented) as the deployed model rather than a single fold.

A 5-fold **ensemble utility** (`ensemble_predict`) is provided to average softmax outputs across all fold checkpoints of the multimodal model for a more robust final artifact than any single fold.

### Global surrogate model (explainability diagnostic, not a classifier)

- **Fidelity to the deep model:** 96.3% (the depth-4 decision tree agrees with the deep model's own predictions 96.3% of the time — this is the metric that matters for a surrogate)
- **Accuracy vs. ground truth:** 90.3%

### Top clinical risk factors by permutation importance (macro-F1 drop when shuffled)

| Feature | Importance |
|---|---|
| `S.No` | 0.090 |
| `Z-Score Value` | 0.033 |
| `Occupation_h.wife` | 0.025 |
| `Obesity_over weight` | 0.021 |
| `height (meter)` | 0.012 |
| `Menopause Age_45` | 0.012 |
| `Occupation_G.E` | 0.012 |
| `Medical History_GERD` | 0.012 |

> ⚠️ **Caveat spotted in the results:** `S.No` (a row/serial number in the source spreadsheet, not a clinical variable) is the single most "important" feature by this metric. That's a strong signal of **spurious correlation / potential leakage through row ordering** rather than a genuine clinical risk factor, and should be investigated (e.g. dropped from the feature set, or checked for correlation with fold/scan-batch) before trusting the tabular branch's importances clinically.

---

## Explainability Outputs (example)

For each sampled validation patient, the notebook produces a bundle like:

**Expert view:**
> Patient OP130: model predicts **osteopenia** (confidence 0.62). Grad-CAM / occlusion sensitivity both localise the most influential image region to: lower-central. Top clinical-feature contributions (LIME-style, signed): `S.No: +0.185`, `Daily Eating habits_low protiens, no sour food: -0.127`, ... Global surrogate decision tree (fidelity to deep model 96%) agrees with this prediction.

**Layperson view:**
> The scan looks most consistent with **osteopenia** (confidence: Medium). The lower-central part of the knee X-ray was the biggest factor, along with the patient's S.No. As always, this is a decision-support suggestion, not a diagnosis — please confirm with a clinician.

Each bundle is then optionally rewritten by an LLM (Claude Sonnet 4.6) into more natural prose for the same two audiences, governed by strict system prompts (preserve every number exactly for the expert version; strip all numbers except a High/Medium/Low confidence word for the layperson version; both must never state a diagnosis).

---

## Limitations & Known Weaknesses

The notebook is explicit about several of these itself; others are worth flagging for anyone reusing it:

- **Small, imbalanced dataset**: 239 images total, only 36 `normal` examples — high variance in per-fold metrics is expected, and the `normal` class F1 is the model's weakest point.
- **Multimodal fusion underperformed the image-only baseline** in this run (macro-F1 −0.028); tabular fusion is not automatically a win here and needs further tuning (loss weighting, feature selection, alternate fusion strategies).
- **Folder label vs. clinical-sheet diagnosis agreement is only ~49.5%.** Roughly half of the images' folder-assigned class disagrees with the sheet's own `Diagnosis` column — this is a meaningful label-quality issue, and results should be interpreted with that noise in mind.
- **`S.No` (a spreadsheet row index) ranks as the top tabular feature** by permutation importance — a red flag for leakage/spurious ordering effects that should be investigated before any clinical interpretation of feature importances.
- **Feature scaling/encoding is fit on the full dataset before CV splitting** for simplicity (explicitly flagged in-notebook); a rigorous evaluation should refit the `StandardScaler`/`OneHotEncoder` inside each fold to avoid statistical leakage from combining train/val distributions.
- **Support counts in evaluation (483) exceed the number of raw images (239)** due to the merge step — indicates duplicated rows from the image↔clinical join that should be de-duplicated for a cleaner evaluation.
- **XAI methods are demonstrated on only 3 sample patients**, not the full validation set — sufficient to prove the pipeline works end-to-end, not to draw dataset-wide conclusions about explanation quality.
- **Not clinically validated.** All patient-facing text explicitly states this is decision support, not a diagnosis, and should always be confirmed against DEXA / a clinician.

---

## Repository / Output Artifacts

Running the notebook (on Kaggle, with `OUT_DIR = /kaggle/working`) produces:

| File | Description |
|---|---|
| `matched_dataset.csv` | Final image+tabular+label+fold table used for training |
| `image_only_fold{0-4}.pt` | Baseline model checkpoints, one per CV fold |
| `multimodal_fold{0-4}.pt` | Fusion model checkpoints, one per CV fold |
| `explanation_review_log.csv` | Per-patient expert/layperson narrated explanations with blank rating columns for human review |

## Requirements

Installed at runtime via `pip`: `openpyxl`, `grad-cam`, `lime`, `shap` (note: `lime`/`shap` are installed but the tabular local-explanation method actually implemented is a custom LIME-style linear surrogate rather than a call into the `lime` package itself).

Core libraries: `torch`, `torchvision`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `pytorch-grad-cam`, `Pillow`. Optional: `anthropic` (only needed if using the live LLM-narration path — a deterministic offline template is used otherwise).

GPU (`CUDA`) is auto-detected and used if available; the pipeline also runs on CPU (slower).

## How to Run

1. Provide the dataset at the path in `CONFIG["ROOT"]` (defaults to the Kaggle mount path — update for local use).
2. Run all cells top to bottom (Setup → EDA → Fusion → Training → Evaluation → XAI Branch A → XAI Branch B → Bundling → LLM Narration).
3. (Optional) Set the `ANTHROPIC_API_KEY` environment variable to enable live LLM-narrated explanations; otherwise the notebook automatically falls back to deterministic offline templates.
4. Trained checkpoints, the matched dataset CSV, and the explanation review log will be written to `CONFIG["OUT_DIR"]`.
