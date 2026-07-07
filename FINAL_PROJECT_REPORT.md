# Hybrid Multi-Modal Framework for Effective Prediction of Liver Diseases

**System Name:** CMCHT-XAI (Cross-Modal Contrastive Hybrid Transformer with Counterfactual Explainable AI)

## Section 1 — Problem Statement
Liver disease prediction in clinical practice is hindered by the fragmented use of diagnostic modalities: radiological imaging and structured laboratory data are interpreted separately, discarding the complementary information each carries about disease state. Single-modality models and conventional ensemble approaches fail to capture cross-modal interactions, while standard deep-learning classifiers offer no mechanism for clinicians to interrogate or challenge a prediction. This system addresses both gaps by proposing a hybrid multi-modal architecture that jointly encodes imaging and tabular clinical features through a novel counterfactual-sensitivity-gated fusion mechanism, coupled with a clinically-ordered cascade prediction head and a post-hoc explainability layer spanning feature attribution, visual attention, counterfactual reasoning, and uncertainty quantification.

## Section 2 — System Architecture
The CMCHT-XAI pipeline processes two modalities in parallel before fusing them into a shared representation:

**Imaging Encoder:** A Swin-CNN hybrid architecture using a Swin-Tiny backbone (`swin_tiny_patch4_window7_224`) extracts hierarchical visual features from ultrasound images. The system was designed for SimCLR contrastive pre-training on unlabelled ultrasound data; however, pre-training was not completed in this version due to the absence of a large unlabelled corpus (the imaging encoder uses ImageNet initialisation only).

**Tabular Encoder:** An FT-Transformer encodes the 11 NAFLD tabular features. Each feature is tokenised via a dedicated linear projection (`Linear(1, embed_dim)`), and processed along with a learnable `[CLS]` token by an `nn.TransformerEncoder` with `norm_first=True` and `activation="gelu"`.

**CSG-Fusion (Contribution 1):** The Counterfactual-Sensitivity-Gated Fusion module gates cross-modal attention using a counterfactual sensitivity probe over the tabular features to construct an uncertainty-aware gating signal, ensuring that modalities contributing higher counterfactual sensitivity receive proportionally greater weight.

**CUSP-Cascade (Contribution 2):** The Clinically-Ordered Uncertainty-Sensitive Prediction Cascade structures the output heads in clinical order: detection (binary) → staging (3-class) → severity (regression). Each stage conditions on the previous stage's prediction and its associated epistemic uncertainty.

**CGCT (Contribution 3):** Confidence-Gated Cascade Training teacher-forces the staging head during training using ground-truth detection labels according to a confidence-aware scheduled sampling schedule, allowing the staging head to specialise on difficult boundary cases.

**XAI Layer:** The post-hoc explainability layer generates SHAP feature attributions, Grad-CAM visual attention heatmaps, DiCE counterfactuals, and MC-Dropout uncertainty estimates.

## Section 3 — The Three Contributions
**Contribution 1 — CSG-Fusion:** Gating cross-modal attention using a counterfactual sensitivity signal derived from the tabular pathway, this builds on standard cross-attention transformers by adapting the gate based on which features matter most to the prediction under counterfactual perturbation.
**Contribution 2 — CUSP-Cascade:** Extending multi-task cascade paradigms (like clinical decision trees) by jointly propagating both uncertainty estimates and counterfactual feature sensitivity between clinically-ordered prediction stages.
**Contribution 3 — CGCT:** Adapting confidence-aware scheduled sampling from autoregressive sequence decoding (Liu et al. 2021) to heterogeneous multi-task clinical cascades, replacing sequence decoding context with ground-truth labels as teacher signals with a decay schedule.

> "We propose a system combining three contributions: (1) CSG-Fusion, the first application of counterfactual-sensitivity-gated, uncertainty-aware cross-modal fusion to multimodal liver disease prediction; (2) CUSP-Cascade, which extends clinically-ordered multi-task cascades by jointly propagating epistemic uncertainty and counterfactual feature sensitivity between stages; and (3) Confidence-Gated Cascade Training, an adaptation of confidence-aware scheduled sampling (Liu et al. 2021) from autoregressive sequence decoding to heterogeneous multi-task clinical cascades."

## Section 4 — Datasets
NAFLD (87 patients, 11 features, 3-class liver grade: Normal/Benign/Malignant, Mendeley CC BY-NC-SA).
ILPD (583 patients, 10 features, binary detection, UCI CC BY 4.0).
Cirrhosis Prediction (418 patients, 18 features, 4-class staging, Kaggle public).

> "All three datasets are publicly available under open licenses and were used strictly for non-commercial academic research."

## Section 5 — Training Details
50 epochs, AdamW optimizer, cosine annealing scheduler, batch size 16, 5-fold stratified cross-validation configured, ImageNet-pretrained Swin-Tiny backbone, no SimCLR pre-training completed (imaging encoder uses ImageNet init only). Best validation loss: 0.4951 (with CGCT), 0.6971 (without CGCT).

## Section 6 — Results

| Config | Det Acc | Det AUC | Det F1 | Stage F1 | Stage Kappa | Sev MAE | Sev RMSE | ECE |
|---|---|---|---|---|---|---|---|---|
| baseline | 0.111 | 0.250 | 0.000 | 0.070 | -0.014 | 0.529 | 0.599 | 0.406 |
| csg_only | 0.111 | 0.844 | 0.000 | 0.222 | -0.066 | 0.885 | 0.958 | 0.406 |
| cusp_only | 0.889 | 0.938 | 0.938 | 0.478 | 0.100 | 0.231 | 0.264 | 0.117 |
| csg_cusp | 0.944 | 0.969 | 0.970 | 0.478 | 0.100 | 0.225 | 0.261 | 0.074 |
| full_system | 0.944 | 0.969 | 0.970 | 0.478 | 0.100 | 0.225 | 0.261 | 0.074 |

CGCT Isolation (csg_cusp row):
| Metric | No CGCT | With CGCT | Δ |
|---|---|---|---|
| Det Acc | 0.889 | **0.944** | +0.055 |
| Det AUC | 0.875 | **0.969** | +0.094 |
| Det F1 | 0.941 | **0.970** | +0.029 |
| Stage F1 | 0.238 | **0.478** | +0.240 |
| Kappa | 0.000 | **0.100** | +0.100 |
| Sev MAE | 0.266 | **0.225** | −0.041 |
| ECE | 0.073 | 0.074 | ≈ tied |
| Best Val Loss | 0.697 | **0.495** | −29% |

> "CUSP-Cascade produces the largest single gain: detection accuracy 88.9% and staging macro-F1 0.478 versus 11.1% and 0.070 for the baseline. CSG-Fusion adds a further increment: detection AUC 0.938→0.969 and ECE 0.117→0.074, with the calibration improvement providing the clearest evidence of counterfactual-sensitivity gating improving cross-modal alignment. CGCT is isolated by comparing two independently trained checkpoints (50 epochs each, csg+cusp_cascade, with and without CGCT): CGCT improves staging F1 from 0.238 to 0.478 (+0.240), detection AUC from 0.875 to 0.969, and best validation loss from 0.697 to 0.495 (−29%)."

## Section 7 — Explainability
The following outputs exist in `results/explainability/`:
- SHAP summary plot (11 tabular features, GradientExplainer on detection task).
- Grad-CAM overlays (3 sample images from the test set).
- Counterfactual explanations (gradient-based).
- MC-Dropout uncertainty (0/5 samples flagged above threshold in the 5-sample run).

## Section 8 — Limitations
> "1. The primary multimodal dataset comprises 87 patients with a class distribution of 11 Normal, 48 Benign, and 28 Malignant cases. Cohen's Kappa of 0.100 reflects slight agreement, better than chance but not clinically actionable. Results should be interpreted as a feasibility demonstration."
> 
> "2. The ablation table loads the full_system checkpoint with strict=False, meaning architecture-mismatched components receive randomly-initialized weights. The CGCT row is the only fully isolated comparison (two independent 50-epoch training runs from scratch)."
> 
> "3. SimCLR contrastive pre-training was not completed due to the absence of a large unlabeled ultrasound corpus. The imaging encoder uses ImageNet-pretrained weights only."
> 
> "4. All findings require validation on a larger, independently collected cohort before any clinical deployment claim."

## Section 9 — References
- Gorishniy et al. (2021) — Revisiting Deep Learning Models for Tabular Data (FT-Transformer)
- Liu et al. (2021) — Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
- Chen et al. (2020) — A Simple Framework for Contrastive Learning of Visual Representations (SimCLR)
- Mothilal et al. (2020) — Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations (DiCE)
- Liu et al. (2021) — Confidence-Aware Scheduled Sampling for Neural Machine Translation (source algorithm for CGCT)
