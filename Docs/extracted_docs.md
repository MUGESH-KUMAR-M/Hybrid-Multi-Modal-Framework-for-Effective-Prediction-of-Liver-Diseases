# CMCHT_XAI_Dataset_Reference.docx

1.  Datasets to Use
These datasets are confirmed accessible, correctly labelled, and aligned to your project tasks. Each entry covers exact contents, data type, known issues, and the specific action you must take before using it.
1.1  ILPD — Indian Liver Patient Dataset
Exact features (10 input + 1 label)
Age — patient age in years
Gender — Male / Female (441 M, 142 F — male-heavy skew)
Total Bilirubin — liver filtration marker, heavily right-skewed
Direct Bilirubin — sub-fraction of total bilirubin
Alkaline Phosphatase (ALP) — enzyme, elevated in liver/bone disease, right-skewed
Alanine Transaminase (SGPT/ALT) — liver cell damage marker
Aspartate Transaminase (SGOT/AST) — liver and heart damage marker
Total Protein — normally distributed
Albumin — protein made by liver, 4 missing values in this column
Albumin/Globulin Ratio (A/G Ratio) — 4 missing values
Class Label — 1 = liver disease, 2 = healthy (binary)
Use in your project
FT-Transformer tabular encoder training and ablation
Binary disease detection task head — your Phase I baseline
Pre-training tabular features before multimodal fusion
1.2  HCC Survival Dataset
What it contains
Demographics: age, gender, ethnicity
Risk factors: hepatitis B/C status, alcohol history, cirrhosis, portal hypertension
Laboratory values: AFP (alpha-fetoprotein), bilirubin, creatinine, AST, ALT, albumin, INR
Tumour features: tumour size, number of nodules, vascular invasion, satellite nodules, BCLC stage
Treatment history: prior surgery, ablation, TACE, systemic therapy
Target: 1-year survival — binary (0 = died, 1 = survived)
Use in your project
HCC-specific severity scoring — your multi-task severity head
Ablation testing of your FT-Transformer tabular encoder
Standalone pre-training before multimodal fusion experiments
1.3  Cirrhosis Prediction Dataset (Kaggle — fedesoriano)
Exact features
N_Days — days between registration and death/transplant/study end
Drug — D-penicillamine vs. placebo
Age — in days
Sex — male / female
Ascites / Hepatomegaly / Spiders / Edema — clinical examination findings (binary/categorical)
Bilirubin — serum bilirubin in mg/dl
Cholesterol — serum cholesterol in mg/dl
Albumin — serum albumin in gm/dl
Copper — urine copper in ug/day
Alk_Phos — alkaline phosphatase in U/liter
SGOT — aspartate aminotransferase in U/ml
Triglycerides — in mg/dl
Platelets — per cubic ml / 1000
Prothrombin — prothrombin time in seconds
Stage — histologic disease stage 1, 2, 3, or 4 (your staging label)
Status — 0 = death, 1 = censored, 2 = censored due to liver transplant
Use in your project
CUSP-Cascade fibrosis staging head — Stage column (1–4) maps directly to fibrosis stage
Replaces BUPA entirely as your second tabular benchmark
Multi-task label source: Stage for staging head, binarised Status for severity head
1.4  NAFLD Ultrasound + Clinical Dataset (Mendeley)
What it contains
Imaging: B-mode ultrasound, standardised to 768 x 1024 pixels, DICOM converted to PNG
Clinical tabular: age, gender, BMI, ALT, AST, and additional liver enzyme markers per patient
Label detail: NAFLD Activity Score (NAS), fibrosis staging, steatosis grading — all linked to biopsy results
The only dataset in your collection with true paired image + tabular data per patient
Use in your project
Primary dataset for CSG-Fusion cross-modal attention experiments
SimCLR contrastive pre-training on unlabelled ultrasound images
Multi-task head evaluation: detection + staging + steatosis grading simultaneously
1.5  LiTS — Liver Tumor Segmentation Benchmark
What it contains
201 contrast-enhanced 3D abdominal CT scans from 7 clinical institutions worldwide
Each volume: 512 x 512 pixels per slice, 74 to 987 slices per volume depending on patient
Training set: 131 volumes with ground truth segmentation masks (liver + tumour separately)
Test set: 70 volumes — ground truth withheld, submit predictions to CodaLab for blind evaluation
194 of 201 scans contain liver lesions — high tumour prevalence
Use in your project
Swin-CNN imaging encoder pre-training on CT modality
Liver segmentation validation for your imaging encoder before multimodal fusion
CT imaging branch of SimCLR contrastive pre-training
1.6  Duke Liver Dataset (DLDS)
What it contains
2,146 unique MRI series across 105 subjects, including patients with cirrhosis and chronic liver disease
310 series include expert-annotated liver segmentation masks (binary DICOM masks)
17 contrast type labels: opposed phase, in-phase, precontrast T1, portal venous phase, and more
Majority of patients have cirrhotic morphological changes — directly relevant to your liver disease focus
Use in your project
MRI branch of your Swin-CNN imaging encoder — use the 17 contrast type labels for series identification
SimCLR MRI pre-training — 2,146 series provides sufficient unlabelled imaging volume
Series classification pre-task: train encoder to identify MRI contrast type, then fine-tune on disease prediction
1.7  MSD-Liver — Medical Segmentation Decathlon Task03
Use in your project
Fallback for LiTS if CodaLab registration is delayed — same modality and task
Conveniently packaged with simpler download than LiTS
2.  Dataset to Remove — BUPA Liver Disorders
3.  Additional Datasets — Verified and Recommended
The following datasets were not in your original document. Each fills a specific gap in your current collection. Priority order is given at the end of this section.
3.1  HCC-TACE-Seg — The Cancer Imaging Archive (TCIA)
What it contains
Pre- and post-TACE treatment CT imaging for 105 HCC patients
5-class segmentation labels: background, liver parenchyma, viable tumour, necrotic tumour, portal vein, abdominal aorta
Clinical data file: demographics, medical history, diagnosis, treatment response variables
Outcome data: overall survival (OS) and time-to-progression — directly supports your severity task head
Unique feature: pre-treatment and post-treatment scans for the same patient — enables treatment response modelling
Why this fills a gap
Your current collection has NO dataset that pairs CT imaging with treatment outcome labels
HCC-TACE-Seg is the only publicly available liver dataset providing both imaging segmentation AND clinical outcomes in one package
The treatment response variable (TACE outcome) gives you a severity signal not available in any of your current datasets
Use in your project
Multi-task severity head: use overall survival and TACE response as severity labels
CT encoder pre-training: 98 patients with annotated scans for fine-tuning beyond LiTS
Ablation: compare model performance with and without treatment outcome supervision
3.2  CirrMRI600+ — Cirrhotic Liver MRI Dataset
What it contains
628 high-resolution abdominal MRI scans totalling approximately 40,000 annotated slices
Expert-validated segmentation labels annotated by radiologists for each scan
3-stage cirrhosis classification per patient in a CSV file (mild, moderate, severe)
Both enhanced and non-enhanced MRI included — multivendor and multiphase
Demographic and clinical parameters included alongside imaging
Why this fills a gap
Your current MRI source (Duke DLDS) provides series labels and segmentation but NO cirrhosis staging labels
CirrMRI600+ is the first MRI dataset designed specifically for cirrhosis — your primary target disease
The 3-stage grading (mild/moderate/severe) maps directly to your CUSP-Cascade fibrosis staging head
Use in your project
Fibrosis staging head: mild/moderate/severe labels are your staging supervision signal in MRI
MRI encoder branch of your Swin-CNN: T1W and T2W diversity strengthens the encoder
SimCLR pre-training: 40,000 annotated slices is substantial unlabelled imaging volume
3.3  CHAOS — Combined Healthy Abdominal Organ Segmentation
What it contains
40 CT scans of healthy abdomens with liver segmentation — no liver disease present
120 MRI scans of healthy livers including in-phase, opposed-phase, and T2-weighted sequences
Expert-annotated segmentation masks for liver, kidneys, and spleen
All subjects are healthy — no pathology
Why this fills a gap
Every imaging dataset in your current collection contains diseased livers only
SimCLR contrastive pre-training requires both positive pairs (similar livers) and negative pairs (healthy vs diseased)
Without healthy liver images, your imaging encoder cannot learn to distinguish normal from abnormal morphology
Use in your project
SimCLR healthy negative samples: pair healthy CHAOS scans against diseased LiTS/Duke scans as contrastive negatives
Imaging encoder validation: test whether your encoder correctly separates healthy from cirrhotic liver representations
3.4  LiQA — Liver Fibrosis Quantification and Analysis (MICCAI 2024)
What it contains
440 patients with hepatobiliary phase MRI from multiple clinical centres and MRI vendors
Fibrosis staging labels for each patient — your staging head's generalisation test
Designed under real-world conditions: domain shift between hospitals, missing modalities, spatial misalignment
The missing modality condition directly tests your CSG-Fusion gating mechanism
Use in your project
Phase II generalisation test: train on NAFLD/CirrMRI600+, evaluate on LiQA to measure cross-centre robustness
Missing modality ablation: test your CSG-Fusion with intentionally missing tabular or imaging inputs
Multi-centre evaluation section of your paper — reviewers expect this for clinical AI papers
3.5  TCGA-LIHC — The Cancer Genome Atlas Liver Hepatocellular Carcinoma
What it contains
RNA sequencing expression profiles — gene expression across thousands of genes
DNA methylation data — epigenetic markers
Copy number variation — structural genomic alterations
Clinical data: survival time, vital status, tumour stage, grade, treatment history
Use in your project
Optional extension: add molecular features to your FT-Transformer tabular encoder in Phase II
Survival prediction task: train an additional task head on survival prediction using TCGA-LIHC clinical + genomic features
Adds a third dimension to your multi-task framework beyond blood tests and imaging
4.  Priority Order and Action Plan
What to do this week, what to do this semester, and what to defer.
Document prepared for CMCHT-XAI Final Year Project — Sri Krishna College of Technology, Coimbatore.

# Liver_Disease_Hybrid_Framework.docx

A Hybrid Model Framework for Effective Prediction of Liver Diseases
Deep Learning - Explainable AI
Final Year Project Framework
1. Problem Statement
Title:
A Cross-Modal Contrastive Hybrid Transformer Framework with Counterfactual Explainable AI for Multi-Task Liver Disease Prediction
Problem:
Liver diseases (cirrhosis, hepatitis, hepatocellular carcinoma) progress silently and are often diagnosed at advanced stages. While AI models have achieved high accuracy in liver disease detection, three critical barriers prevent clinical adoption:
Black-box predictions: Clinicians cannot trust models that do not explain why a decision was made or what clinical factors drove the prediction.
Modality isolation: Existing models either process medical images (CT/MRI/Ultrasound) or clinical blood tests separately. They fail to exploit the complementary information between imaging morphology and biochemical markers (e.g., bilirubin, albumin, SGOT) in a dynamically weighted fusion.
Single-task limitation: Most models perform only binary classification (disease vs. healthy). They do not simultaneously predict disease presence, fibrosis stage, and progression risk—limiting clinical utility.
Objective:
Develop an executable hybrid deep learning framework that:
Uses contrastive self-supervised pre-training on unlabeled liver imaging to learn robust visual representations.
Fuses imaging (CNN-Transformer) and tabular clinical data (Tabular Transformer) via cross-modal attention.
Performs multi-task learning (detection + staging + severity scoring).
Provides counterfactual explanations ("If the patient's bilirubin were X instead of Y, the model would predict...") alongside SHAP and Grad-CAM.
Quantifies prediction uncertainty to flag low-confidence cases for human review.
2. Legal & Open-Source Datasets
You can legally use these datasets for academic research without copyright infringement:
Recommendation for your project:
Use ILPD or BUPA for the tabular pipeline (blood-based prediction) and LiTS or Duke Liver Dataset for the imaging pipeline. For a truly novel multimodal project, create a synthetic aligned dataset by pairing publicly available imaging with tabular features from the same disease categories (this is a common academic practice when paired multimodal data is scarce).
3. Literature Survey & Technology Gap Analysis
3.1 Existing Technologies (State-of-the-Art)
3.2 Identified Research Gaps (Your Novelty)
Based on the current literature, the following have NOT been implemented together in any liver disease prediction system:
Contrastive Self-Supervised Pre-training + Cross-Modal Fusion: No existing work first pre-trains an imaging encoder on unlabeled liver images using contrastive learning (SimCLR/MoCo), then fuses it with a tabular transformer via cross-modal attention (where imaging features query clinical features and vice versa).
Counterfactual Explainable AI in Liver Prediction: While SHAP and Grad-CAM are common, no liver disease framework provides counterfactual explanations (e.g., "If this patient's albumin increased by 0.5 g/dL, the model would downgrade the severity from Stage 3 to Stage 2"). This is critical for clinical decision support.
Multi-Task Learning with Uncertainty Quantification: Existing models predict one task (classification OR staging). None jointly predict (a) disease presence, (b) fibrosis stage, (c) MELD/Child-Pugh severity score in a single architecture with Monte Carlo Dropout uncertainty to flag ambiguous cases.
Tabular Transformer + Vision Hybrid: Most works use basic MLPs or Random Forest for tabular data. Using a FT-Transformer (Feature Tokenizer Transformer) for clinical data and a Swin-CNN hybrid for imaging, then fusing them via transformer cross-attention, is unexplored in hepatology AI.
4. Proposed Novel Methodology: CMCHT-XAI
(Cross-Modal Contrastive Hybrid Transformer with Counterfactual Explainable AI)
4.1 Architecture Overview
The proposed architecture consists of the following layers:
Input Layer: Liver CT/MRI Scan (3D Volume or 2D Slice) and Clinical Tabular Data (Bilirubin, Albumin, SGOT, Age, Gender, Platelets, etc.)
Imaging Encoder: Swin-Transformer + CNN Hybrid Backbone, pre-trained via SimCLR on unlabeled liver images
Tabular Encoder: FT-Transformer (Feature Tokenizer Transformer) with self-attention over clinical features
Cross-Modal Attention Fusion: Imaging queries Tabular and Tabular queries Imaging via Multi-Head Cross-Attention
Multi-Task Heads: Disease Detection (Binary), Stage Classification (Multi-class), Severity Score (MELD Regression)
Uncertainty Quantification: Monte Carlo Dropout / Deep Ensembles for flagging ambiguous cases
Explainability Layer: SHAP (Global/Local), Grad-CAM (Imaging), Counterfactual Generator (DiCE / Alibi)
4.2 Key Components
5. Step-by-Step Implementation Roadmap
Phase 1: Environment & Data Preparation (Weeks 1-3)
Setup: Python 3.10+, PyTorch 2.0+, MONAI (medical imaging), Captum (XAI), Alibi/DiCE (counterfactuals), SHAP, OpenCV, SimpleITK.
Download Datasets: ILPD/BUPA for tabular experiments; LiTS or Duke Liver Dataset for imaging.
Data Preprocessing:
Imaging: NIfTI → numpy arrays. Resample to 1.5mm isotropic. Window/level for liver (window=400, level=50). Extract 2D axial slices. Augment: rotation, elastic deformation, intensity shift.
Tabular: Handle missing values (median imputation). Normalize via Min-Max or Z-score. Encode gender. Apply SMOTE for class imbalance.
Alignment: If using both modalities, create patient-level pairing. If no direct pairing exists, use disease-category stratified sampling to create synthetic multimodal batches.
Phase 2: Self-Supervised Pre-training (Weeks 4-6)
Build a SimCLR pipeline for 2D liver slices.
Use ResNet-50 or Swin-Tiny as encoder.
Train on unlabeled LiTS/Duke slices with NT-Xent loss.
Save pre-trained weights. This is your novel contribution—most liver papers use ImageNet pre-training, not liver-specific contrastive pre-training.
Evaluation: Fine-tune a small classifier on 10% of labeled data to verify representation quality vs. random initialization.
Phase 3: Hybrid Model Development (Weeks 7-12)
Imaging Encoder: Load contrastive pre-trained Swin-CNN hybrid. Add segmentation-aware attention if using LiTS (multi-task with segmentation auxiliary loss).
Tabular Encoder: Implement FT-Transformer (from "Revisiting Deep Learning Models for Tabular Data" paper). Input: 10-20 clinical features → Output: 128-dim embedding.
Cross-Modal Fusion: Implement Bidirectional Cross-Attention: Imaging_embedding (Q) × Tabular_embedding (K,V) → Imaging-aware-of-clinical; Tabular_embedding (Q) × Imaging_embedding (K,V) → Clinical-aware-of-imaging. Concatenate both + residual connection.
Multi-Task Heads: Detection head (Binary, Sigmoid + BCE); Staging head (3-class/4-class, Softmax + Focal Loss); Severity head (MELD score regression, Linear + Huber Loss).
Training Strategy: Use Gradient Blending (weighted loss combination: λ₁L_det + λ₂L_stage + λ₃L_sev). Optimizer: AdamW with cosine annealing. Regularization: Dropout (0.3), Label Smoothing, MixUp/CutMix on imaging.
Phase 4: Explainability & Uncertainty (Weeks 13-15)
Uncertainty Quantification: Enable Dropout at inference. Run 10 stochastic forward passes. Compute mean prediction and epistemic uncertainty (variance). Reject predictions where uncertainty > 0.15 (configurable).
XAI Integration:
SHAP: DeepExplainer on tabular encoder. Generate summary plots showing bilirubin, albumin, SGOT importance.
Grad-CAM: On imaging encoder for localization.
Counterfactuals: Use DiCE library. Input patient profile → generate minimal changes to flip prediction.
Build a clinical dashboard (Gradio/Streamlit) showing: original scan + Grad-CAM overlay + SHAP bar chart + 3 counterfactual cards.
Phase 5: Evaluation & Benchmarking (Weeks 16-18)
Metrics:
Detection: Accuracy, AUC-ROC, Sensitivity, Specificity.
Staging: F1-score (macro), Cohen's Kappa.
Severity: MAE, RMSE.
Uncertainty: Calibration plots (Expected Calibration Error).
Explainability: Fidelity (explanation accuracy), Completeness, Sparsity.
Ablation Studies:
Remove contrastive pre-training → measure drop.
Replace Cross-Modal Attention with simple concatenation → measure drop.
Remove multi-task → compare single-task vs. multi-task performance.
Remove uncertainty → measure calibration error.
Phase 6: Documentation & Deployment (Weeks 19-20)
Final Deliverables: Executable Jupyter notebooks / Python scripts; Trained model weights (.pth files); Clinical dashboard (Streamlit app); Technical report with all ablation tables; IEEE-formatted research paper draft.
6. Key Research Papers & Journals to Reference
Additional foundational papers to include:
Gorishniy et al. (2021): "Revisiting Deep Learning Models for Tabular Data" (FT-Transformer).
Liu et al. (2021): "Swin Transformer: Hierarchical Vision Transformer" (for imaging backbone).
Chen et al. (2020): "Simple Framework for Contrastive Learning (SimCLR)" (for pre-training).
Mothilal et al. (2020): "Explaining Machine Learning Classifiers through Diverse Counterfactual Explanations" (DiCE).
7. Why This Will Work (Clinical & Technical Justification)
Clinical Alignment: The features your model will highlight (bilirubin, albumin, SGOT, platelets) are the exact biomarkers used in the MELD and Child-Pugh scores. When SHAP shows these as important, clinicians will trust the model.
Technical Edge: Cross-modal attention has revolutionized vision-language models (CLIP, DALL-E). Applying it to medical imaging + clinical data is a natural but unexplored extension for hepatology.
Counterfactuals for Actionability: Unlike SHAP (which only says "bilirubin is important"), counterfactuals tell the clinician what to change to improve patient outcome—making your system a decision-support tool, not just a classifier.
Uncertainty for Safety: In healthcare, saying "I don't know" is better than a wrong answer. Monte Carlo Dropout adds this safety layer.
8. Tools & Libraries Checklist
Next Step:
Begin with Phase 1 immediately. Download the ILPD dataset and LiTS subset, and build a simple baseline (ResNet-50 + MLP) within one week. Once your baseline runs, incrementally add the novel components (contrastive pre-training → cross-modal attention → counterfactuals). This ensures you always have an executable pipeline at every stage.

# Liver_Disease_Hybrid_Framework_Updated.docx

A Hybrid Model Framework for Effective Prediction of Liver Diseases
Deep Learning – Explainable AI
Final Year Project Framework — Updated Edition (3 algorithmic contributions, weaknesses reviewed and fixed, full implementation scaffold)
1. Problem Statement
Title
A Cross-Modal Contrastive Hybrid Transformer Framework with Counterfactual Explainable AI for Multi-Task Liver Disease Prediction (CMCHT-XAI)
Problem
Liver diseases (cirrhosis, hepatitis, hepatocellular carcinoma) progress silently and are often diagnosed at advanced stages. While AI models have achieved high accuracy in liver disease detection, three critical barriers prevent clinical adoption:
Black-box predictions: Clinicians cannot trust models that do not explain why a decision was made or what clinical factors drove the prediction.
Modality isolation: Existing models either process medical images (CT/MRI/Ultrasound) or clinical blood tests separately, failing to exploit the complementary information between imaging morphology and biochemical markers (bilirubin, albumin, SGOT) in a dynamically weighted fusion.
Single-task limitation: Most models perform only binary classification (disease vs. healthy) and do not simultaneously predict disease presence, fibrosis stage, and progression risk — limiting clinical utility.
Objective
Develop an executable hybrid deep learning framework that:
Uses contrastive self-supervised pre-training on unlabeled liver imaging to learn robust visual representations.
Fuses imaging (CNN-Transformer) and tabular clinical data (Tabular Transformer) via cross-modal attention.
Performs multi-task learning (detection + staging + severity scoring).
Provides counterfactual explanations alongside SHAP and Grad-CAM.
Quantifies prediction uncertainty to flag low-confidence cases for human review.
2. Legal & Open-Source Datasets
These datasets may be legally used for academic research without copyright infringement. Links verified directly — three from the project's review presentation, three added after a fresh search for additional liver-disease datasets.
2.1 Previously identified (from review presentation)
2.2 Newly added (verified during this update)
Suggested use: HCC Survival and Cirrhosis Prediction extend the tabular benchmark pool beyond ILPD/BUPA — useful for pre-training or sanity-checking the FT-Transformer tabular encoder in isolation before full multimodal pairing. MSD-Liver gives a second, independently-licensed source of segmented CT volumes alongside LiTS/Duke, useful if more imaging pre-training data is needed for SimCLR (Phase 2) beyond the primary NAFLD ultrasound set.
Recommendation: use ILPD/BUPA/HCC-Survival/Cirrhosis-Prediction for tabular pipeline experiments and LiTS/Duke/MSD-Liver for imaging-only experiments. The primary paired multimodal experiments should use the NAFLD Ultrasound + Clinical dataset; where direct pairing is unavailable, disease-category stratified sampling can construct synthetic multimodal batches (a recognized academic practice when paired multimodal data is scarce).
3. Literature Survey & Technology Gap Analysis
3.1 Existing State-of-the-Art
3.2 Identified Research Gaps (Original Novelty Claim)
Based on the literature reviewed, the following had NOT been implemented together in any liver disease prediction system at the time of the original framework:
Contrastive self-supervised pre-training + cross-modal fusion (image encoder pre-trained via SimCLR/MoCo, then fused with a tabular transformer via bidirectional cross-attention).
Counterfactual explainable AI in liver prediction (most works use SHAP/Grad-CAM only, not actionable counterfactuals).
Multi-task learning with uncertainty quantification (joint detection + staging + severity with Monte Carlo Dropout, in one architecture).
Tabular transformer (FT-Transformer) + vision hybrid fused via cross-attention, specifically for hepatology.
This document's Section 6 narrows and updates these claims after further literature verification conducted during development — read Section 6 before finalizing the report's contributions section.
4. System Architecture: CMCHT-XAI
Cross-Modal Contrastive Hybrid Transformer with Counterfactual Explainable AI
5. The Three Algorithmic Contributions
Each contribution below was only added after a literature search specifically targeting that mechanism. All three are implemented and unit-tested with synthetic data (see Section 9); none have yet been trained on real liver data, since that depends on the still-unimplemented encoder modules (see Section 8).
5.1 Contribution 1 — CSG-Fusion (Counterfactual-Sensitivity-Gated Fusion)
File: src/models/csg_fusion.py
Mechanism: runs a cheap, differentiable counterfactual probe during every forward pass — perturbing tabular features and measuring the resulting prediction shift — and uses that sensitivity signal, combined with per-modality Monte Carlo Dropout uncertainty, to gate the cross-modal attention fusion. A counterfactual consistency loss during training keeps the model's sensitivity aligned with clinically meaningful features (bilirubin, albumin, SGOT).
Literature checked: counterfactual-guided attention exists for unimodal fine-grained image recognition; uncertainty-gated image+tabular fusion exists for other diseases (chest X-ray, bone health). The combination of both signals gating fusion for image+tabular liver disease prediction was not found in either search pass.
5.2 Contribution 2 — CUSP-Cascade (Cascaded Uncertainty-and-Sensitivity Propagation)
File: src/models/cascade_heads.py
Mechanism: replaces three independent task heads with a clinically-ordered cascade (detect → stage → assess severity). Each head passes forward its own epistemic uncertainty and the counterfactual sensitivity vector from CSG-Fusion to the next head.
Literature checked: clinically-ordered cascaded multi-task heads exist already (bone disease, GI lesions, COVID/pneumonia, dementia staging); uncertainty propagation through cascaded medical pipelines exists already (MRI reconstruction cascades). Jointly propagating both uncertainty and counterfactual sensitivity through a clinically-ordered cascade built on a gated fusion layer was not found in either search pass.
Architecture analysis
Expected benefit (hypothesis, not guaranteed): error containment — downstream heads can hedge when upstream confidence is low; tighter agreement between internal reasoning and post-hoc explanations.
Known risk: cascade error propagation if the detection head is wrong early in training. Addressed directly by Contribution 3.
Cost: wider head inputs and multi-pass training-time uncertainty estimation roughly triple each head's forward cost — small relative to the imaging encoder, but must be reported honestly.
5.3 Contribution 3 — CGCT (Confidence-Gated Cascade Training)
File: src/training/confidence_gated_training.py
Mechanism: a training algorithm, not an architecture. Decides, per sample, whether CUSP-Cascade conditions the next stage on its own upstream prediction or on ground truth, based on the upstream head's own uncertainty, with a decaying global schedule.
Honest lineage — state exactly this in the report: this is a direct adaptation of Confidence-Aware Scheduled Sampling (Liu et al., 2021, Neural Machine Translation), which already uses model confidence to choose between ground truth and self-prediction during training. That mechanism was built for autoregressive token-by-token decoding. What is new here is applying it to a non-autoregressive, heterogeneous multi-task clinical cascade (binary → multi-class → regression) instead of sequence decoding — that combination was not found in any search pass run for this project.
6. How to State the Novelty Claim
Use wording close to this in the report's contributions section:
"We propose a system combining three contributions: (1) CSG-Fusion, the first application of counterfactual-sensitivity-gated, uncertainty-aware cross-modal fusion to multimodal liver disease prediction; (2) CUSP-Cascade, which extends clinically-ordered multi-task cascades by jointly propagating epistemic uncertainty and counterfactual feature sensitivity between stages; and (3) Confidence-Gated Cascade Training, an adaptation of confidence-aware scheduled sampling (Liu et al. 2021) from autoregressive sequence decoding to heterogeneous multi-task clinical cascades."
Do not claim unqualified world-first novelty for any single mechanism in isolation — every individual building block has prior art outside this specific application. The defensible claim is the specific combination, applied to this specific problem. Run one more targeted search before final submission; this is a fast-moving research area.
7. Weaknesses Found and Fixed (Self-Review)
Weakness 3 is intentionally left as a documented trade-off: every available fix changes the accuracy of the uncertainty signal that both the consistency loss and CGCT's confidence gate depend on. Guessing the fix before real profiling data exists risks quietly corrupting that signal.
8. Folder Structure
liver-cmcht-xai/
README.md — architecture diagram, run order, contribution writeups, ablation plan (this document's companion)
requirements.txt — torch, monai, timm, shap, captum, dice-ml, streamlit, etc.
config/config.yaml — every hyperparameter and the fusion_type / head_type / use_cgct ablation toggles
data/raw/ — untouched dataset downloads (never edited directly)
data/processed/ — cleaned tensors / normalized CSVs / split indices, output of preprocessing.py
notebooks/ — exploration and sanity-check notebooks
src/data/preprocessing.py — Phase 1: imaging resample/window-level, tabular impute/normalize/SMOTE, patient-level pairing
src/data/dataset.py — PyTorch Dataset classes (unlabeled imaging for SimCLR; paired multimodal for training/eval)
src/models/imaging_encoder.py — Swin-CNN hybrid encoder wrapper
src/models/tabular_encoder.py — FT-Transformer encoder
src/models/fusion.py — baseline bidirectional cross-attention fusion (ablation baseline)
src/models/csg_fusion.py — Contribution 1: CSG-Fusion
src/models/heads.py — baseline independent multi-task heads (ablation baseline)
src/models/cascade_heads.py — Contribution 2: CUSP-Cascade
src/models/cmcht_model.py — wires encoders + fusion + heads into the full model, with ablation toggles
src/training/confidence_gated_training.py — Contribution 3: CGCT
src/pretrain/simclr.py — Phase 2: SimCLR self-supervised pre-training
src/explainability/shap_utils.py, gradcam.py, counterfactual.py, uncertainty.py, run_explain.py — Phase 4 XAI layer
src/utils/metrics.py, logger.py — shared metrics and reproducibility helpers
src/train.py — Phase 3 main training loop, wires in CGCT and the consistency loss
src/evaluate.py — Phase 5 evaluation + ablation runner
dashboard/app.py — Phase 6 Streamlit clinical dashboard
checkpoints/, results/ — saved weights and metrics/plots (gitignored)
tests/test_models.py — unit tests, including the three contributions
9. Verification Performed
All three new modules (csg_fusion.py, cascade_heads.py, confidence_gated_training.py) were syntax-checked AND executed end-to-end with synthetic tensors — including a full forward pass through CUSP-Cascade with CGCT mixing active and the fixed UncertaintyAwareHead — confirming correct output shapes and value ranges. This confirms the code runs as designed; it does not replace training on real data.
10. Required Ablation Plan
1. cross_attention fusion + independent heads (full baseline)
2. csg fusion + independent heads (Contribution 1 alone)
3. cross_attention fusion + cusp cascade, no CGCT (Contribution 2 alone)
4. csg fusion + cusp cascade, no CGCT (Contributions 1+2)
5. csg fusion + cusp cascade + CGCT (full proposed system)
11. Implementation Roadmap (20 Weeks)
12. Key References
Gorishniy et al. (2021) — Revisiting Deep Learning Models for Tabular Data (FT-Transformer)
Liu et al. (2021) — Swin Transformer: Hierarchical Vision Transformer
Chen et al. (2020) — A Simple Framework for Contrastive Learning (SimCLR)
Mothilal et al. (2020) — Explaining ML Classifiers through Diverse Counterfactual Explanations (DiCE)
Liu et al. (2021) — Confidence-Aware Scheduled Sampling for Neural Machine Translation (source algorithm for Contribution 3)
13. Tools & Libraries Checklist
Next step: begin with Phase 1. Implement src/data/preprocessing.py against a real download of the NAFLD/ILPD/LiTS datasets, get a simple baseline (ResNet + MLP, baseline fusion, independent heads) running end-to-end first, then switch on each contribution one at a time, re-running the ablation table at each step.

# Liver_Disease_Hybrid_Framework_v3.docx

CMCHT-XAI: Liver Disease Hybrid Framework
Cross-Modal Contrastive Hybrid Transformer with Counterfactual Explainable AI
Updated Framework Document — Revision 3 (three algorithmic contributions, weaknesses reviewed and fixed)
1. Overview
This document supersedes the original Liver Disease Hybrid Framework. It records three specific algorithmic/architectural contributions added during development, the literature search conducted before each was built, the exact (narrow) novelty claim each one supports, and a self-review of weaknesses found in the implementation along with the fixes applied.
Read Section 5 ("How to state the novelty claim") before writing the final report's contributions section. The wording there is intentionally precise and should be used close to verbatim.
2. System Architecture
Ultrasound image -> Swin-CNN Encoder (SimCLR pre-trained); Clinical tabular data -> FT-Transformer Encoder. Both feed into the fusion stage, which feeds a multi-task cascade, which feeds the XAI layer (SHAP, Grad-CAM, DiCE counterfactuals, MC-Dropout uncertainty).
3. Contribution 1 — CSG-Fusion (Counterfactual-Sensitivity-Gated Fusion)
Location: src/models/csg_fusion.py
What it does: runs a cheap, differentiable counterfactual probe during every forward pass — perturbing tabular features and measuring how much the prediction shifts — and uses that sensitivity signal, combined with per-modality MC-Dropout uncertainty, to gate the cross-modal attention fusion. A counterfactual consistency loss during training keeps the model's sensitivity aligned with clinically meaningful features (bilirubin, albumin, SGOT).
Literature checked
Counterfactual-guided attention exists already for unimodal fine-grained image recognition (counterfactual intervention used as a training signal for attention maps).
Uncertainty-gated cross-modal (image + tabular) fusion exists already for other diseases (chest X-ray, bone health, outdoor health monitoring).
The combination of both signals gating fusion, applied to image+tabular liver disease prediction with multi-task heads, was not found in either literature pass run for this project.
4. Contribution 2 — CUSP-Cascade (Cascaded Uncertainty-and-Sensitivity Propagation)
Location: src/models/cascade_heads.py
What it does: replaces three independent task heads with a clinically-ordered cascade (detect -> stage -> assess severity). Each head passes forward its own epistemic uncertainty and the counterfactual sensitivity vector from CSG-Fusion to the next head, so e.g. the staging head knows how confident the detection step was before making its own call.
Literature checked
Clinically-ordered cascaded multi-task heads exist already (bone disease, GI lesions, COVID/pneumonia, dementia staging).
Uncertainty propagation through cascaded medical imaging pipelines exists already (MRI reconstruction -> downstream prediction).
Propagating BOTH uncertainty and counterfactual sensitivity jointly through a clinically-ordered cascade, sourced from a gated multimodal fusion layer, was not found in either literature pass for this project.
Architecture analysis
Expected benefit (hypothesis to test, not a guarantee): error containment (downstream heads can hedge when upstream confidence is low) and tighter agreement between the model's internal reasoning and its later post-hoc explanations.
Known risk: cascade error propagation — if the detection head is wrong early in training, that error can bias the staging head. This is the documented failure mode in the cascade literature reviewed above. Contribution 3 (CGCT) was built specifically to address this.
Cost: wider head inputs (propagated signals) and a training-time multi-pass uncertainty estimate roughly triple each head's forward cost. Small relative to the imaging encoder; report it honestly in a parameter-count / inference-time table.
5. Contribution 3 — Confidence-Gated Cascade Training (CGCT)
Location: src/training/confidence_gated_training.py
What it does: a training algorithm (not an architecture) that decides, per sample, whether CUSP-Cascade should condition the next stage on its own upstream prediction or on ground truth — based on the upstream head's own uncertainty — with a decaying global schedule. Built specifically to fix the cascade-error-propagation risk named in Contribution 2.
Honest lineage — state this exactly
This is a direct adaptation of Confidence-Aware Scheduled Sampling (Liu et al., 2021, Neural Machine Translation), which already uses model confidence to choose between ground truth and self-prediction during training, including with MC-Dropout-based confidence in parts of that line of work. That mechanism was built for autoregressive token-by-token sequence decoding.
What is new here: applying it to a non-autoregressive, heterogeneous multi-task clinical cascade (binary detection -> multi-class staging -> continuous severity regression) instead of token sequence decoding. This combination was not found in any literature pass run for this project.
6. How to state the novelty claim (use this wording)
"We propose a system combining three contributions: (1) CSG-Fusion, the first application of counterfactual-sensitivity-gated, uncertainty-aware cross-modal fusion to multimodal liver disease prediction; (2) CUSP-Cascade, which extends clinically-ordered multi-task cascades by jointly propagating epistemic uncertainty and counterfactual feature sensitivity between stages; and (3) Confidence-Gated Cascade Training, an adaptation of confidence-aware scheduled sampling (Liu et al. 2021) from autoregressive sequence decoding to heterogeneous multi-task clinical cascades."
Do not claim unqualified world-first novelty for any individual mechanism in isolation — every individual building block (counterfactual-guided attention, uncertainty-gated fusion, cascaded multi-task heads, confidence-aware sampling) has prior art outside this specific application. The defensible claim is the specific combination, applied to this specific problem. Verify with one more targeted search yourself before final submission, since this is a fast-moving research area.
7. Weaknesses Found and Fixed (self-review)
A self-review pass was conducted while integrating Contribution 3. Three real weaknesses were found in the prior implementation.
Weakness 3 is intentionally left as a documented trade-off rather than papered over: every available fix changes the accuracy of the uncertainty signal that both CSG-Fusion's consistency loss and CGCT's confidence gate depend on. Guessing the right fix now, before real profiling data exists, risks quietly corrupting that signal.
8. Updated Folder Structure
liver-cmcht-xai/config, data/{raw,processed}, notebooks/, src/{data, models (csg_fusion.py, cascade_heads.py, fusion.py, heads.py, cmcht_model.py, imaging_encoder.py, tabular_encoder.py), training/ (confidence_gated_training.py — new), pretrain/, explainability/, utils/, train.py, evaluate.py}, dashboard/, checkpoints/, results/, tests/.
9. Required Ablation Plan
Five rows minimum, to isolate each contribution's effect rather than only reporting the full system vs. baseline:
1. cross_attention fusion + independent heads (full baseline)
2. csg fusion + independent heads (Contribution 1 alone)
3. cross_attention fusion + cusp cascade, no CGCT (Contribution 2 alone)
4. csg fusion + cusp cascade, no CGCT (Contributions 1+2)
5. csg fusion + cusp cascade + CGCT (full proposed system, Contributions 1+2+3)
10. Verification Performed
All three new modules (csg_fusion.py, cascade_heads.py, confidence_gated_training.py) were syntax-checked and executed end-to-end with synthetic tensors during this revision — including a full forward pass through CUSP-Cascade with CGCT mixing active — confirming shapes and value ranges are correct. This does not replace training on real data; it confirms the code runs as designed.

# Zeroth review template (3).pptx

Department of Artificial Intelligence and Data Science
Supervisor: Ms. R Kalaivani  
Batch Members:
Mugesh Kumar M - 727823TUAD101
Rohit T J – 727823TUAD129
Roshan S – 727823TUAD130 
Batch No.: 1
Date: 19/06/2026 (AN)
A Hybrid Multi-Modal Framework for
Effective Prediction of Liver Diseases

Project Work Phase – I
Zeroth Review

19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
2


01
Silent Progression
NAFLD progresses silently through stages of steatosis and fibrosis, often undetected until advanced. Liver diseases (cirrhosis, hepatitis, HCC) are similarly diagnosed late, limiting treatment options.


02
Black-Box AI
Existing AI models achieve high accuracy but lack explainability — clinicians cannot trust predictions without understanding the clinical reasoning.


03
Modality Isolation
Current models process ultrasound or CT/MRI images OR blood tests in isolation, failing to exploit complementary information between imaging and biochemical markers.

Objective: Build a multimodal, multi-task, explainable AI framework for NAFLD detection and fibrosis staging from paired ultrasound imaging and clinical blood tests, with clinical trust. 
Problem Statement

Objectives


01
Contrastive Pre-training
Use SimCLR to self-supervise a Swin-CNN encoder on unlabeled NAFLD ultrasound images (10,352 images from 384 patients) before fine-tuning — extracting robust visual representations.


02
Cross-Modal Fusion
Fuse imaging (Swin-CNN) and tabular clinical data (FT-Transformer) via bidirectional cross-attention for dynamic per-patient weighting.


03
Multi-Task Learning
Jointly predict: (a) NAFLD detection (Normal vs Diseased), and (b) fibrosis stage (F0–F3) using shared backbone with task-specific heads.


04
Counterfactual XAI
SimCLR on 10,352 unlabeled NAFLD ultrasound slices — liver-specific visual priors instead of generic ImageNet weights.


05
Uncertainty Quantification
Flag low-confidence predictions using Monte Carlo Dropout (10 forward passes) for human-in-the-loop clinical safety review.


06
Clinical Dashboard
Build a Streamlit/Gradio interface showing scan overlays, SHAP charts, and counterfactual cards for clinician decision support.
19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
3

19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
4
Literature Survey & Technology Gap

Method / Paper

Approach

Limitation / Gap

XAIHO (Springer 2025)

ResNet-50 + SHAP + Adam — 92.35% accuracy

Single modality; no cross-modal fusion; post-hoc SHAP lacks counterfactuals

BiLSTM-AM-VMD (2025)

Multimodal BiLSTM + attention, AUC 0.963 for HCC

Feature concatenation (not cross-attention); no uncertainty; SHAP-only

NMD-FusionNet (2025)

CT + multi-window fusion for liver cancer segmentation

Segmentation-only; no tabular integration; no predictive staging

Self-Supervised SNN (2025)

SimCLR-based Siamese Net — 99.90% on ultrasound

Single modality; no tabular fusion; minimal explainability

FL-XAI (2025)

Federated Learning + Random Forest / XGBoost + XAI

Ensemble ML only; no imaging modality; no multi-task prediction

Research Gap: No existing work combines contrastive pre-training + cross-modal attention + multi-task learning + counterfactual XAI in a single hepatology framework.

19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
5
Proposed Methodology: CMCHT-XAI
Cross-Modal Contrastive Hybrid Transformer with Counterfactual Explainable AI

Input Layer
NAFLD Ultrasound
B-mode + Tabular Clinical Data


Imaging Encoder
Swin-CNN Hybrid
(SimCLR
Pre-trained)


Tabular Encoder
FT-Transformer
(Feature Tokenizer)


Cross-Modal
Attention
Bidirectional
Multi-Head
Cross-Attention


Multi-Task
Heads
 NAFLD Detection
(Binary) Fibrosis Staging (Multi-class)


XAI Layer
SHAP+GradCAM
+Counterfactuals
+Uncertainty


★
Contrastive Pre-training
SimCLR on unlabeled liver images — liver-specific visual priors instead of generic ImageNet weights.


★
Counterfactual XAI
DiCE generates actionable "what-if" patient-level recommendations for clinical decision support.


★
Uncertainty Quantification
Monte Carlo Dropout (10 forward passes) flags ambiguous predictions for human review.

19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
6
Datasets
Legal Open-Source Datasets
Sample Datasets
NAFLD Ultrasound + Clinical provides paired imaging and tabular data for the same 384 patients, enabling true cross-modal fusion without synthetic alignment. ILPD and BUPA serve as supplementary tabular benchmarks.




19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
7
Implementation Roadmap


Phase 1
Wk 1–3
Environment & Data
Setup PyTorch/MONAI, Download NAFLD Ultrasound + Clinical (Mendeley); ILPD as supplementary tabular benchmark. Preprocessing (resample, augment, SMOTE)


Phase 2
Wk 4–6
Self-Supervised Pre-training
SimCLR on unlabeled NAFLD ultrasound slices (10,352 images), NT-Xent loss, Evaluate representations on 10% labeled data


Phase 3
Wk 7–12
Hybrid Model Development
Swin-CNN + FT-Transformer, Cross-Modal Attention Fusion, Multi-Task training (AdamW + Focal Loss)

Phase 4
Wk 13–15
XAI & Uncertainty
SHAP + Grad-CAM + DiCE counterfactuals, MC Dropout (10 passes), Clinical dashboard (Streamlit)

Phase 5
Wk 16–18
Evaluation & Ablation
AUC-ROC, F1 (macro), Stage Accuracy, ECE metrics — ablation: remove pre-training / fusion / multi-task / uncertainty

Phase 6
Wk 19–20
Documentation & Submission
Jupyter notebooks, Model weights (.pth), IEEE paper draft, Project report

19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
8
Expected Outcomes & Deliverables

>90%
Detection AUC-ROC
(NAFLD Detection)

>0.85
Staging F1 (macro)
Multi-class fibrosis

>80%
Fibrosis Stage Accuracy
Multi-class staging

<0.10
Expected Calibration Error
Uncertainty quality
Final Deliverables


✓
Executable Jupyter notebooks and Python scripts for end-to-end pipeline


✓
Trained model weights (.pth) for all components


✓
Clinical Streamlit dashboard with Grad-CAM, SHAP charts, and counterfactual cards


✓
Journal paper draft with ablation study tables


✓
Technical project report with full methodology and results


✓
Ablation: contrastive pre-training vs random init, cross-modal vs concatenation, multi-task vs single-task

19-06-2026
| Dept. of Artificial Intelligence and Data Science | PWP-I | Batch - 1 | Zeroth review | 
9
References


Thank You

Thank You

