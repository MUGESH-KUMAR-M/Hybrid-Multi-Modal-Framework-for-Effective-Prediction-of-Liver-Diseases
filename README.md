# CMCHT-XAI

**A Cross-Modal Contrastive Hybrid Transformer Framework with Counterfactual Explainable AI for Multi-Task Liver Disease Prediction**

A hybrid deep learning framework that fuses liver imaging (Swin-CNN encoder, SimCLR pre-trained) with clinical tabular data (FT-Transformer) via cross-modal attention, performs multi-task prediction (detection + staging + severity), and provides counterfactual explanations alongside SHAP, Grad-CAM, and Monte-Carlo-Dropout uncertainty.

---

## 1. The Three Algorithmic Contributions

| # | Name | File | What it does |
|---|------|------|--------------|
| 1 | **CSG-Fusion** | `src/models/csg_fusion.py` | Counterfactual-sensitivity-gated, uncertainty-aware cross-modal fusion. Runs a differentiable counterfactual probe on tabular features each forward pass and uses the sensitivity signal + per-modality MC-Dropout uncertainty to gate cross-modal attention. A consistency loss keeps sensitivity aligned with clinically meaningful features (bilirubin, albumin, SGOT). |
| 2 | **CUSP-Cascade** | `src/models/cascade_heads.py` | Clinically-ordered cascade (detect → stage → severity) that propagates both epistemic uncertainty and the counterfactual sensitivity vector between stages, so downstream heads can hedge when upstream confidence is low. |
| 3 | **CGCT** | `src/training/confidence_gated_training.py` | Confidence-Gated Cascade Training — an adaptation of confidence-aware scheduled sampling (Liu et al. 2021) from autoregressive sequence decoding to a heterogeneous multi-task clinical cascade. Per sample, decides whether the next stage conditions on its own upstream prediction or ground truth, based on upstream uncertainty, with a decaying global schedule. |

**Novelty claim (use close to verbatim in the report):**

> "We propose a system combining three contributions: (1) CSG-Fusion, the first application of counterfactual-sensitivity-gated, uncertainty-aware cross-modal fusion to multimodal liver disease prediction; (2) CUSP-Cascade, which extends clinically-ordered multi-task cascades by jointly propagating epistemic uncertainty and counterfactual feature sensitivity between stages; and (3) Confidence-Gated Cascade Training, an adaptation of confidence-aware scheduled sampling (Liu et al. 2021) from autoregressive sequence decoding to heterogeneous multi-task clinical cascades."

The defensible claim is the **specific combination applied to this specific problem** — every individual building block has prior art outside hepatology AI.

---

## 2. Architecture

```
Liver Image (CT/MRI/US) ──► Swin-CNN Encoder (SimCLR pre-trained) ──┐
                                                                     │
Clinical Tabular Data   ──► FT-Transformer Encoder ─────────────────┤
                                                                     ▼
                                              ┌────────────────────────────────┐
                                              │  CSG-Fusion (Contribution 1)   │
                                              │  counterfactual probe +        │
                                              │  uncertainty-gated cross-attn  │
                                              └───────────────┬────────────────┘
                                                              │
                                                              ▼
                                              ┌────────────────────────────────┐
                                              │  CUSP-Cascade (Contribution 2) │
                                              │  detect → stage → severity     │
                                              │  (uncertainty + sensitivity    │
                                              │   propagated between stages)   │
                                              └───────────────┬────────────────┘
                                                              │
                                                              ▼
                                              ┌────────────────────────────────┐
                                              │  XAI Layer                     │
                                              │  SHAP · Grad-CAM · DiCE CFs    │
                                              │  MC-Dropout uncertainty        │
                                              └────────────────────────────────┘
```

Training is driven by **CGCT (Contribution 3)**, which gates each cascade stage between self-prediction and ground truth using upstream uncertainty.

---

## 3. Folder Structure

```
liver-cmcht-xai/
├── README.md
├── requirements.txt
├── config/config.yaml
├── data/{raw,processed}/
├── notebooks/
├── src/
│   ├── data/{preprocessing.py, dataset.py}
│   ├── models/{imaging_encoder.py, tabular_encoder.py, fusion.py,
│   │          csg_fusion.py, heads.py, cascade_heads.py, cmcht_model.py}
│   ├── training/confidence_gated_training.py
│   ├── pretrain/simclr.py
│   ├── explainability/{shap_utils.py, gradcam.py, counterfactual.py,
│   │                    uncertainty.py, run_explain.py}
│   ├── utils/{metrics.py, logger.py}
│   ├── train.py
│   └── evaluate.py
├── api/server.py
├── frontend/
├── checkpoints/
├── results/
└── tests/test_models.py
```

---

## 4. Run Order (Implementation Roadmap)

| Phase | Weeks | Command | Deliverable |
|-------|-------|---------|-------------|
| 1. Data prep | 1–3 | `python -m src.data.preprocessing` | Cleaned CSVs in `data/processed/` |
| 2. Pre-training | 4–6 | `python src/pretrain/simclr.py` | `checkpoints/simclr_encoder.pth` |
| 3. Hybrid training | 7–12 | `python src/train.py` | `checkpoints/cmcht_xai_best.pth` |
| 4. XAI & uncertainty | 13–15 | `python src/explainability/run_explain.py` | SHAP/Grad-CAM/counterfactual outputs in `results/` |
| 5. Evaluation | 16–18 | `python src/evaluate.py` | Metrics + ablation tables in `results/` |
| 6. Dashboard & docs | 19–20 | `uvicorn api.server:app` + `npm run dev` in `frontend/` | Next.js clinical dashboard |

---

## 5. Ablation Plan (Section 10)

Five rows isolate each contribution's effect:

| # | Config | Fusion | Heads | CGCT |
|---|--------|--------|-------|------|
| 1 | `baseline` | cross_attention | independent | off |
| 2 | `csg_only` | csg | independent | off |
| 3 | `cusp_only` | cross_attention | cusp_cascade | off |
| 4 | `csg_cusp` | csg | cusp_cascade | off |
| 5 | `full_system` | csg | cusp_cascade | on |

Run all five via:
```bash
python src/evaluate.py --ablation
```

---

## 6. Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Verify model architecture (unit tests — no dataset required)
python tests/test_models.py

# 3. Place raw datasets under data/raw/ then preprocess
#    data/raw/ILPD/ilpd.csv
#    data/raw/Cirrhosis/cirrhosis.csv
python -m src.data.preprocessing

# 4. SimCLR pre-training
python src/pretrain/simclr.py --config config/config.yaml

# 5. Hybrid training
python src/train.py --config config/config.yaml

# 6. Explainability
python src/explainability/run_explain.py --config config/config.yaml

# 7. Evaluation + ablation
python src/evaluate.py --config config/config.yaml --ablation

# 8. Dashboard (FastAPI + Next.js)
uvicorn api.server:app --reload --port 8000
cd frontend && npm install && npm run dev
# Open http://localhost:3000
```

---

## 7. Datasets (Legal & Open-Source)

| Dataset | Type | Use |
|---------|------|-----|
| ILPD / BUPA | Tabular | Tabular pipeline benchmark |
| HCC Survival / Cirrhosis Prediction | Tabular | Extra tabular benchmark / pre-training |
| NAFLD Ultrasound + Clinical | Paired multimodal | **Primary** multimodal experiments |
| LiTS / Duke / MSD-Liver | CT/MRI volumes | Imaging-only experiments + SimCLR pre-training |

When direct multimodal pairing is unavailable, use the NAFLD ultrasound dataset with clinical tabular features per patient (see `Docs/CMCHT_XAI_Dataset_Reference.docx`).

---

## 8. Key References

- Gorishniy et al. (2021) — Revisiting Deep Learning Models for Tabular Data (FT-Transformer)
- Liu et al. (2021) — Swin Transformer: Hierarchical Vision Transformer
- Chen et al. (2020) — A Simple Framework for Contrastive Learning (SimCLR)
- Mothilal et al. (2020) — Diverse Counterfactual Explanations (DiCE)
- Liu et al. (2021) — Confidence-Aware Scheduled Sampling for Neural Machine Translation (source algorithm for CGCT)

---

## 9. Status

All three contributions are **implemented and unit-tested** (see `tests/test_models.py`), confirming correct output shapes and value ranges end-to-end. Training and evaluation require processed data in `data/processed/` (Phase 1).