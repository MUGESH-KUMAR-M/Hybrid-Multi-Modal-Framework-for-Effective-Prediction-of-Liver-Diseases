"""
FastAPI backend for CMCHT-XAI clinical dashboard.

Serves model predictions, Grad-CAM overlays, SHAP values, counterfactuals,
and MC-Dropout uncertainty to the Next.js frontend.

Run:
    uvicorn api.server:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import MultimodalLiverDataset
from src.explainability.counterfactual import (
    format_counterfactual_text,
    generate_counterfactuals,
)
from src.explainability.gradcam import generate_gradcam_for_sample, overlay_cam_on_image
from src.explainability.shap_utils import compute_shap_values
from src.explainability.uncertainty import mc_dropout_predict
from src.models.cmcht_model import build_model
from src.utils.logger import get_device, load_config, set_seed

app = FastAPI(title="CMCHT-XAI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_model = None
_cfg = None
_device = None
_dataset = None


def _build_dashboard_dataset(cfg):
    """Load processed multimodal data for the dashboard."""
    from src.data.dataset import resolve_experiment_dataset
    dataset_info = resolve_experiment_dataset(cfg, require_train=False, require_test=True)
    feature_names = list(dataset_info["feature_names"])
    
    return MultimodalLiverDataset(
        csv_path=str(dataset_info["test_csv"]),
        feature_names=feature_names,
        image_dir=dataset_info["image_dir"],
        image_size=cfg.data.image_size,
    )


def _get_resources():
    global _model, _cfg, _device, _dataset
    if _model is None:
        _cfg = load_config("config/config.yaml")
        set_seed(_cfg.seed)
        _device = get_device()
        
        # 1. Resolve dataset first to update cfg with proper feature counts
        _dataset = _build_dashboard_dataset(_cfg)
        
        # 2. Build model with updated cfg
        _model = build_model(_cfg).to(_device)
        
        # 3. Dummy forward pass to initialize lazy layers (like 832-dim projection)
        dummy_img = torch.zeros(1, 3, _cfg.data.image_size, _cfg.data.image_size).to(_device)
        dummy_tab = torch.zeros(1, _cfg.data.num_tabular_features).to(_device)
        with torch.no_grad():
            _ = _model(dummy_img, dummy_tab)
            
        # 4. Load trained weights
        ckpt_path = Path("checkpoints/cmcht_xai_best.pth")
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=_device)
            _model.load_state_dict(state.get("model_state_dict", state))
        
        _model.eval()
    return _model, _cfg, _device, _dataset


def _array_to_png_b64(arr: np.ndarray) -> str:
    from PIL import Image

    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.clip(arr, 0, 1)
    img = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class PatientSummary(BaseModel):
    id: int
    label: str
    features: dict[str, float]


class AnalyzeResponse(BaseModel):
    patient_id: int
    feature_names: list[str]
    features: dict[str, float]
    predictions: dict
    uncertainty: dict
    gradcam_b64: str | None
    shap: list[dict]
    counterfactuals: list[str]


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/patients")
def list_patients():
    _, cfg, _, dataset = _get_resources()
    stage_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]
    patients = []
    for i in range(len(dataset)):
        sample = dataset[i]
        stage = int(sample["staging_label"].item() if hasattr(sample["staging_label"], "item") else sample["staging_label"])
        det = float(sample["detection_label"].item() if hasattr(sample["detection_label"], "item") else sample["detection_label"])
        patients.append({
            "id": i,
            "stage": stage_names[min(stage, len(stage_names) - 1)],
            "detection": det,
        })
    return {
        "count": len(dataset),
        "feature_names": list(cfg.data.tabular_features),
        "patients": patients,
    }


@app.get("/api/analyze/{patient_id}", response_model=AnalyzeResponse)
def analyze_patient(patient_id: int):
    model, cfg, device, dataset = _get_resources()
    if patient_id < 0 or patient_id >= len(dataset):
        raise HTTPException(status_code=404, detail="Patient not found")

    sample = dataset[patient_id]
    feature_names = list(cfg.data.tabular_features)
    feat_vals = sample["tabular"].numpy()
    features = {name: float(feat_vals[i]) for i, name in enumerate(feature_names)}

    image = sample["image"].unsqueeze(0).to(device)
    tabular = sample["tabular"].unsqueeze(0).to(device)

    with torch.no_grad():
        out = model(image, tabular, return_uncertainty=True)

    det_prob = float(torch.sigmoid(out["detection_logits"]).item())
    stage_pred = int(out["staging_logits"].argmax(dim=-1).item())
    sev_pred = float(out["severity_pred"].item())
    stage_names = ["Stage 1", "Stage 2", "Stage 3", "Stage 4"]

    mc_out = mc_dropout_predict(model, image, tabular, n_passes=10)
    det_unc = float(mc_out["detection_uncertainty"].item())
    threshold = cfg.explainability.uncertainty.threshold
    needs_review = det_unc > threshold

    gradcam_b64 = None
    try:
        cam = generate_gradcam_for_sample(model, image, tabular, task="detection")
        img_np = image.squeeze(0).cpu().numpy()
        overlay = overlay_cam_on_image(img_np, cam)
        gradcam_b64 = _array_to_png_b64(overlay)
    except Exception:
        pass

    shap_items: list[dict] = []
    try:
        shap_values, _ = compute_shap_values(
            model, tabular, tabular, feature_names,
            task="detection", n_background=20,
        )
        vals = shap_values[0] if shap_values.ndim > 1 else shap_values
        for i, name in enumerate(feature_names):
            shap_items.append({"feature": name, "value": float(vals[i])})
        shap_items.sort(key=lambda x: abs(x["value"]), reverse=True)
    except Exception:
        pass

    cf_texts: list[str] = []
    try:
        cfs = generate_counterfactuals(
            model, sample["tabular"].numpy(), feature_names,
            desired_class=0, num_cf=3, method="gradient",
        )
        for cf in cfs:
            cf_texts.append(format_counterfactual_text(cf, feature_names))
    except Exception:
        pass

    return AnalyzeResponse(
        patient_id=patient_id,
        feature_names=feature_names,
        features=features,
        predictions={
            "detection_probability": det_prob,
            "detection_label": "Diseased" if det_prob >= 0.5 else "Healthy",
            "staging": stage_names[stage_pred],
            "staging_index": stage_pred,
            "severity_meld": sev_pred,
        },
        uncertainty={
            "detection_variance": det_unc,
            "threshold": threshold,
            "needs_review": needs_review,
            "confidence": "low" if needs_review else "high",
        },
        gradcam_b64=gradcam_b64,
        shap=shap_items,
        counterfactuals=cf_texts,
    )
