"""
Phase 4 — Explainability orchestrator for CMCHT-XAI.

Runs SHAP (tabular), Grad-CAM (imaging), DiCE counterfactuals, and MC-Dropout
uncertainty on a trained model and saves all outputs to results/.

Usage:
    python src/explainability/run_explain.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalLiverDataset, resolve_experiment_dataset
from src.explainability.counterfactual import (
    generate_counterfactuals,
    format_counterfactual_text,
)
from src.explainability.gradcam import generate_gradcam_for_sample
from src.explainability.shap_utils import compute_shap_values, plot_shap_summary
from src.explainability.uncertainty import mc_dropout_predict, flag_uncertain_cases
from src.models.cmcht_model import build_model
from src.utils.logger import get_logger, load_config, set_seed, get_device

logger = get_logger(__name__)


def run_explainability(
    config_path: str = "config/config.yaml",
    checkpoint_path: Optional[str] = None,
    n_samples: int = 10,
) -> None:
    """Run the full Phase-4 explainability pipeline."""
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    device = get_device()

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    xai_dir = results_dir / "explainability"
    xai_dir.mkdir(parents=True, exist_ok=True)

    dataset_info = resolve_experiment_dataset(cfg, require_train=False, require_test=True)
    feature_names = list(dataset_info["feature_names"])
    cfg.data.num_tabular_features = len(feature_names)
    cfg.data.tabular_features = feature_names
    cfg.model.num_staging_classes = int(dataset_info["stage_num_classes"])
    logger.info(
        "XAI using %s dataset (%d features, %d staging classes)",
        dataset_info["name"],
        len(feature_names),
        int(dataset_info["stage_num_classes"]),
    )

    # Build model (now with correct num_tabular_features)
    model = build_model(cfg).to(device)
    
    # Dummy forward pass to initialize lazy layers (like the 832-dim imaging projection)
    dummy_img = torch.zeros(1, 3, cfg.data.image_size, cfg.data.image_size).to(device)
    dummy_tab = torch.zeros(1, cfg.data.num_tabular_features).to(device)
    with torch.no_grad():
        _ = model(dummy_img, dummy_tab)
        
    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.warning("No checkpoint loaded; using randomly initialized weights for XAI demo.")
    model.eval()

    dataset = MultimodalLiverDataset(
        csv_path=str(dataset_info["test_csv"]),
        feature_names=feature_names,
        image_dir=dataset_info["image_dir"],
        image_size=cfg.data.image_size,
    )

    loader = DataLoader(dataset, batch_size=min(n_samples, len(dataset)), shuffle=False)
    batch = next(iter(loader))
    images = batch["image"].to(device)
    tabular = batch["tabular"].to(device)

    # ---- 1. SHAP (tabular) ----------------------------------------------
    logger.info("Computing SHAP values for tabular pathway...")
    try:
        # Use a larger background from the training set if available, or sample up to 100
        background_dataset = dataset
        train_csv = dataset_info.get("train_csv")
        if train_csv and Path(train_csv).exists():
            background_dataset = MultimodalLiverDataset(
                csv_path=str(train_csv),
                feature_names=feature_names,
                image_dir=dataset_info["image_dir"],
                image_size=cfg.data.image_size,
            )
        
        bg_loader = DataLoader(background_dataset, batch_size=100, shuffle=True)
        bg_batch = next(iter(bg_loader))
        bg_tabular = bg_batch["tabular"].to(device)

        import shap
        # Fallback to shap.sample if it's a numpy array, but for tensors we slice/randperm in compute_shap_values
        shap_values, fnames = compute_shap_values(
            model, bg_tabular, tabular, feature_names, task="detection",
            n_background=100,
        )
        plot_shap_summary(
            shap_values, tabular.cpu().numpy(), feature_names,
            save_path=str(xai_dir / "shap_summary.png"),
        )
        np.save(xai_dir / "shap_values.npy", shap_values)
        logger.info("SHAP values saved.")
    except Exception as exc:
        logger.warning("SHAP computation failed: %s", exc)

    # ---- 2. Grad-CAM (imaging) -------------------------------------------
    logger.info("Computing Grad-CAM for imaging pathway...")
    for i in range(min(3, images.size(0))):
        try:
            generate_gradcam_for_sample(
                model, images[i:i+1], tabular[i:i+1], task="detection",
                save_path=str(xai_dir / f"gradcam_sample_{i}.png"),
            )
        except Exception as exc:
            logger.warning("Grad-CAM for sample %d failed: %s", i, exc)

    # ---- 3. Counterfactuals (DiCE / gradient) ----------------------------
    logger.info("Generating counterfactual explanations...")
    cf_texts = []
    for i in range(min(3, tabular.size(0))):
        try:
            cfs = generate_counterfactuals(
                model, tabular[i].cpu().numpy(), feature_names,
                desired_class=0, num_cf=cfg.explainability.counterfactual.num_cf,
                method=cfg.explainability.counterfactual.method,
            )
            for j, cf in enumerate(cfs):
                text = format_counterfactual_text(cf, feature_names)
                cf_texts.append(f"Sample {i}, CF {j+1}: {text}")
        except Exception as exc:
            logger.warning("Counterfactual generation for sample %d failed: %s", i, exc)

    with open(xai_dir / "counterfactuals.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(cf_texts))
    logger.info("Counterfactuals saved.")

    # ---- 4. MC-Dropout uncertainty ---------------------------------------
    logger.info("Computing MC-Dropout uncertainty...")
    try:
        mc_out = mc_dropout_predict(
            model, images, tabular,
            n_passes=cfg.explainability.uncertainty.mc_passes,
        )
        needs_review = flag_uncertain_cases(
            mc_out["detection_uncertainty"],
            threshold=cfg.explainability.uncertainty.threshold,
        )
        np.save(xai_dir / "mc_predictions.npy", {
            "detection_prob": mc_out["detection_prob"].cpu().numpy(),
            "detection_uncertainty": mc_out["detection_uncertainty"].cpu().numpy(),
            "staging_uncertainty": mc_out["staging_uncertainty"].cpu().numpy(),
            "severity_uncertainty": mc_out["severity_uncertainty"].cpu().numpy(),
            "needs_review": needs_review.cpu().numpy(),
        }, allow_pickle=True)
        n_review = int(needs_review.sum().item())
        logger.info("Uncertainty computed. %d/%d samples flagged for review.",
                    n_review, images.size(0))
    except Exception as exc:
        logger.warning("MC-Dropout uncertainty computation failed: %s", exc)

    logger.info("Phase 4 explainability complete. Outputs in %s", xai_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 4: Explainability")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()
    run_explainability(args.config, args.checkpoint, args.n_samples)