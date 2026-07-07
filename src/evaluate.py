"""
Phase 5 — Evaluation and ablation runner for CMCHT-XAI.

Evaluates a trained model on the test set and, optionally, runs the full
five-row ablation plan (Section 10) to isolate each contribution's effect.

Ablation rows:
    1. baseline:    cross_attention + independent heads, no CGCT
    2. csg_only:    csg + independent heads, no CGCT
    3. cusp_only:   cross_attention + cusp_cascade, no CGCT
    4. csg_cusp:    csg + cusp_cascade, no CGCT
    5. full_system: csg + cusp_cascade + CGCT

Usage:
    python src/evaluate.py --config config/config.yaml
    python src/evaluate.py --config config/config.yaml --ablation
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalLiverDataset, resolve_experiment_dataset
from src.models.cmcht_model import build_model, build_model_from_ablation
from src.utils.logger import get_logger, load_config, set_seed, get_device
from src.utils.metrics import aggregate_metrics

logger = get_logger(__name__)


def build_test_loader(cfg):
    """Build a test dataloader from the resolved experiment dataset."""
    dataset_info = resolve_experiment_dataset(cfg, require_train=False, require_test=True)
    feature_names = list(dataset_info["feature_names"])
    logger.info(
        "Evaluating on %s dataset (%d features, %d staging classes)",
        dataset_info["name"],
        len(feature_names),
        int(dataset_info["stage_num_classes"]),
    )

    dataset = MultimodalLiverDataset(
        csv_path=str(dataset_info["test_csv"]),
        feature_names=feature_names,
        image_dir=dataset_info["image_dir"],
        image_size=cfg.data.image_size,
    )
    return DataLoader(dataset, batch_size=cfg.training.batch_size, shuffle=False,
                      num_workers=0)


@torch.no_grad()
def evaluate_model(model, loader, device, cfg) -> Dict:
    """Evaluate a model on a dataloader and return metrics + uncertainty."""
    model.eval()
    all_det_prob = []
    all_det_gt = []
    all_stage_pred = []
    all_stage_gt = []
    all_sev_pred = []
    all_sev_gt = []
    all_unc = []

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        det_gt = batch["detection_label"].to(device)
        stage_gt = batch["staging_label"].to(device)
        sev_gt = batch["severity_label"].to(device)

        out = model(images, tabular, return_uncertainty=True)

        all_det_prob.append(torch.sigmoid(out["detection_logits"]).cpu().numpy())
        all_det_gt.append(det_gt.cpu().numpy())
        all_stage_pred.append(out["staging_logits"].argmax(dim=-1).cpu().numpy())
        all_stage_gt.append(stage_gt.cpu().numpy())
        all_sev_pred.append(out["severity_pred"].cpu().numpy())
        all_sev_gt.append(sev_gt.cpu().numpy())
        if "detection_uncertainty" in out:
            all_unc.append(out["detection_uncertainty"].cpu().numpy())

    det_prob = np.concatenate(all_det_prob).ravel()
    det_gt = np.concatenate(all_det_gt).ravel()
    stage_pred = np.concatenate(all_stage_pred).ravel()
    stage_gt = np.concatenate(all_stage_gt).ravel()
    sev_pred = np.concatenate(all_sev_pred).ravel()
    sev_gt = np.concatenate(all_sev_gt).ravel()

    metrics = aggregate_metrics(
        det_true=det_gt, det_prob=det_prob,
        stage_true=stage_gt, stage_pred=stage_pred,
        sev_true=sev_gt, sev_pred=sev_pred,
        det_confidence=det_prob,
    )
    if all_unc:
        unc = np.concatenate(all_unc).ravel()
        metrics["uncertainty"]["mean_detection_uncertainty"] = float(np.mean(unc))
        n_review = int(np.sum(unc > cfg.training.uncertainty_threshold))
        metrics["uncertainty"]["n_flagged_for_review"] = n_review
        metrics["uncertainty"]["pct_flagged"] = n_review / len(unc) * 100
    return metrics


def run_single_evaluation(config_path: str = "config/config.yaml",
                          checkpoint_path: str = None) -> Dict:
    """Evaluate the model defined by the current config."""
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    device = get_device()

    # Build loader first so it can dynamically update cfg if NAFLD data exists
    loader = build_test_loader(cfg)

    model = build_model(cfg).to(device)
    
    # Dummy forward pass to initialize lazy layers
    dummy_img = torch.zeros(1, 3, cfg.data.image_size, cfg.data.image_size).to(device)
    dummy_tab = torch.zeros(1, cfg.data.num_tabular_features).to(device)
    with torch.no_grad():
        _ = model(dummy_img, dummy_tab)

    if checkpoint_path and Path(checkpoint_path).exists():
        state = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state.get("model_state_dict", state))
        logger.info("Loaded checkpoint from %s", checkpoint_path)
    else:
        logger.warning("No checkpoint; evaluating randomly initialized model.")

    metrics = evaluate_model(model, loader, device, cfg)

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_file = results_dir / "evaluation_metrics.json"
    with open(out_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info("Evaluation metrics saved to %s", out_file)
    _log_metrics("current_config", metrics)
    return metrics


def _load_checkpoint_into(model, checkpoint_path: str, device) -> bool:
    """Load a checkpoint into a model with strict=False.

    strict=False means ablation variants that differ architecturally from the
    trained full_system still load all weights they share (imaging encoder,
    tabular encoder, etc.).  Keys that don't exist in the model are silently
    skipped; keys missing from the checkpoint keep their random init.

    Returns True if the checkpoint was loaded, False otherwise.
    """
    ckpt_path = Path(checkpoint_path) if checkpoint_path else None
    if not ckpt_path or not ckpt_path.exists():
        return False
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    state_dict = state.get("model_state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logger.debug("Checkpoint missing keys (expected for ablation variants): %s", missing[:5])
    if unexpected:
        logger.debug("Checkpoint unexpected keys (expected for ablation variants): %s", unexpected[:5])
    return True


def run_ablation(config_path: str = "config/config.yaml",
                 checkpoint_path: str = None) -> Dict[str, Dict]:
    """Run the full five-row ablation plan.

    Each ablation variant is initialised from the trained checkpoint (using
    strict=False so mismatched heads still load their shared weights).  This
    isolates each architectural contribution rather than conflating it with
    'random weights vs trained weights'.
    """
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    device = get_device()

    loader = build_test_loader(cfg)
    ablation_configs = cfg.evaluation.ablation_configs
    all_results = {}

    results_dir = Path(cfg.paths.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve checkpoint: CLI arg > config default > canonical path
    ckpt = checkpoint_path or str(Path(cfg.paths.checkpoints_dir) / "cmcht_xai_best.pth")
    ckpt_exists = Path(ckpt).exists()
    if ckpt_exists:
        logger.info("Ablation: loading shared weights from checkpoint %s (strict=False)", ckpt)
    else:
        logger.warning(
            "Ablation: checkpoint not found at %s — all rows use random weights. "
            "Run training first: python -m src.train --config %s",
            ckpt, config_path,
        )

    for abl in ablation_configs:
        name = abl["name"]
        logger.info("=== Ablation: %s ===", name)
        # Deep-copy config so each ablation is independent
        cfg_copy = copy.deepcopy(cfg)
        model = build_model_from_ablation(cfg_copy, abl).to(device)

        loaded = _load_checkpoint_into(model, ckpt if ckpt_exists else None, device)
        if loaded:
            logger.info("  [%s] checkpoint weights loaded (strict=False)", name)
        else:
            logger.warning("  [%s] using random weights — results are not meaningful", name)

        metrics = evaluate_model(model, loader, device, cfg_copy)
        metrics["checkpoint_loaded"] = loaded
        all_results[name] = metrics
        _log_metrics(name, metrics)

    # Save ablation table
    out_file = results_dir / "ablation_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save a readable summary table
    _save_ablation_table(all_results, results_dir / "ablation_table.md")
    logger.info("Ablation results saved to %s", out_file)
    return all_results


def _log_metrics(name: str, metrics: Dict) -> None:
    det = metrics.get("detection", {})
    stage = metrics.get("staging", {})
    sev = metrics.get("severity", {})
    unc = metrics.get("uncertainty", {})
    logger.info("[%s] det: acc=%.3f auc=%.3f f1=%.3f | stage: f1m=%.3f kappa=%.3f | "
                "sev: mae=%.3f rmse=%.3f | ece=%.3f",
                name,
                det.get("accuracy", 0), det.get("auc_roc", 0), det.get("f1", 0),
                stage.get("f1_macro", 0), stage.get("cohen_kappa", 0),
                sev.get("mae", 0), sev.get("rmse", 0),
                unc.get("ece", 0))


def _save_ablation_table(results: Dict, path: Path) -> None:
    """Save a Markdown ablation table."""
    headers = ["Config", "Det Acc", "Det AUC", "Det F1", "Stage F1", "Stage Kappa",
               "Sev MAE", "Sev RMSE", "ECE"]
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for name, m in results.items():
        det = m.get("detection", {})
        stage = m.get("staging", {})
        sev = m.get("severity", {})
        unc = m.get("uncertainty", {})
        row = [
            name,
            f"{det.get('accuracy', 0):.3f}",
            f"{det.get('auc_roc', 0):.3f}",
            f"{det.get('f1', 0):.3f}",
            f"{stage.get('f1_macro', 0):.3f}",
            f"{stage.get('cohen_kappa', 0):.3f}",
            f"{sev.get('mae', 0):.3f}",
            f"{sev.get('rmse', 0):.3f}",
            f"{unc.get('ece', 0):.3f}",
        ]
        lines.append("| " + " | ".join(row) + " |")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info("Ablation table saved to %s", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 5: Evaluation & Ablation")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/cmcht_xai_best.pth",
        help="Path to trained checkpoint (default: checkpoints/cmcht_xai_best.pth)",
    )
    parser.add_argument("--ablation", action="store_true", help="Run the full 5-row ablation plan")
    args = parser.parse_args()

    if args.ablation:
        run_ablation(args.config, checkpoint_path=args.checkpoint)
    else:
        run_single_evaluation(args.config, args.checkpoint)