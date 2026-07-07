"""
Phase 3 — Main training loop for CMCHT-XAI.

Wires together:
    - The full model (encoders + fusion + heads, per ablation toggles)
    - CGCT (Contribution 3) teacher-forcing schedule
    - The CSG counterfactual consistency loss (Contribution 1)
    - Gradient blending across the three tasks
    - AdamW + cosine annealing + warmup
    - Checkpointing of the best model

Usage:
    python src/train.py --config config/config.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import MultimodalLiverDataset, resolve_experiment_dataset
from src.models.cmcht_model import build_model
from src.training.confidence_gated_training import (
    confidence_gated_mask,
    MultiTaskLoss,
    build_loss,
)
from src.utils.logger import get_logger, load_config, set_seed, get_device

logger = get_logger(__name__)


from torchvision import transforms

def build_dataloaders(cfg):
    """Build train/val dataloaders from the resolved experiment dataset."""
    dataset_info = resolve_experiment_dataset(cfg, require_train=True, require_test=True)
    feature_names = list(dataset_info["feature_names"])
    logger.info(
        "Using %s dataset (%d features, %d staging classes)",
        dataset_info["name"],
        len(feature_names),
        int(dataset_info["stage_num_classes"]),
    )

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
    ])

    train_ds = MultimodalLiverDataset(
        csv_path=str(dataset_info["train_csv"]),
        feature_names=feature_names,
        image_dir=dataset_info["image_dir"],
        image_size=cfg.data.image_size,
        transform=train_transform,
    )
    val_csv_path = dataset_info.get("val_csv", dataset_info["test_csv"])
    val_ds = MultimodalLiverDataset(
        csv_path=str(val_csv_path),
        feature_names=feature_names,
        image_dir=dataset_info["image_dir"],
        image_size=cfg.data.image_size,
        transform=None,
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True,
        num_workers=cfg.training.num_workers, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False,
        num_workers=cfg.training.num_workers,
    )
    return train_loader, val_loader, dataset_info



def train_one_epoch(
    model, loader, loss_fn, optimizer, device, cfg, epoch: int, feature_names: list
):
    """Train for one epoch with soft CGCT teacher forcing + consistency loss."""
    model.train()

    # Calculate global tf rate
    initial_tf = cfg.training.cgct.initial_teacher_forcing if cfg.training.use_cgct else 1.0
    final_tf = cfg.training.cgct.final_teacher_forcing if cfg.training.use_cgct else 1.0
    total_epochs = max(cfg.training.epochs, 1)
    current_tf = initial_tf + (min(epoch, total_epochs) / total_epochs) * (final_tf - initial_tf)

    # Column mapping for CSG-Fusion — match clinically meaningful features
    # by case-insensitive name, supporting both Cirrhosis and NAFLD feature sets
    probe_keywords = ["bilirubin", "sgot", "albumin", "ast", "alt"]
    feature_names_lower = [f.lower() for f in feature_names]
    probe_features = [
        i for i, fl in enumerate(feature_names_lower)
        if any(kw in fl for kw in probe_keywords)
    ]
    if model.fusion_type == "csg" and hasattr(model.fusion, "probe"):
        model.fusion.probe.probe_features = probe_features

    total_loss = 0.0
    total_det = 0.0
    total_stage = 0.0
    total_sev = 0.0
    total_cons = 0.0
    n_batches = 0

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        det_gt = batch["detection_label"].to(device)
        stage_gt = batch["staging_label"].to(device)
        sev_gt = batch["severity_label"].to(device)

        # --- CGCT soft masking ---------------------------------
        self_prob_dict = None
        if cfg.training.use_cgct and model.head_type == "cusp_cascade":
            with torch.no_grad():
                probe_out = model(images, tabular, return_uncertainty=True)
            self_prob_dict = {
                "detection": confidence_gated_mask(
                    probe_out.get("detection_uncertainty"), current_tf, threshold=cfg.training.cgct.uncertainty_threshold
                ),
                "staging": confidence_gated_mask(
                    probe_out.get("staging_uncertainty"), current_tf, threshold=cfg.training.cgct.uncertainty_threshold
                )
            }

        # --- Forward pass ------------------------------------------------
        outputs = model(
            images, tabular,
            detection_gt=det_gt if cfg.training.use_cgct else None,
            staging_gt=stage_gt if cfg.training.use_cgct else None,
            self_prob_dict=self_prob_dict,
            return_uncertainty=False,
        )

        # --- CSG consistency loss ----------------------------------------
        consistency_loss = None
        if model.fusion_type == "csg" and hasattr(model.fusion, "consistency_loss"):
            consistency_loss = model.fusion.consistency_loss()

        # --- Multi-task loss ---------------------------------------------
        losses = loss_fn(
            outputs, det_gt, stage_gt, sev_gt,
            consistency_loss=consistency_loss,
        )

        optimizer.zero_grad()
        losses["total"].backward()
        if cfg.training.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip)
        optimizer.step()

        total_loss += losses["total"].item()
        total_det += losses["detection"].item()
        total_stage += losses["staging"].item()
        total_sev += losses["severity"].item()
        total_cons += float(losses["consistency"])
        n_batches += 1

    n = max(n_batches, 1)
    return {
        "total": total_loss / n,
        "detection": total_det / n,
        "staging": total_stage / n,
        "severity": total_sev / n,
        "consistency": total_cons / n,
    }


@torch.no_grad()
def validate(model, loader, loss_fn, device, cfg):
    """Validate on the val set."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_det_prob = []
    all_det_gt = []
    all_stage_pred = []
    all_stage_gt = []
    all_sev_pred = []
    all_sev_gt = []

    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        det_gt = batch["detection_label"].to(device)
        stage_gt = batch["staging_label"].to(device)
        sev_gt = batch["severity_label"].to(device)

        outputs = model(images, tabular, return_uncertainty=False)
        losses = loss_fn(outputs, det_gt, stage_gt, sev_gt)
        total_loss += losses["total"].item()
        n_batches += 1

        all_det_prob.append(torch.sigmoid(outputs["detection_logits"]).cpu().numpy())
        all_det_gt.append(det_gt.cpu().numpy())
        all_stage_pred.append(outputs["staging_logits"].argmax(dim=-1).cpu().numpy())
        all_stage_gt.append(stage_gt.cpu().numpy())
        all_sev_pred.append(outputs["severity_pred"].cpu().numpy())
        all_sev_gt.append(sev_gt.cpu().numpy())

    from src.utils.metrics import aggregate_metrics

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
    metrics["val_loss"] = total_loss / max(n_batches, 1)
    return metrics


def train(
    config_path: str = "config/config.yaml",
    epochs: int | None = None,
    checkpoint_out: str | None = None,
    use_cgct: bool | None = None,
) -> str:
    """Run the full Phase-3 training loop.

    Args:
        config_path:    Path to config YAML.
        epochs:         Override epoch count from config.
        checkpoint_out: Override output checkpoint filename (stem only, e.g.
                        ``'cmcht_xai_no_cgct'``). Saved under checkpoints_dir.
        use_cgct:       If not None, overrides ``training.use_cgct`` in config.
    """
    cfg = load_config(config_path)
    if epochs is not None:
        cfg.training.epochs = epochs
    if use_cgct is not None:
        cfg.training.use_cgct = use_cgct
    set_seed(cfg.seed)
    device = get_device()

    train_loader, val_loader, dataset_info = build_dataloaders(cfg)
    feature_names = list(dataset_info["feature_names"])
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )

    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_stem = checkpoint_out if checkpoint_out else "cmcht_xai_best"
    best_ckpt = ckpt_dir / f"{ckpt_stem}.pth"
    best_val_loss = float("inf")

    logger.info(
        "Starting training: %d epochs | fusion=%s | heads=%s | cgct=%s",
        cfg.training.epochs, cfg.model.fusion.fusion_type,
        cfg.model.heads.head_type, cfg.training.use_cgct,
    )

    for epoch in range(cfg.training.epochs):
        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, device, cfg, epoch, feature_names
        )
        val_metrics = validate(model, val_loader, loss_fn, device, cfg)
        scheduler.step()

        val_loss = val_metrics["val_loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_metrics": val_metrics,
                "config": {
                    "fusion_type": cfg.model.fusion.fusion_type,
                    "head_type": cfg.model.heads.head_type,
                    "use_cgct": cfg.training.use_cgct,
                },
            }, best_ckpt)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            logger.info(
                "Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | "
                "det_acc=%.3f | stage_f1=%.3f | sev_mae=%.3f",
                epoch + 1, cfg.training.epochs,
                train_metrics["total"], val_loss,
                val_metrics.get("detection", {}).get("accuracy", 0),
                val_metrics.get("staging", {}).get("f1_macro", 0),
                val_metrics.get("severity", {}).get("mae", 0),
            )

    logger.info("Training complete. Best checkpoint: %s (val_loss=%.4f)", best_ckpt, best_val_loss)
    return str(best_ckpt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3: Hybrid model training")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count from config")
    parser.add_argument(
        "--no-cgct", dest="no_cgct", action="store_true",
        help="Disable CGCT teacher forcing (sets training.use_cgct=False)",
    )
    parser.add_argument(
        "--checkpoint-out", dest="checkpoint_out", default=None,
        help="Output checkpoint stem, e.g. 'cmcht_xai_no_cgct' (saved under checkpoints_dir)",
    )
    args = parser.parse_args()
    train(
        args.config,
        epochs=args.epochs,
        checkpoint_out=args.checkpoint_out,
        use_cgct=False if args.no_cgct else None,
    )