"""
Phase 2 — SimCLR self-supervised pre-training for the imaging encoder.

Reference: Chen et al. (2020), "A Simple Framework for Contrastive Learning of
Visual Representations" (SimCLR).

Pipeline:
    - For each unlabeled liver image, generate two augmented views.
    - Encode both views with the Swin-CNN hybrid encoder + a projection head.
    - Compute the NT-Xent (normalized temperature-scaled cross-entropy) loss so
      that views from the same image are pulled together and views from different
      images are pushed apart.
    - Save the encoder weights to checkpoints/simclr_encoder.pth for later use
      by the CMCHT-XAI model.

This is the novel pre-training contribution — most liver papers use ImageNet
pre-training, not liver-specific contrastive pre-training.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import UnlabeledImagingDataset, SimCLRAugment
from src.models.imaging_encoder import SwinCNNHybridEncoder, build_imaging_encoder
from src.utils.logger import get_logger, load_config, set_seed, get_device

logger = get_logger(__name__)


class SimCLRProjectionHead(nn.Module):
    """MLP projection head (2-layer) as in the original SimCLR paper."""

    def __init__(self, in_dim: int, hidden_dim: int = 512, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLRModel(nn.Module):
    """Encoder + projection head for SimCLR pre-training."""

    def __init__(self, encoder: SwinCNNHybridEncoder, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.projector = SimCLRProjectionHead(encoder.embed_dim, encoder.embed_dim, proj_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)            # representation (B, embed_dim)
        z = self.projector(h)          # projection (B, proj_dim)
        return h, z


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """
    Normalized Temperature-scaled Cross-Entropy loss (NT-Xent).

    z1, z2: projections (B, proj_dim) from the two augmented views.
    """
    batch_size = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # (2B, proj_dim)
    z = F.normalize(z, dim=-1)

    # cosine similarity matrix (2B, 2B)
    sim = torch.matmul(z, z.T) / temperature

    # mask out self-similarity
    mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float('-inf'))

    # positive pairs: (i, i+B) and (i+B, i)
    labels = torch.cat([
        torch.arange(batch_size, 2 * batch_size, device=z.device),
        torch.arange(0, batch_size, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def train_simclr(
    config_path: str = "config/config.yaml",
    image_dir: Optional[str] = None,
    epochs: Optional[int] = None,
) -> str:
    """
    Run SimCLR pre-training and save encoder weights.

    Returns the path to the saved checkpoint.
    """
    cfg = load_config(config_path)
    set_seed(cfg.seed)
    device = get_device()

    simclr_cfg = cfg.pretrain.simclr
    epochs = epochs or simclr_cfg.epochs
    batch_size = simclr_cfg.batch_size
    lr = simclr_cfg.lr
    weight_decay = simclr_cfg.weight_decay
    temperature = simclr_cfg.temperature
    proj_dim = simclr_cfg.proj_dim
    image_size = cfg.data.image_size

    # Dataset
    if image_dir is None:
        image_dir = str(Path(cfg.data.processed_dir) / "unlabeled_slices")
    if not Path(image_dir).exists():
        raise FileNotFoundError(
            f"Unlabeled image directory not found: {image_dir}. "
            "Place preprocessed slices under data/processed/unlabeled_slices/ "
            "or pass --image_dir."
        )

    augment = SimCLRAugment(image_size=image_size, jitter_strength=simclr_cfg.jitter_strength)
    dataset = UnlabeledImagingDataset(image_dir, image_size=image_size, augment=augment)
    if len(dataset) == 0:
        logger.error("No unlabeled images found. Aborting SimCLR pre-training.")
        return ""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)

    # Model
    encoder = build_imaging_encoder(cfg)
    model = SimCLRModel(encoder, proj_dim=proj_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    logger.info("Starting SimCLR pre-training: %d epochs, %d images", epochs, len(dataset))
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for view1, view2 in loader:
            view1, view2 = view1.to(device), view2.to(device)
            _, z1 = model(view1)
            _, z2 = model(view2)
            loss = nt_xent_loss(z1, z2, temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info("SimCLR epoch %d/%d | NT-Xent loss: %.4f", epoch + 1, epochs, avg_loss)

    # Save encoder weights
    ckpt_dir = Path(cfg.paths.checkpoints_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "simclr_encoder.pth"
    torch.save({
        "encoder_state_dict": model.encoder.state_dict(),
        "config": {
            "embed_dim": encoder.embed_dim,
            "image_size": image_size,
            "epochs": epochs,
        },
    }, ckpt_path)
    logger.info("SimCLR encoder saved to %s", ckpt_path)
    return str(ckpt_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2: SimCLR pre-training")
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--image_dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_simclr(args.config, args.image_dir, args.epochs)