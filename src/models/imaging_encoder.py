"""
Imaging encoder for CMCHT-XAI.

A Swin-Transformer + CNN hybrid backbone:
    - A lightweight CNN stem (ResNet-34 first block) captures fine-grained lesion
      textures.
    - A Swin-Tiny transformer captures long-range spatial dependencies.
    - Outputs a projected embedding of dim `embed_dim`.

Pre-trained via SimCLR on unlabeled liver images (Phase 2). If SimCLR weights
are available they overwrite the ImageNet initialization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


class CNNStem(nn.Module):
    """Lightweight CNN stem (first layers of ResNet-34) for local texture features."""

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        try:
            import torchvision.models as tvm

            weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = tvm.resnet34(weights=weights)
            # stem + layer1: keeps spatial resolution high enough for texture
            self.stem = nn.Sequential(
                resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
            )
            out_channels = 64
        except Exception as exc:
            logger.warning("torchvision ResNet unavailable (%s); using conv stem.", exc)
            self.stem = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.MaxPool2d(3, stride=2, padding=1),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            )
            out_channels = 64
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.stem(x)


class SwinCNNHybridEncoder(nn.Module):
    """
    Swin-CNN hybrid imaging encoder.

    Pipeline: image -> CNN stem -> Swin Transformer -> global pool -> projection.
    Falls back to a pure CNN encoder if timm/Swin is unavailable, so the pipeline
    always runs end-to-end.
    """

    def __init__(
        self,
        swin_name: str = "swin_tiny_patch4_window7_224",
        cnn_stem: str = "resnet34",
        pretrained: bool = True,
        embed_dim: int = 256,
        image_size: int = 224,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.image_size = image_size

        self.cnn_stem = CNNStem(in_channels=3, pretrained=pretrained)
        cnn_out_channels = self.cnn_stem.out_channels

        # Swin transformer backbone via timm
        self.swin = None
        self.swin_out_dim = None
        try:
            import timm

            self.swin = timm.create_model(
                swin_name, pretrained=pretrained, in_chans=3,
                num_classes=0, global_pool="avg",
            )
            self.swin_out_dim = self.swin.num_features
            logger.info("Swin backbone loaded: %s (features=%d)", swin_name, self.swin_out_dim)
        except Exception as exc:
            logger.warning("timm/Swin unavailable (%s); using CNN-only encoder.", exc)
            self.swin = None

        if self.swin is not None:
            feat_dim = self._compute_concat_dim()
        else:
            # CNN-only fallback: global pool the CNN stem output
            self.cnn_pool = nn.AdaptiveAvgPool2d(1)
            feat_dim = cnn_out_channels

        self.projection = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _compute_concat_dim(self) -> int:
        """Dynamically compute the concatenated feature dimension using a dummy forward pass."""
        device = next(self.cnn_stem.parameters()).device
        dummy_x = torch.zeros(1, 3, self.image_size, self.image_size, device=device)
        with torch.no_grad():
            cnn_feat = self.cnn_stem(dummy_x)
            cnn_feat = cnn_feat.flatten(2).mean(dim=-1)
            swin_feat = self.swin(dummy_x)
            feat = torch.cat([swin_feat, cnn_feat], dim=-1)
        return feat.shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) -> (B, embed_dim)."""
        if self.swin is not None:
            # Swin expects the raw image; CNN stem is used as an auxiliary path
            # whose features are fused for richer texture representation.
            cnn_feat = self.cnn_stem(x)            # (B, 64, H', W')
            cnn_feat = cnn_feat.flatten(2).mean(dim=-1)  # (B, 64) global texture summary
            swin_feat = self.swin(x)               # (B, swin_out_dim)
            feat = torch.cat([swin_feat, cnn_feat], dim=-1)
            # adapt projection input if we concatenated
            if feat.shape[-1] != self.projection[0].normalized_shape[0]:
                self._rebuild_projection(feat.shape[-1])
        else:
            feat = self.cnn_pool(self.cnn_stem(x)).flatten(1)
        return self.projection(feat)

    def _rebuild_projection(self, in_dim: int) -> None:
        """Rebuild the projection head to match a new input dimension."""
        device = next(self.projection.parameters()).device
        self.projection = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        ).to(device)

    def load_simclr_weights(self, path: str) -> bool:
        """Load SimCLR pre-trained encoder weights if available."""
        if not path or not Path(path).exists():
            logger.info("No SimCLR weights at %s; using current init.", path)
            return False
        try:
            state = torch.load(path, map_location="cpu")
            if "encoder_state_dict" in state:
                state = state["encoder_state_dict"]
            missing, unexpected = self.load_state_dict(state, strict=False)
            logger.info("SimCLR weights loaded from %s (missing=%d, unexpected=%d)",
                        path, len(missing), len(unexpected))
            return True
        except Exception as exc:
            logger.warning("Failed to load SimCLR weights: %s", exc)
            return False


def build_imaging_encoder(cfg) -> SwinCNNHybridEncoder:
    """Build the imaging encoder from a config object."""
    enc_cfg = cfg.model.imaging_encoder
    encoder = SwinCNNHybridEncoder(
        swin_name=enc_cfg.swin_name,
        cnn_stem=enc_cfg.cnn_stem,
        pretrained=enc_cfg.pretrained,
        embed_dim=enc_cfg.embed_dim,
        image_size=cfg.data.image_size,
    )
    if enc_cfg.simclr_weights:
        encoder.load_simclr_weights(enc_cfg.simclr_weights)
    return encoder