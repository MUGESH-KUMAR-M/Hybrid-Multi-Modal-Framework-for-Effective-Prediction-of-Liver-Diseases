"""
Baseline independent multi-task heads for CMCHT-XAI (ablation baseline).

Three independent heads operating on the fused embedding:
    - Detection head: binary classification (sigmoid + BCE)
    - Staging head:   multi-class classification (softmax + Focal/CrossEntropy)
    - Staging head:   MELD severity regression (linear + Huber)

This is the ablation baseline against which CUSP-Cascade (Contribution 2) is
compared. The heads do NOT propagate uncertainty or sensitivity between stages.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class DetectionHead(nn.Module):
    """Binary disease detection head."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, 1) — apply sigmoid for probability."""
        return self.net(x)


class StagingHead(nn.Module):
    """Multi-class fibrosis staging head."""

    def __init__(self, in_dim: int, num_classes: int = 4,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits (B, num_classes)."""
        return self.net(x)


class SeverityHead(nn.Module):
    """MELD severity score regression head."""

    def __init__(self, in_dim: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns predicted severity score (B, 1)."""
        return self.net(x)


class IndependentMultiTaskHeads(nn.Module):
    """
    Baseline: three independent task heads sharing the fused embedding.

    No cascade, no uncertainty propagation, no sensitivity propagation.
    """

    def __init__(self, in_dim: int, num_classes: int = 4,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.detection = DetectionHead(in_dim, hidden_dim, dropout)
        self.staging = StagingHead(in_dim, num_classes, hidden_dim, dropout)
        self.severity = SeverityHead(in_dim, hidden_dim, dropout)

    def forward(self, fused: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {
            "detection_logits": self.detection(fused),
            "staging_logits": self.staging(fused),
            "severity_pred": self.severity(fused),
        }


def build_independent_heads(cfg) -> IndependentMultiTaskHeads:
    """Build the baseline independent heads from config."""
    hcfg = cfg.model.heads
    in_dim = cfg.model.fusion.embed_dim * 2  # fused output is 2*embed_dim
    return IndependentMultiTaskHeads(
        in_dim=in_dim,
        num_classes=hcfg.staging.num_classes,
        hidden_dim=hcfg.detection.hidden_dim,
        dropout=hcfg.detection.dropout,
    )