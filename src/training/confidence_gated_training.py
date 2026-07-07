"""
Contribution 3 — CGCT (Confidence-Gated Cascade Training).

Location: src/training/confidence_gated_training.py

Mechanism:
    A training algorithm (not an architecture) that decides, per sample, whether
    CUSP-Cascade conditions the next stage on its own upstream prediction or on
    ground truth — based on the upstream head's own uncertainty — with a decaying
    global schedule.

Honest lineage — state exactly this in the report:
    This is a direct adaptation of Confidence-Aware Scheduled Sampling
    (Liu et al., 2021, Neural Machine Translation), which already uses model
    confidence to choose between ground truth and self-prediction during training.
    That mechanism was built for autoregressive token-by-token decoding. What is
    new here is applying it to a non-autoregressive, heterogeneous multi-task
    clinical cascade (binary -> multi-class -> regression) instead of sequence
    decoding.

Built specifically to fix the cascade-error-propagation risk named in
Contribution 2 (CUSP-Cascade).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


def confidence_gated_mask(
    upstream_uncertainty: Optional[torch.Tensor],
    global_tf_rate: float,
    threshold: float = 0.15
) -> torch.Tensor:
    """
    Returns self_prob (probability weight toward the model's own prediction).
    High uncertainty -> low self_prob (lean on Ground Truth).
    Low uncertainty -> high self_prob (lean on Model Prediction).
    """
    if upstream_uncertainty is None:
        # no uncertainty signal -> just use the global schedule
        return torch.ones_like(upstream_uncertainty) * (1.0 - global_tf_rate) if upstream_uncertainty is not None else torch.tensor(1.0 - global_tf_rate)
    
    # Soft mixing:
    # self_prob decreases as uncertainty increases.
    unc = upstream_uncertainty.squeeze(-1)
    base_self_prob = 1.0 - global_tf_rate
    self_prob = base_self_prob * torch.clamp(1.0 - (unc / threshold), min=0.0)
    return self_prob.unsqueeze(-1)

def mix_with_ground_truth(
    predictions: torch.Tensor,
    ground_truth: torch.Tensor,
    self_prob: torch.Tensor
) -> torch.Tensor:
    """
    Safely blends predictions and ground truth using self_prob.
    predictions * self_prob + ground_truth * (1 - self_prob)
    """
    if ground_truth.dim() < predictions.dim():
        ground_truth = ground_truth.unsqueeze(-1)
    gt_float = ground_truth.to(dtype=predictions.dtype, device=predictions.device)
    
    return predictions * self_prob + gt_float * (1.0 - self_prob)


# ==============================================================================
# Multi-task loss with gradient blending + CSG consistency loss
# ==============================================================================
class MultiTaskLoss(nn.Module):
    """
    Combined loss for the three tasks + the CSG counterfactual consistency loss.

    Detection:  BCEWithLogitsLoss (+ optional label smoothing)
    Staging:    CrossEntropyLoss (+ optional label smoothing) / Focal
    Severity:   SmoothL1Loss (Huber)
    Consistency: CSG margin loss (only active when fusion_type == "csg")
    """

    def __init__(
        self,
        lambda_detection: float = 1.0,
        lambda_staging: float = 1.0,
        lambda_severity: float = 0.5,
        lambda_consistency: float = 0.1,
        label_smoothing: float = 0.1,
        use_focal: bool = True,
        staging_gamma: float = 2.0,
    ):
        super().__init__()
        self.lambda_detection = lambda_detection
        self.lambda_staging = lambda_staging
        self.lambda_severity = lambda_severity
        self.lambda_consistency = lambda_consistency
        self.label_smoothing = label_smoothing
        self.use_focal = use_focal
        self.gamma = staging_gamma

        self.bce = nn.BCEWithLogitsLoss()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.huber = nn.SmoothL1Loss()

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Focal loss for class-imbalanced staging."""
        ce = nn.functional.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        detection_label: torch.Tensor,
        staging_label: torch.Tensor,
        severity_label: torch.Tensor,
        consistency_loss: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        det_loss = self.bce(
            outputs["detection_logits"].squeeze(-1),
            detection_label.squeeze(-1) if detection_label.dim() > 1 else detection_label,
        )
        if self.use_focal:
            stage_loss = self.focal_loss(outputs["staging_logits"], staging_label.long())
        else:
            stage_loss = self.ce(outputs["staging_logits"], staging_label.long())
        sev_loss = self.huber(
            outputs["severity_pred"].squeeze(-1),
            severity_label.squeeze(-1) if severity_label.dim() > 1 else severity_label,
        )

        total = (
            self.lambda_detection * det_loss
            + self.lambda_staging * stage_loss
            + self.lambda_severity * sev_loss
        )
        if consistency_loss is not None and self.lambda_consistency > 0:
            total = total + self.lambda_consistency * consistency_loss

        return {
            "total": total,
            "detection": det_loss,
            "staging": stage_loss,
            "severity": sev_loss,
            "consistency": consistency_loss if consistency_loss is not None else torch.tensor(0.0),
        }


def build_loss(cfg) -> MultiTaskLoss:
    """Build the multi-task loss from config."""
    tcfg = cfg.training
    return MultiTaskLoss(
        lambda_detection=tcfg.lambda_detection,
        lambda_staging=tcfg.lambda_staging,
        lambda_severity=tcfg.lambda_severity,
        lambda_consistency=tcfg.lambda_consistency,
        label_smoothing=tcfg.label_smoothing,
    )