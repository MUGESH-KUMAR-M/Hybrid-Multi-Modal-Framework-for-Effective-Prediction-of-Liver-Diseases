"""
Contribution 2 — CUSP-Cascade (Cascaded Uncertainty-and-Sensitivity Propagation).

Location: src/models/cascade_heads.py

Mechanism:
    Replaces three independent task heads with a clinically-ordered cascade
    (detect -> stage -> assess severity). Each head passes forward its own
    epistemic uncertainty and the counterfactual sensitivity vector from
    CSG-Fusion to the next head, so e.g. the staging head knows how confident the
    detection step was before making its own call.

Literature checked:
    - Clinically-ordered cascaded multi-task heads exist already (bone disease,
      GI lesions, COVID/pneumonia, dementia staging).
    - Uncertainty propagation through cascaded medical pipelines exists already
      (MRI reconstruction cascades).
    - Jointly propagating BOTH uncertainty and counterfactual sensitivity through
      a clinically-ordered cascade built on a gated fusion layer was not found in
      either search pass.

Known risk: cascade error propagation if the detection head is wrong early in
training. Addressed directly by Contribution 3 (CGCT).

Weakness 2 fix (Section 7): training-time uncertainty now averages 3 stochastic
passes, matching the inference-time approach (previously a single pair of passes
with ~1 degree of freedom).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class UncertaintyAwareHead(nn.Module):
    """
    A task head that estimates its own epistemic uncertainty via Monte Carlo
    Dropout and can receive propagated uncertainty + sensitivity from an upstream
    head.

    The head's input is augmented with:
        - the upstream prediction (scalar or one-hot),
        - the upstream uncertainty (scalar),
        - the sensitivity vector (from CSG-Fusion).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        task_type: str = "binary",          # binary | multiclass | regression
        upstream_pred_dim: int = 1,         # dim of the upstream prediction signal
        sensitivity_dim: int = 10,          # n_features
        mc_passes_train: int = 3,
        mc_passes_eval: int = 10,
    ):
        super().__init__()
        self.task_type = task_type
        self.out_dim = out_dim
        self.mc_passes_train = mc_passes_train
        self.mc_passes_eval = mc_passes_eval

        # Augmented input: fused emb + upstream pred + upstream unc + sensitivity
        aug_dim = in_dim + upstream_pred_dim + 1 + sensitivity_dim
        self.input_proj = nn.Sequential(
            nn.Linear(aug_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, out_dim)
        # Dropout layer used for MC-Dropout uncertainty estimation
        self.mc_dropout = nn.Dropout(dropout)

    def forward(
        self,
        fused: torch.Tensor,
        upstream_pred: Optional[torch.Tensor] = None,
        upstream_uncertainty: Optional[torch.Tensor] = None,
        sensitivity: Optional[torch.Tensor] = None,
        return_uncertainty: bool = False,
        n_mc_passes: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns (output, uncertainty).
        - binary:     output = logits (B, 1), uncertainty (B, 1)
        - multiclass: output = logits (B, C), uncertainty (B, 1)
        - regression: output = pred (B, 1), uncertainty (B, 1)
        """
        B = fused.size(0)
        device = fused.device

        if upstream_pred is None:
            upstream_pred = torch.zeros(B, 1, device=device)
        if upstream_uncertainty is None:
            upstream_uncertainty = torch.zeros(B, 1, device=device)
        if sensitivity is None:
            sensitivity = torch.zeros(B, 1, device=device)
        # Ensure 2D
        if upstream_pred.dim() == 1:
            upstream_pred = upstream_pred.unsqueeze(-1)
        if upstream_uncertainty.dim() == 1:
            upstream_uncertainty = upstream_uncertainty.unsqueeze(-1)
        if sensitivity.dim() == 1:
            sensitivity = sensitivity.unsqueeze(-1)

        aug = torch.cat([fused, upstream_pred, upstream_uncertainty, sensitivity], dim=-1)
        h = self.input_proj(aug)

        if return_uncertainty:
            n = n_mc_passes or (self.mc_passes_train if self.training else self.mc_passes_eval)
            outs = []
            for _ in range(n):
                outs.append(self.head(self.mc_dropout(h)))
            outs = torch.stack(outs, dim=0)        # (n, B, out_dim)
            mean_out = outs.mean(dim=0)
            # Epistemic uncertainty: variance of the output, summarized per sample
            variance = outs.var(dim=0)             # (B, out_dim)
            uncertainty = variance.mean(dim=-1, keepdim=True)  # (B, 1)
            return mean_out, uncertainty
        else:
            out = self.head(h)
            return out, None


class CUSPCascade(nn.Module):
    """
    Contribution 2: Cascaded Uncertainty-and-Sensitivity Propagation.

    Clinically-ordered cascade: detect -> stage -> severity.

    Each stage receives:
        - the fused embedding,
        - the upstream head's prediction (as a conditioning signal),
        - the upstream head's epistemic uncertainty,
        - the counterfactual sensitivity vector from CSG-Fusion.

    During training, CGCT (Contribution 3) decides whether the upstream
    prediction passed downstream is the model's own prediction or the ground
    truth; that mixing is controlled via the `teacher_forcing_mask` argument.
    """

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 4,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        sensitivity_dim: int = 10,
        mc_passes_train: int = 3,
        mc_passes_eval: int = 10,
        propagate_uncertainty: bool = True,
        propagate_sensitivity: bool = True,
    ):
        super().__init__()
        self.propagate_uncertainty = propagate_uncertainty
        self.propagate_sensitivity = propagate_sensitivity

        # Stage 1: Detection (binary)
        self.detection = UncertaintyAwareHead(
            in_dim=in_dim, out_dim=1, hidden_dim=hidden_dim, dropout=dropout,
            task_type="binary", upstream_pred_dim=1,
            sensitivity_dim=sensitivity_dim,
            mc_passes_train=mc_passes_train, mc_passes_eval=mc_passes_eval,
        )
        # Stage 2: Staging (multi-class), conditioned on detection
        self.staging = UncertaintyAwareHead(
            in_dim=in_dim, out_dim=num_classes, hidden_dim=hidden_dim, dropout=dropout,
            task_type="multiclass", upstream_pred_dim=1,
            sensitivity_dim=sensitivity_dim,
            mc_passes_train=mc_passes_train, mc_passes_eval=mc_passes_eval,
        )
        # Stage 3: Severity (regression), conditioned on staging
        self.severity = UncertaintyAwareHead(
            in_dim=in_dim, out_dim=1, hidden_dim=hidden_dim, dropout=dropout,
            task_type="regression", upstream_pred_dim=num_classes,
            sensitivity_dim=sensitivity_dim,
            mc_passes_train=mc_passes_train, mc_passes_eval=mc_passes_eval,
        )

    def forward(
        self,
        fused: torch.Tensor,
        sensitivity: Optional[torch.Tensor] = None,
        detection_gt: Optional[torch.Tensor] = None,
        staging_gt: Optional[torch.Tensor] = None,
        self_prob_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_uncertainty: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            fused: (B, in_dim)
            sensitivity: (B, n_features) from CSG-Fusion
            detection_gt / staging_gt: ground-truth labels for CGCT teacher forcing
            self_prob_dict: dict with 'detection' and 'staging' soft weights (B, 1);
                High self_prob => use model's own prediction.
            return_uncertainty: if True, estimate per-stage epistemic uncertainty.
        """
        B = fused.size(0)
        device = fused.device
        if sensitivity is None:
            sensitivity = torch.zeros(B, 10, device=device)

        from src.training.confidence_gated_training import mix_with_ground_truth

        # ---- Stage 1: Detection --------------------------------------------
        det_logits, det_unc = self.detection(
            fused, upstream_pred=None, upstream_uncertainty=None,
            sensitivity=sensitivity if self.propagate_sensitivity else None,
            return_uncertainty=return_uncertainty,
        )

        det_prob = torch.sigmoid(det_logits)
        if detection_gt is not None and self_prob_dict is not None:
            self_prob = self_prob_dict.get("detection", torch.ones(B, 1, device=device))
            det_signal = mix_with_ground_truth(det_prob, detection_gt, self_prob)
        else:
            det_signal = det_prob

        det_unc_signal = det_unc if (return_uncertainty and self.propagate_uncertainty) else None

        # ---- Stage 2: Staging ----------------------------------------------
        stage_logits, stage_unc = self.staging(
            fused, upstream_pred=det_signal, upstream_uncertainty=det_unc_signal,
            sensitivity=sensitivity if self.propagate_sensitivity else None,
            return_uncertainty=return_uncertainty,
        )

        stage_prob = F.softmax(stage_logits, dim=-1)
        if staging_gt is not None and self_prob_dict is not None:
            self_prob = self_prob_dict.get("staging", torch.ones(B, 1, device=device))
            gt_onehot = F.one_hot(staging_gt.long(), num_classes=stage_prob.size(-1)).float()
            stage_signal = mix_with_ground_truth(stage_prob, gt_onehot, self_prob)
        else:
            stage_signal = stage_prob

        stage_unc_signal = stage_unc if (return_uncertainty and self.propagate_uncertainty) else None

        # ---- Stage 3: Severity ---------------------------------------------
        sev_pred, sev_unc = self.severity(
            fused, upstream_pred=stage_signal, upstream_uncertainty=stage_unc_signal,
            sensitivity=sensitivity if self.propagate_sensitivity else None,
            return_uncertainty=return_uncertainty,
        )

        out = {
            "detection_logits": det_logits,
            "staging_logits": stage_logits,
            "severity_pred": sev_pred,
        }
        if return_uncertainty:
            out["detection_uncertainty"] = det_unc
            out["staging_uncertainty"] = stage_unc
            out["severity_uncertainty"] = sev_unc
        return out


def build_cusp_cascade(cfg) -> CUSPCascade:
    """Build CUSP-Cascade from config."""
    hcfg = cfg.model.heads
    cusp = hcfg.cusp
    in_dim = cfg.model.fusion.embed_dim * 2  # fused output is 2*embed_dim
    return CUSPCascade(
        in_dim=in_dim,
        num_classes=hcfg.staging.num_classes,
        hidden_dim=hcfg.detection.hidden_dim,
        dropout=hcfg.detection.dropout,
        sensitivity_dim=cfg.data.num_tabular_features,
        mc_passes_train=cusp.mc_passes_train,
        mc_passes_eval=cusp.mc_passes_eval,
        propagate_uncertainty=cusp.propagate_uncertainty,
        propagate_sensitivity=cusp.propagate_sensitivity,
    )