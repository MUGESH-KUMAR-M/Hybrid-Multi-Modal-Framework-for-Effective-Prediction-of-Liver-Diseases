"""
Monte Carlo Dropout uncertainty quantification for CMCHT-XAI.

Applies MC-Dropout at inference (N stochastic forward passes) and computes:
    - mean prediction across passes,
    - epistemic uncertainty (variance) per sample,
    - a 'needs_review' flag for cases above the uncertainty threshold.

This is the clinical safety layer: in healthcare, saying "I don't know" is
better than a wrong answer.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


def enable_mc_dropout(model: nn.Module) -> None:
    """Set the model to eval mode but re-enable Dropout layers for MC sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()


def mc_dropout_predict(
    model: nn.Module,
    image: torch.Tensor,
    tabular: torch.Tensor,
    n_passes: int = 10,
    return_individual: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Run n_passes stochastic forward passes and return mean predictions +
    epistemic uncertainty.

    Returns dict with:
        - detection_prob (B, 1)        mean sigmoid prob
        - staging_probs (B, C)         mean softmax probs
        - severity_pred (B, 1)         mean regression output
        - detection_uncertainty (B, 1) variance of detection prob
        - staging_uncertainty (B, 1)   mean variance over classes
        - severity_uncertainty (B, 1)  variance of severity
        - needs_review (B, 1)          1 if detection_uncertainty > threshold
    """
    enable_mc_dropout(model)

    det_probs = []
    stage_probs = []
    sev_preds = []
    for _ in range(n_passes):
        with torch.no_grad():
            out = model(image, tabular, return_uncertainty=False)
        det_probs.append(torch.sigmoid(out["detection_logits"]))
        stage_probs.append(torch.softmax(out["staging_logits"], dim=-1))
        sev_preds.append(out["severity_pred"])

    det_probs = torch.stack(det_probs, dim=0)       # (n, B, 1)
    stage_probs = torch.stack(stage_probs, dim=0)   # (n, B, C)
    sev_preds = torch.stack(sev_preds, dim=0)       # (n, B, 1)

    mean_det = det_probs.mean(dim=0)
    mean_stage = stage_probs.mean(dim=0)
    mean_sev = sev_preds.mean(dim=0)

    det_unc = det_probs.var(dim=0)
    stage_unc = stage_probs.var(dim=0).mean(dim=-1, keepdim=True)
    sev_unc = sev_preds.var(dim=0)

    result = {
        "detection_prob": mean_det,
        "staging_probs": mean_stage,
        "staging_pred": mean_stage.argmax(dim=-1),
        "severity_pred": mean_sev,
        "detection_uncertainty": det_unc,
        "staging_uncertainty": stage_unc,
        "severity_uncertainty": sev_unc,
    }
    if return_individual:
        result["detection_individual"] = det_probs
        result["staging_individual"] = stage_probs
        result["severity_individual"] = sev_preds
    return result


def flag_uncertain_cases(
    uncertainty: torch.Tensor,
    threshold: float = 0.15,
) -> torch.Tensor:
    """Return a boolean mask of samples above the uncertainty threshold."""
    return (uncertainty.squeeze(-1) > threshold).float().unsqueeze(-1)


def calibration_metrics(
    y_true: np.ndarray,
    mean_probs: np.ndarray,
    uncertainties: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute uncertainty-quality metrics:
        - ECE (Expected Calibration Error)
        - Pearson correlation between uncertainty and error
        - AURC (Area Under Risk-Coverage curve, simplified)
    """
    from src.utils.metrics import expected_calibration_error

    y_true = np.asarray(y_true).ravel()
    mean_probs = np.asarray(mean_probs).ravel()
    uncertainties = np.asarray(uncertainties).ravel()

    ece = expected_calibration_error(y_true, mean_probs, n_bins=n_bins)

    errors = (y_true - mean_probs) ** 2
    if len(uncertainties) > 2 and uncertainties.std() > 0:
        corr = float(np.corrcoef(uncertainties, errors)[0, 1])
    else:
        corr = 0.0

    # AURC: sort by uncertainty, compute cumulative risk
    order = np.argsort(-uncertainties)
    sorted_errors = errors[order]
    cumulative_risk = np.cumsum(sorted_errors) / (np.arange(len(sorted_errors)) + 1)
    aurc = float(np.mean(cumulative_risk))

    return {"ece": ece, "uncertainty_error_correlation": corr, "aurc": aurc}