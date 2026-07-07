"""
Evaluation metrics for CMCHT-XAI multi-task liver disease prediction.

Covers:
    - Detection (binary): accuracy, AUC-ROC, sensitivity, specificity, F1
    - Staging (multi-class): macro-F1, Cohen's Kappa, accuracy
    - Severity (regression): MAE, RMSE
    - Uncertainty: Expected Calibration Error (ECE)
"""
from __future__ import annotations

from typing import Dict

import numpy as np


# ------------------------------------------------------------------------------
# Detection (binary classification)
# ------------------------------------------------------------------------------
def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    return float(np.mean(y_true == y_pred))


def auc_roc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """AUC-ROC via the trapezoidal rule on the ROC curve."""
    y_true = np.asarray(y_true).ravel()
    y_score = np.asarray(y_score).ravel()
    if len(np.unique(y_true)) < 2:
        return float("nan")

    desc_idx = np.argsort(-y_score)
    y_sorted = y_true[desc_idx]
    s_sorted = y_score[desc_idx]

    P = float(np.sum(y_sorted == 1))
    N = float(np.sum(y_sorted == 0))
    if P == 0 or N == 0:
        return float("nan")

    tpr_points = [0.0]
    fpr_points = [0.0]
    tp = 0.0
    fp = 0.0
    for i in range(len(y_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        tpr_points.append(tp / P)
        fpr_points.append(fp / N)
    tpr_points.append(1.0)
    fpr_points.append(1.0)
    _trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(_trapezoid(tpr_points, fpr_points))


def sensitivity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall for the positive class."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tn = float(np.sum((y_pred == 0) & (y_true == 0)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    return tn / (tn + fp) if (tn + fp) > 0 else 0.0


def f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    tp = float(np.sum((y_pred == 1) & (y_true == 1)))
    fp = float(np.sum((y_pred == 1) & (y_true == 0)))
    fn = float(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ------------------------------------------------------------------------------
# Staging (multi-class)
# ------------------------------------------------------------------------------
def macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    f1s = []
    for c in classes:
        tp = float(np.sum((y_pred == c) & (y_true == c)))
        fp = float(np.sum((y_pred == c) & (y_true != c)))
        fn = float(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1s.append((2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0)
    return float(np.mean(f1s)) if f1s else 0.0


def cohen_kappa(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    classes = np.unique(np.concatenate([y_true, y_pred]))
    n = len(y_true)
    po = float(np.mean(y_true == y_pred))

    pe = 0.0
    for c in classes:
        p_true = float(np.sum(y_true == c) / n)
        p_pred = float(np.sum(y_pred == c) / n)
        pe += p_true * p_pred
    return (po - pe) / (1.0 - pe) if (1.0 - pe) > 0 else 0.0


# ------------------------------------------------------------------------------
# Severity (regression)
# ------------------------------------------------------------------------------
def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


# ------------------------------------------------------------------------------
# Uncertainty / calibration
# ------------------------------------------------------------------------------
def expected_calibration_error(
    y_true: np.ndarray, confidences: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error for binary detection confidence."""
    y_true = np.asarray(y_true).ravel()
    confidences = np.asarray(confidences, dtype=float).ravel()
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(y_true[mask] == 1))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


# ------------------------------------------------------------------------------
# Aggregators
# ------------------------------------------------------------------------------
def detection_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """Compute the full detection metric set from probabilities."""
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    y_pred = (y_prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy(y_true, y_pred),
        "auc_roc": auc_roc(y_true, y_prob),
        "sensitivity": sensitivity(y_true, y_pred),
        "specificity": specificity(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def staging_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "f1_macro": macro_f1(y_true, y_pred),
        "cohen_kappa": cohen_kappa(y_true, y_pred),
        "accuracy": accuracy(y_true, y_pred),
    }


def severity_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }


def aggregate_metrics(
    det_true=None, det_prob=None,
    stage_true=None, stage_pred=None,
    sev_true=None, sev_pred=None,
    det_confidence=None,
) -> Dict[str, Dict[str, float]]:
    """Compute metrics for all three tasks at once."""
    out: Dict[str, Dict[str, float]] = {}
    if det_true is not None and det_prob is not None:
        out["detection"] = detection_metrics(det_true, det_prob)
        if det_confidence is not None:
            out["uncertainty"] = {
                "ece": expected_calibration_error(det_true, det_confidence)
            }
    if stage_true is not None and stage_pred is not None:
        out["staging"] = staging_metrics(stage_true, stage_pred)
    if sev_true is not None and sev_pred is not None:
        out["severity"] = severity_metrics(sev_true, sev_pred)
    return out