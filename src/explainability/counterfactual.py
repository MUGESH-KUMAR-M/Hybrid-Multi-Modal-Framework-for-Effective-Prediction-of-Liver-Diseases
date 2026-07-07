"""
Counterfactual explainability for CMCHT-XAI.

Generates actionable counterfactual explanations: "If the patient's bilirubin
were X instead of Y, the model would predict healthy." This is the key
differentiator from SHAP (which only says "bilirubin is important") —
counterfactuals tell the clinician what to change to improve the outcome.

Uses DiCE (Mothilal et al. 2020) when available, with a gradient-based
counterfactual fallback so the pipeline always runs.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def generate_counterfactuals(
    model: torch.nn.Module,
    query_instance: np.ndarray,
    feature_names: List[str],
    desired_class: int = 0,
    num_cf: int = 3,
    method: str = "dice",
    background_df=None,
    save_path: Optional[str] = None,
) -> List[Dict]:
    """
    Generate counterfactual explanations for a single patient.

    Args:
        model: the CMCHT-XAI model (eval mode).
        query_instance: (n_features,) numpy array — the patient's tabular features.
        feature_names: names of the features.
        desired_class: target prediction (0 = healthy, 1 = diseased).
        num_cf: number of counterfactuals to generate.
        method: "dice" | "gradient".
        background_df: optional pandas DataFrame of training data for DiCE.

    Returns:
        List of counterfactual dicts, each with 'features' and 'changes'.
    """
    model.eval()

    if method == "dice" and background_df is not None:
        try:
            return _dice_counterfactuals(
                model, query_instance, feature_names, desired_class, num_cf, background_df
            )
        except Exception as exc:
            logger.warning("DiCE counterfactual generation failed (%s); using gradient fallback.", exc)

    return _gradient_counterfactuals(
        model, query_instance, feature_names, desired_class, num_cf
    )


def _dice_counterfactuals(
    model, query_instance, feature_names, desired_class, num_cf, background_df
) -> List[Dict]:
    """Generate counterfactuals using the DiCE library."""
    import dice_ml
    import pandas as pd

    # Build a DiCE data object
    df = background_df.copy()
    # Ensure the outcome column exists
    if "detection_label" not in df.columns:
        raise ValueError("background_df must contain 'detection_label' column")

    d = dice_ml.Data(
        dataframe=df,
        continuous_features=feature_names,
        outcome_name="detection_label",
    )

    # Wrapper model for DiCE
    class _ModelWrapper:
        def __init__(self, model, feature_names):
            self.model = model
            self.feature_names = feature_names

        def predict(self, x):
            if isinstance(x, pd.DataFrame):
                x = x[self.feature_names].values
            x_t = torch.tensor(x, dtype=torch.float32)
            with torch.no_grad():
                B = x_t.size(0)
                dummy_img = torch.zeros(B, 3, 224, 224)
                out = self.model(dummy_img, x_t)
                prob = torch.sigmoid(out["detection_logits"]).numpy()
            return np.hstack([1 - prob, prob])

        def predict_proba(self, x):
            return self.predict(x)

    backend = dice_ml.Model(model=_ModelWrapper(model, feature_names), backend="sklearn")
    exp = dice_ml.Dice(d, backend)

    query_df = pd.DataFrame([query_instance], columns=feature_names)
    cfs = exp.generate_counterfactuals(
        query_df, total_CFs=num_cf, desired_class=desired_class
    )
    cf_df = cfs.cf_examples_list[0].final_cfs_df
    results = []
    for _, row in cf_df.iterrows():
        cf_features = row[feature_names].values.astype(float)
        changes = {}
        for i, fname in enumerate(feature_names):
            orig = float(query_instance[i])
            cf_val = float(cf_features[i])
            if abs(orig - cf_val) > 1e-4:
                changes[fname] = {"original": orig, "counterfactual": cf_val}
        results.append({"features": cf_features, "changes": changes})
    return results


def _gradient_counterfactuals(
    model, query_instance, feature_names, desired_class, num_cf
) -> List[Dict]:
    """
    Gradient-based counterfactual fallback.

    Optimizes a perturbation of the input features to flip the prediction toward
    `desired_class`, minimizing the L1 distance from the original input.
    """
    device = next(model.parameters()).device
    x_orig = torch.tensor(query_instance, dtype=torch.float32, device=device).unsqueeze(0)
    dummy_img = torch.zeros(1, 3, 224, 224, device=device)

    results = []
    for k in range(num_cf):
        # Initialize perturbation with different random seeds for diversity
        torch.manual_seed(42 + k)
        delta = torch.randn_like(x_orig) * 0.1
        delta.requires_grad_(True)

        optimizer = torch.optim.Adam([delta], lr=0.05)
        target = float(desired_class)

        for step in range(200):
            x_pert = x_orig + delta
            out = model(dummy_img, x_pert)
            prob = torch.sigmoid(out["detection_logits"])

            # Loss: move prediction toward target + L1 sparsity
            bce = torch.nn.functional.binary_cross_entropy_with_logits(
                out["detection_logits"], torch.tensor([[target]], device=device)
            )
            l1 = delta.abs().mean()
            loss = bce + 0.1 * l1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Check if prediction flipped
            with torch.no_grad():
                if (prob.item() >= 0.5) == (target >= 0.5):
                    break

        with torch.no_grad():
            cf_features = (x_orig + delta).squeeze(0).cpu().numpy()
            changes = {}
            for i, fname in enumerate(feature_names):
                orig = float(query_instance[i])
                cf_val = float(cf_features[i])
                if abs(orig - cf_val) > 1e-4:
                    changes[fname] = {"original": orig, "counterfactual": cf_val}
            results.append({"features": cf_features, "changes": changes})

    return results


def format_counterfactual_text(cf: Dict, feature_names: List[str]) -> str:
    """Format a counterfactual dict into a human-readable string."""
    if not cf["changes"]:
        return "No changes needed."
    parts = []
    for fname, vals in cf["changes"].items():
        parts.append(f"change {fname} from {vals['original']:.2f} to {vals['counterfactual']:.2f}")
    return "If you " + " and ".join(parts) + ", the prediction would flip."