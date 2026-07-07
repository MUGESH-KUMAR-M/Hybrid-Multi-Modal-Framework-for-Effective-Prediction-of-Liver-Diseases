"""
SHAP explainability for the tabular pathway of CMCHT-XAI.

Uses SHAP (DeepExplainer / KernelExplainer) on the FT-Transformer tabular
encoder to generate global and local feature importance plots showing which
clinical features (bilirubin, albumin, SGOT, ...) drive each prediction.

Falls back to a gradient-based attribution if the `shap` library is unavailable,
so the pipeline always runs.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def compute_shap_values(
    model: torch.nn.Module,
    background: torch.Tensor,
    test_samples: torch.Tensor,
    feature_names: List[str],
    task: str = "detection",
    n_background: int = 100,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute SHAP values for the tabular pathway.

    Args:
        model: the full CMCHT-XAI model (eval mode).
        background: (N, n_features) reference tabular data for the explainer.
        test_samples: (M, n_features) samples to explain.
        feature_names: names of the tabular features.
        task: which task output to explain ("detection" | "staging" | "severity").
        n_background: number of background samples to use.

    Returns:
        shap_values: (M, n_features) array of SHAP values.
        feature_names: the feature names (for plotting).
    """
    model.eval()
    # subsample background
    if background.size(0) > n_background:
        idx = torch.randperm(background.size(0))[:n_background]
        background = background[idx]

    try:
        import shap

        # TabularOnlyWrapper that isolates the forward pass of the model
        class TabularOnlyWrapper(torch.nn.Module):
            def __init__(self, full_model: torch.nn.Module, task: str):
                super().__init__()
                self.full_model = full_model
                self.task = task
                
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                B = x.size(0)
                # Create a dummy image tensor on the same device
                dummy_image = torch.zeros(B, 3, 224, 224, device=x.device)
                out = self.full_model(dummy_image, x)
                
                if self.task == "detection":
                    return out["detection_logits"]
                elif self.task == "staging":
                    return out["staging_logits"]
                else:
                    return out["severity_pred"]

        wrapper = TabularOnlyWrapper(model, task)
        wrapper.eval()

        # Try GradientExplainer first (works well for differentiable models, safe for Transformers)
        try:
            # GradientExplainer expects a model and background data
            explainer = shap.GradientExplainer(wrapper, background)
            shap_values = explainer.shap_values(test_samples)
            
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.array(shap_values)
            logger.info("SHAP values computed via GradientExplainer: shape %s", shap_values.shape)
            return shap_values, feature_names
        except Exception as exc:
            logger.warning("GradientExplainer failed (%s); falling back to KernelExplainer.", exc)

        # Fallback: KernelExplainer
        def predict_fn_numpy(x_np: np.ndarray) -> np.ndarray:
            x_tensor = torch.tensor(x_np, dtype=torch.float32, device=background.device)
            with torch.no_grad():
                return wrapper(x_tensor).cpu().numpy()
                
        bg_np = background.cpu().numpy()
        test_np = test_samples.cpu().numpy()
        explainer = shap.KernelExplainer(predict_fn_numpy, bg_np)
        shap_values = explainer.shap_values(test_np, nsamples=100)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        shap_values = np.array(shap_values)
        logger.info("SHAP values computed via KernelExplainer: shape %s", shap_values.shape)
        return shap_values, feature_names

    except ImportError:
        logger.warning("shap library unavailable; using gradient-based attribution fallback.")
        return _gradient_attribution(model, test_samples, feature_names, task), feature_names


def _gradient_attribution(
    model: torch.nn.Module,
    samples: torch.Tensor,
    feature_names: List[str],
    task: str,
) -> np.ndarray:
    """
    Fallback: integrated-gradients-style attribution via input gradients.
    Uses Captum if available, else a simple input-gradient saliency.
    """
    samples = samples.clone().requires_grad_(True)
    B = samples.size(0)
    dummy_image = torch.zeros(B, 3, 224, 224, device=samples.device)

    try:
        from captum.attr import IntegratedGradients

        def forward_fn(x):
            out = model(dummy_image, x)
            if task == "detection":
                return out["detection_logits"].squeeze(-1)
            elif task == "staging":
                return out["staging_logits"]
            else:
                return out["severity_pred"].squeeze(-1)

        ig = IntegratedGradients(forward_fn)
        target = 0 if task != "staging" else 0
        attr = ig.attribute(samples, target=target)
        return attr.detach().numpy()
    except Exception:
        # Simple input-gradient saliency
        out = model(dummy_image, samples)
        if task == "detection":
            score = out["detection_logits"].sum()
        elif task == "staging":
            score = out["staging_logits"][:, 0].sum()
        else:
            score = out["severity_pred"].sum()
        grad = torch.autograd.grad(score, samples)[0]
        return grad.detach().numpy()


def plot_shap_summary(
    shap_values: np.ndarray,
    test_samples: np.ndarray,
    feature_names: List[str],
    save_path: Optional[str] = None,
) -> None:
    """Generate and optionally save a SHAP summary plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # force non-interactive backend before any pyplot import
        import matplotlib.pyplot as plt
        import shap

        # GradientExplainer may return (N, F, 1) for single-output tasks;
        # summary_plot expects (N, F) — squeeze any trailing singleton dim.
        sv = np.array(shap_values)
        if sv.ndim == 3 and sv.shape[-1] == 1:
            sv = sv[:, :, 0]
        ts = np.array(test_samples)
        if ts.ndim == 3 and ts.shape[-1] == 1:
            ts = ts[:, :, 0]

        shap.summary_plot(sv, ts, feature_names=feature_names, show=False)
        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=150)
            logger.info("SHAP summary plot saved to %s", save_path)
        plt.close()
    except Exception as exc:
        logger.warning("Could not generate SHAP plot: %s", exc)