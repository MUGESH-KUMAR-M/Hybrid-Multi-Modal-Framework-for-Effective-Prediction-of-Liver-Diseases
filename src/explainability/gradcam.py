"""
Grad-CAM explainability for the imaging pathway of CMCHT-XAI.

Generates localization heatmaps over liver images showing which regions drove
the prediction. Uses Captum's LayerGradCam if available, with a manual
gradient-based fallback so the pipeline always runs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.logger import get_logger

logger = get_logger(__name__)


class GradCAM:
    """Grad-CAM for the imaging encoder of CMCHT-XAI."""

    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.model.eval()

        # Find a suitable target layer in the imaging encoder
        if target_layer is None:
            target_layer = self._find_target_layer()
        self.target_layer = target_layer

        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None
        self._hooks = []

        if self.target_layer is not None:
            self._register_hooks()

    def _find_target_layer(self) -> Optional[torch.nn.Module]:
        """Find the last convolutional layer of the imaging encoder."""
        encoder = self.model.imaging_encoder
        # Prefer the CNN stem's last conv layer (good spatial resolution)
        last_conv = None
        for module in encoder.cnn_stem.modules():
            if isinstance(module, (torch.nn.Conv2d,)):
                last_conv = module
        if last_conv is not None:
            return last_conv
        # Fallback: any conv in the whole model
        for module in self.model.modules():
            if isinstance(module, torch.nn.Conv2d):
                last_conv = module
        return last_conv

    def _register_hooks(self) -> None:
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self._hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self._hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        task: str = "detection",
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for a single image.

        Args:
            image: (1, 3, H, W)
            tabular: (1, n_features)
            task: "detection" | "staging" | "severity"
            target_class: for staging, which class to explain.

        Returns:
            cam: (H, W) numpy array in [0, 1].
        """
        if self.target_layer is None:
            logger.warning("No target layer found for Grad-CAM; returning zeros.")
            return np.zeros((image.size(-2), image.size(-1)), dtype=np.float32)

        self.model.eval()
        image = image.clone().requires_grad_(True)

        # Forward pass
        out = self.model(image, tabular)

        # Select the target score
        if task == "detection":
            score = out["detection_logits"].sum()
        elif task == "staging":
            if target_class is None:
                target_class = out["staging_logits"].argmax(dim=-1).item()
            score = out["staging_logits"][0, target_class]
        else:
            score = out["severity_pred"].sum()

        # Backward pass
        self.model.zero_grad()
        score.backward(retain_graph=False)

        if self.gradients is None or self.activations is None:
            logger.warning("Grad-CAM gradients/activations not captured; returning zeros.")
            return np.zeros((image.size(-2), image.size(-1)), dtype=np.float32)

        # Compute CAM++
        grad2 = self.gradients ** 2
        grad3 = self.gradients ** 3
        
        # Correct (keepdims=True — shape becomes [B, C, 1, 1] in the spatial sum):
        denom = 2 * grad2 + (self.activations * grad3).sum(dim=(2, 3), keepdim=True) + 1e-7
        alpha = grad2 / denom
        weights = (alpha * F.relu(self.gradients)).sum(dim=(2, 3), keepdim=True)
        
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(image.size(-2), image.size(-1)),
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().detach().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def remove_hooks(self) -> None:
        for hook in self._hooks:
            hook.remove()
        self._hooks = []


def overlay_cam_on_image(image: np.ndarray, cam: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay a Grad-CAM heatmap on an image. Returns RGB uint8 array."""
    try:
        import cv2

        if image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        image = ((image - image.min()) / (image.max() - image.min() + 1e-8) * 255).astype(np.uint8)

        heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
        return overlay
    except Exception:
        # Fallback without cv2
        if image.ndim == 3 and image.shape[0] in (1, 3):
            image = np.transpose(image, (1, 2, 0))
        return ((image * 0.5 + cam[..., None] * 0.5) * 255).astype(np.uint8)


def generate_gradcam_for_sample(
    model: torch.nn.Module,
    image: torch.Tensor,
    tabular: torch.Tensor,
    task: str = "detection",
    save_path: Optional[str] = None,
) -> np.ndarray:
    """Convenience: generate + optionally save a Grad-CAM overlay for one sample."""
    gradcam = GradCAM(model)
    try:
        cam = gradcam.generate(image, tabular, task=task)
        img_np = image.squeeze(0).detach().cpu().numpy()
        overlay = overlay_cam_on_image(img_np, cam)
        if save_path:
            try:
                import matplotlib.pyplot as plt
                plt.imsave(save_path, overlay)
                logger.info("Grad-CAM overlay saved to %s", save_path)
            except Exception as exc:
                logger.warning("Could not save Grad-CAM image: %s", exc)
        return cam
    finally:
        gradcam.remove_hooks()