"""
Full CMCHT-XAI model.

Wires together:
    - Imaging encoder (Swin-CNN hybrid, SimCLR pre-trained)
    - Tabular encoder (FT-Transformer)
    - Fusion (baseline cross-attention OR CSG-Fusion, per ablation toggle)
    - Heads (independent baseline OR CUSP-Cascade, per ablation toggle)

Ablation toggles (from config):
    - fusion_type:  "cross_attention" (baseline) | "csg" (Contribution 1)
    - head_type:    "independent" (baseline) | "cusp_cascade" (Contribution 2)
    - use_cgct:     bool (Contribution 3, applied during training)

Weakness 1 fix (Section 7): _quick_mc_uncertainty is implemented (forces dropout
active, runs N stochastic passes, returns per-sample variance) instead of raising
NotImplementedError, so the default "csg" fusion path runs end-to-end.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from src.models.csg_fusion import CSGFusion, build_csg_fusion
from src.models.cascade_heads import CUSPCascade, build_cusp_cascade
from src.models.fusion import CrossAttentionFusion, build_fusion
from src.models.heads import IndependentMultiTaskHeads, build_independent_heads
from src.models.imaging_encoder import SwinCNNHybridEncoder, build_imaging_encoder
from src.models.tabular_encoder import FTTransformerEncoder, build_tabular_encoder
from src.utils.logger import get_logger

logger = get_logger(__name__)


class CMCHTXAIModel(nn.Module):
    """Full Cross-Modal Contrastive Hybrid Transformer model."""

    def __init__(
        self,
        imaging_encoder: SwinCNNHybridEncoder,
        tabular_encoder: FTTransformerEncoder,
        fusion: nn.Module,
        heads: nn.Module,
        fusion_type: str = "csg",
        head_type: str = "cusp_cascade",
        use_cgct: bool = True,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.imaging_encoder = imaging_encoder
        self.tabular_encoder = tabular_encoder
        self.fusion = fusion
        self.heads = heads
        self.fusion_type = fusion_type
        self.head_type = head_type
        self.use_cgct = use_cgct
        self.embed_dim = embed_dim

    def forward(
        self,
        image: torch.Tensor,
        tabular: torch.Tensor,
        detection_gt: Optional[torch.Tensor] = None,
        staging_gt: Optional[torch.Tensor] = None,
        self_prob_dict: Optional[Dict[str, torch.Tensor]] = None,
        return_uncertainty: bool = False,
        return_embeddings: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        image: (B, 3, H, W)
        tabular: (B, n_features)

        Returns dict with task outputs (+ uncertainty + embeddings if requested).
        """
        # --- Encoders --------------------------------------------------------
        image_emb = self.imaging_encoder(image)        # (B, embed_dim)
        tabular_emb = self.tabular_encoder(tabular)    # (B, embed_dim)

        # --- Fusion ----------------------------------------------------------
        sensitivity = None
        if self.fusion_type == "csg":
            fused, sensitivity = self.fusion(
                image_emb=image_emb,
                tabular_emb=tabular_emb,
                tabular_raw=tabular,
                imaging_encoder=self.imaging_encoder,
                tabular_encoder=self.tabular_encoder,
                training=self.training,
                compute_uncertainty=return_uncertainty,
            )
        else:
            # baseline cross-attention fusion
            fused = self.fusion(image_emb, tabular_emb)
            sensitivity = torch.zeros(image_emb.size(0), tabular.size(1), device=image_emb.device)

        # --- Heads -----------------------------------------------------------
        if self.head_type == "cusp_cascade":
            out = self.heads(
                fused=fused,
                sensitivity=sensitivity,
                detection_gt=detection_gt,
                staging_gt=staging_gt,
                self_prob_dict=self_prob_dict,
                return_uncertainty=return_uncertainty,
            )
        else:
            out = self.heads(fused)

        if return_embeddings:
            out["image_embedding"] = image_emb
            out["tabular_embedding"] = tabular_emb
            out["fused_embedding"] = fused
            out["sensitivity"] = sensitivity
        return out

    # --------------------------------------------------------------------------
    # Weakness 1 fix: implemented MC-Dropout uncertainty for the full model
    # --------------------------------------------------------------------------
    def _quick_mc_uncertainty(
        self, image: torch.Tensor, tabular: torch.Tensor, n_passes: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Run n stochastic forward passes (dropout active) and return per-sample
        variance for each task output. Used for inference-time uncertainty
        flagging of low-confidence cases.

        Previously this raised NotImplementedError, which broke the default "csg"
        fusion path end-to-end. Now implemented: forces dropout active, runs N
        stochastic passes, returns per-sample variance.
        """
        was_training = self.training
        self.train()  # ensure dropout is active
        # but disable batchnorm updates by setting modules to eval selectively
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        det_logits = []
        stage_logits = []
        sev_preds = []
        for _ in range(n_passes):
            out = self.forward(image, tabular, return_uncertainty=False)
            det_logits.append(out["detection_logits"])
            stage_logits.append(out["staging_logits"])
            sev_preds.append(out["severity_pred"])

        det_logits = torch.stack(det_logits, dim=0)     # (n, B, 1)
        stage_logits = torch.stack(stage_logits, dim=0) # (n, B, C)
        sev_preds = torch.stack(sev_preds, dim=0)       # (n, B, 1)

        det_unc = det_logits.var(dim=0).mean(dim=-1, keepdim=True)
        stage_unc = stage_logits.var(dim=0).mean(dim=-1, keepdim=True)
        sev_unc = sev_preds.var(dim=0).mean(dim=-1, keepdim=True)

        if not was_training:
            self.eval()
        return {
            "detection_uncertainty": det_unc,
            "staging_uncertainty": stage_unc,
            "severity_uncertainty": sev_unc,
        }

    def predict_with_uncertainty(
        self, image: torch.Tensor, tabular: torch.Tensor, n_passes: int = 10,
        uncertainty_threshold: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        """
        Inference helper: mean prediction over MC passes + per-sample uncertainty
        + a 'needs_review' flag for cases above the uncertainty threshold.
        """
        was_training = self.training
        self.train()
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

        det_logits = []
        stage_logits = []
        sev_preds = []
        for _ in range(n_passes):
            out = self.forward(image, tabular, return_uncertainty=False)
            det_logits.append(out["detection_logits"])
            stage_logits.append(out["staging_logits"])
            sev_preds.append(out["severity_pred"])

        det_logits = torch.stack(det_logits, dim=0).mean(dim=0)
        stage_logits = torch.stack(stage_logits, dim=0).mean(dim=0)
        sev_preds = torch.stack(sev_preds, dim=0).mean(dim=0)

        unc = self._quick_mc_uncertainty(image, tabular, n_passes=n_passes)
        if not was_training:
            self.eval()

        det_prob = torch.sigmoid(det_logits)
        needs_review = (unc["detection_uncertainty"] > uncertainty_threshold).float()
        return {
            "detection_prob": det_prob,
            "detection_pred": (det_prob >= 0.5).float(),
            "staging_logits": stage_logits,
            "staging_pred": stage_logits.argmax(dim=-1),
            "severity_pred": sev_preds,
            "detection_uncertainty": unc["detection_uncertainty"],
            "staging_uncertainty": unc["staging_uncertainty"],
            "severity_uncertainty": unc["severity_uncertainty"],
            "needs_review": needs_review,
        }


def build_model(cfg) -> CMCHTXAIModel:
    """Build the full CMCHT-XAI model from config, respecting ablation toggles."""
    imaging_encoder = build_imaging_encoder(cfg)
    tabular_encoder = build_tabular_encoder(cfg)

    fusion_type = cfg.model.fusion.fusion_type
    head_type = cfg.model.heads.head_type
    use_cgct = cfg.training.use_cgct

    if fusion_type == "csg":
        fusion = build_csg_fusion(cfg)
    else:
        fusion = build_fusion(cfg)

    if head_type == "cusp_cascade":
        heads = build_cusp_cascade(cfg)
    else:
        heads = build_independent_heads(cfg)

    model = CMCHTXAIModel(
        imaging_encoder=imaging_encoder,
        tabular_encoder=tabular_encoder,
        fusion=fusion,
        heads=heads,
        fusion_type=fusion_type,
        head_type=head_type,
        use_cgct=use_cgct,
        embed_dim=cfg.model.fusion.embed_dim,
    )
    logger.info(
        "Built CMCHT-XAI model | fusion=%s | heads=%s | cgct=%s",
        fusion_type, head_type, use_cgct,
    )
    return model


def build_model_from_ablation(cfg, ablation: dict) -> CMCHTXAIModel:
    """Build a model variant for a specific ablation config row."""
    # Override the ablation toggles on the config
    cfg.model.fusion.fusion_type = ablation["fusion_type"]
    cfg.model.heads.head_type = ablation["head_type"]
    cfg.training.use_cgct = ablation["use_cgct"]
    return build_model(cfg)