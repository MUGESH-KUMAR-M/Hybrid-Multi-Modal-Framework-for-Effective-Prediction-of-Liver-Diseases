"""
Tabular encoder for CMCHT-XAI: FT-Transformer (Feature Tokenizer Transformer).

Reference: Gorishniy et al. (2021), "Revisiting Deep Learning Models for
Tabular Data".

Each clinical feature (bilirubin, albumin, SGOT, ...) is embedded as a token;
self-attention learns feature interactions (e.g., bilirubin <-> albumin ratio).
A [CLS]-style pooled token yields the final embedding.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# Feature Tokenizer
# ==============================================================================

class FeatureTokenizer(nn.Module):
    """Tokenize each numerical feature into an embedding vector.

    Each feature gets its own Linear(1, embed_dim) implemented as a
    ``(num_features, embed_dim)`` weight parameter (+ bias) rather than
    a Python list of ``nn.Linear`` layers.
    """

    def __init__(self, num_features: int, embed_dim: int):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        # Weight matrix — each row is the weight vector for one feature's
        # Linear(1, embed_dim) projection.
        self.weight = nn.Parameter(torch.empty(num_features, embed_dim))
        self.bias = nn.Parameter(torch.empty(num_features, embed_dim))
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        # Kaiming-uniform, consistent with nn.Linear default init
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in = 1  # each feature is a scalar
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, num_features) -> tokens: (B, num_features, embed_dim)."""
        # x[..., None] broadcasts over the embed dimension
        return x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)


# ==============================================================================
# FT-Transformer Encoder
# ==============================================================================

class FTTransformerEncoder(nn.Module):
    """FT-Transformer tabular encoder (Gorishniy et al. 2021).

    Pipeline:
        features -> FeatureTokenizer -> prepend [CLS] ->
        nn.TransformerEncoder (norm_first, GELU) -> extract CLS output.

    The CLS token output serves as the patient embedding of shape
    ``(batch, embed_dim)``.
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim

        # --- Feature tokenizer ------------------------------------------------
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)

        # --- Learnable [CLS] token --------------------------------------------
        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # --- Transformer encoder (pre-norm, GELU) -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        # --- Final layer-norm on CLS output -----------------------------------
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_features) float tensor of clinical features.

        Returns:
            (batch, embed_dim) patient embedding extracted from [CLS] token.
        """
        tokens = self.tokenizer(x)                               # (B, F, D)
        cls = self.cls_token.expand(x.size(0), -1, -1)           # (B, 1, D)
        seq = torch.cat([cls, tokens], dim=1)                    # (B, F+1, D)
        seq = self.transformer(seq)                              # (B, F+1, D)
        cls_out = seq[:, 0, :]                                   # (B, D)
        return self.final_norm(cls_out)                          # (B, D)


# ==============================================================================
# Config-driven factory
# ==============================================================================

def build_tabular_encoder(cfg) -> FTTransformerEncoder:
    """Build the FT-Transformer encoder from a config object.

    Maps the legacy config fields (``d_token``, ``n_blocks``, ``n_heads``)
    to the canonical constructor arguments.
    """
    enc_cfg = cfg.model.tabular_encoder
    return FTTransformerEncoder(
        num_features=cfg.data.num_tabular_features,
        embed_dim=getattr(enc_cfg, "embed_dim", 128),
        num_heads=getattr(enc_cfg, "n_heads", 8),
        num_layers=getattr(enc_cfg, "n_blocks", 3),
        dropout=getattr(enc_cfg, "ffn_dropout", 0.1),
    )