"""
Baseline fusion for CMCHT-XAI (ablation baseline).

Bidirectional cross-attention fusion:
    - Imaging queries Tabular (image-aware-of-clinical)
    - Tabular queries Imaging (clinical-aware-of-imaging)
    - Concatenate both + residual connection.

This is the ablation baseline against which CSG-Fusion (Contribution 1) is
compared. It performs standard cross-modal attention WITHOUT the
counterfactual-sensitivity gating or uncertainty-awareness.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Baseline bidirectional cross-modal attention fusion."""

    def __init__(self, embed_dim: int = 256, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim

        # Image -> queries Tabular
        self.img_to_tab = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        # Tabular -> queries Image
        self.tab_to_img = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)

        self.norm_img = nn.LayerNorm(embed_dim)
        self.norm_tab = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim * 2)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        image_emb: torch.Tensor,
        tabular_emb: torch.Tensor,
        return_attn: bool = False,
    ) -> torch.Tensor:
        """
        image_emb: (B, embed_dim)  -> treated as single-token sequences (B, 1, D)
        tabular_emb: (B, embed_dim)
        Returns fused embedding (B, 2*embed_dim) -> typically projected later.
        """
        img_seq = image_emb.unsqueeze(1)      # (B, 1, D)
        tab_seq = tabular_emb.unsqueeze(1)    # (B, 1, D)

        # Image queries tabular
        img_aware, attn_i2t = self.img_to_tab(
            query=self.norm_img(img_seq),
            key=self.norm_tab(tab_seq),
            value=self.norm_tab(tab_seq),
            need_weights=return_attn,
        )
        img_out = img_seq + self.dropout(img_aware)

        # Tabular queries image
        tab_aware, attn_t2i = self.tab_to_img(
            query=self.norm_tab(tab_seq),
            key=self.norm_img(img_seq),
            value=self.norm_img(img_seq),
            need_weights=return_attn,
        )
        tab_out = tab_seq + self.dropout(tab_aware)

        fused = torch.cat([img_out, tab_out], dim=-1)   # (B, 1, 2D)
        fused = self.norm_out(fused).squeeze(1)         # (B, 2D)
        return fused


def build_fusion(cfg) -> CrossAttentionFusion:
    """Build the baseline cross-attention fusion from config."""
    fcfg = cfg.model.fusion
    return CrossAttentionFusion(
        embed_dim=fcfg.embed_dim,
        n_heads=fcfg.n_heads,
        dropout=fcfg.dropout,
    )