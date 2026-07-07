"""
Unit tests for CMCHT-XAI — verifies all three contributions run end-to-end with
in-memory random tensors, confirming correct output shapes and value ranges.

Run:
    python tests/test_models.py
    python -m pytest tests/test_models.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from src.models.cmcht_model import build_model, build_model_from_ablation
from src.models.csg_fusion import CSGFusion, CounterfactualProbe
from src.models.cascade_heads import CUSPCascade, UncertaintyAwareHead
from src.training.confidence_gated_training import (
    confidence_gated_mask,
    mix_with_ground_truth,
    MultiTaskLoss,
)
from src.utils.logger import load_config, set_seed


def _make_config():
    """Load config and reduce sizes for fast testing."""
    cfg = load_config("config/config.yaml")
    cfg.data.num_tabular_features = 10
    cfg.data.tabular_features = cfg.data.tabular_features[:10]
    cfg.model.fusion.embed_dim = 64
    cfg.model.imaging_encoder.embed_dim = 64
    cfg.model.tabular_encoder.embed_dim = 64
    cfg.model.tabular_encoder.d_token = 32
    cfg.model.tabular_encoder.n_blocks = 1
    cfg.model.fusion.n_heads = 4
    cfg.model.heads.cusp.mc_passes_train = 2
    cfg.model.heads.cusp.mc_passes_eval = 3
    cfg.model.fusion.csg.mc_passes_fusion = 2
    cfg.model.fusion.csg.sensitivity_dim = 32
    cfg.training.batch_size = 4
    cfg.training.num_workers = 0
    return cfg


def _make_batch(cfg, batch_size=4):
    """Create a small in-memory batch for unit tests."""
    size = cfg.data.image_size
    n_feat = cfg.data.num_tabular_features
    return {
        "image": torch.randn(batch_size, 3, size, size),
        "tabular": torch.randn(batch_size, n_feat),
        "detection_label": torch.randint(0, 2, (batch_size,)).float(),
        "staging_label": torch.randint(0, 4, (batch_size,)),
        "severity_label": torch.randn(batch_size) * 5 + 15,
    }


# ==============================================================================
# Contribution 1: CSG-Fusion
# ==============================================================================
def test_csg_fusion():
    """Test CSG-Fusion produces correct shapes + sensitivity vector."""
    print("\n=== Test: CSG-Fusion (Contribution 1) ===")
    cfg = _make_config()
    set_seed(cfg.seed)
    B, D, F = 4, 64, 10

    fusion = CSGFusion(
        embed_dim=D, n_heads=4, dropout=0.1, n_features=F,
        probe_features=[2, 6, 7], probe_epsilon=0.1, sensitivity_dim=32,
        mc_passes_fusion=2,
    )
    image_emb = torch.randn(B, D)
    tabular_emb = torch.randn(B, D)
    tabular_raw = torch.randn(B, F)

    fused, sensitivity = fusion(image_emb, tabular_emb, tabular_raw)

    assert fused.shape == (B, 2 * D), f"Expected fused shape ({B}, {2*D}), got {fused.shape}"
    assert sensitivity.shape == (B, F), f"Expected sensitivity shape ({B}, {F}), got {sensitivity.shape}"
    assert sensitivity.min() >= 0.0, "Sensitivity should be non-negative"
    assert sensitivity.max() <= 1.0 + 1e-5, "Sensitivity should be <= 1"

    # Test consistency loss
    loss = fusion.consistency_loss()
    assert loss.item() >= 0.0, "Consistency loss should be non-negative"
    print(f"  fused shape: {fused.shape}")
    print(f"  sensitivity shape: {sensitivity.shape}")
    print(f"  sensitivity range: [{sensitivity.min():.4f}, {sensitivity.max():.4f}]")
    print(f"  consistency loss: {loss.item():.4f}")
    print("  PASSED ✓")


def test_counterfactual_probe():
    """Test the counterfactual probe in isolation."""
    print("\n=== Test: Counterfactual Probe ===")
    cfg = _make_config()
    F, D = 10, 32
    probe = CounterfactualProbe(n_features=F, probe_features=[2, 6, 7],
                                epsilon=0.1, sensitivity_dim=D)
    encoder = torch.nn.Sequential(torch.nn.Linear(F, D), torch.nn.GELU())
    tabular = torch.randn(4, F)
    sens = probe(tabular, encoder)
    assert sens.shape == (4, F)
    # Non-probe features should have 0 sensitivity
    non_probe = [i for i in range(F) if i not in [2, 6, 7]]
    assert torch.allclose(sens[:, non_probe], torch.zeros_like(sens[:, non_probe]))
    print(f"  sensitivity shape: {sens.shape}")
    print("  PASSED ✓")


# ==============================================================================
# Contribution 2: CUSP-Cascade
# ==============================================================================
def test_cusp_cascade():
    """Test CUSP-Cascade produces correct shapes + uncertainty propagation."""
    print("\n=== Test: CUSP-Cascade (Contribution 2) ===")
    cfg = _make_config()
    set_seed(cfg.seed)
    B, in_dim, C, F = 4, 128, 4, 10

    cascade = CUSPCascade(
        in_dim=in_dim, num_classes=C, hidden_dim=64, dropout=0.3,
        sensitivity_dim=F, mc_passes_train=2, mc_passes_eval=3,
    )
    fused = torch.randn(B, in_dim)
    sensitivity = torch.randn(B, F)

    # Without uncertainty (training mode)
    cascade.train()
    out = cascade(fused, sensitivity=sensitivity, return_uncertainty=False)
    assert out["detection_logits"].shape == (B, 1)
    assert out["staging_logits"].shape == (B, C)
    assert out["severity_pred"].shape == (B, 1)

    # With uncertainty (eval mode)
    cascade.eval()
    out = cascade(fused, sensitivity=sensitivity, return_uncertainty=True)
    assert out["detection_uncertainty"].shape == (B, 1)
    assert out["staging_uncertainty"].shape == (B, 1)
    assert out["severity_uncertainty"].shape == (B, 1)
    assert (out["detection_uncertainty"] >= 0).all()

    # Test with CGCT teacher forcing
    det_gt = torch.tensor([1.0, 0.0, 1.0, 0.0])
    stage_gt = torch.tensor([2, 0, 3, 1])
    self_prob = {"detection": torch.zeros(B, 1), "staging": torch.zeros(B, 1)}
    out = cascade(
        fused,
        sensitivity=sensitivity,
        detection_gt=det_gt,
        staging_gt=stage_gt,
        self_prob_dict=self_prob,
        return_uncertainty=False,
    )
    assert out["detection_logits"].shape == (B, 1)
    print(f"  detection_logits: {out['detection_logits'].shape}")
    print(f"  staging_logits: {out['staging_logits'].shape}")
    print(f"  severity_pred: {out['severity_pred'].shape}")
    print("  PASSED ✓")


# ==============================================================================
# Contribution 3: CGCT
# ==============================================================================
def test_cgct_scheduler():
    """Test CGCT confidence-gated mixing."""
    print("\n=== Test: CGCT (Contribution 3) ===")
    high_unc = torch.tensor([[0.5], [0.01], [0.3]])
    low_self = confidence_gated_mask(high_unc, global_tf_rate=0.0, threshold=0.15)
    assert low_self[0, 0] < low_self[1, 0], "High uncertainty should reduce self-prediction weight"

    preds = torch.tensor([[0.9], [0.1], [0.8]])
    gt = torch.tensor([1.0, 0.0, 0.0])
    mixed = mix_with_ground_truth(preds, gt, low_self)
    assert mixed.shape == preds.shape

    print(f"  self_prob (high-unc batch): {low_self.squeeze().tolist()}")
    print(f"  mixed predictions shape: {mixed.shape}")
    print("  PASSED ✓")


# ==============================================================================
# Full model end-to-end
# ==============================================================================
def test_full_model_forward():
    """Test the full CMCHT-XAI model forward pass (full system config)."""
    print("\n=== Test: Full Model Forward (full system) ===")
    cfg = _make_config()
    set_seed(cfg.seed)
    model = build_model(cfg)
    batch = _make_batch(cfg)

    image = batch["image"]
    tabular = batch["tabular"]
    det_gt = batch["detection_label"]
    stage_gt = batch["staging_label"]
    sev_gt = batch["severity_label"]

    # Forward without uncertainty
    model.train()
    out = model(image, tabular, return_uncertainty=False)
    assert out["detection_logits"].shape[0] == image.size(0)
    assert out["staging_logits"].shape[1] == cfg.model.heads.staging.num_classes
    assert out["severity_pred"].shape[0] == image.size(0)

    # Forward with uncertainty + CGCT teacher forcing
    model.eval()
    self_prob = {
        "detection": torch.zeros(image.size(0), 1),
        "staging": torch.zeros(image.size(0), 1),
    }
    out = model(
        image, tabular,
        detection_gt=det_gt, staging_gt=stage_gt,
        self_prob_dict=self_prob, return_uncertainty=True,
    )
    assert "detection_uncertainty" in out
    assert "staging_uncertainty" in out
    assert "severity_uncertainty" in out

    print(f"  fusion_type: {model.fusion_type}")
    print(f"  head_type: {model.head_type}")
    print(f"  use_cgct: {model.use_cgct}")
    print(f"  detection_logits: {out['detection_logits'].shape}")
    print(f"  staging_logits: {out['staging_logits'].shape}")
    print(f"  severity_pred: {out['severity_pred'].shape}")
    print(f"  detection_uncertainty: {out['detection_uncertainty'].shape}")
    print("  PASSED ✓")


def test_mc_dropout_uncertainty():
    """Test the fixed _quick_mc_uncertainty (Weakness 1 fix)."""
    print("\n=== Test: MC-Dropout Uncertainty (Weakness 1 fix) ===")
    cfg = _make_config()
    set_seed(cfg.seed)
    model = build_model(cfg)
    batch = _make_batch(cfg)

    model.eval()
    unc = model._quick_mc_uncertainty(batch["image"], batch["tabular"], n_passes=3)
    assert "detection_uncertainty" in unc
    assert "staging_uncertainty" in unc
    assert "severity_uncertainty" in unc
    assert unc["detection_uncertainty"].shape[0] == batch["image"].size(0)
    assert (unc["detection_uncertainty"] >= 0).all()
    print(f"  detection_uncertainty: {unc['detection_uncertainty'].shape}")
    print(f"  mean uncertainty: {unc['detection_uncertainty'].mean():.6f}")
    print("  PASSED ✓")


def test_ablation_configs():
    """Test that all five ablation configs build and run a forward pass."""
    print("\n=== Test: All 5 Ablation Configs ===")
    cfg = _make_config()
    set_seed(cfg.seed)
    batch = _make_batch(cfg)

    for abl in cfg.evaluation.ablation_configs:
        name = abl["name"]
        import copy
        cfg_copy = copy.deepcopy(cfg)
        model = build_model_from_ablation(cfg_copy, abl)
        model.eval()
        with torch.no_grad():
            out = model(batch["image"], batch["tabular"], return_uncertainty=False)
        assert out["detection_logits"].shape[0] == batch["image"].size(0)
        print(f"  {name:15s} (fusion={abl['fusion_type']:15s}, "
              f"heads={abl['head_type']:13s}, cgct={str(abl['use_cgct']):5s}) ✓")
    print("  PASSED ✓")


def test_training_step():
    """Test one training step with CGCT + consistency loss."""
    print("\n=== Test: Training Step (CGCT + consistency loss) ===")
    cfg = _make_config()
    set_seed(cfg.seed)
    model = build_model(cfg)
    loss_fn = MultiTaskLoss(lambda_consistency=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    batch = _make_batch(cfg)

    model.train()
    out = model(batch["image"], batch["tabular"], return_uncertainty=False)

    consistency_loss = None
    if hasattr(model.fusion, "consistency_loss"):
        consistency_loss = model.fusion.consistency_loss()

    losses = loss_fn(out, batch["detection_label"], batch["staging_label"],
                     batch["severity_label"], consistency_loss=consistency_loss)

    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    assert losses["total"].item() > 0
    print(f"  total loss: {losses['total'].item():.4f}")
    print(f"  detection: {losses['detection'].item():.4f}")
    print(f"  staging: {losses['staging'].item():.4f}")
    print(f"  severity: {losses['severity'].item():.4f}")
    print(f"  consistency: {float(losses['consistency']):.4f}")
    print("  PASSED ✓")


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 70)
    print("CMCHT-XAI Unit Tests — Verifying All Three Contributions")
    print("=" * 70)

    tests = [
        ("Counterfactual Probe", test_counterfactual_probe),
        ("CSG-Fusion (Contribution 1)", test_csg_fusion),
        ("CUSP-Cascade (Contribution 2)", test_cusp_cascade),
        ("CGCT Scheduler (Contribution 3)", test_cgct_scheduler),
        ("Full Model Forward", test_full_model_forward),
        ("MC-Dropout Uncertainty (Weakness 1 fix)", test_mc_dropout_uncertainty),
        ("All 5 Ablation Configs", test_ablation_configs),
        ("Training Step (CGCT + consistency)", test_training_step),
    ]

    passed = 0
    failed = 0
    for name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except Exception as exc:
            print(f"\n  FAILED ✗: {name}: {exc}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{passed + failed} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED ✓")
    else:
        print(f"{failed} TEST(S) FAILED ✗")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)