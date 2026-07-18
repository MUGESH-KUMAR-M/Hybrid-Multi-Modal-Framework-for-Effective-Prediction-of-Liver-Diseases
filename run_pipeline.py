"""
CMCHT-XAI Full Pipeline — Single-Command Orchestrator
=====================================================
Runs all 6 phases of the Hybrid Multimodal Liver Disease project:

    Phase 1: Data Preprocessing   (ILPD + Cirrhosis + NAFLD images + NAFLD pairing)
    Phase 2: SimCLR Pre-training  (self-supervised on unlabeled ultrasound slices)
    Phase 3: Hybrid Model Training (multimodal fusion with CGCT)
    Phase 4: Explainability        (SHAP + Grad-CAM + Counterfactuals + Uncertainty)
    Phase 5: Evaluation + Ablation (5-row ablation table)
    Phase 6: Summary Report

Usage:
    python run_pipeline.py                      # full pipeline, default epochs
    python run_pipeline.py --epochs 5           # override training epochs
    python run_pipeline.py --skip-pretrain      # skip SimCLR pre-training
    python run_pipeline.py --skip-xai           # skip explainability phase
    python run_pipeline.py --skip-ablation      # skip ablation study
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _phase_banner(phase_num: int, title: str) -> None:
    """Print a formatted phase banner."""
    width = 70
    print("\n" + "=" * width)
    print(f"  PHASE {phase_num}: {title}")
    print("=" * width)


def _phase_result(phase_num: int, title: str, elapsed: float, success: bool, detail: str = "") -> dict:
    """Return a result dict for this phase."""
    status = "SUCCESS" if success else "FAILED"
    icon = "✓" if success else "✗"
    print(f"\n  {icon} Phase {phase_num} ({title}): {status} in {elapsed:.1f}s")
    if detail:
        print(f"    {detail}")
    return {
        "phase": phase_num,
        "title": title,
        "status": status,
        "elapsed_seconds": round(elapsed, 2),
        "detail": detail,
    }


def run_pipeline(
    config_path: str = "config/config.yaml",
    epochs: int | None = None,
    simclr_epochs: int | None = None,
    skip_pretrain: bool = False,
    skip_xai: bool = False,
    skip_ablation: bool = False,
) -> None:
    """Run the full CMCHT-XAI pipeline."""
    total_start = time.time()
    results = []
    print("\n" + "#" * 70)
    print("  CMCHT-XAI: Full Pipeline Execution")
    print("  Cross-Modal Contrastive Hybrid Transformer")
    print("  with Counterfactual Explainable AI")
    print("#" * 70)

    # ====================================================================
    # PHASE 1: Data Preprocessing
    # ====================================================================
    _phase_banner(1, "DATA PREPROCESSING")
    t0 = time.time()
    try:
        from src.data.preprocessing import preprocess_all
        report = preprocess_all()
        
        # Validate actual file outputs exist and are fresh
        proc_dir = Path("data/processed")
        validation_errors = []
        
        # Check for expected CSV files and their freshness
        expected_files = [
            "nafld_paired_train.csv",
            "nafld_paired_val.csv",
            "nafld_paired_test.csv", 
            "preprocessing_report.json"
        ]
        
        # CirrMRI600+ outputs (pretraining-only — no cirrmri_paired_*.csv expected)
        cirrmri_raw = Path("data/raw/CirrMRI600+")
        if cirrmri_raw.exists():
            expected_files.append("cirrmri_unlabeled_manifest.json")
        
        for expected in expected_files:
            file_path = proc_dir / expected
            if not file_path.exists():
                validation_errors.append(f"Missing expected file: {expected}")
            elif file_path.stat().st_size == 0:
                validation_errors.append(f"Empty file: {expected}")
            elif file_path.stat().st_mtime < t0:
                validation_errors.append(f"Stale file (not regenerated in this run): {expected}")
        
        # Check for unlabeled slices directory and freshness
        unlabeled_dir = proc_dir / "unlabeled_slices"
        if not unlabeled_dir.exists():
            validation_errors.append("Missing unlabeled_slices directory")
        else:
            slice_files = list(unlabeled_dir.glob("slice_*.npy"))
            if len(slice_files) == 0:
                validation_errors.append("No unlabeled slices generated")
            else:
                # Check the first generated slice to ensure it's fresh
                if slice_files[0].stat().st_mtime < t0:
                    validation_errors.append("Stale unlabeled slices (not regenerated in this run)")
        
        if validation_errors:
            raise RuntimeError(f"Preprocessing validation failed: {'; '.join(validation_errors)}")
        
        n_datasets = len([k for k, v in report.items() if "error" not in v])
        results.append(_phase_result(1, "Data Preprocessing", time.time() - t0, True,
                                     f"{n_datasets} datasets processed successfully"))
    except Exception as exc:
        traceback.print_exc()
        results.append(_phase_result(1, "Data Preprocessing", time.time() - t0, False, str(exc)))
        print("\n  FATAL: Cannot continue without preprocessed data.")
        _save_report(results, time.time() - total_start)
        return

    # ====================================================================
    # PHASE 2: SimCLR Pre-training
    # ====================================================================
    if skip_pretrain:
        print("\n  ⏭ Phase 2 (SimCLR Pre-training) SKIPPED by user request.")
        results.append({"phase": 2, "title": "SimCLR Pre-training", "status": "SKIPPED",
                        "elapsed_seconds": 0, "detail": "--skip-pretrain flag"})
    else:
        _phase_banner(2, "SimCLR PRE-TRAINING")
        t0 = time.time()
        try:
            from src.pretrain.simclr import train_simclr
            ckpt = train_simclr(config_path, epochs=simclr_epochs)
            results.append(_phase_result(2, "SimCLR Pre-training", time.time() - t0, True,
                                         f"Encoder saved to {ckpt}"))
        except Exception as exc:
            traceback.print_exc()
            results.append(_phase_result(2, "SimCLR Pre-training", time.time() - t0, False, str(exc)))
            print("  WARNING: Continuing without SimCLR pre-trained weights.")

    # ====================================================================
    # PHASE 3: Hybrid Model Training
    # ====================================================================
    _phase_banner(3, "HYBRID MODEL TRAINING")
    t0 = time.time()
    try:
        from src.train import train
        ckpt = train(config_path, epochs=epochs)
        results.append(_phase_result(3, "Hybrid Model Training", time.time() - t0, True,
                                     f"Best checkpoint: {ckpt}"))
    except Exception as exc:
        traceback.print_exc()
        results.append(_phase_result(3, "Hybrid Model Training", time.time() - t0, False, str(exc)))
        print("  WARNING: Continuing with untrained model for evaluation.")

    # ====================================================================
    # PHASE 4: Explainability
    # ====================================================================
    best_ckpt = str(Path("checkpoints") / "cmcht_xai_best.pth")
    if skip_xai:
        print("\n  ⏭ Phase 4 (Explainability) SKIPPED by user request.")
        results.append({"phase": 4, "title": "Explainability", "status": "SKIPPED",
                        "elapsed_seconds": 0, "detail": "--skip-xai flag"})
    else:
        _phase_banner(4, "EXPLAINABILITY (SHAP / Grad-CAM / Counterfactuals / Uncertainty)")
        t0 = time.time()
        try:
            from src.explainability.run_explain import run_explainability
            run_explainability(config_path, checkpoint_path=best_ckpt, n_samples=10)
            results.append(_phase_result(4, "Explainability", time.time() - t0, True,
                                         "SHAP + Grad-CAM + Counterfactuals + MC-Dropout"))
        except Exception as exc:
            traceback.print_exc()
            results.append(_phase_result(4, "Explainability", time.time() - t0, False, str(exc)))

    # ====================================================================
    # PHASE 5: Evaluation + Ablation
    # ====================================================================
    _phase_banner(5, "EVALUATION & ABLATION")
    t0 = time.time()
    try:
        from src.evaluate import run_single_evaluation, run_ablation
        metrics = run_single_evaluation(config_path, checkpoint_path=best_ckpt)
        det_acc = metrics.get("detection", {}).get("accuracy", 0)
        stage_f1 = metrics.get("staging", {}).get("f1_macro", 0)
        detail = f"Det Acc={det_acc:.3f} | Stage F1={stage_f1:.3f}"

        if not skip_ablation:
            ablation = run_ablation(config_path)
            detail += f" | Ablation: {len(ablation)} configs evaluated"

        results.append(_phase_result(5, "Evaluation & Ablation", time.time() - t0, True, detail))
    except Exception as exc:
        traceback.print_exc()
        results.append(_phase_result(5, "Evaluation & Ablation", time.time() - t0, False, str(exc)))

    # ====================================================================
    # PHASE 6: Summary Report
    # ====================================================================
    _phase_banner(6, "SUMMARY REPORT")
    _save_report(results, time.time() - total_start)


def _save_report(results: list, total_elapsed: float) -> None:
    """Print and save the pipeline summary."""
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "total_elapsed_seconds": round(total_elapsed, 2),
        "phases": results,
    }
    report_path = results_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # Print summary table
    print("\n" + "-" * 70)
    print("  PIPELINE SUMMARY")
    print("-" * 70)
    for r in results:
        icon = {"SUCCESS": "✓", "FAILED": "✗", "SKIPPED": "⏭"}.get(r["status"], "?")
        print(f"  {icon} Phase {r['phase']}: {r['title']:30s} {r['status']:8s} ({r['elapsed_seconds']:.1f}s)")
    print("-" * 70)
    print(f"  Total elapsed: {total_elapsed:.1f}s")
    print(f"  Report saved: {report_path}")
    print("-" * 70)

    # List generated outputs
    print("\n  Generated outputs:")
    outputs = [
        ("data/processed/", "Preprocessed datasets"),
        ("checkpoints/simclr_encoder.pth", "SimCLR encoder weights"),
        ("checkpoints/cmcht_xai_best.pth", "Best trained model"),
        ("results/evaluation_metrics.json", "Evaluation metrics"),
        ("results/ablation_results.json", "Ablation study results"),
        ("results/ablation_table.md", "Ablation comparison table"),
        ("results/explainability/", "XAI outputs (SHAP, Grad-CAM, CFs)"),
        ("results/pipeline_report.json", "Pipeline execution report"),
    ]
    for path, desc in outputs:
        exists = Path(path).exists()
        icon = "✓" if exists else "–"
        print(f"    {icon} {path:45s} {desc}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CMCHT-XAI: Run the full pipeline (all 6 phases) with a single command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                    # Full pipeline, default config
  python run_pipeline.py --epochs 5         # Override training epochs
  python run_pipeline.py --skip-pretrain    # Skip SimCLR (use existing weights)
  python run_pipeline.py --skip-xai         # Skip explainability phase
  python run_pipeline.py --skip-ablation    # Skip 5-row ablation study
""",
    )
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epoch count")
    parser.add_argument("--simclr-epochs", type=int, default=None, help="Override SimCLR epoch count")
    parser.add_argument("--skip-pretrain", action="store_true", help="Skip Phase 2 (SimCLR)")
    parser.add_argument("--skip-xai", action="store_true", help="Skip Phase 4 (Explainability)")
    parser.add_argument("--skip-ablation", action="store_true", help="Skip ablation in Phase 5")
    args = parser.parse_args()

    run_pipeline(
        config_path=args.config,
        epochs=args.epochs,
        simclr_epochs=args.simclr_epochs,
        skip_pretrain=args.skip_pretrain,
        skip_xai=args.skip_xai,
        skip_ablation=args.skip_ablation,
    )
