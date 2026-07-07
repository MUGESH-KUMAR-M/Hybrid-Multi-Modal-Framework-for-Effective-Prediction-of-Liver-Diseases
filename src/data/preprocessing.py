"""
CMCHT-XAI Data Preprocessing
=============================
Phase 1 preprocessing for tabular datasets placed under ``data/raw/``.

Implements dataset-specific actions from ``Docs/CMCHT_XAI_Dataset_Reference.docx``:

  * ILPD  - median impute missing A/G ratios, log1p on bilirubin/ALP, SMOTE.
  * Cirrhosis - MICE imputation, Stage (1-4) staging label, binarised Status.

Processed outputs are written to ``data/processed/`` as CSVs plus
``preprocessing_report.json``.

Place raw CSVs at:
  data/raw/ILPD/ilpd.csv
  data/raw/Cirrhosis/cirrhosis.csv

Usage:
    python -m src.data.preprocessing
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    from imblearn.over_sampling import SMOTE
    _HAS_SMOTE = True
except ImportError:  # pragma: no cover
    _HAS_SMOTE = False

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

# ILPD column name mappings (ucimlrepo returns short names).
ILPD_RENAME = {
    "TB": "Total_Bilirubin",
    "DB": "Direct_Bilirubin",
    "Alkphos": "Alkaline_Phosphatase",
    "Sgpt": "Alamine_Aminotransferase",
    "Sgot": "Aspartate_Aminotransferase",
    "TP": "Total_Proteins",
    "ALB": "Albumin",
    "A/G Ratio": "Albumin_and_Globulin_Ratio",
}

# Columns to log1p transform (heavily right-skewed).
LOG_SKEW_COLS = [
    "Total_Bilirubin", "Direct_Bilirubin",
    "Alkaline_Phosphatase", "Alamine_Aminotransferase",
    "Aspartate_Aminotransferase",
]


# --------------------------------------------------------------------------- #
# 1. ILPD
# --------------------------------------------------------------------------- #
def preprocess_ilpd() -> Dict:
    """ILPD - Indian Liver Patient Dataset (UCI id=225)."""
    print("[ILPD] preprocessing ...")
    raw_path = RAW_DIR / "ILPD" / "ilpd.csv"
    df = pd.read_csv(raw_path)
    report: Dict = {"dataset": "ILPD", "raw_shape": list(df.shape), "steps": []}

    # 1a. Rename columns to descriptive names.
    df = df.rename(columns=ILPD_RENAME)
    report["steps"].append(f"Renamed columns: {ILPD_RENAME}")

    # 1b. Encode Gender (Male=1, Female=0).
    df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})
    if df["Gender"].isna().any():
        df["Gender"] = LabelEncoder().fit_transform(df["Gender"])
    report["steps"].append("Gender label-encoded (Male=1, Female=0)")

    # 1c. Median-impute missing A/G ratios (4 values).
    ag_col = "Albumin_and_Globulin_Ratio"
    n_missing = int(df[ag_col].isna().sum())
    if n_missing > 0:
        med = df[ag_col].median()
        df[ag_col] = df[ag_col].fillna(med)
        report["steps"].append(f"Median-imputed {n_missing} missing A/G ratio values (median={med:.3f})")
    else:
        report["steps"].append("No missing A/G ratio values found")

    # 1d. log1p on heavily right-skewed columns.
    for col in LOG_SKEW_COLS:
        if col in df.columns:
            df[col] = np.log1p(df[col].clip(lower=0))
    report["steps"].append(f"log1p transform applied to {LOG_SKEW_COLS}")

    # 1e. Target: 1 = disease, 2 = healthy -> remap to 1/0.
    if "Dataset" in df.columns:
        df["label"] = (df["Dataset"] == 1).astype(int)
        df = df.drop(columns=["Dataset"])
    report["steps"].append("Target remapped: 1=disease, 0=healthy")

    # 1f. Train/test split BEFORE SMOTE to avoid leakage.
    y = df["label"].values
    X = df.drop(columns=["label"])
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    report["steps"].append(f"Train/test split (stratified): train={X_tr.shape}, test={X_te.shape}")

    # 1g. Standardise.
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    # 1h. SMOTE on training only.
    if _HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_tr_res, y_tr_res = smote.fit_resample(X_tr_scaled, y_tr)
        report["steps"].append(
            f"SMOTE applied to training set: {dict(zip(*np.unique(y_tr, return_counts=True)))} "
            f"-> {dict(zip(*np.unique(y_tr_res, return_counts=True)))}"
        )
    else:
        X_tr_res, y_tr_res = X_tr_scaled, y_tr
        report["steps"].append("SMOTE skipped (imbalanced-learn not installed)")

    # Save.
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cols = list(X.columns)
    pd.DataFrame(X_tr_res, columns=cols).assign(label=y_tr_res).to_csv(
        PROC_DIR / "ilpd_train.csv", index=False
    )
    pd.DataFrame(X_te_scaled, columns=cols).assign(label=y_te).to_csv(
        PROC_DIR / "ilpd_test.csv", index=False
    )

    report["processed_shape"] = {"train": list(X_tr_res.shape), "test": list(X_te_scaled.shape)}
    report["files"] = ["ilpd_train.csv", "ilpd_test.csv"]
    print(f"[ILPD] done. train={X_tr_res.shape} test={X_te_scaled.shape}")
    return report


# --------------------------------------------------------------------------- #
# 2. Cirrhosis Prediction (Mayo PBC)
# --------------------------------------------------------------------------- #
def preprocess_cirrhosis() -> Dict:
    """Cirrhosis Prediction Dataset - staging (Stage 1-4) + severity (Status)."""
    print("[Cirrhosis] preprocessing ...")
    raw_path = RAW_DIR / "Cirrhosis" / "cirrhosis.csv"
    df = pd.read_csv(raw_path)
    report: Dict = {"dataset": "Cirrhosis", "raw_shape": list(df.shape), "steps": []}

    # Drop ID-like columns.
    for c in ["ID", "id"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    # Encode categoricals.
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for c in cat_cols:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    report["steps"].append(f"Label-encoded {len(cat_cols)} categorical columns: {cat_cols}")

    # MICE imputation.
    n_missing_total = int(df.isna().sum().sum())
    if n_missing_total > 0:
        mice = IterativeImputer(random_state=42, max_iter=10, sample_posterior=True)
        df_imputed = pd.DataFrame(mice.fit_transform(df), columns=df.columns)
        report["steps"].append(
            f"MICE imputation: {n_missing_total} missing cells filled "
            f"(~{n_missing_total / df.size * 100:.1f}% of matrix)"
        )
    else:
        df_imputed = df
        report["steps"].append("No missing values - imputation skipped")

    # Multi-task labels.
    if "Stage" in df_imputed.columns:
        df_imputed["staging_label"] = (df_imputed["Stage"].astype(int).clip(1, 4)) - 1
    if "Status" in df_imputed.columns:
        df_imputed["severity_label"] = (df_imputed["Status"] == 0).astype(int)
        df_imputed["detection_label"] = 1
        report["steps"].append(
            "Multi-task labels created: staging_label (0-3), severity_label (Status binarised), detection_label=1"
        )

    feature_drop = ["Stage", "Status", "N_Days"]
    feature_cols = [c for c in df_imputed.columns if c not in feature_drop + ["staging_label", "severity_label", "detection_label"]]
    X = df_imputed[feature_cols]

    stratify_col = df_imputed.get("staging_label", df_imputed.get("detection_label"))
    X_tr, X_te, idx_tr, idx_te = train_test_split(
        X, range(len(X)), test_size=0.2, random_state=42, stratify=stratify_col
    )
    report["steps"].append(f"Train/test split (stratified on stage): train={X_tr.shape}, test={X_te.shape}")

    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr)
    X_te_scaled = scaler.transform(X_te)

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    label_cols = [c for c in ["staging_label", "severity_label", "detection_label"] if c in df_imputed.columns]
    train_out = pd.DataFrame(X_tr_scaled, columns=feature_cols)
    test_out = pd.DataFrame(X_te_scaled, columns=feature_cols)
    for lc in label_cols:
        train_out[lc] = df_imputed.iloc[idx_tr][lc].values
        test_out[lc] = df_imputed.iloc[idx_te][lc].values
    train_out.to_csv(PROC_DIR / "cirrhosis_train.csv", index=False)
    test_out.to_csv(PROC_DIR / "cirrhosis_test.csv", index=False)

    report["processed_shape"] = {"train": list(X_tr_scaled.shape), "test": list(X_te_scaled.shape)}
    report["files"] = ["cirrhosis_train.csv", "cirrhosis_test.csv"]
    print(f"[Cirrhosis] done. train={X_tr_scaled.shape} test={X_te_scaled.shape}")
    return report


# --------------------------------------------------------------------------- #
# 4. Image Preprocessing (DICOM/NIfTI Windowing)
# --------------------------------------------------------------------------- #
def preprocess_images(input_dir: Path, output_dir: Path, modality: str = "CT") -> Dict:
    """
    Preprocess medical images (CT/MRI/Ultrasound).
    Applies Hounsfield Unit windowing for CT, resampling, and normalization.
    """
    print(f"[Images] preprocessing {modality} from {input_dir} ...")
    report: Dict = {"dataset": "Images", "steps": []}
    
    if not input_dir.exists():
        msg = f"Input directory not found: {input_dir}"
        print(f"[Images] SKIPPED: {msg}")
        return {"error": msg}
        
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        from monai.transforms import (
            Compose, LoadImage, ScaleIntensityRange, Resize, EnsureType, EnsureChannelFirst
        )
    except ImportError:
        msg = "MONAI not installed. Skipping image preprocessing."
        print(f"[Images] SKIPPED: {msg}")
        return {"error": msg}

    # Define transforms based on modality
    transforms_list = [LoadImage(image_only=True), EnsureChannelFirst()]
    
    if modality.upper() == "CT":
        # Liver window: W=400, L=50 -> range is [-150, 250]
        transforms_list.append(ScaleIntensityRange(a_min=-150, a_max=250, b_min=0.0, b_max=1.0, clip=True))
        report["steps"].append("Applied CT Liver windowing (W:400, L:50)")
    else:
        # Default min-max scaling for MRI/Ultrasound
        transforms_list.append(ScaleIntensityRange(a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True))
        report["steps"].append("Applied standard intensity scaling [0, 1]")
        
    # Resize to standard 224x224 (assuming 2D slices for now)
    transforms_list.append(Resize((224, 224)))
    transforms_list.append(EnsureType())
    
    transform = Compose(transforms_list)
    
    processed_count = 0
    # Process supported formats
    for ext in ["*.nii", "*.nii.gz", "*.dcm", "*.png", "*.jpg"]:
        for img_path in input_dir.rglob(ext):
            try:
                out_tensor = transform(img_path)
                # Save processed tensor (e.g., as numpy array or similar)
                out_file = output_dir / f"{img_path.stem}_processed.npy"
                np.save(out_file, out_tensor.numpy())
                processed_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
    report["steps"].append(f"Processed and saved {processed_count} images to {output_dir}")
    print(f"[Images] done. Processed {processed_count} files.")
    return report


# --------------------------------------------------------------------------- #
# 5. NAFLD Paired Multimodal Dataset (clinical_data.csv + images)
# --------------------------------------------------------------------------- #
def preprocess_nafld_paired() -> Dict:
    """
    Build the paired multimodal NAFLD dataset: clinical tabular + ultrasound images.

    Reads data/raw/nafld_ultrasound/clinical_data.csv, pairs each patient row
    with its processed .npy image in data/processed/nafld_images/, creates
    multi-task labels (detection, staging, severity), normalises tabular features,
    and writes stratified train/test CSVs.
    """
    print("[NAFLD Paired] preprocessing ...")
    report: Dict = {"dataset": "NAFLD_Paired", "steps": []}

    clinical_path = RAW_DIR / "nafld_ultrasound" / "clinical_data.csv"
    if not clinical_path.exists():
        msg = f"Clinical data not found: {clinical_path}"
        print(f"[NAFLD Paired] SKIPPED: {msg}")
        return {"error": msg}

    df = pd.read_csv(clinical_path)
    report["raw_shape"] = list(df.shape)

    # --- Resolve image paths -------------------------------------------------
    img_dir = PROC_DIR / "nafld_images"
    if not img_dir.exists():
        msg = f"Processed NAFLD images not found at {img_dir}. Run image preprocessing first."
        print(f"[NAFLD Paired] SKIPPED: {msg}")
        return {"error": msg}

    # The ID column in clinical_data.csv is like "id1", "id2", ...
    # Processed images are named "id1_processed.npy", "id2_processed.npy", ...
    id_col = "ID"
    image_paths = []
    valid_mask = []
    for _, row in df.iterrows():
        pid = str(row[id_col])
        npy_path = img_dir / f"{pid}_processed.npy"
        if npy_path.exists():
            image_paths.append(str(npy_path))
            valid_mask.append(True)
        else:
            image_paths.append("")
            valid_mask.append(False)

    df["image_path"] = image_paths
    n_paired = sum(valid_mask)
    df = df[valid_mask].reset_index(drop=True)
    report["steps"].append(f"Paired {n_paired}/{len(valid_mask)} patients with processed images")

    if len(df) == 0:
        msg = "No patients could be paired with images."
        print(f"[NAFLD Paired] SKIPPED: {msg}")
        return {"error": msg}

    # --- Create multi-task labels --------------------------------------------
    # Liver Grade: Normal=0, Benign=1, Malignant=2
    grade_col = [c for c in df.columns if "Liver Grade" in c or "liver grade" in c.lower()]
    if grade_col:
        grade_col = grade_col[0]
        # detection_label: 0 = normal, 1 = any disease (benign or malignant)
        df["detection_label"] = (df[grade_col] > 0).astype(int)
        # staging_label: 0 = normal, 1 = benign, 2 = malignant (direct mapping)
        df["staging_label"] = df[grade_col].astype(int).clip(0, 2)
        # severity_label: continuous 0-1 (0=normal, 0.5=benign, 1.0=malignant)
        df["severity_label"] = df[grade_col].astype(float) / 2.0
        report["steps"].append(
            "Multi-task labels: detection (binary), staging (0-2), severity (0-1)"
        )
    else:
        df["detection_label"] = 1
        df["staging_label"] = 0
        df["severity_label"] = 0.0
        report["steps"].append("No grade column found; default labels assigned")

    # --- Select and normalise tabular features -------------------------------
    feature_cols = ["Age", "Gender(Female=1,Male=2)", "BMI", "Waist_cm",
                    "ALT", "AST", "Glucose", "Cholesterol", "LDL", "HDL", "Triglycerides"]
    # Keep only columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    # Impute missing values with median
    n_missing = int(df[feature_cols].isna().sum().sum())
    if n_missing > 0:
        for c in feature_cols:
            if df[c].isna().any():
                med = df[c].median()
                df[c] = df[c].fillna(med)
        report["steps"].append(f"Median-imputed {n_missing} missing tabular values")

    # Stratified train/val/test split
    stratify_col = df["staging_label"] if "staging_label" in df.columns else df["detection_label"]
    try:
        train_idx, temp_idx = train_test_split(
            range(len(df)), test_size=0.3, random_state=42, stratify=stratify_col
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42, stratify=stratify_col.iloc[temp_idx]
        )
    except ValueError:
        # Fallback if stratification fails (too few samples per class)
        train_idx, temp_idx = train_test_split(
            range(len(df)), test_size=0.3, random_state=42
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.5, random_state=42
        )
    report["steps"].append(f"Train/val/test split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Standardise tabular features
    scaler = StandardScaler()
    df_train = df.iloc[train_idx].copy()
    df_val = df.iloc[val_idx].copy()
    df_test = df.iloc[test_idx].copy()

    df_train[feature_cols] = scaler.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler.transform(df_test[feature_cols])

    # --- Select output columns -----------------------------------------------
    label_cols = ["detection_label", "staging_label", "severity_label"]
    meta_cols = ["image_path"]
    out_cols = feature_cols + label_cols + meta_cols

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    df_train[out_cols].to_csv(PROC_DIR / "nafld_paired_train.csv", index=False)
    df_val[out_cols].to_csv(PROC_DIR / "nafld_paired_val.csv", index=False)
    df_test[out_cols].to_csv(PROC_DIR / "nafld_paired_test.csv", index=False)

    report["processed_shape"] = {
        "train": [len(df_train), len(out_cols)],
        "val": [len(df_val), len(out_cols)],
        "test": [len(df_test), len(out_cols)],
    }
    report["files"] = ["nafld_paired_train.csv", "nafld_paired_val.csv", "nafld_paired_test.csv"]
    report["feature_columns"] = feature_cols
    print(f"[NAFLD Paired] done. train={len(df_train)} val={len(df_val)} test={len(df_test)}")
    return report


def _generate_unlabeled_slices() -> None:
    """
    Copy processed NAFLD images to data/processed/unlabeled_slices/ as
    slice_XXXX.npy for SimCLR pre-training (Phase 2).
    """
    src_dir = PROC_DIR / "nafld_images"
    dst_dir = PROC_DIR / "unlabeled_slices"
    if not src_dir.exists():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)

    existing = list(dst_dir.glob("slice_*.npy"))
    if len(existing) > 0:
        print(f"[Unlabeled Slices] {len(existing)} slices already exist, regenerating...")
        for f in existing:
            f.unlink()

    idx = 0
    for npy_file in sorted(src_dir.glob("*.npy")):
        arr = np.load(npy_file)
        out_path = dst_dir / f"slice_{idx:04d}.npy"
        # Ensure shape is (C, H, W) with C=1 for SimCLR augment to expand to 3
        if arr.ndim == 3 and arr.shape[0] == 3:
            # Already (3, H, W) — take single channel for diversity in augmentation
            np.save(out_path, arr[:1])
        elif arr.ndim == 3 and arr.shape[0] == 1:
            np.save(out_path, arr)
        elif arr.ndim == 2:
            np.save(out_path, arr[None, ...])
        else:
            np.save(out_path, arr)
        idx += 1

    print(f"[Unlabeled Slices] Generated {idx} slices in {dst_dir}")


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def preprocess_all() -> Dict:
    """Run preprocessing for all collected datasets and write a summary report."""
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    results: Dict = {}

    for fn in (preprocess_ilpd, preprocess_cirrhosis):
        try:
            results[fn.__name__] = fn()
        except Exception as exc:
            results[fn.__name__] = {"error": str(exc)}
            print(f"[{fn.__name__}] FAILED: {exc}")

    nafld_img_dir = RAW_DIR / "nafld_ultrasound" / "images"
    if nafld_img_dir.exists():
        results["preprocess_images_nafld"] = preprocess_images(nafld_img_dir, PROC_DIR / "nafld_images", modality="US")

    # NAFLD paired multimodal dataset (requires images to be processed first)
    try:
        results["preprocess_nafld_paired"] = preprocess_nafld_paired()
    except Exception as exc:
        results["preprocess_nafld_paired"] = {"error": str(exc)}
        print(f"[preprocess_nafld_paired] FAILED: {exc}")

    # Generate unlabeled slices for SimCLR pre-training
    try:
        _generate_unlabeled_slices()
    except Exception as exc:
        print(f"[unlabeled_slices] FAILED: {exc}")

    out = PROC_DIR / "preprocessing_report.json"
    out.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nPreprocessing report -> {out}")
    return results


if __name__ == "__main__":
    preprocess_all()