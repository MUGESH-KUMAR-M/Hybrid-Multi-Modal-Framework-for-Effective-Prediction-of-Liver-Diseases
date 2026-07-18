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
from typing import Dict, Set

from PIL import Image

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


# --------------------------------------------------------------------------- #
# 6. CirrMRI600+ — Unlabeled Pretraining Corpus (SimCLR only)
# --------------------------------------------------------------------------- #
def preprocess_cirrmri_unlabeled() -> Dict:
    """
    CirrMRI600+ — unlabeled image preprocessing for SimCLR pre-training.

    Extracts representative 2D liver slices from CirrMRI600+ T2-weighted data
    (cirrhosis patients from T2_2D PNGs + healthy subjects from T2 NIfTI).

    These slices are used ONLY for unsupervised contrastive pre-training.
    No tabular features, no multi-task labels, no staging-head integration.
    CirrMRI600+'s Radiological Evaluation labels are deliberately ignored.
    """
    print("[CirrMRI600+ Unlabeled] preprocessing ...")
    report: Dict = {"dataset": "CirrMRI600+_Unlabeled", "steps": []}

    cirrmri_root = RAW_DIR / "CirrMRI600+"
    if not cirrmri_root.exists():
        msg = f"CirrMRI600+ not found at {cirrmri_root}"
        print(f"[CirrMRI600+ Unlabeled] SKIPPED: {msg}")
        return {"error": msg}

    slices_per_patient = 3
    output_dir = PROC_DIR / "cirrmri_unlabeled_slices"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous run
    for old_file in output_dir.glob("*.npy"):
        old_file.unlink()

    manifest: Dict = {
        "cirrhosis": {},
        "healthy": {},
        "source_splits": {},
    }
    cirrhosis_count = 0

    # ------------------------------------------------------------------
    # 1. Cirrhosis T2_2D patients (PNG slices — no NIfTI processing)
    # ------------------------------------------------------------------
    t2_2d_root = cirrmri_root / "Cirrhosis_T2_2D" / "Cirrhosis_T2_2D"

    for split in ["train", "test", "valid"]:
        split_dir = t2_2d_root / split
        if not split_dir.exists():
            continue

        for patient_dir in sorted(split_dir.iterdir()):
            if not patient_dir.is_dir():
                continue
            patient_id = patient_dir.name
            images_dir = patient_dir / "images"
            masks_dir = patient_dir / "masks"

            if not images_dir.exists():
                continue

            # Score each slice by liver mask cross-sectional area
            slice_scores = []
            for img_path in sorted(images_dir.glob("*.png")):
                mask_path = masks_dir / img_path.name
                if mask_path.exists():
                    mask_arr = np.array(Image.open(mask_path).convert("L"))
                    area = int((mask_arr > 0).sum())
                else:
                    area = 0
                slice_scores.append((img_path, area))

            # Select top-N slices by liver mask area
            slice_scores.sort(key=lambda x: x[1], reverse=True)
            selected = slice_scores[:slices_per_patient]

            patient_slices = []
            for idx, (img_path, _area) in enumerate(selected):
                # Load grayscale, resize to 224x224, normalize [0,1]
                img_pil = Image.open(img_path).convert("L")
                img_pil = img_pil.resize((224, 224), Image.BILINEAR)
                arr = np.array(img_pil, dtype=np.float32) / 255.0
                # Shape: (1, 224, 224) — single channel for SimCLR
                arr = arr[None, ...]

                out_name = f"cirrmri_{patient_id}_{idx}.npy"
                np.save(output_dir / out_name, arr)
                patient_slices.append(out_name)
                cirrhosis_count += 1

            manifest["cirrhosis"][patient_id] = patient_slices
            manifest["source_splits"][patient_id] = split

    report["steps"].append(
        f"Processed {cirrhosis_count} cirrhosis T2_2D slices from "
        f"{len(manifest['cirrhosis'])} patients"
    )

    # ------------------------------------------------------------------
    # 2. Healthy T2 subjects (NIfTI volumes — extract axial slices)
    # ------------------------------------------------------------------
    healthy_t2_imgs = cirrmri_root / "Healthy_subjects" / "T2_W_Healthy" / "T2_images"
    healthy_t2_masks = cirrmri_root / "Healthy_subjects" / "T2_W_Healthy" / "T2_masks"
    healthy_count = 0

    if healthy_t2_imgs.exists():
        try:
            import nibabel as nib
        except ImportError:
            nib = None
            print(
                "[CirrMRI600+ Unlabeled] WARNING: nibabel not installed — "
                "skipping 55 healthy T2 subjects"
            )
            report["steps"].append("Skipped healthy subjects (nibabel not installed)")

        if nib is not None:
            for nii_path in sorted(healthy_t2_imgs.glob("*.nii.gz")):
                # Subject ID: stem without .nii extension  (e.g. "1" from "1.nii.gz")
                subject_id = nii_path.name.replace(".nii.gz", "")
                mask_path = healthy_t2_masks / nii_path.name

                try:
                    vol = nib.load(str(nii_path)).get_fdata(dtype=np.float32)
                    has_mask = mask_path.exists()
                    if has_mask:
                        mask_vol = nib.load(str(mask_path)).get_fdata(dtype=np.float32)
                    else:
                        mask_vol = None

                    # Determine the slice axis (smallest spatial dim = through-plane)
                    if vol.ndim < 3:
                        continue
                    n_slices = vol.shape[2]  # standard axial for abdominal MRI

                    # Score each axial slice by liver mask area
                    slice_scores = []
                    for s in range(n_slices):
                        if mask_vol is not None:
                            area = int((mask_vol[:, :, s] > 0).sum())
                        else:
                            # No mask — use intensity variance as a proxy
                            area = int(vol[:, :, s].var())
                        slice_scores.append((s, area))

                    slice_scores.sort(key=lambda x: x[1], reverse=True)
                    selected = slice_scores[:slices_per_patient]

                    subject_slices = []
                    for idx, (s_idx, _area) in enumerate(selected):
                        img_2d = vol[:, :, s_idx]
                        # Per-slice min-max to [0, 1] (MRI intensities are arbitrary)
                        vmin, vmax = float(img_2d.min()), float(img_2d.max())
                        if vmax > vmin:
                            img_2d = (img_2d - vmin) / (vmax - vmin)
                        else:
                            img_2d = np.zeros_like(img_2d)
                        # Resize to 224x224 via PIL (consistent with PNG path)
                        img_uint8 = (img_2d * 255).clip(0, 255).astype(np.uint8)
                        img_pil = Image.fromarray(img_uint8, mode="L")
                        img_pil = img_pil.resize((224, 224), Image.BILINEAR)
                        arr = np.array(img_pil, dtype=np.float32) / 255.0
                        arr = arr[None, ...]  # (1, 224, 224)

                        out_name = f"cirrmri_healthy_{subject_id}_{idx}.npy"
                        np.save(output_dir / out_name, arr)
                        subject_slices.append(out_name)
                        healthy_count += 1

                    manifest["healthy"][subject_id] = subject_slices
                except Exception as exc:
                    print(
                        f"[CirrMRI600+ Unlabeled] Error processing healthy "
                        f"subject {subject_id}: {exc}"
                    )

    report["steps"].append(
        f"Processed {healthy_count} healthy T2 NIfTI slices from "
        f"{len(manifest['healthy'])} subjects"
    )

    # ------------------------------------------------------------------
    # 3. Write manifest (for provenance / leakage checks)
    # ------------------------------------------------------------------
    manifest["total_cirrhosis_slices"] = cirrhosis_count
    manifest["total_healthy_slices"] = healthy_count
    manifest["total_slices"] = cirrhosis_count + healthy_count
    manifest["slices_per_patient"] = slices_per_patient

    manifest_path = PROC_DIR / "cirrmri_unlabeled_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    report["steps"].append(f"Manifest saved to {manifest_path}")

    total = cirrhosis_count + healthy_count
    print(
        f"[CirrMRI600+ Unlabeled] done. {total} total slices "
        f"({cirrhosis_count} cirrhosis + {healthy_count} healthy)"
    )
    report["total_slices"] = total
    report["files"] = [str(manifest_path)]
    return report


def _assert_no_patient_id_collision(nafld_ids: Set[str], cirrmri_ids: Set[str]) -> None:
    """Hard-fail if NAFLD and CirrMRI600+ patient IDs overlap.

    A silent collision would contaminate the unlabeled pretraining corpus
    provenance without any visible symptom in training metrics.
    """
    overlap = nafld_ids & cirrmri_ids
    if overlap:
        raise RuntimeError(
            f"Patient ID collision between NAFLD and CirrMRI600+: {overlap}. "
            "This would corrupt the unlabeled pretraining corpus provenance. "
            "Aborting."
        )


def _generate_unlabeled_slices() -> None:
    """
    Merge NAFLD processed images and CirrMRI600+ unlabeled slices into
    a single ``data/processed/unlabeled_slices/`` directory for SimCLR
    pre-training (Phase 2).
    """
    dst_dir = PROC_DIR / "unlabeled_slices"
    dst_dir.mkdir(parents=True, exist_ok=True)

    # Clear any previous run
    existing = list(dst_dir.glob("slice_*.npy"))
    if len(existing) > 0:
        print(f"[Unlabeled Slices] {len(existing)} slices already exist, regenerating...")
        for f in existing:
            f.unlink()

    idx = 0
    nafld_count = 0
    cirrmri_count = 0

    # --- Source 1: NAFLD processed images ---
    nafld_src = PROC_DIR / "nafld_images"
    if nafld_src.exists():
        for npy_file in sorted(nafld_src.glob("*.npy")):
            arr = np.load(npy_file)
            out_path = dst_dir / f"slice_{idx:04d}.npy"
            # Ensure shape is (C, H, W) with C=1 for SimCLR augment to expand to 3
            if arr.ndim == 3 and arr.shape[0] == 3:
                np.save(out_path, arr[:1])
            elif arr.ndim == 3 and arr.shape[0] == 1:
                np.save(out_path, arr)
            elif arr.ndim == 2:
                np.save(out_path, arr[None, ...])
            else:
                np.save(out_path, arr)
            idx += 1
            nafld_count += 1

    # --- Source 2: CirrMRI600+ unlabeled slices ---
    cirrmri_src = PROC_DIR / "cirrmri_unlabeled_slices"
    if cirrmri_src.exists():
        for npy_file in sorted(cirrmri_src.glob("*.npy")):
            arr = np.load(npy_file)
            out_path = dst_dir / f"slice_{idx:04d}.npy"
            # CirrMRI slices are already (1, 224, 224), copy directly
            if arr.ndim == 2:
                np.save(out_path, arr[None, ...])
            else:
                np.save(out_path, arr)
            idx += 1
            cirrmri_count += 1

    print(
        f"[Unlabeled Slices] Generated {idx} slices in {dst_dir} "
        f"(NAFLD: {nafld_count}, CirrMRI600+: {cirrmri_count})"
    )


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

    # CirrMRI600+ unlabeled pretraining corpus (T2_2D slices for SimCLR)
    try:
        results["preprocess_cirrmri_unlabeled"] = preprocess_cirrmri_unlabeled()
    except Exception as exc:
        results["preprocess_cirrmri_unlabeled"] = {"error": str(exc)}
        print(f"[preprocess_cirrmri_unlabeled] FAILED: {exc}")

    # --- Patient ID collision assertion ---
    # NAFLD IDs are strings like "id1", "id2"; CirrMRI IDs are numeric strings
    # like "10", "100". They should never overlap, but assert rather than assume.
    try:
        nafld_ids: Set[str] = set()
        nafld_csv = PROC_DIR / "nafld_paired_train.csv"
        if nafld_csv.exists():
            ndf = pd.read_csv(nafld_csv)
            if "image_path" in ndf.columns:
                for p in ndf["image_path"].dropna():
                    # Extract patient ID from path like "...nafld_images/id5_processed.npy"
                    stem = Path(str(p)).stem  # "id5_processed"
                    pid = stem.replace("_processed", "")
                    nafld_ids.add(pid)

        cirrmri_ids: Set[str] = set()
        manifest_path = PROC_DIR / "cirrmri_unlabeled_manifest.json"
        if manifest_path.exists():
            mf = json.loads(manifest_path.read_text(encoding="utf-8"))
            cirrmri_ids.update(f"cirrmri_{k}" for k in mf.get("cirrhosis", {}))
            cirrmri_ids.update(f"cirrmri_healthy_{k}" for k in mf.get("healthy", {}))

        if nafld_ids and cirrmri_ids:
            _assert_no_patient_id_collision(nafld_ids, cirrmri_ids)
            print(f"[ID Collision Check] OK — {len(nafld_ids)} NAFLD vs {len(cirrmri_ids)} CirrMRI IDs, no overlap")
    except RuntimeError:
        raise  # Re-raise collision errors — these are fatal
    except Exception as exc:
        print(f"[ID Collision Check] WARNING: could not verify — {exc}")

    # Generate merged unlabeled slices for SimCLR pre-training
    # (must run AFTER both NAFLD and CirrMRI preprocessing)
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