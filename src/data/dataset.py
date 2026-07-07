"""
PyTorch Dataset classes for CMCHT-XAI.

    - UnlabeledImagingDataset: for SimCLR self-supervised pre-training (Phase 2).
    - MultimodalLiverDataset: paired image + tabular data for training/eval.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ==============================================================================
# SimCLR augmentations
# ==============================================================================
class SimCLRAugment:
    """Stochastic augmentations producing two correlated views of an image."""

    def __init__(self, image_size: int = 224, jitter_strength: float = 0.4):
        self.image_size = image_size
        self.jitter = jitter_strength

    def __call__(self, img: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # img: (C, H, W) float tensor
        def _view():
            x = img.clone()
            # random crop (pad if needed)
            x = self._random_crop(x)
            # color jitter
            x = x + torch.randn_like(x) * self.jitter * 0.1
            # random horizontal flip
            if torch.rand(1).item() < 0.5:
                x = torch.flip(x, dims=[-1])
            # gaussian blur
            if torch.rand(1).item() < 0.5:
                k = 3
                pad = k // 2
                kernel = torch.ones(1, 1, k, k) / (k * k)
                c = x.shape[0]
                x = torch.nn.functional.conv2d(
                    x.unsqueeze(0),
                    kernel.expand(c, 1, k, k),
                    padding=pad,
                    groups=c,
                ).squeeze(0)
            return x

        return _view(), _view()

    def _random_crop(self, x: torch.Tensor) -> torch.Tensor:
        _, h, w = x.shape
        if h >= self.image_size and w >= self.image_size:
            top = torch.randint(0, h - self.image_size + 1, (1,)).item()
            left = torch.randint(0, w - self.image_size + 1, (1,)).item()
            return x[:, top:top + self.image_size, left:left + self.image_size]
        # resize via interpolation if smaller
        return torch.nn.functional.interpolate(
            x.unsqueeze(0), size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False
        ).squeeze(0)


# ==============================================================================
# Unlabeled imaging dataset (SimCLR)
# ==============================================================================
class UnlabeledImagingDataset(Dataset):
    """Dataset of unlabeled 2D liver slices for SimCLR pre-training."""

    def __init__(self, image_dir: str, image_size: int = 224,
                 augment: Optional[SimCLRAugment] = None,
                 file_pattern: str = "*.npy"):
        self.image_dir = Path(image_dir)
        self.files = sorted(self.image_dir.glob(file_pattern))
        self.image_size = image_size
        self.augment = augment or SimCLRAugment(image_size)
        if not self.files:
            logger.warning("No images found in %s", image_dir)

    def __len__(self) -> int:
        return len(self.files)

    def _load_image(self, path: Path) -> torch.Tensor:
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        return torch.from_numpy(arr).float()

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self._load_image(self.files[idx])
        view1, view2 = self.augment(img)
        return view1, view2


# ==============================================================================
# Multimodal paired dataset
# ==============================================================================
class MultimodalLiverDataset(Dataset):
    """
    Paired image + tabular dataset.

    Expects a CSV with feature columns + detection_label / staging_label /
    severity_label, and either an `image_path` column or an `image_index`
    column referencing a stacked .npy array.
    """

    def __init__(
        self,
        csv_path: str,
        feature_names: Sequence[str],
        image_array_path: Optional[str] = None,
        image_dir: Optional[str] = None,
        image_size: int = 224,
        image_channels: int = 3,
        transform: Optional[Callable] = None,
    ):
        self.df = pd.read_csv(csv_path)
        self.feature_names = list(feature_names)
        self.image_size = image_size
        self.image_channels = image_channels
        self.transform = transform

        self.image_array = None
        if image_array_path and Path(image_array_path).exists():
            self.image_array = np.load(image_array_path)
            logger.info("Loaded image array %s with shape %s", image_array_path, self.image_array.shape)
        self.image_dir = Path(image_dir) if image_dir else None

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, row: pd.Series) -> torch.Tensor:
        if self.image_array is not None and "image_index" in row:
            img = self.image_array[int(row["image_index"])]
        elif "image_path" in row and pd.notna(row.get("image_path", None)) and str(row["image_path"]).strip():
            img_path = Path(str(row["image_path"]))
            # Support both absolute paths and relative paths under image_dir
            if not img_path.is_absolute() and self.image_dir is not None:
                img_path = self.image_dir / img_path
            if img_path.exists():
                img = np.load(str(img_path))
            else:
                img = np.zeros((self.image_size, self.image_size), dtype=np.float32)
        else:
            # fallback: blank image
            img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        if img.ndim == 3 and img.shape[0] == 1:
            img = img[0]
        if img.ndim == 2:
            img = np.stack([img] * self.image_channels, axis=0)
        elif img.ndim == 3 and img.shape[0] != self.image_channels:
            img = img[:self.image_channels]

        t = torch.from_numpy(img).float()
        if t.shape[-1] != self.image_size or t.shape[-2] != self.image_size:
            t = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            ).squeeze(0)
        if self.transform is not None:
            t = self.transform(t)
        return t

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        tabular = torch.tensor(
            [float(row[f]) for f in self.feature_names], dtype=torch.float32
        )
        image = self._load_image(row)

        return {
            "image": image,
            "tabular": tabular,
            "detection_label": torch.tensor(float(row.get("detection_label", 0)), dtype=torch.float32),
            "staging_label": torch.tensor(int(row.get("staging_label", 0)), dtype=torch.long),
            "severity_label": torch.tensor(float(row.get("severity_label", 0.0)), dtype=torch.float32),
        }


def _infer_stage_num_classes(csv_path: Path, fallback: int) -> int:
    """Infer the staging class count from a processed CSV when available."""
    if not csv_path.exists():
        return fallback

    try:
        df = pd.read_csv(csv_path, usecols=["staging_label"])
    except Exception:
        return fallback

    if "staging_label" not in df.columns or df.empty:
        return fallback

    unique_labels = sorted(pd.Series(df["staging_label"]).dropna().astype(int).unique().tolist())
    if not unique_labels:
        return fallback
    return max(unique_labels) + 1


def apply_dataset_config(
    cfg,
    feature_names: Sequence[str],
    stage_num_classes: Optional[int] = None,
) -> None:
    """Mutate the config so model shapes match the selected processed dataset."""
    feature_names = list(feature_names)
    cfg.data.tabular_features = feature_names
    cfg.data.num_tabular_features = len(feature_names)
    if stage_num_classes is not None:
        cfg.model.heads.staging.num_classes = int(stage_num_classes)


def resolve_experiment_dataset(
    cfg,
    require_train: bool = True,
    require_test: bool = True,
) -> Dict[str, object]:
    """
    Resolve the processed dataset to use for the hybrid pipeline.

    Prefers the configured primary dataset and falls back when its processed
    files are unavailable. The config is updated in-place so downstream model
    builders use the same tabular feature count and stage-class count.
    """
    processed = Path(cfg.data.processed_dir)
    primary = str(cfg.data.get("primary_dataset", "nafld")).lower()
    nafld_features = list(cfg.data.get("nafld_tabular_features", cfg.data.tabular_features))
    default_stage_classes = int(cfg.model.heads.staging.num_classes)

    nafld_val_csv = processed / "nafld_paired_val.csv"
    nafld_candidate = {
        "name": "nafld",
        "train_csv": processed / "nafld_paired_train.csv",
        "test_csv": processed / "nafld_paired_test.csv",
        "feature_names": nafld_features,
        "image_dir": processed / "nafld_images",
        "stage_num_classes": 3,
    }
    if nafld_val_csv.exists():
        nafld_candidate["val_csv"] = nafld_val_csv

    candidates = {
        "nafld": nafld_candidate,
        "cirrhosis": {
            "name": "cirrhosis",
            "train_csv": processed / "cirrhosis_train.csv",
            "test_csv": processed / "cirrhosis_test.csv",
            "feature_names": list(cfg.data.tabular_features),
            "image_dir": None,
            "stage_num_classes": default_stage_classes,
        },
    }

    order = [primary] if primary in candidates else []
    order.extend(name for name in ("nafld", "cirrhosis") if name not in order)

    missing_reasons = []
    for name in order:
        candidate = dict(candidates[name])
        train_csv = candidate["train_csv"]
        test_csv = candidate["test_csv"]

        if require_train and not train_csv.exists():
            missing_reasons.append(f"{name}: missing {train_csv}")
            continue
        if require_test and not test_csv.exists():
            missing_reasons.append(f"{name}: missing {test_csv}")
            continue

        reference_csv = train_csv if train_csv.exists() else test_csv
        stage_num_classes = _infer_stage_num_classes(
            reference_csv,
            int(candidate["stage_num_classes"]),
        )
        apply_dataset_config(cfg, candidate["feature_names"], stage_num_classes=stage_num_classes)

        image_dir = candidate["image_dir"]
        image_dir_str = str(image_dir) if image_dir is not None and image_dir.exists() else None
        candidate["image_dir"] = image_dir_str
        candidate["stage_num_classes"] = stage_num_classes
        return candidate

    detail = "; ".join(missing_reasons) if missing_reasons else "no processed dataset candidates found"
    raise FileNotFoundError(
        f"Missing processed data in {processed}. Run: python -m src.data.preprocessing. Details: {detail}"
    )

