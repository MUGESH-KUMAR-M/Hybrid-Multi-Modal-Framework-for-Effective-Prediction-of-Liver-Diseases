"""
Logging and reproducibility helpers for CMCHT-XAI.

Provides:
    - get_logger(name): configured console + file logger
    - set_seed(seed): deterministic seeding across python/random/numpy/torch
    - load_config(path): YAML config loader with attribute-style access
"""
from __future__ import annotations

import logging
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore


# ------------------------------------------------------------------------------
# Config loading
# ------------------------------------------------------------------------------
class Config(dict):
    """Dict that also supports attribute access (cfg.training.epochs)."""

    def __init__(self, data: Dict[str, Any] | None = None):
        super().__init__()
        if data:
            for k, v in data.items():
                self[k] = self._wrap(v)

    @staticmethod
    def _wrap(v: Any) -> Any:
        if isinstance(v, dict):
            return Config(v)
        if isinstance(v, list):
            return [Config._wrap(x) for x in v]
        return v

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = self._wrap(value)


def load_config(path: str | os.PathLike) -> Config:
    """Load a YAML config file into a Config (attribute-accessible dict)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return Config(data)


# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str = "cmcht_xai",
    log_dir: str | os.PathLike | None = None,
    level: int = logging.INFO,
    log_to_file: bool = True,
) -> logging.Logger:
    """Return a configured logger with console + optional file handler."""
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    if not logger.handlers:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(fmt)
        logger.addHandler(ch)

        if log_to_file and log_dir is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(Path(log_dir) / f"{name}.log", encoding="utf-8")
            fh.setFormatter(fmt)
            logger.addHandler(fh)

    _LOGGERS[name] = logger
    return logger


# ------------------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    """Seed python, numpy, and torch for reproducibility."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if np is not None:
        np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_device(prefer_cuda: bool = True) -> str:
    """Return 'cuda' if available and preferred, else 'cpu'."""
    if torch is not None and prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"