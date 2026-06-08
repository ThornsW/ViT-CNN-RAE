"""Paths + run_id. Single source of truth for the project.

Env overrides:
    DATA_ROOT       — dataset root holding <dataset>/ images  (default: <repo>/data)
    DATASET         — dataset name under data/splits/  (default: caltech256)
    CHECKPOINT_DIR  — target classifier weights (default: <repo>/checkpoints)
    OUTPUT_DIR      — training products (default: <repo>/outputs)
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]

DATA_ROOT: Path = Path(os.environ.get("DATA_ROOT", PROJECT_ROOT / "data"))
DATASET: str = os.environ.get("DATASET", "caltech256")
SPLITS_DIR: Path = PROJECT_ROOT / "data" / "splits" / DATASET
CHECKPOINT_DIR: Path = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
OUTPUT_DIR: Path = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs"))
MODELS_OUT: Path = OUTPUT_DIR / "models"

NUM_CLASSES: int = 257
IMAGE_SIZE: int = 224
IMAGE_CHANNELS: int = 3


def split_path(name: str) -> Path:
    return SPLITS_DIR / name


def checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / name


def ensure_output_dirs() -> None:
    MODELS_OUT.mkdir(parents=True, exist_ok=True)


def run_dir(tag: str, model: str = "srae") -> Path:
    """Make a timestamped run directory under outputs/<YYYYMMDD_HHMMSS_model_tag>/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_tag = "".join(c if c.isalnum() or c in "_-" else "_" for c in tag.lower())
    d = OUTPUT_DIR / f"{ts}_{model}_{safe_tag}"
    d.mkdir(parents=True, exist_ok=True)
    return d
