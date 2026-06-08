"""Paths + run_id. Single source of truth for the project.

Env overrides:
    DATA_ROOT       — dataset root holding <dataset>/ images  (default: <repo>/data)
    DATASET         — dataset name under data/splits/  (default: caltech256)
    CHECKPOINT_DIR  — target classifier weights (default: <repo>/checkpoints)
    OUTPUT_DIR      — training products (default: <repo>/outputs)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
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


def _git_sha() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)


def save_run_config(run_dir: Path, params: dict, model: str = "") -> Path:
    """Record a run's initial hyperparameters to <run_dir>/run_config.json.

    Called once at training start so every run is self-describing: the argparse
    params plus git commit, start time, dataset and library versions.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_dir": run_dir.name,
        "model": model,
        "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": _git_sha(),
        "dataset": DATASET,
        "python": sys.version.split()[0],
        "params": {k: _jsonable(v) for k, v in params.items()},
    }
    try:
        import torch
        record["torch"] = torch.__version__
        record["cuda"] = bool(torch.cuda.is_available())
    except Exception:
        pass
    f = run_dir / "run_config.json"
    f.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return f
