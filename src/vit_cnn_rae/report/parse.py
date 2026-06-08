"""Parse outputs/<run>/ products for the report web app.

Stdlib only (ast/re/pathlib) + config for paths — deliberately does NOT import
torch so the web app stays light. test.txt / loss1.txt hold Python literals
(repr of a dict / lists), so we parse them with ast.literal_eval, not json.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path

from .. import config

IMAGE_PIXELS = config.IMAGE_SIZE * config.IMAGE_SIZE * config.IMAGE_CHANNELS  # 150528

# Top-level dirs under outputs/ that are shared buckets, not training runs.
_NON_RUN = {"models", "logs", "visualizations", "attention_viz"}
# <YYYYMMDD>_<HHMMSS>_<model>_<tag...>[_s<seed>]
_RUN_NAME_RE = re.compile(r"^(\d{8})_(\d{6})_([A-Za-z0-9]+?)_(.+?)(?:_s(\d+))?$")
_TS_PREFIX_RE = re.compile(r"^\d{8}_\d{6}_")


@dataclass
class RunInfo:
    name: str
    path: Path
    date: str | None          # "2026-06-02 02:46:46"
    model: str | None
    tag: str | None
    seed: int | None
    has_metrics: bool
    has_loss: bool


def parse_run_name(name: str) -> dict:
    """Split a run dir name into date / model / tag / seed (best effort)."""
    m = _RUN_NAME_RE.match(name)
    if not m:
        return {"date": None, "model": None, "tag": name, "seed": None}
    d, t, model, tag, seed = m.groups()
    date = f"{d[:4]}-{d[4:6]}-{d[6:8]} {t[:2]}:{t[2:4]}:{t[4:6]}"
    return {"date": date, "model": model, "tag": tag, "seed": int(seed) if seed else None}


def _is_run_dir(p: Path) -> bool:
    if not p.is_dir() or p.name in _NON_RUN:
        return False
    if _TS_PREFIX_RE.match(p.name):
        return True
    return (p / "models").is_dir() or (p / "test.txt").is_file()


def discover_runs(outputs_dir: Path | None = None) -> list[RunInfo]:
    """All run dirs under outputs/, newest first (timestamp prefix sorts)."""
    outputs_dir = Path(outputs_dir) if outputs_dir else config.OUTPUT_DIR
    runs: list[RunInfo] = []
    if not outputs_dir.is_dir():
        return runs
    for p in outputs_dir.iterdir():
        if not _is_run_dir(p):
            continue
        meta = parse_run_name(p.name)
        runs.append(RunInfo(
            name=p.name, path=p,
            date=meta["date"], model=meta["model"], tag=meta["tag"], seed=meta["seed"],
            has_metrics=(p / "test.txt").is_file(),
            has_loss=(p / "models" / "loss1.txt").is_file(),
        ))
    runs.sort(key=lambda r: r.name, reverse=True)
    return runs


def get_run(name: str, outputs_dir: Path | None = None) -> RunInfo | None:
    for r in discover_runs(outputs_dir):
        if r.name == name:
            return r
    return None


def parse_metrics(run_path: Path) -> list[dict]:
    """Each non-empty line of test.txt is one eval summary dict (repr)."""
    f = Path(run_path) / "test.txt"
    if not f.is_file():
        return []
    out: list[dict] = []
    for line in f.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            val = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue
        if isinstance(val, dict):
            out.append(val)
    return out


def _mean(d: dict, key: str):
    v = d.get(key)
    return v.get("mean") if isinstance(v, dict) else None


def metric_summary(d: dict) -> dict:
    """Flatten one eval dict to the headline fields the tables show."""
    l0a, l0r = _mean(d, "l0_adv_ori"), _mean(d, "l0_r_ori")
    return {
        "asr": d.get("generator_error_rate"),
        "clean_err": d.get("target_error_rate"),
        "recovered_err": d.get("remover_error_rate"),
        "adv_psnr": _mean(d, "psnr_adv_ori"),
        "adv_ssim": _mean(d, "ssim_adv_ori"),
        "adv_l2": _mean(d, "l2_adv_ori"),
        "adv_linf": _mean(d, "l_inf_adv_ori"),
        "adv_l0_pct": (l0a / IMAGE_PIXELS * 100) if l0a is not None else None,
        "rec_psnr": _mean(d, "psnr_r_ori"),
        "rec_ssim": _mean(d, "ssim_r_ori"),
        "rec_l2": _mean(d, "l2_r_ori"),
        "rec_linf": _mean(d, "l_inf_r_ori"),
        "rec_l0_pct": (l0r / IMAGE_PIXELS * 100) if l0r is not None else None,
    }


def parse_loss(run_path: Path) -> dict[str, list[float]]:
    """models/loss1.txt: 6 lines of `loss_name:[float, ...]`."""
    f = Path(run_path) / "models" / "loss1.txt"
    if not f.is_file():
        return {}
    out: dict[str, list[float]] = {}
    for line in f.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        name, _, rest = line.partition(":")
        try:
            vals = ast.literal_eval(rest.strip())
        except (ValueError, SyntaxError):
            continue
        if isinstance(vals, list):
            out[name.strip()] = [float(x) for x in vals]
    return out


def _rel(p: Path, base: Path) -> str:
    return str(p.resolve().relative_to(base.resolve()))


def find_global_images(outputs_dir: Path | None = None) -> dict[str, list[str]]:
    """Global image galleries under outputs/ — NOT tied to any single run.

    attention : outputs/attention_viz/*.png                (ViT attention over dataset)
    samples   : outputs/visualizations/save_test/**/*.png   (ori/adv/r_adv exemplars)

    Both sit at the outputs/ root (siblings of run dirs), so they belong to the
    project, not a run. Paths are relative to outputs/ (for the /file route).

    Note: a run's models/*.png are training-time loss snapshots — already covered
    by the re-rendered loss chart, so they are intentionally not surfaced.
    """
    outputs_dir = Path(outputs_dir) if outputs_dir else config.OUTPUT_DIR
    attn_dir = outputs_dir / "attention_viz"
    save_test = outputs_dir / "visualizations" / "save_test"
    attention = sorted(attn_dir.glob("*.png")) if attn_dir.is_dir() else []
    samples = sorted(save_test.rglob("*.png")) if save_test.is_dir() else []
    return {
        "attention": [_rel(p, outputs_dir) for p in attention],
        "samples": [_rel(p, outputs_dir) for p in samples],
    }


def find_logs(run_path: Path) -> list[Path]:
    return sorted(Path(run_path).glob("*.log"))


def safe_under(base: Path, target: Path) -> bool:
    """True iff resolved `target` lives inside resolved `base` (path-traversal guard)."""
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except (ValueError, RuntimeError):
        return False


def parse_run_config(run_path: Path) -> dict | None:
    """run_config.json written at training start; None for absent / pre-feature runs."""
    f = Path(run_path) / "run_config.json"
    if not f.is_file():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None
