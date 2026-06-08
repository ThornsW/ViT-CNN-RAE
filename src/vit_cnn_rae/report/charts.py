"""Render training loss curves to PNG bytes (matplotlib, headless Agg).

Re-plots from the parsed loss1.txt instead of reusing the run's stored single
PNGs, so the report controls the layout. Results are cached by (run, mtime).
"""
from __future__ import annotations

import io
import math

import matplotlib

matplotlib.use("Agg")  # no display on cloud / inside the web server
import matplotlib.pyplot as plt  # noqa: E402

_cache: dict = {}

_LABELS = {
    "loss_D": "Discriminator (loss_D)",
    "loss_G_fake": "Generator → fool target (loss_G_fake)",
    "loss_perturb": "Perturbation magnitude (loss_perturb)",
    "loss_adv": "Adversarial (loss_adv)",
    "loss_r_adv": "Recovered → correct (loss_r_adv)",
    "loss_l2_r_pert": "Recover residual L2 (loss_l2_r_pert)",
}


def render_loss_figure(loss: dict[str, list[float]]) -> bytes:
    """One subplot per loss series in a 2-column grid; return PNG bytes."""
    items = [(k, v) for k, v in loss.items() if v]
    n = len(items)
    if n == 0:
        raise ValueError("empty loss dict")
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.9 * rows), squeeze=False)
    flat = axes.flatten()
    for ax, (name, vals) in zip(flat, items):
        ax.plot(range(1, len(vals) + 1), vals, color="#2563eb", linewidth=1.4)
        ax.set_title(_LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
    for ax in flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def render_loss_cached(key: str, mtime: float, loss: dict[str, list[float]]) -> bytes:
    """Memoize the rendered PNG; invalidates when loss1.txt mtime changes."""
    ck = (key, mtime)
    png = _cache.get(ck)
    if png is None:
        png = _cache[ck] = render_loss_figure(loss)
    return png


_LOSS_ORDER = ["loss_D", "loss_G_fake", "loss_perturb", "loss_adv", "loss_r_adv", "loss_l2_r_pert"]


def render_compare_figure(runs) -> bytes:
    """Overlay same-named loss series across runs.

    `runs` = [(label, loss_dict), ...]; one subplot per loss name, one line per run.
    """
    runs = [(lbl, d) for lbl, d in runs if d]
    if not runs:
        raise ValueError("nothing to compare")
    present = {n for _, d in runs for n in d}
    names = [n for n in _LOSS_ORDER if n in present] + sorted(present - set(_LOSS_ORDER))
    n = len(names)
    cols = 2 if n > 1 else 1
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.9 * rows), squeeze=False)
    flat = axes.flatten()
    for ax, name in zip(flat, names):
        for label, d in runs:
            vals = d.get(name)
            if vals:
                ax.plot(range(1, len(vals) + 1), vals, linewidth=1.3, label=label)
        ax.set_title(_LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    for ax in flat[n:]:
        ax.axis("off")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def render_compare_cached(key, runs) -> bytes:
    """Memoize the overlay PNG; `key` should encode each run's loss1.txt mtime."""
    png = _cache.get(key)
    if png is None:
        png = _cache[key] = render_compare_figure(runs)
    return png
