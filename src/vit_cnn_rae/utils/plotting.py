"""Loss curve plotting helper."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def draw_loss_curve(out_dir: Path, data: list[float], label: str) -> None:
    """Save a single loss curve as <out_dir>/<label>1.png."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.title("Loss During Training")
    plt.plot(data, label=str(label))
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(out_dir / f"{label}1.png")
    plt.close()
