"""Fine-tune DenseNet121 on Caltech-256 → checkpoints/DenseNet121.pth.

The repo ships a pre-trained checkpoint; only re-run this if you need to retrain.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae.targets import train_classifier


if __name__ == '__main__':
    train_classifier('densenet121')
