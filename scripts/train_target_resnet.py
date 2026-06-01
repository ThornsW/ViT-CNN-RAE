"""Fine-tune ResNet50 on Caltech-256 → checkpoints/ResNet50.pth."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae.targets import train_classifier


if __name__ == '__main__':
    train_classifier('resnet50')
