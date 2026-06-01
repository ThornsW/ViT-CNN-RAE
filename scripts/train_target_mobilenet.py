"""Fine-tune MobileNetV3-Large on Caltech-256 → checkpoints/MoblieNetV3.pth."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae.targets import train_classifier


if __name__ == '__main__':
    train_classifier('mobilenet_v3_large')
