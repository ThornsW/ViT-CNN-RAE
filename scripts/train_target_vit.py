"""Fine-tune a timm ViT/DeiT/Swin victim on Caltech-256 for the transfer matrix.

    python scripts/train_target_vit.py --model vit_base_patch16_224
    python scripts/train_target_vit.py --model deit3_small_patch16_224   # off-family

The transfer matrix needs victims the attack has NOT been trained on. Include at
least one model from a different family than the vit_base attention extractor
(deit3 / swin) so a CNN->ViT transfer result is not "self-attack". Only the head
is trained (frozen backbone), so a few epochs to ~80% val acc is enough — this
is an evaluation target, not a contribution.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # 见 memory: 拉 HF 权重须禁 Xet,否则 SSL 断

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae.targets import train_classifier


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', default='vit_base_patch16_224',
                   choices=['vit_base_patch16_224', 'deit3_small_patch16_224',
                            'swin_tiny_patch4_window7_224'])
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch-size', type=int, default=64)
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    train_classifier(a.model, epochs=a.epochs, batch_size=a.batch_size)
