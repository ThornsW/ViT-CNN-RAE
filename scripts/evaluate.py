"""Evaluate trained G and R: print ASR + PSNR/SSIM/L0/L2/Linf.

    DATA_ROOT=~/data python scripts/evaluate.py [--g <ckpt>] [--r <ckpt>]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae.evaluation import evaluate


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='densenet121')
    p.add_argument('--g', default='netG_epoch_150_1.pth')
    p.add_argument('--r', default='netR_epoch_150_1.pth')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--models-dir', type=Path, default=None,
                   help='directory with the G and R checkpoints (default: outputs/models)')
    return p.parse_args()


def main():
    args = parse_args()
    evaluate(target_name=args.target,
             g_ckpt=args.g,
             r_ckpt=args.r,
             batch_size=args.batch_size,
             models_dir=args.models_dir)


if __name__ == '__main__':
    main()
