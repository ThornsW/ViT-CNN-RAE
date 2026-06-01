"""Save 9 per-sample images (ori / adv / r_adv / perturbations / diffs).

    DATA_ROOT=~/data python scripts/save_visualizations.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from vit_cnn_rae.visualization import save_visualizations


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', default='densenet121')
    p.add_argument('--g', default='netG_epoch_150_1.pth')
    p.add_argument('--r', default='netR_epoch_150_1.pth')
    p.add_argument('--batch-size', type=int, default=30)
    p.add_argument('--models-dir', type=Path, default=None)
    p.add_argument('--out-dir', type=Path, default=None,
                   help='where to write images (default: outputs/visualizations/save_test)')
    return p.parse_args()


def main():
    args = parse_args()
    out = save_visualizations(target_name=args.target,
                              g_ckpt=args.g,
                              r_ckpt=args.r,
                              batch_size=args.batch_size,
                              models_dir=args.models_dir,
                              out_dir=args.out_dir)
    print(f"saved visualizations to {out}")


if __name__ == '__main__':
    main()
