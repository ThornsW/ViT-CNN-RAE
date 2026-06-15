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
    p.add_argument('--clip', type=float, default=1.0,
                   help='扰动 L∞ 上限;local run 必须等于训练时的 eps(否则对抗图与评估不一致)')
    p.add_argument('--models-dir', type=Path, default=None)
    p.add_argument('--out-dir', type=Path, default=None,
                   help='where to write images (default: outputs/visualizations/save_test)')
    p.add_argument('--local', action='store_true',
                   help='LocalAttack run:G 用 ViT top-k mask 门控还原局部扰动(需 timm+ViT)')
    p.add_argument('--top-k', type=float, default=0.2, help='local: top-k 区域比例')
    p.add_argument('--bg-weight', type=float, default=0.0, help='local: 背景区域扰动权重')
    p.add_argument('--attn-model', default='vit_base_patch16_224')
    p.add_argument('--limit', type=int, default=None,
                   help='只生成前 N 张样本(展示用);默认遍历整个 val 集(2570 张)')
    return p.parse_args()


def main():
    args = parse_args()
    out = save_visualizations(target_name=args.target,
                              g_ckpt=args.g,
                              r_ckpt=args.r,
                              batch_size=args.batch_size,
                              clip=args.clip,
                              models_dir=args.models_dir,
                              out_dir=args.out_dir,
                              local=args.local,
                              top_k_ratio=args.top_k,
                              bg_weight=args.bg_weight,
                              attn_model=args.attn_model,
                              limit=args.limit)
    print(f"saved visualizations to {out}")


if __name__ == '__main__':
    main()
