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
    p.add_argument('--target', default=None)
    p.add_argument('--g', default='netG_epoch_150_1.pth')
    p.add_argument('--r', default='netR_epoch_150_1.pth')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--models-dir', type=Path, default=None,
                   help='G/R checkpoint 目录(默认 outputs/models);其余参数默认从上级 run_config.json 读')
    p.add_argument('--local', action=argparse.BooleanOptionalAction, default=None,
                   help='是否 LocalAttack run;默认从 run_config 的 model 字段自动判断')
    p.add_argument('--top-k', type=float, default=None)
    p.add_argument('--bg-weight', type=float, default=None)
    p.add_argument('--eps', type=float, default=None,
                   help='L∞扰动上限;默认从 run_config 读,显式传参覆盖')
    p.add_argument('--attn-model', default=None)
    p.add_argument('--gated-recovery', action='store_true',
                   help='恢复时 R 输出乘 mask;同时报告密钥门控(clean 算 mask)与自恢复(adv 算 mask)')
    return p.parse_args()


def main():
    args = parse_args()
    evaluate(target_name=args.target,
             g_ckpt=args.g,
             r_ckpt=args.r,
             batch_size=args.batch_size,
             clip=args.eps,
             models_dir=args.models_dir,
             local=args.local,
             top_k_ratio=args.top_k,
             bg_weight=args.bg_weight,
             attn_model=args.attn_model,
             gated_recovery=args.gated_recovery)


if __name__ == '__main__':
    main()
