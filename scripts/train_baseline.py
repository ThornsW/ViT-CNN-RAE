"""Train SRAE baseline (full-image perturbation).

    DATA_ROOT=~/data python scripts/train_baseline.py
    python scripts/train_baseline.py --seed 123
    python scripts/train_baseline.py --resume outputs/<run>/models/last.pth
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from vit_cnn_rae import config
from vit_cnn_rae.attacks import Attack
from vit_cnn_rae.data import MyDataset, default_transform
from vit_cnn_rae.targets import load_target_model
from vit_cnn_rae.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--target', default='densenet121',
                   choices=['densenet121', 'resnet50', 'mobilenet_v3_large'])
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=22)
    p.add_argument('--eps', type=float, default=1.0,
                   help='L∞ 扰动上限;baseline 默认 1.0(不限);对照组用 0.125=32/255')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--trn-split', default='dataset-trn.txt',
                   help='训练 split 文件名;轻量扫参用 dataset-trn-n30.txt')
    p.add_argument('--lr-drops', type=int, nargs=2, default=[50, 100],
                   metavar=('E1', 'E2'),
                   help='lr 从 1e-3 降到 1e-4/1e-5 的两个 epoch;轻量: 20 40')
    return p.parse_args()


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    args = parse_args()
    set_seed(args.seed, deterministic=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed}")

    target = load_target_model(args.target, device=device)
    train_data = MyDataset(txt=config.split_path(args.trn_split),
                           root=config.DATA_ROOT, transform=default_transform)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        pin_memory=torch.cuda.is_available(), num_workers=4)

    tag = f"baseline_eps{args.eps}_s{args.seed}" if args.eps != 1.0 else f"baseline_s{args.seed}"
    models_dir = config.run_dir(tag=tag) / "models"
    print(f"output: {models_dir.parent}")
    config.save_run_config(models_dir.parent, vars(args), model="srae")

    attacker = Attack(device, target, config.NUM_CLASSES, config.IMAGE_CHANNELS,
                      box_min=0, box_max=1, clip=args.eps, models_path=models_dir)
    if args.resume:
        attacker.load_checkpoint(args.resume)

    attacker.train(loader, args.epochs, lr_drops=args.lr_drops)


if __name__ == '__main__':
    main()
