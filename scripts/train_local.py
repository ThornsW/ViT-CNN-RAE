"""Train the attention-guided LOCAL variant.

    python scripts/train_local.py --top-k 0.2
    python scripts/train_local.py --top-k 0.2 --bg-weight 0.3 --eps 0.062745098   # 16/255
    python scripts/train_local.py --resume outputs/<run>/models/last.pth
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
from vit_cnn_rae.attacks import LocalAttack
from vit_cnn_rae.data import MyDataset, default_transform
from vit_cnn_rae.targets import load_target_model
from vit_cnn_rae.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--top-k', type=float, default=0.2)
    p.add_argument('--bg-weight', type=float, default=0.0,
                   help='非 top-k(背景)区域的扰动权重: 0=硬mask, 0<a<1=软mask, 1=全图')
    p.add_argument('--eps', type=float, default=1.0,
                   help='L∞扰动上限,图像范围为[0,1];16/255≈0.062745')
    p.add_argument('--attn-model', default='vit_base_patch16_224')
    p.add_argument('--target', default='densenet121',
                   choices=['densenet121', 'resnet50', 'mobilenet_v3_large'])
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=22)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--resume', type=Path, default=None)
    return p.parse_args()


def main():
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"device={device} seed={args.seed} top_k={args.top_k} "
          f"bg_weight={args.bg_weight} eps={args.eps}")

    target = load_target_model(args.target, device=device)
    train_data = MyDataset(txt=config.split_path('dataset-trn.txt'),
                           root=config.DATA_ROOT, transform=default_transform)
    loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                        pin_memory=torch.cuda.is_available(), num_workers=1)

    if args.resume:
        models_dir = args.resume.resolve().parent  # 续训:产物写回原 run 目录,不新建
    else:
        tag = f"local_topk{args.top_k}_bg{args.bg_weight}_eps{args.eps}_s{args.seed}"
        models_dir = config.run_dir(tag=tag, model="srae_local") / "models"
    print(f"output: {models_dir.parent}")
    if not args.resume:
        config.save_run_config(models_dir.parent, vars(args), model="srae_local")

    attacker = LocalAttack(device, target, config.NUM_CLASSES, config.IMAGE_CHANNELS,
                           box_min=0, box_max=1, clip=args.eps,
                           top_k_ratio=args.top_k, bg_weight=args.bg_weight,
                           attn_model_name=args.attn_model,
                           models_path=models_dir)
    if args.resume:
        attacker.load_checkpoint(args.resume)

    attacker.train(loader, args.epochs)


if __name__ == '__main__':
    main()
