"""可视化 ViT 注意力 → top-k mask,Go/No-Go 决策 + 组会汇报用图。

每张图出 4-panel:
    1. 原图
    2. ViT attention rollout(jet heatmap 叠在原图)
    3. top-k 二值 mask(红色 overlay)
    4. 模拟局部扰动效果(noise × mask + 原图)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from vit_cnn_rae import config
from vit_cnn_rae.attention import (
    ViTAttentionExtractor, make_topk_mask, normalize_for_vit,
)
from vit_cnn_rae.data import MyDataset, default_transform


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=8, help='要可视化的图片张数')
    p.add_argument('--top-k', type=float, default=0.2, help='ViT mask 覆盖率')
    p.add_argument('--attn-model', default='vit_base_patch16_224')
    p.add_argument('--seed', type=int, default=42, help='随机选图的 seed')
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'device = {device}')

    out_dir = config.OUTPUT_DIR / 'attention_viz'
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. 随机抽 N 张 val 图
    val = MyDataset(txt=config.split_path('dataset-val.txt'),
                    root=config.DATA_ROOT, transform=default_transform)
    loader = DataLoader(val, batch_size=args.n, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    images = images.to(device)
    print(f'sampled {len(images)} images, labels = {labels.tolist()}')

    # 2. 跑 ViT attention rollout
    ext = ViTAttentionExtractor(model_name=args.attn_model,
                                pretrained=True, device=device)
    att = ext.get_attention_map(normalize_for_vit(images))      # (B, 14, 14)
    # bilinear upsample 给 heatmap 用
    att_up = torch.nn.functional.interpolate(
        att.unsqueeze(1), size=224, mode='bilinear', align_corners=False
    ).squeeze(1).cpu()
    mask = make_topk_mask(att, top_k_ratio=args.top_k, out_size=224).cpu()
    print(f'attention map: shape={tuple(att.shape)}, '
          f'mask coverage = {mask.mean().item():.4f}  (target {args.top_k})')

    # 3. 出图
    for i in range(len(images)):
        img = images[i].cpu().permute(1, 2, 0).numpy()
        heat = att_up[i].numpy()
        heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-9)
        m = mask[i, 0].numpy()

        # 模拟局部扰动:在 mask 区域加点强随机噪声,看视觉效果
        rng = np.random.default_rng(args.seed + i)
        noise = rng.standard_normal((224, 224, 3)) * 0.3
        adv_sim = np.clip(img + noise * m[..., None], 0, 1)

        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
        axes[0].imshow(img)
        axes[0].set_title(f'(a) original  ·  label={labels[i].item()}', fontsize=11)

        axes[1].imshow(img)
        axes[1].imshow(heat, cmap='jet', alpha=0.55)
        axes[1].set_title('(b) ViT attention rollout', fontsize=11)

        axes[2].imshow(img)
        axes[2].imshow(m, cmap='Reds', alpha=0.55)
        axes[2].set_title(f'(c) top-{args.top_k:.0%} binary mask', fontsize=11)

        axes[3].imshow(adv_sim)
        axes[3].set_title('(d) simulated local perturbation', fontsize=11)

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        out_path = out_dir / f'attn_{i:02d}_label{labels[i].item():03d}.png'
        plt.savefig(out_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'  saved {out_path.name}')

    ext.close()
    print(f'\ndone. {len(images)} figures → {out_dir}')


if __name__ == '__main__':
    main()
