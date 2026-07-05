"""Measure how recoverable the attention mask is from the adversarial image.

    python scripts/mask_overlap.py --models-dir outputs/<run>/models

The mask "key" story only holds if an attacker CANNOT reconstruct the top-k
positions from the adv image they received. This script quantifies that: for
each val image it compares the ViT top-k patch set computed on the clean image
(the real key) against the top-k set the attacker would get by re-running the
same public ViT on the adv image.

Reported against a random-guess floor (k/P): a hit rate near 1.0 means the mask
is essentially public -> drop the "key" narrative, use "self-recovery" instead;
a low hit rate means the perturbation moves attention enough to hide the key.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from vit_cnn_rae import config
from vit_cnn_rae.attacks.local import MaskedGenerator
from vit_cnn_rae.attention import ViTAttentionExtractor, make_topk_mask, normalize_for_vit
from vit_cnn_rae.data import MyDataset, default_transform
from vit_cnn_rae.evaluation.evaluator import _load_component, _load_run_config
from vit_cnn_rae.models import Generator


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models-dir', type=Path, required=True)
    p.add_argument('--g', default='netG_epoch_150_1.pth',
                   help='G checkpoint 文件名;不存在则回退 last.pth')
    p.add_argument('--top-k', type=float, default=None, help='默认从 run_config 读')
    p.add_argument('--eps', type=float, default=None, help='默认从 run_config 读')
    p.add_argument('--attn-model', default=None)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-images', type=int, default=1000,
                   help='评估图片数上限(0=全部 val)')
    return p.parse_args()


def _save_hist(hits, ious, rand_floor, out: Path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print("[overlap] matplotlib 不可用,跳过直方图")
        return None
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].hist(hits, bins=30, color='steelblue')
    ax[0].axvline(rand_floor, color='r', ls='--', label=f'random={rand_floor:.2f}')
    ax[0].set_title('top-k hit rate (clean vs adv)')
    ax[0].set_xlabel('hit rate'); ax[0].legend()
    ax[1].hist(ious, bins=30, color='seagreen')
    ax[1].set_title('IoU (clean vs adv top-k)'); ax[1].set_xlabel('IoU')
    fig.tight_layout(); fig.savefig(out, dpi=120); plt.close(fig)
    return out


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = args.models_dir

    cfg = _load_run_config(models_dir)
    p = cfg.get("params", {})
    top_k = args.top_k if args.top_k is not None else p.get("top_k", 0.2)
    eps = args.eps if args.eps is not None else float(p.get("eps", 1.0))
    bg_weight = float(p.get("bg_weight", 0.0))
    attn_model = args.attn_model or p.get("attn_model", "vit_base_patch16_224")
    print(f"[overlap] run={models_dir.parent.name} top_k={top_k} eps={eps} "
          f"bg={bg_weight} attn={attn_model}")

    attn = ViTAttentionExtractor(model_name=attn_model, pretrained=True, device=device)
    n_patches = attn.num_patches_side ** 2
    k = max(1, round(n_patches * top_k))
    rand_floor = k / n_patches

    def mask_fn(x):
        a = attn.get_attention_map(normalize_for_vit(x))
        return make_topk_mask(a, top_k_ratio=top_k, out_size=x.shape[-1], bg_weight=bg_weight)

    nc = config.IMAGE_CHANNELS
    netG = MaskedGenerator(Generator(nc, nc), mask_fn, cache=False, perturb_clip=eps).to(device)
    g_path = models_dir / args.g
    if not g_path.exists() and (models_dir / "last.pth").exists():
        g_path = models_dir / "last.pth"
        print(f"[overlap] {args.g} 不存在,回退到 last.pth")
    netG.load_state_dict(_load_component(g_path, "netG", device))
    netG.eval()

    val = MyDataset(txt=config.split_path('dataset-val.txt'),
                    root=config.DATA_ROOT, transform=default_transform)
    loader = DataLoader(val, batch_size=args.batch_size,
                        pin_memory=torch.cuda.is_available(), num_workers=1)

    hits, ious = [], []
    seen = 0
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            pert = torch.clamp(netG(x), -eps, eps)
            adv = torch.clamp(pert + x, 0, 1)
            ac = attn.get_attention_map(normalize_for_vit(x)).flatten(1)
            aa = attn.get_attention_map(normalize_for_vit(adv)).flatten(1)
            ic = ac.topk(k, dim=-1).indices
            ia = aa.topk(k, dim=-1).indices
            for i in range(x.size(0)):
                a, b = set(ic[i].tolist()), set(ia[i].tolist())
                inter = len(a & b)
                hits.append(inter / k)
                ious.append(inter / len(a | b))
            seen += x.size(0)
            if args.max_images and seen >= args.max_images:
                break

    hits, ious = np.asarray(hits), np.asarray(ious)
    print(f"[overlap] images={len(hits)} k={k}/{n_patches} random-floor={rand_floor:.3f}")
    print(f"[overlap] hit  mean={hits.mean():.4f} median={np.median(hits):.4f} "
          f"min={hits.min():.4f} max={hits.max():.4f}")
    print(f"[overlap] IoU  mean={ious.mean():.4f} median={np.median(ious):.4f}")

    out_dir = models_dir.parent
    _save_hist(hits, ious, rand_floor, out_dir / 'mask_overlap.png')
    with open(out_dir / 'mask_overlap.txt', 'a', encoding='utf-8') as f:
        f.write(f"top_k={top_k} k={k}/{n_patches} random_floor={rand_floor:.4f} "
                f"hit_mean={hits.mean():.4f} hit_median={np.median(hits):.4f} "
                f"iou_mean={ious.mean():.4f} n={len(hits)}\n")
    print(f"[overlap] wrote {out_dir/'mask_overlap.png'} + mask_overlap.txt")


if __name__ == '__main__':
    main()
