"""Cross-evaluate G checkpoints against multiple victims (conditional ASR).

    python scripts/transfer_matrix.py \
        --runs outputs/20260602_*baseline* outputs/20260622_*bg0_0_eps0_125* \
        --victims densenet121 resnet50 mobilenet_v3_large vit_base_patch16_224

Conditional ASR = of the images a victim classifies CORRECTLY when clean, the
fraction the adv image flips. The conditional form makes victims with different
clean accuracy comparable (standard transfer-attack convention). Each G's config
(local / top_k / eps / bg) is read from its run_config.json, so DenseNet-trained
G vs any unseen victim is a pure black-box transfer number.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
from torch.utils.data import DataLoader

from vit_cnn_rae import config
from vit_cnn_rae.attacks.local import MaskedGenerator
from vit_cnn_rae.attention import ViTAttentionExtractor, make_topk_mask, normalize_for_vit
from vit_cnn_rae.data import MyDataset, default_transform
from vit_cnn_rae.evaluation.evaluator import _load_component, _load_run_config
from vit_cnn_rae.models import Generator
from vit_cnn_rae.targets import load_target_model


def _build_g(models_dir: Path, device, g_ckpt: str):
    """Rebuild a run's generator from its run_config (local G gets its ViT mask)."""
    cfg = _load_run_config(models_dir)
    p = cfg.get('params', {})
    top_k = p.get('top_k', 0.2)
    eps = float(p.get('eps', 1.0))
    bg = float(p.get('bg_weight', 0.0))
    attn_model = p.get('attn_model', 'vit_base_patch16_224')
    nc = config.IMAGE_CHANNELS

    gp = models_dir / g_ckpt
    if not gp.exists() and (models_dir / 'last.pth').exists():
        gp = models_dir / 'last.pth'
    g_state = _load_component(gp, 'netG', device)
    local = any(k.startswith('inner.') for k in g_state)  # checkpoint 前缀为准,不靠 run_config

    if local:
        attn = ViTAttentionExtractor(model_name=attn_model, pretrained=True, device=device)

        def mask_fn(x):
            a = attn.get_attention_map(normalize_for_vit(x))
            return make_topk_mask(a, top_k_ratio=top_k, out_size=x.shape[-1], bg_weight=bg)

        netG = MaskedGenerator(Generator(nc, nc), mask_fn, cache=False, perturb_clip=eps).to(device)
    else:
        netG = Generator(nc, nc).to(device)
    netG.load_state_dict(g_state)
    netG.eval()
    return netG, eps, local


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--runs', type=Path, nargs='+', required=True,
                   help='run 目录(含 models/)或直接给 models/ 目录')
    p.add_argument('--victims', nargs='+',
                   default=['densenet121', 'resnet50', 'mobilenet_v3_large',
                            'vit_base_patch16_224'])
    p.add_argument('--g', default='netG_epoch_150_1.pth')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--max-images', type=int, default=0, help='0=全部 val')
    p.add_argument('--out', type=Path, default=config.OUTPUT_DIR / 'transfer_matrix.md')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    victims = {v: load_target_model(v, device=device) for v in args.victims}
    val = MyDataset(txt=config.split_path('dataset-val.txt'),
                    root=config.DATA_ROOT, transform=default_transform)
    loader = DataLoader(val, batch_size=args.batch_size,
                        pin_memory=torch.cuda.is_available(), num_workers=1)

    rows = []
    for run in args.runs:
        models_dir = run / 'models' if (run / 'models').exists() else run
        netG, eps, local = _build_g(models_dir, device, args.g)
        cc = {v: 0 for v in args.victims}   # clean-correct count
        aw = {v: 0 for v in args.victims}   # adv-wrong among clean-correct
        seen = 0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                pert = torch.clamp(netG(x), -eps, eps)
                adv = torch.clamp(pert + x, 0, 1)
                for v, m in victims.items():
                    correct = torch.argmax(m(x), 1) == y
                    adv_pred = torch.argmax(m(adv), 1)
                    cc[v] += int(correct.sum())
                    aw[v] += int((correct & (adv_pred != y)).sum())
                seen += x.size(0)
                if args.max_images and seen >= args.max_images:
                    break
        casr = {v: 100.0 * aw[v] / max(cc[v], 1) for v in args.victims}
        tag = models_dir.parent.name
        rows.append((tag, casr))
        print(f"[transfer] {tag} ({'local' if local else 'global'}): "
              + " ".join(f"{v}={casr[v]:.1f}" for v in args.victims))

    header = '| G run \\ victim | ' + ' | '.join(args.victims) + ' |'
    sep = '|---|' + '|'.join(['---'] * len(args.victims)) + '|'
    lines = [header, sep]
    for tag, casr in rows:
        lines.append('| ' + tag + ' | ' + ' | '.join(f"{casr[v]:.1f}" for v in args.victims) + ' |')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f"[transfer] conditional ASR (%) matrix -> {args.out}")


if __name__ == '__main__':
    main()
