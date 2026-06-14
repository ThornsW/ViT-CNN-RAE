"""Evaluation loop: ASR + PSNR/SSIM/L0/L2/Linf on adv and recovered images."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import config
from ..data import MyDataset, default_transform
from ..models import Generator, Recover
from ..targets import load_target_model
from .metrics import compare_psnr, compare_ssim


def _channel_first_to_last(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img


def evaluate(target_name: str = 'densenet121',
             g_ckpt: str | Path = 'netG_epoch_150_1.pth',
             r_ckpt: str | Path = 'netR_epoch_150_1.pth',
             batch_size: int = 32,
             clip: float = 1.0,
             models_dir: Path | None = None,
             local: bool = False,
             top_k_ratio: float = 0.2,
             bg_weight: float = 0.0,
             attn_model: str = 'vit_base_patch16_224') -> dict:
    """Load G/R/target, run val set, print and return metric summary.

    local=True evaluates a LocalAttack run: G is wrapped in MaskedGenerator so the
    perturbation is gated by the same ViT top-k attention mask used in training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(models_dir) if models_dir else config.MODELS_OUT

    image_nc = config.IMAGE_CHANNELS

    if local:
        from ..attacks.local import MaskedGenerator
        from ..attention import ViTAttentionExtractor, make_topk_mask, normalize_for_vit
        _attn = ViTAttentionExtractor(model_name=attn_model, pretrained=True, device=device)

        def _mask_fn(x):
            att = _attn.get_attention_map(normalize_for_vit(x))
            return make_topk_mask(att, top_k_ratio=top_k_ratio,
                                  out_size=x.shape[-1], bg_weight=bg_weight)

        netG = MaskedGenerator(Generator(image_nc, image_nc), _mask_fn,
                               cache=False, perturb_clip=clip).to(device)
    else:
        netG = Generator(image_nc, image_nc).to(device)
    netG.load_state_dict(torch.load(models_dir / g_ckpt, map_location=device))
    netG.eval()

    netR = Recover(image_nc, image_nc).to(device)
    netR.load_state_dict(torch.load(models_dir / r_ckpt, map_location=device))
    netR.eval()

    target = load_target_model(target_name, device=device)

    test_data = MyDataset(txt=config.split_path('dataset-val.txt'),
                          root=config.DATA_ROOT,
                          transform=default_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             pin_memory=torch.cuda.is_available(), num_workers=1)

    num = 0
    num_correct = 0
    num_correct_r = 0

    metrics = {key: [] for key in [
        'l0_adv_ori', 'l2_adv_ori', 'l_inf_adv_ori', 'psnr_adv_ori', 'ssim_adv_ori',
        'l0_r_ori', 'l2_r_ori', 'l_inf_r_ori', 'psnr_r_ori', 'ssim_r_ori',
    ]}

    with torch.no_grad():
        for test_img, test_label in test_loader:
            test_img, test_label = test_img.to(device), test_label.to(device)

            perturbation = torch.clamp(netG(test_img), -clip, clip)
            adv_img = torch.clamp(perturbation + test_img, 0, 1)

            pred_lab = torch.argmax(target(adv_img), 1)
            num_correct += int(torch.sum(pred_lab == test_label))

            r_perturbation = netR(adv_img)
            r_adv = torch.clamp(adv_img - r_perturbation, 0, 1)

            pred_r_adv = torch.argmax(target(r_adv), 1)
            num_correct_r += int(torch.sum(pred_r_adv == test_label))

            ori_pred = torch.argmax(target(test_img), 1)
            num += int(torch.sum(ori_pred == test_label))

            for j in range(len(test_img)):
                r_a = _channel_first_to_last((r_adv[j] * 255.).cpu().numpy().astype('uint8').squeeze())
                ori = _channel_first_to_last((test_img[j] * 255.).cpu().numpy().astype('uint8').squeeze())
                adv = _channel_first_to_last((adv_img[j] * 255.).cpu().numpy().astype('uint8').squeeze())

                metrics['l0_adv_ori'].append(torch.norm((adv_img[j] - test_img[j]), p=0).item())
                metrics['l2_adv_ori'].append(torch.norm(adv_img[j] - test_img[j]).item())
                metrics['l_inf_adv_ori'].append(torch.norm((adv_img[j] - test_img[j]), p=float('inf')).item())
                metrics['psnr_adv_ori'].append(compare_psnr(adv, ori, data_range=255))
                metrics['ssim_adv_ori'].append(compare_ssim(adv, ori, data_range=255, multichannel=True))

                metrics['l0_r_ori'].append(torch.norm((r_adv[j] - test_img[j]), p=0).item())
                metrics['l2_r_ori'].append(torch.norm(r_adv[j] - test_img[j]).item())
                metrics['l_inf_r_ori'].append(torch.norm((r_adv[j] - test_img[j]), p=float('inf')).item())
                metrics['psnr_r_ori'].append(compare_psnr(r_a, ori, data_range=255))
                metrics['ssim_r_ori'].append(compare_ssim(r_a, ori, data_range=255, multichannel=True))

    n = len(test_data)
    summary = {
        'target_error_rate': 100 * (1 - num / n),
        'generator_error_rate': 100 * (1 - num_correct / n),
        'remover_error_rate': 100 * (1 - num_correct_r / n),
    }
    print(f"target error rate    {summary['target_error_rate']:.3f}%")
    print(f"generator error rate {summary['generator_error_rate']:.3f}%")
    print(f"remover error rate   {summary['remover_error_rate']:.3f}%")
    for key, values in metrics.items():
        a = np.asarray(values)
        print(f"{key:>14} max={a.max():.4f} min={a.min():.4f} mean={a.mean():.4f} "
              f"median={np.median(a):.4f} var={a.var():.4f}")
        summary[key] = {
            'max': float(a.max()), 'min': float(a.min()),
            'mean': float(a.mean()), 'median': float(np.median(a)),
            'var': float(a.var()),
        }

    results_dir = models_dir.parent
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / 'test.txt', 'a', encoding='utf-8') as f:
        f.write(repr(summary) + '\n')

    return summary
