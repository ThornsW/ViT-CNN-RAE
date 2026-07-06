"""Evaluation loop: ASR + PSNR/SSIM/L0/L2/Linf on adv and recovered images."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import config
from ..data import MyDataset, default_transform
from ..models import Generator, Recover
from ..targets import load_target_model
from .metrics import compare_psnr, compare_ssim


def _load_run_config(models_dir: Path) -> dict:
    """Read the run's run_config.json (sits one level above models/), else {}."""
    for cand in (models_dir.parent / "run_config.json", models_dir / "run_config.json"):
        if cand.exists():
            try:
                return json.loads(cand.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _load_component(path: Path, key: str, device) -> dict:
    """Load a state_dict, handling both a bare state_dict file (netG_epoch_N_1.pth)
    and a full checkpoint dict (last.pth carrying netG/netR/... sub-dicts)."""
    obj = torch.load(path, map_location=device)
    if isinstance(obj, dict) and key in obj and isinstance(obj[key], dict):
        return obj[key]
    return obj


def _channel_first_to_last(img):
    img = img.swapaxes(0, 2)
    img = img.swapaxes(0, 1)
    return img


def evaluate(target_name: str | None = None,
             g_ckpt: str | Path = 'netG_epoch_150_1.pth',
             r_ckpt: str | Path = 'netR_epoch_150_1.pth',
             batch_size: int = 32,
             clip: float | None = None,
             models_dir: Path | None = None,
             local: bool | None = None,
             top_k_ratio: float | None = None,
             bg_weight: float | None = None,
             attn_model: str | None = None,
             gated_recovery: bool = False) -> dict:
    """Load G/R/target, run val set, print and return metric summary.

    local=True evaluates a LocalAttack run: G is wrapped in MaskedGenerator so the
    perturbation is gated by the same ViT top-k attention mask used in training.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(models_dir) if models_dir else config.MODELS_OUT

    # run_config.json 是配置的唯一真源:未显式指定(None)的参数从这里回填,避免手动
    # 传 --bg-weight/--eps 与训练时错配而静默评出错误数字。显式传参最高优先级。
    cfg = _load_run_config(models_dir)
    p = cfg.get("params", {})
    if local is None:
        local = (cfg.get("model") == "srae_local")
    if target_name is None:
        target_name = p.get("target", "densenet121")
    if top_k_ratio is None:
        top_k_ratio = p.get("top_k", 0.2)
    if bg_weight is None:
        bg_weight = p.get("bg_weight", 0.0)
    if clip is None:
        clip = float(p.get("eps", 1.0))
    if attn_model is None:
        attn_model = p.get("attn_model", "vit_base_patch16_224")

    # ckpt: 优先给定文件名;不存在则回退到 last.pth(完整 checkpoint,含 netG/netR),
    # 这样轻量 run(未到 ckpt_interval、无 netG_epoch_150)也能直接评估。
    g_path, r_path = models_dir / g_ckpt, models_dir / r_ckpt
    if not g_path.exists() and (models_dir / "last.pth").exists():
        g_path = r_path = models_dir / "last.pth"
        print(f"[eval] {g_ckpt} 不存在,回退到 last.pth")
    # local 以 checkpoint 前缀为准(MaskedGenerator 存的 netG 有 inner. 前缀),比 run_config
    # 的 model 字段更可靠——最早的 local run 可能缺 run_config 或字段不符。
    g_state = _load_component(g_path, "netG", device)
    if local != (ckpt_local := any(k.startswith("inner.") for k in g_state)):
        print(f"[eval] local 修正: run_config={local} -> checkpoint={ckpt_local}")
        local = ckpt_local
    if gated_recovery and not local:
        print("[eval] --gated-recovery 仅对 local run 有意义(需要 mask),已忽略")
        gated_recovery = False
    print(f"[eval] run={models_dir.parent.name} local={local} target={target_name} "
          f"top_k={top_k_ratio} bg={bg_weight} eps={clip} attn={attn_model} gated={gated_recovery}")

    image_nc = config.IMAGE_CHANNELS
    _mask_fn = None  # local 分支会赋值;门控恢复(gated_recovery)复用它

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
    netG.load_state_dict(g_state)
    netG.eval()

    netR = Recover(image_nc, image_nc).to(device)
    netR.load_state_dict(_load_component(r_path, "netR", device))
    netR.eval()

    target = load_target_model(target_name, device=device)

    _lpips_net = None
    try:
        import lpips as _lpips_mod
        _lpips_net = _lpips_mod.LPIPS(net='alex').to(device).eval()
    except Exception as e:
        print(f"[eval] LPIPS 不可用({type(e).__name__}),跳过该指标: pip install lpips")

    test_data = MyDataset(txt=config.split_path('dataset-val.txt'),
                          root=config.DATA_ROOT,
                          transform=default_transform)
    test_loader = DataLoader(test_data, batch_size=batch_size,
                             pin_memory=torch.cuda.is_available(), num_workers=1)

    num = 0
    num_correct = 0
    num_correct_r = 0

    metric_keys = [
        'l0_adv_ori', 'l2_adv_ori', 'l_inf_adv_ori', 'psnr_adv_ori', 'ssim_adv_ori',
        'l0_r_ori', 'l2_r_ori', 'l_inf_r_ori', 'psnr_r_ori', 'ssim_r_ori',
    ]
    if gated_recovery:
        # r_key: mask 用 clean 图算(持有密钥场景); r_self: mask 用 adv 图重算(自恢复场景)
        metric_keys += ['psnr_r_key_ori', 'ssim_r_key_ori', 'psnr_r_self_ori', 'ssim_r_self_ori']
    if _lpips_net is not None:
        metric_keys += ['lpips_adv_ori', 'lpips_r_ori']
    metrics = {key: [] for key in metric_keys}

    with torch.no_grad():
        for test_img, test_label in test_loader:
            test_img, test_label = test_img.to(device), test_label.to(device)

            perturbation = torch.clamp(netG(test_img), -clip, clip)
            adv_img = torch.clamp(perturbation + test_img, 0, 1)

            pred_lab = torch.argmax(target(adv_img), 1)
            num_correct += int(torch.sum(pred_lab == test_label))

            r_perturbation = netR(adv_img)
            r_adv = torch.clamp(adv_img - r_perturbation, 0, 1)

            if gated_recovery:
                m_clean = _mask_fn(test_img).to(adv_img.dtype).to(device)
                r_key = torch.clamp(adv_img - r_perturbation * m_clean, 0, 1)
                m_adv = _mask_fn(adv_img).to(adv_img.dtype).to(device)
                r_self = torch.clamp(adv_img - r_perturbation * m_adv, 0, 1)

            pred_r_adv = torch.argmax(target(r_adv), 1)
            num_correct_r += int(torch.sum(pred_r_adv == test_label))

            ori_pred = torch.argmax(target(test_img), 1)
            num += int(torch.sum(ori_pred == test_label))

            if _lpips_net is not None:
                lp_adv = _lpips_net(adv_img * 2 - 1, test_img * 2 - 1).flatten()
                lp_r = _lpips_net(r_adv * 2 - 1, test_img * 2 - 1).flatten()

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

                if gated_recovery:
                    rk = _channel_first_to_last((r_key[j] * 255.).cpu().numpy().astype('uint8').squeeze())
                    rs = _channel_first_to_last((r_self[j] * 255.).cpu().numpy().astype('uint8').squeeze())
                    metrics['psnr_r_key_ori'].append(compare_psnr(rk, ori, data_range=255))
                    metrics['ssim_r_key_ori'].append(compare_ssim(rk, ori, data_range=255, multichannel=True))
                    metrics['psnr_r_self_ori'].append(compare_psnr(rs, ori, data_range=255))
                    metrics['ssim_r_self_ori'].append(compare_ssim(rs, ori, data_range=255, multichannel=True))
                if _lpips_net is not None:
                    metrics['lpips_adv_ori'].append(lp_adv[j].item())
                    metrics['lpips_r_ori'].append(lp_r[j].item())

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
