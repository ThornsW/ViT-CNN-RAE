"""Save 9 per-sample images (ori / adv / r_adv / perturbations / diffs)."""
from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from .. import config
from ..data import MyDataset, default_transform
from ..models import Generator, Recover
from ..targets import load_target_model


_SUBDIRS = ('img', 'adv', 'r_adv', 'pert1', 'pert5', 'r_pert1', 'r_pert5', 'diff1', 'diff5')


def save_visualizations(target_name: str = 'densenet121',
                        g_ckpt: str | Path = 'netG_epoch_150_1.pth',
                        r_ckpt: str | Path = 'netR_epoch_150_1.pth',
                        batch_size: int = 30,
                        clip: float = 1.0,
                        models_dir: Path | None = None,
                        out_dir: Path | None = None) -> Path:
    """Iterate val set, save 9 images per sample to <out_dir>/save_test/."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models_dir = Path(models_dir) if models_dir else config.MODELS_OUT
    out_dir = Path(out_dir) if out_dir else config.VIZ_OUT / 'save_test'

    image_nc = config.IMAGE_CHANNELS

    for sub in _SUBDIRS:
        (out_dir / sub).mkdir(parents=True, exist_ok=True)

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

    to_pil = transforms.ToPILImage()
    count = 0

    with torch.no_grad():
        for test_img, test_label in test_loader:
            test_img, test_label = test_img.to(device), test_label.to(device)

            perturbation = torch.clamp(netG(test_img), -clip, clip)
            adv_img = torch.clamp(perturbation + test_img, 0, 1)
            pred_lab = torch.argmax(target(adv_img), 1)

            r_perturbation = netR(adv_img)
            r_adv = torch.clamp(adv_img - r_perturbation, 0, 1)
            pred_r_adv = torch.argmax(target(r_adv), 1)

            for j in range(len(test_img)):
                p = torch.abs(adv_img[j] - test_img[j])
                r_p = torch.abs(r_adv[j] - adv_img[j])
                diff = torch.abs(r_p - p)

                ori = to_pil(test_img[j].cpu()).convert('RGB')
                adv_pil = to_pil(adv_img[j].cpu()).convert('RGB')
                r_a = to_pil(r_adv[j].cpu()).convert('RGB')

                p1 = to_pil(p.cpu()).convert('RGB')
                p5 = to_pil((p * 10).clamp(0, 1).cpu()).convert('RGB')
                r_p1 = to_pil(r_p.cpu()).convert('RGB')
                r_p5 = to_pil((r_p * 10).clamp(0, 1).cpu()).convert('RGB')
                diff1 = to_pil(diff.cpu()).convert('RGB')
                diff5 = to_pil((diff * 10).clamp(0, 1).cpu()).convert('RGB')

                ori.save(out_dir / 'img' / f'{count}_ori_.png')
                adv_pil.save(out_dir / 'adv' / f'{count}_{pred_lab[j].item()}_{test_label[j].item()}_.png')
                r_a.save(out_dir / 'r_adv' /
                         f'{count}_{pred_lab[j].item()}_{test_label[j].item()}_{pred_r_adv[j].item()}_r_a_.png')

                p1.save(out_dir / 'pert1' / f'{count}_noise1_.png')
                p5.save(out_dir / 'pert5' / f'{count}_noise5_.png')
                r_p1.save(out_dir / 'r_pert1' / f'{count}_rnoise1_.png')
                r_p5.save(out_dir / 'r_pert5' / f'{count}_rnoise5_.png')
                diff1.save(out_dir / 'diff1' / f'{count}_diff1_.png')
                diff5.save(out_dir / 'diff5' / f'{count}_diff5_.png')

                count += 1

    return out_dir
