"""SRAE training engine (Zhang et al., TCSVT 2022).

Three-network adversarial training: Generator (G), Discriminator (D), Recover (R).
Don't modify train_batch — extend via subclass (see LocalAttack).
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .. import config
from ..models import Discriminator, Generator, Recover
from ..utils import draw_loss_curve, weights_init


class Attack:
    def __init__(self, device, model, model_num_labels, image_nc,
                 box_min, box_max, clip,
                 models_path: Path | str | None = None):
        self.device = device
        self.model = model
        self.model_num_labels = model_num_labels
        self.box_min = box_min
        self.box_max = box_max
        self.clip = clip

        self.netG = Generator(image_nc, image_nc).to(device)
        self.netDisc = Discriminator(image_nc).to(device)
        self.netR = Recover(image_nc, image_nc).to(device)

        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)
        self.netR.apply(weights_init)

        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1e-3)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=1e-3)
        self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=1e-3)

        self.models_path = Path(models_path) if models_path is not None else config.MODELS_OUT
        self.models_path.mkdir(parents=True, exist_ok=True)

        self.start_epoch = 1
        self.loss_history: dict[str, list[float]] = {
            k: [] for k in ("loss_D", "loss_G_fake", "loss_perturb",
                            "loss_adv", "loss_r_adv", "loss_l2_r_pert")
        }

    def save_checkpoint(self, epoch: int, tag: str = "last") -> Path:
        """Save full state (G/D/R + optimizers + history) to <tag>.pth."""
        path = self.models_path / f"{tag}.pth"
        torch.save({
            "epoch": epoch,
            "netG": self.netG.state_dict(),
            "netD": self.netDisc.state_dict(),
            "netR": self.netR.state_dict(),
            "optimizer_G": self.optimizer_G.state_dict(),
            "optimizer_D": self.optimizer_D.state_dict(),
            "optimizer_R": self.optimizer_R.state_dict(),
            "loss_history": self.loss_history,
        }, path)
        return path

    def load_checkpoint(self, path: Path | str) -> int:
        """Restore state. Returns epoch to resume from (start_epoch)."""
        ckpt = torch.load(Path(path), map_location=self.device)
        self.netG.load_state_dict(ckpt["netG"])
        self.netDisc.load_state_dict(ckpt["netD"])
        self.netR.load_state_dict(ckpt["netR"])
        self.optimizer_G.load_state_dict(ckpt["optimizer_G"])
        self.optimizer_D.load_state_dict(ckpt["optimizer_D"])
        self.optimizer_R.load_state_dict(ckpt["optimizer_R"])
        self.loss_history = ckpt.get("loss_history", self.loss_history)
        self.start_epoch = int(ckpt["epoch"]) + 1
        print(f"[resume] from {path} at epoch {self.start_epoch}")
        return self.start_epoch

    def train_batch(self, x, labels):
        perturbation = self.netG(x)
        adv_images = torch.clamp(perturbation, -self.clip, self.clip) + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        # ---- D ----
        self.optimizer_D.zero_grad()
        pred_real = self.netDisc(x)
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real))
        loss_D_real.backward()
        pred_fake = self.netDisc(adv_images.detach())
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake))
        loss_D_fake.backward()
        loss_D_GAN = loss_D_fake + loss_D_real
        self.optimizer_D.step()

        # ---- R (pure L2, M3 mode) ----
        self.optimizer_R.zero_grad()
        r_pert = self.netR(adv_images.detach())
        r_adv = torch.clamp(adv_images - r_pert, self.box_min, self.box_max)
        loss_l2_r_pert = F.mse_loss(r_adv, x)
        loss_r_adv = F.cross_entropy(self.model(r_adv), labels)
        loss_l2_r_pert.backward(retain_graph=True)
        self.optimizer_R.step()

        # ---- G (GAN + C&W + perturbation magnitude) ----
        self.optimizer_G.zero_grad()
        pred_fake = self.netDisc(adv_images)
        loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake))
        loss_G_fake.backward(retain_graph=True)
        loss_perturb = torch.mean(torch.norm(
            perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        probs = F.softmax(self.model(adv_images), dim=1)
        onehot = torch.eye(self.model_num_labels, device=self.device)[labels]
        real = torch.sum(onehot * probs, dim=1)
        other, _ = torch.max((1 - onehot) * probs - onehot * 10000, dim=1)
        loss_adv = torch.sum(torch.max(real - other, torch.zeros_like(other)))
        (10 * loss_adv + 1 * loss_perturb).backward()
        self.optimizer_G.step()

        return (loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(),
                loss_adv.item(), loss_r_adv.item(), loss_l2_r_pert.item())

    def _rebuild_optimizers(self, lr: float) -> None:
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=lr)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(), lr=lr)
        self.optimizer_R = torch.optim.Adam(self.netR.parameters(), lr=lr)

    def train(self, train_dataloader, epochs: int, ckpt_interval: int = 50):
        """Main training loop. Saves last.pth every epoch + epoch_NNN.pth at intervals."""
        for epoch in range(self.start_epoch, epochs + 1):
            if epoch == 50:
                self._rebuild_optimizers(1e-4)
            if epoch == 100:
                self._rebuild_optimizers(1e-5)

            sums = {k: 0.0 for k in self.loss_history}
            pbar = tqdm(train_dataloader, desc=f"epoch {epoch}/{epochs}", leave=False)
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                losses = self.train_batch(images, labels)
                for k, v in zip(self.loss_history, losses):
                    sums[k] += v
                pbar.set_postfix(D=f"{losses[0]:.3f}", G=f"{losses[3]:.3f}",
                                 perturb=f"{losses[2]:.1f}")

            n = len(train_dataloader)
            avgs = {k: sums[k] / n for k in self.loss_history}
            for k, v in avgs.items():
                self.loss_history[k].append(v)

            print(f"epoch {epoch:3d} | D={avgs['loss_D']:.4f} G_fake={avgs['loss_G_fake']:.4f} "
                  f"perturb={avgs['loss_perturb']:.4f} adv={avgs['loss_adv']:.4f} "
                  f"r_adv={avgs['loss_r_adv']:.4f} l2_r={avgs['loss_l2_r_pert']:.4f}")

            self.save_checkpoint(epoch, tag="last")
            if epoch % ckpt_interval == 0:
                self.save_checkpoint(epoch, tag=f"epoch_{epoch:03d}")
                # legacy compat: G/R only with v1 filename
                torch.save(self.netG.state_dict(), self.models_path / f"netG_epoch_{epoch}_1.pth")
                torch.save(self.netR.state_dict(), self.models_path / f"netR_epoch_{epoch}_1.pth")

        # final dump
        with open(self.models_path / "loss1.txt", "a", encoding="utf-8") as f:
            for k, v in self.loss_history.items():
                f.write(f"{k}:{v}\n")
        for k, v in self.loss_history.items():
            draw_loss_curve(self.models_path, v, label=k.replace("loss_", ""))

        return self.loss_history
