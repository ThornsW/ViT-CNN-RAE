"""
Attention-guided local variant of SRAE.

The Generator is wrapped so that its perturbation output is multiplied by a
binary mask whose positions are the top-k attention patches of a ViT applied
to the input image. The mask positions become the recovery key.

Nothing in the upstream RGAN.py is modified — the baseline stays bisectable.

Usage:
    from attack_local import LocalAttack
    attacker = LocalAttack(device, target_model, num_labels,
                           image_nc=3, box_min=0, box_max=1, clip=1,
                           top_k_ratio=0.2)
    attacker.train(dataloader, epochs=150)
"""
from __future__ import annotations

import torch
import torch.nn as nn

from RGAN import Attack
from attention import ViTAttentionExtractor, make_topk_mask, normalize_for_vit


class MaskedGenerator(nn.Module):
    """Wrap a Generator so its perturbation is zero outside an attention mask."""

    def __init__(self, inner: nn.Module, mask_fn):
        super().__init__()
        self.inner = inner
        self._mask_fn = mask_fn  # x -> (B, 1, H, W) binary mask, no grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pert = self.inner(x)
        with torch.no_grad():
            mask = self._mask_fn(x).to(pert.dtype).to(pert.device)
        return pert * mask


class LocalAttack(Attack):
    """
    Same training loop as SRAE's Attack, but generator output is gated by a
    ViT-attention top-k mask. Pass top_k_ratio=1.0 to recover baseline behaviour.
    """

    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 clip,
                 top_k_ratio: float = 0.2,
                 attn_model_name: str = 'vit_base_patch16_224',
                 attn_pretrained: bool = True,
                 attn_device=None):
        super().__init__(device, model, model_num_labels, image_nc, box_min, box_max, clip)

        self.top_k_ratio = top_k_ratio
        self._attn_extractor = ViTAttentionExtractor(
            model_name=attn_model_name,
            pretrained=attn_pretrained,
            device=attn_device if attn_device is not None else device,
        )

        def mask_fn(x: torch.Tensor) -> torch.Tensor:
            x_vit = normalize_for_vit(x)
            att = self._attn_extractor.get_attention_map(x_vit)
            return make_topk_mask(att, top_k_ratio=self.top_k_ratio, out_size=x.shape[-1])

        # Wrap netG so the parent train loop can stay unmodified.
        wrapped = MaskedGenerator(self.netG, mask_fn).to(device)
        self.netG = wrapped
        # The optimizer captured the OLD netG.parameters() reference. Re-bind it
        # to the wrapped module so masked-generator params are what we optimise.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=0.001)


if __name__ == '__main__':
    # Self-test: build LocalAttack with a dummy classifier and run one batch.
    torch.manual_seed(0)
    from torchvision import models as tv_models
    from torch import nn as _nn

    # Lightweight dummy target classifier (won't load real weights for self-test).
    target = tv_models.densenet121(weights=None)
    target.classifier = _nn.Linear(target.classifier.in_features, 257)
    target.eval()
    for p in target.parameters():
        p.requires_grad = False

    device = torch.device('cpu')
    target = target.to(device)

    atk = LocalAttack(
        device=device,
        model=target,
        model_num_labels=257,
        image_nc=3, box_min=0, box_max=1, clip=1,
        top_k_ratio=0.2,
        attn_pretrained=False,  # avoid downloading weights for self-test
    )

    x = torch.rand(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    losses = atk.train_batch(x, labels)
    names = ['D_GAN', 'G_fake', 'perturb', 'adv', 'r_adv', 'l2_r_pert']
    print('one train_batch ran end-to-end on CPU:')
    for n, v in zip(names, losses):
        print(f'  {n:>9}: {v:.6f}')

    # Verify that the generator's effective output is zero outside the mask.
    with torch.no_grad():
        pert = atk.netG(x)
        nonzero_per_image = (pert.abs().sum(dim=1) > 1e-8).float().mean(dim=(-1, -2))
        print(f'fraction of pixels with nonzero perturbation per image: '
              f'{nonzero_per_image.tolist()}  (expect ~0.2)')
