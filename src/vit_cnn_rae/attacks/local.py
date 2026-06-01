"""Attention-guided local variant of SRAE.

The Generator is wrapped so its perturbation is multiplied by a binary mask
whose positions are the top-k attention patches of a ViT applied to the input.
The mask positions become the recovery key.

Nothing in base.py's train_batch is modified — baseline stays bisectable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..attention import ViTAttentionExtractor, make_topk_mask, normalize_for_vit
from .base import Attack


class MaskedGenerator(nn.Module):
    """Wrap a Generator so its perturbation is zero outside an attention mask."""

    def __init__(self, inner: nn.Module, mask_fn):
        super().__init__()
        self.inner = inner
        self._mask_fn = mask_fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pert = self.inner(x)
        with torch.no_grad():
            mask = self._mask_fn(x).to(pert.dtype).to(pert.device)
        return pert * mask


class LocalAttack(Attack):
    """Same training loop as Attack, but G's output is gated by a ViT top-k mask."""

    def __init__(self, device, model, model_num_labels, image_nc,
                 box_min, box_max, clip,
                 top_k_ratio: float = 0.2,
                 attn_model_name: str = 'vit_base_patch16_224',
                 attn_pretrained: bool = True,
                 models_path=None):
        super().__init__(device, model, model_num_labels, image_nc,
                         box_min, box_max, clip, models_path=models_path)

        self.top_k_ratio = top_k_ratio
        self._attn = ViTAttentionExtractor(
            model_name=attn_model_name, pretrained=attn_pretrained, device=device)

        def mask_fn(x: torch.Tensor) -> torch.Tensor:
            att = self._attn.get_attention_map(normalize_for_vit(x))
            return make_topk_mask(att, top_k_ratio=self.top_k_ratio, out_size=x.shape[-1])

        self.netG = MaskedGenerator(self.netG, mask_fn).to(device)
        # Parent captured the OLD netG.parameters(); re-bind.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1e-3)
