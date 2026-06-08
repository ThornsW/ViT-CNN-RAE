"""Attention-guided local variant of SRAE.

The Generator is wrapped so its perturbation is multiplied by a mask whose
top-k positions (high ViT attention) get weight 1 and the background gets
`bg_weight` (0 = hard local mask, 0<a<1 = soft mask). The mask positions
become the recovery key.

Nothing in base.py's train_batch is modified — baseline stays bisectable.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..attention import ViTAttentionExtractor, make_topk_mask, normalize_for_vit
from .base import Attack


class MaskedGenerator(nn.Module):
    """Wrap a Generator so its perturbation is gated by an attention mask.

    The mask depends only on the (deterministic) clean image, so it is identical
    every epoch. We cache it per image (keyed by content hash) and skip the
    expensive ViT forward from epoch 2 onwards. Cached masks live on CPU as
    float16 to save memory. Pass cache=False to disable (e.g. for evaluation).
    """

    def __init__(self, inner: nn.Module, mask_fn, cache: bool = True):
        super().__init__()
        self.inner = inner
        self._mask_fn = mask_fn
        self._cache: dict | None = {} if cache else None

    @torch.no_grad()
    def _masks_for(self, x: torch.Tensor) -> torch.Tensor:
        if self._cache is None:                       # caching off: always recompute
            return self._mask_fn(x)
        keys = [hash(x[i].detach().cpu().numpy().tobytes()) for i in range(x.shape[0])]
        miss = [i for i, k in enumerate(keys) if k not in self._cache]
        if miss:                                      # only run ViT on uncached images
            computed = self._mask_fn(x[miss])
            for j, i in enumerate(miss):
                self._cache[keys[i]] = computed[j].half().cpu()
        return torch.stack([self._cache[k] for k in keys])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pert = self.inner(x)
        mask = self._masks_for(x).to(pert.dtype).to(pert.device)
        return pert * mask


class LocalAttack(Attack):
    """Same training loop as Attack, but G's output is gated by a ViT top-k mask."""

    def __init__(self, device, model, model_num_labels, image_nc,
                 box_min, box_max, clip,
                 top_k_ratio: float = 0.2,
                 bg_weight: float = 0.0,
                 attn_model_name: str = 'vit_base_patch16_224',
                 attn_pretrained: bool = True,
                 models_path=None):
        super().__init__(device, model, model_num_labels, image_nc,
                         box_min, box_max, clip, models_path=models_path)

        self.top_k_ratio = top_k_ratio
        self.bg_weight = bg_weight
        self._attn = ViTAttentionExtractor(
            model_name=attn_model_name, pretrained=attn_pretrained, device=device)

        def mask_fn(x: torch.Tensor) -> torch.Tensor:
            att = self._attn.get_attention_map(normalize_for_vit(x))
            return make_topk_mask(att, top_k_ratio=self.top_k_ratio,
                                  out_size=x.shape[-1], bg_weight=self.bg_weight)

        self.netG = MaskedGenerator(self.netG, mask_fn).to(device)
        # Parent captured the OLD netG.parameters(); re-bind.
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=1e-3)
