"""ViTAttentionExtractor: wrap a timm ViT and expose the rolled-out attention."""
from __future__ import annotations

import timm
import torch

from .rollout import attention_rollout


class ViTAttentionExtractor:
    """Wrap a timm ViT and expose the [CLS]→patches attention map.

    For vit_base_patch16_224 the patch grid is 14x14.
    """

    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 device: str | torch.device = 'cpu'):
        self.device = torch.device(device)
        self.model = timm.create_model(model_name, pretrained=pretrained).to(self.device).eval()

        self.num_patches_side = int(self.model.patch_embed.num_patches ** 0.5)
        assert self.num_patches_side ** 2 == self.model.patch_embed.num_patches, \
            "patch grid must be square"

        self._stored_attn: list[torch.Tensor] = []
        self._hooks = []
        # timm >= 0.9 defaults attn.fused_attn=True, routing through
        # F.scaled_dot_product_attention and bypassing the explicit softmax /
        # attn_drop path. We need the intermediate attention matrix, so force
        # the explicit path on every block.
        for blk in self.model.blocks:
            if hasattr(blk.attn, 'fused_attn'):
                blk.attn.fused_attn = False
            h = blk.attn.attn_drop.register_forward_hook(self._save_attn)
            self._hooks.append(h)

    def _save_attn(self, module, input, output):
        self._stored_attn.append(input[0].detach())

    def close(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def get_attention_map(self,
                          x: torch.Tensor,
                          head_fusion: str = 'mean',
                          discard_ratio: float = 0.0) -> torch.Tensor:
        """Run a forward pass and return (B, H, W) attention map.

        x must be in the normalisation expected by the model — use normalize_for_vit
        if your input is in [0,1].
        """
        self._stored_attn.clear()
        _ = self.model(x.to(self.device))
        if not self._stored_attn:
            raise RuntimeError(
                "no attention captured — hooks did not fire. "
                "Check that blocks[i].attn.attn_drop exists and fused_attn is off."
            )
        rollout = attention_rollout(self._stored_attn, head_fusion, discard_ratio)
        cls_to_patches = rollout[:, 0, 1:]
        side = self.num_patches_side
        return cls_to_patches.view(-1, side, side)
