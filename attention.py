"""
ViT attention extraction via attention rollout.

Reference: Abnar & Zuidema, "Quantifying Attention Flow in Transformers" (ACL 2020).

The extracted (B, H, W) map is the [CLS]-token attention to image patches,
rolled across all transformer blocks. For vit_base_patch16_224 the grid is 14x14.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import timm


def _attention_rollout(attentions: list[torch.Tensor],
                       head_fusion: str = 'mean',
                       discard_ratio: float = 0.0) -> torch.Tensor:
    """
    attentions : list of (B, H, T, T) post-softmax attention tensors, one per block
    Returns    : (B, T, T) rolled-out attention.
    """
    rollout = None
    for att in attentions:
        if head_fusion == 'mean':
            a = att.mean(dim=1)
        elif head_fusion == 'max':
            a = att.amax(dim=1)
        elif head_fusion == 'min':
            a = att.amin(dim=1)
        else:
            raise ValueError(f"unknown head_fusion: {head_fusion}")

        if discard_ratio > 0:
            flat = a.flatten(1)
            k = int(flat.shape[-1] * discard_ratio)
            if k > 0:
                _, idx = flat.topk(k, dim=-1, largest=False)
                flat.scatter_(-1, idx, 0.0)
            a = flat.view_as(a)

        # Add identity for residual stream, then re-normalise rows.
        I = torch.eye(a.size(-1), device=a.device).expand_as(a)
        a = (a + I) / 2
        a = a / a.sum(dim=-1, keepdim=True)

        rollout = a if rollout is None else torch.bmm(a, rollout)

    return rollout


class ViTAttentionExtractor:
    """Wrap a timm ViT and expose the rolled-out [CLS]→patches attention map."""

    def __init__(self,
                 model_name: str = 'vit_base_patch16_224',
                 pretrained: bool = True,
                 device: str | torch.device = 'cpu'):
        self.device = torch.device(device)
        self.model = timm.create_model(model_name, pretrained=pretrained).to(self.device).eval()

        # Patch grid size: vit_base_patch16_224 → 14x14, deit_small_patch16_224 → 14x14 too.
        self.num_patches_side = int(self.model.patch_embed.num_patches ** 0.5)
        assert self.num_patches_side ** 2 == self.model.patch_embed.num_patches, \
            "patch grid must be square"

        self._stored_attn: list[torch.Tensor] = []
        self._hooks = []
        # timm >= 0.9 defaults attn.fused_attn=True, which routes through
        # F.scaled_dot_product_attention and BYPASSES the explicit softmax /
        # attn_drop path. We need the intermediate attention matrix, so force
        # the explicit path on every block.
        for blk in self.model.blocks:
            if hasattr(blk.attn, 'fused_attn'):
                blk.attn.fused_attn = False
            # Hook attn_drop: its INPUT is the post-softmax (B, H, T, T) attention
            # we want for rollout.
            h = blk.attn.attn_drop.register_forward_hook(self._save_attn)
            self._hooks.append(h)

    def _save_attn(self, module, input, output):
        # input is a tuple; element 0 is (B, H, T, T)
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
        """
        x : (B, 3, 224, 224) image tensor in the normalisation expected by the model.
            Use timm.data.create_transform(...) or normalise with ImageNet mean/std.
        Returns : (B, H, W) attention map with H=W=num_patches_side (14 for vit_base_patch16_224).
        """
        self._stored_attn.clear()
        _ = self.model(x.to(self.device))
        if not self._stored_attn:
            raise RuntimeError(
                "no attention captured — hooks did not fire. "
                "Check that blocks[i].attn.attn_drop exists and fused_attn is off."
            )
        rollout = _attention_rollout(self._stored_attn, head_fusion, discard_ratio)
        # Row 0 is the [CLS] token; columns 1: are the patch tokens.
        cls_to_patches = rollout[:, 0, 1:]
        side = self.num_patches_side
        return cls_to_patches.view(-1, side, side)


_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_for_vit(x: torch.Tensor) -> torch.Tensor:
    """
    Map [0,1] RGB to ImageNet-normalised input expected by timm ViT.
    Input  : (B, 3, H, W) in [0,1]
    Output : same shape, normalised. No-op friendly to autograd.
    """
    mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def make_topk_mask(attention_map: torch.Tensor,
                   top_k_ratio: float = 0.2,
                   out_size: int = 224,
                   mode: str = 'nearest') -> torch.Tensor:
    """
    Binary mask marking the top-k patches by attention.

    attention_map : (B, H, W), e.g. (B, 14, 14)
    top_k_ratio   : fraction of patches kept (1.0 = whole image, 0.2 = top 20%)
    out_size      : target spatial size, typically 224 (so each patch becomes a 16x16 block)
    mode          : 'nearest' (hard patch boundaries) or 'bilinear' (soft, for visualisation)

    Returns: (B, 1, out_size, out_size) mask in {0,1} (or [0,1] if bilinear).
    """
    B, H, W = attention_map.shape
    flat = attention_map.flatten(1)
    k = max(1, int(round(flat.shape[-1] * top_k_ratio)))
    _, idx = flat.topk(k, dim=-1)
    mask_flat = torch.zeros_like(flat)
    mask_flat.scatter_(-1, idx, 1.0)
    mask = mask_flat.view(B, 1, H, W)
    if out_size != H:
        scale = out_size / H
        align = False if mode == 'bilinear' else None
        if mode == 'bilinear':
            mask = F.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=align)
        else:
            mask = F.interpolate(mask, scale_factor=scale, mode='nearest')
    return mask


if __name__ == '__main__':
    # Self-test: random input, verify shapes + value ranges.
    torch.manual_seed(0)
    extractor = ViTAttentionExtractor(pretrained=False)  # pretrained=False for offline self-test
    x = torch.randn(2, 3, 224, 224)
    att = extractor.get_attention_map(x)
    print(f'attention_map: shape={tuple(att.shape)}, '
          f'min={att.min().item():.4f}, max={att.max().item():.4f}, '
          f'mean={att.mean().item():.4f}')
    print(f'row sums (each batch should sum to ~1):',
          att.sum(dim=(-1, -2)).tolist())
    mask = make_topk_mask(att, top_k_ratio=0.2)
    print(f'mask: shape={tuple(mask.shape)}, '
          f'unique={torch.unique(mask).tolist()}, '
          f'coverage={mask.mean().item():.4f}  (expect ~0.2)')
    extractor.close()
