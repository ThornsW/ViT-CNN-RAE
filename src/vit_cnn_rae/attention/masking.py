"""Top-k mask generation and ImageNet normalisation helpers."""
from __future__ import annotations

import torch
import torch.nn.functional as F

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_for_vit(x: torch.Tensor) -> torch.Tensor:
    """Map [0,1] RGB to ImageNet-normalised input expected by timm ViT."""
    mean = x.new_tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = x.new_tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    return (x - mean) / std


def make_topk_mask(attention_map: torch.Tensor,
                   top_k_ratio: float = 0.2,
                   out_size: int = 224,
                   mode: str = 'nearest') -> torch.Tensor:
    """Binary mask marking the top-k patches by attention.

    attention_map : (B, H, W), e.g. (B, 14, 14)
    top_k_ratio   : fraction of patches kept (1.0 = whole image, 0.2 = top 20%)
    out_size      : target spatial size, typically 224
    mode          : 'nearest' (hard patch boundaries) or 'bilinear' (soft)

    Returns: (B, 1, out_size, out_size).
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
        if mode == 'bilinear':
            mask = F.interpolate(mask, scale_factor=scale, mode='bilinear', align_corners=False)
        else:
            mask = F.interpolate(mask, scale_factor=scale, mode='nearest')
    return mask
