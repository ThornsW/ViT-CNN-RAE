"""Attention rollout: Abnar & Zuidema, "Quantifying Attention Flow" (ACL 2020)."""
from __future__ import annotations

import torch


def attention_rollout(attentions: list[torch.Tensor],
                      head_fusion: str = 'mean',
                      discard_ratio: float = 0.0) -> torch.Tensor:
    """Roll out per-block attention into a single (B, T, T) attention map.

    attentions   : list of (B, H, T, T) post-softmax tensors, one per block
    head_fusion  : 'mean' | 'max' | 'min' across attention heads
    discard_ratio: zero out this fraction of smallest weights per row before rolling
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

        # residual stream: add identity then renormalise rows
        I = torch.eye(a.size(-1), device=a.device).expand_as(a)
        a = (a + I) / 2
        a = a / a.sum(dim=-1, keepdim=True)

        rollout = a if rollout is None else torch.bmm(a, rollout)

    return rollout
