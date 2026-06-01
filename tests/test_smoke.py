"""End-to-end smoke: ViT attention extractor + LocalAttack train_batch."""
from __future__ import annotations

import torch
from torch import nn
from torchvision import models as tv_models

from vit_cnn_rae.attacks import LocalAttack
from vit_cnn_rae.attention import ViTAttentionExtractor, make_topk_mask


def test_attention_extractor_and_mask_coverage():
    torch.manual_seed(0)
    ext = ViTAttentionExtractor(pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    att = ext.get_attention_map(x)
    assert att.shape == (2, 14, 14)

    mask = make_topk_mask(att, top_k_ratio=0.2)
    assert mask.shape == (2, 1, 224, 224)
    assert set(torch.unique(mask).tolist()) == {0.0, 1.0}
    # 14*14=196, round(196*0.2)=39, 39/196≈0.1990
    assert 0.19 < mask.mean().item() < 0.21
    ext.close()


def test_local_attack_end_to_end_on_cpu():
    torch.manual_seed(0)
    target = tv_models.densenet121(weights=None)
    target.classifier = nn.Linear(target.classifier.in_features, 257)
    target.eval()
    for p in target.parameters():
        p.requires_grad = False

    atk = LocalAttack(device=torch.device('cpu'), model=target, model_num_labels=257,
                      image_nc=3, box_min=0, box_max=1, clip=1,
                      top_k_ratio=0.2, attn_pretrained=False)
    x = torch.rand(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    losses = atk.train_batch(x, labels)
    assert len(losses) == 6
    assert all(v == v for v in losses)  # no NaN

    # verify mask gating: perturbation only in top-20% pixels
    with torch.no_grad():
        pert = atk.netG(x)
    nonzero = (pert.abs().sum(dim=1) > 1e-8).float().mean(dim=(-1, -2))
    for v in nonzero.tolist():
        assert 0.19 < v < 0.21
