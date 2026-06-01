"""Unified loader for the target (victim) classifier."""
from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models as tv_models

from .. import config


_BUILDERS = {
    'densenet121': {
        'ctor': lambda: tv_models.densenet121(weights=None),
        'replace': lambda m, n: setattr(m, 'classifier', nn.Linear(m.classifier.in_features, n)),
        'default_ckpt': 'DenseNet121.pth',
    },
    'resnet50': {
        'ctor': lambda: tv_models.resnet50(weights=None),
        'replace': lambda m, n: setattr(m, 'fc', nn.Linear(m.fc.in_features, n)),
        'default_ckpt': 'ResNet50.pth',
    },
    'mobilenet_v3_large': {
        'ctor': lambda: tv_models.mobilenet_v3_large(weights=None),
        'replace': lambda m, n: m.classifier.__setitem__(
            3, nn.Linear(m.classifier[3].in_features, n)),
        'default_ckpt': 'MoblieNetV3.pth',
    },
}


def load_target_model(name: str = 'densenet121',
                      num_classes: int = config.NUM_CLASSES,
                      checkpoint: str | Path | None = None,
                      device: str | torch.device = 'cpu',
                      eval_mode: bool = True) -> nn.Module:
    """Build a torchvision classifier, replace head, load weights, send to device."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown target model: {name}. Choose from {list(_BUILDERS)}.")
    spec = _BUILDERS[name]

    model = spec['ctor']()
    spec['replace'](model, num_classes)

    ckpt_path = Path(checkpoint) if checkpoint else config.checkpoint_path(spec['default_ckpt'])
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"target classifier checkpoint not found at {ckpt_path}. "
            f"Run scripts/train_target_{name.split('_')[0]}.py to create it.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    if eval_mode:
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
    return model
