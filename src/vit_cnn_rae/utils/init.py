"""Custom weight initialisation for Conv/BatchNorm layers."""
from __future__ import annotations

import torch.nn as nn


def weights_init(m: nn.Module) -> None:
    """DCGAN-style init: Conv weights N(0, 0.02), BatchNorm weights N(1, 0.02)."""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
