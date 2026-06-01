"""Reusable building blocks: ResnetBlock and DimensionReducerBlock."""
from __future__ import annotations

import torch
import torch.nn as nn


class ResnetBlock(nn.Module):
    """Two-conv residual block, adapted from junyanz/pytorch-CycleGAN-and-pix2pix."""

    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d,
                 use_dropout=False, use_bias=False):
        super().__init__()
        self.conv_block = self._build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    @staticmethod
    def _build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias):
        def pad(p_type):
            if p_type == 'reflect':
                return [nn.ReflectionPad2d(1)], 0
            if p_type == 'replicate':
                return [nn.ReplicationPad2d(1)], 0
            if p_type == 'zero':
                return [], 1
            raise NotImplementedError(f'padding [{p_type}] is not implemented')

        layers = []
        pad_layers, p = pad(padding_type)
        layers += pad_layers
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim),
                   nn.ReLU(True)]
        if use_dropout:
            layers += [nn.Dropout(0.5)]

        pad_layers, p = pad(padding_type)
        layers += pad_layers
        layers += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                   norm_layer(dim)]
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class DimensionReducerBlock(nn.Module):
    """Downsample-then-upsample with a residual skip; used only inside the Generator."""

    def __init__(self):
        super().__init__()
        self.pooling1 = nn.Conv2d(3, 3, kernel_size=(2, 2), stride=2)
        self.pooling2 = nn.AdaptiveAvgPool2d(224)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        x = self.pooling1(x)
        x = self.pooling2(x)
        return identity + x
