"""Thin wrappers over skimage.metrics with backwards-compatible kwargs."""
from __future__ import annotations

from skimage.metrics import peak_signal_noise_ratio as _psnr
from skimage.metrics import structural_similarity as _ssim


def compare_psnr(a, b, data_range=None):
    return _psnr(a, b, data_range=data_range)


def compare_ssim(a, b, data_range=None, multichannel=True):
    return _ssim(a, b, channel_axis=-1 if multichannel else None, data_range=data_range)
