"""Random seed control for reproducibility."""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed every RNG that PyTorch training touches.

    deterministic=True trades a bit of speed for bit-exact reproducibility:
    - torch.backends.cudnn.deterministic = True
    - torch.backends.cudnn.benchmark = False
    - PYTHONHASHSEED env var
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True
