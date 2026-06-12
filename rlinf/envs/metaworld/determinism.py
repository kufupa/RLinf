from __future__ import annotations

import random

import numpy as np


def seed_metaworld_process(seed: int) -> None:
    value = int(seed)
    random.seed(value)
    np.random.seed(value)
    try:
        import torch
    except ModuleNotFoundError:
        return
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
