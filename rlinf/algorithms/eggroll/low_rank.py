from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np


@dataclass(frozen=True)
class LowRankDelta:
    left: np.ndarray
    right: np.ndarray
    scale: float = 1.0

    def materialize(self) -> np.ndarray:
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("low-rank factors must be rank-2 arrays")
        if self.left.shape[1] != self.right.shape[0]:
            raise ValueError(
                f"incompatible low-rank factors: {self.left.shape} x {self.right.shape}"
            )
        return (self.left @ self.right) * self.scale


@contextmanager
def temporary_low_rank_perturbation(module: Any, delta: LowRankDelta) -> Iterator[None]:
    """Temporarily add a low-rank delta to a module weight and always restore it."""
    weight = module.weight.data
    materialized = delta.materialize()
    if materialized.shape != tuple(weight.shape):
        raise ValueError(
            f"delta shape {materialized.shape} does not match weight shape {tuple(weight.shape)}"
        )

    if hasattr(weight, "detach"):
        import torch

        original = weight.detach().clone()
        torch_delta = torch.as_tensor(materialized, dtype=weight.dtype, device=weight.device)
        with torch.no_grad():
            weight.add_(torch_delta)
        try:
            yield
        finally:
            with torch.no_grad():
                weight.copy_(original)
        return

    original = weight.copy()
    materialized = materialized.astype(weight.dtype, copy=False)
    module.weight.data[...] = weight + materialized
    try:
        yield
    finally:
        module.weight.data[...] = original
