"""Memory-mapped torch.load helper for owned RLinf SmolVLA scripts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import torch

LoadMode = Literal["mmap", "eager"]


def torch_load_mmap_default(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
) -> Any:
    """Load a path-like checkpoint with mmap=True when PyTorch supports it."""
    payload, _ = torch_load_mmap_with_mode(
        path,
        map_location=map_location,
        weights_only=weights_only,
    )
    return payload


def torch_load_mmap_with_mode(
    path: str | Path,
    *,
    map_location: str | torch.device | None = None,
    weights_only: bool = False,
) -> tuple[Any, LoadMode]:
    """Load a path-like checkpoint and report whether mmap or eager load was used."""
    try:
        return torch.load(path, map_location=map_location, weights_only=weights_only, mmap=True), "mmap"
    except TypeError:
        try:
            return torch.load(path, map_location=map_location, weights_only=weights_only), "eager"
        except TypeError:
            return torch.load(path, map_location=map_location), "eager"
