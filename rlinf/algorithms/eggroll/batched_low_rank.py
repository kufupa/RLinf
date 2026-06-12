from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import MethodType
from typing import Any

import numpy as np

from rlinf.algorithms.eggroll.low_rank import LowRankDelta


def batched_low_rank_linear_numpy(
    inputs: np.ndarray,
    *,
    weight: np.ndarray,
    bias: np.ndarray | None,
    deltas: Sequence[LowRankDelta],
    member_positions: np.ndarray,
) -> np.ndarray:
    inputs = np.asarray(inputs, dtype=np.float32)
    weight = np.asarray(weight, dtype=np.float32)
    member_positions = np.asarray(member_positions, dtype=np.int64).reshape(-1)
    if inputs.ndim != 2:
        raise ValueError(f"expected rank-2 inputs, got {inputs.shape}")
    if weight.ndim != 2:
        raise ValueError(f"expected rank-2 weight, got {weight.shape}")
    if inputs.shape[0] != member_positions.size:
        raise ValueError("member_positions must have one entry per input row")
    if inputs.shape[1] != weight.shape[1]:
        raise ValueError(f"input dim {inputs.shape[1]} does not match weight dim {weight.shape[1]}")
    _validate_member_positions(member_positions, len(deltas))

    output = inputs @ weight.T
    if bias is not None:
        output = output + np.asarray(bias, dtype=np.float32)
    for position, delta in enumerate(deltas):
        mask = member_positions == position
        if not np.any(mask):
            continue
        output[mask] += inputs[mask] @ delta.materialize().T
    return output.astype(np.float32, copy=False)


@contextmanager
def batched_low_rank_module_patch(
    module: Any,
    *,
    deltas: Sequence[LowRankDelta],
    member_positions: np.ndarray,
    allow_flattened_batch: bool = False,
) -> Iterator[None]:
    import torch
    import torch.nn.functional as functional

    original_forward = module.forward
    positions_np = np.asarray(member_positions, dtype=np.int64).reshape(-1)
    _validate_member_positions(positions_np, len(deltas))
    _validate_torch_delta_shapes(module, deltas)

    positions = torch.as_tensor(positions_np, dtype=torch.long, device=module.weight.device)
    torch_delta_factors = [
        (
            torch.as_tensor(delta.left, dtype=module.weight.dtype, device=module.weight.device),
            torch.as_tensor(delta.right, dtype=module.weight.dtype, device=module.weight.device),
            torch.as_tensor(delta.scale, dtype=module.weight.dtype, device=module.weight.device),
        )
        for delta in deltas
    ]

    def row_positions_for(input_tensor: torch.Tensor) -> torch.Tensor:
        if input_tensor.ndim < 2:
            raise ValueError(f"expected input with rank >= 2, got {tuple(input_tensor.shape)}")
        if input_tensor.shape[-1] != module.weight.shape[1]:
            raise ValueError(
                f"input dim {input_tensor.shape[-1]} does not match weight dim {module.weight.shape[1]}"
            )
        if input_tensor.shape[0] == positions.numel():
            return positions
        if not allow_flattened_batch:
            raise ValueError(
                f"expected first input dimension {positions.numel()} "
                f"to match member positions, got {input_tensor.shape[0]}"
            )
        if input_tensor.shape[0] % positions.numel() != 0:
            raise ValueError(
                f"flattened first dimension {input_tensor.shape[0]} is not divisible by "
                f"member positions {positions.numel()}"
            )
        return positions.repeat_interleave(input_tensor.shape[0] // positions.numel())

    def patched_forward(self: Any, input_tensor: torch.Tensor) -> torch.Tensor:
        row_positions = row_positions_for(input_tensor)
        output = functional.linear(input_tensor, self.weight, self.bias)
        for position, (left, right, scale) in enumerate(torch_delta_factors):
            mask = row_positions == position
            if not bool(mask.any()):
                continue
            low_rank_output = (
                functional.linear(functional.linear(input_tensor[mask], right, None), left, None)
                * scale
            )
            output[mask] = output[mask] + low_rank_output
        return output

    module.forward = MethodType(patched_forward, module)
    try:
        yield
    finally:
        module.forward = original_forward


def _validate_member_positions(member_positions: np.ndarray, delta_count: int) -> None:
    if delta_count < 1:
        raise ValueError("deltas must not be empty")
    if member_positions.size < 1:
        raise ValueError("member_positions must not be empty")
    if member_positions.min() < 0 or member_positions.max() >= delta_count:
        raise ValueError("member_positions contains an out-of-range delta index")


def _validate_torch_delta_shapes(module: Any, deltas: Sequence[LowRankDelta]) -> None:
    weight_shape = tuple(module.weight.shape)
    for delta in deltas:
        if delta.left.ndim != 2 or delta.right.ndim != 2:
            raise ValueError("low-rank factors must be rank-2 arrays")
        if delta.left.shape[1] != delta.right.shape[0]:
            raise ValueError(f"incompatible low-rank factors: {delta.left.shape} x {delta.right.shape}")
        delta_shape = (delta.left.shape[0], delta.right.shape[1])
        if delta_shape != weight_shape:
            raise ValueError(f"delta shape {delta_shape} does not match weight shape {weight_shape}")
