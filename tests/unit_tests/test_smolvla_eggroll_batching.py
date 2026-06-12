from __future__ import annotations

import numpy as np
import pytest
import torch

from rlinf.algorithms.eggroll.batched_low_rank import (
    batched_low_rank_linear_numpy,
    batched_low_rank_module_patch,
)
from rlinf.algorithms.eggroll.low_rank import LowRankDelta


def _linear() -> torch.nn.Linear:
    module = torch.nn.Linear(4, 3)
    with torch.no_grad():
        module.weight.copy_(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10.0)
        module.bias.copy_(torch.tensor([0.1, -0.2, 0.3], dtype=torch.float32))
    return module


def _deltas() -> list[LowRankDelta]:
    return [
        LowRankDelta(
            left=np.arange(6, dtype=np.float32).reshape(3, 2) / 5.0,
            right=np.arange(8, dtype=np.float32).reshape(2, 4) / 11.0,
            scale=0.01,
        ),
        LowRankDelta(
            left=np.arange(6, 12, dtype=np.float32).reshape(3, 2) / 13.0,
            right=np.arange(8, 16, dtype=np.float32).reshape(2, 4) / 17.0,
            scale=-0.02,
        ),
    ]


def test_batched_low_rank_linear_matches_serial_numpy() -> None:
    module = _linear()
    deltas = _deltas()
    inputs = np.arange(20, dtype=np.float32).reshape(5, 4) / 7.0
    member_positions = np.array([0, 1, 1, 0, 1], dtype=np.int64)

    result = batched_low_rank_linear_numpy(
        inputs,
        weight=module.weight.detach().numpy(),
        bias=module.bias.detach().numpy(),
        deltas=deltas,
        member_positions=member_positions,
    )

    expected = []
    for row, position in zip(inputs, member_positions, strict=True):
        delta_weight = deltas[int(position)].materialize()
        expected.append(row @ (module.weight.detach().numpy() + delta_weight).T + module.bias.detach().numpy())
    assert np.allclose(result, np.asarray(expected, dtype=np.float32), atol=1e-6)


def test_module_patch_matches_serial_for_batch_token_inputs() -> None:
    module = _linear()
    inputs = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4) / 7.0
    deltas = _deltas()
    member_positions = np.array([0, 1], dtype=np.int64)

    with batched_low_rank_module_patch(module, deltas=deltas, member_positions=member_positions):
        result = module(inputs)

    expected = []
    for batch_index, position in enumerate(member_positions):
        delta_weight = torch.as_tensor(deltas[int(position)].materialize())
        expected.append(torch.nn.functional.linear(inputs[batch_index], module.weight + delta_weight, module.bias))
    assert torch.allclose(result, torch.stack(expected), atol=1e-6)


def test_module_patch_matches_serial_for_flattened_batch_horizon_inputs() -> None:
    module = _linear()
    batch = 2
    horizon = 3
    inputs = torch.arange(batch * horizon * 4, dtype=torch.float32).reshape(batch * horizon, 4) / 7.0
    deltas = _deltas()
    member_positions = np.array([0, 1], dtype=np.int64)

    with batched_low_rank_module_patch(
        module,
        deltas=deltas,
        member_positions=member_positions,
        allow_flattened_batch=True,
    ):
        result = module(inputs)

    expected = []
    for batch_index, position in enumerate(member_positions):
        delta_weight = torch.as_tensor(deltas[int(position)].materialize())
        rows = inputs[batch_index * horizon : (batch_index + 1) * horizon]
        expected.append(torch.nn.functional.linear(rows, module.weight + delta_weight, module.bias))
    assert torch.allclose(result.reshape(batch, horizon, -1), torch.stack(expected), atol=1e-6)


def test_module_patch_rejects_flattened_inputs_without_opt_in() -> None:
    module = _linear()
    inputs = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(6, 4) / 7.0

    with batched_low_rank_module_patch(module, deltas=_deltas(), member_positions=np.array([0, 1])):
        with pytest.raises(ValueError, match="first input dimension"):
            module(inputs)


def test_module_patch_restores_forward_after_exception() -> None:
    module = _linear()
    inputs = torch.arange(2 * 4, dtype=torch.float32).reshape(2, 4) / 7.0
    expected = module(inputs)

    with pytest.raises(RuntimeError, match="boom"):
        with batched_low_rank_module_patch(module, deltas=_deltas(), member_positions=np.array([0, 1])):
            raise RuntimeError("boom")

    assert torch.equal(module(inputs), expected)
