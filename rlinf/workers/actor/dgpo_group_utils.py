# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""True-group utilities for Direct Group Preference Optimization (DGPO) in NFT training."""

from __future__ import annotations

import torch


def build_dgpo_group_ids(n_chunk: int, batch_size: int, group_size: int) -> torch.Tensor:
    """Build per-row GRPO group ids before train flatten/shuffle.

    Groups are (chunk_step, env_seed_block): envs sharing a reset seed at the same
    chunk index form one Direct-DGPO group.

    Returns:
        Tensor of shape ``[n_chunk, batch_size]`` with dtype int64.
    """
    if batch_size % group_size != 0:
        raise ValueError(
            f"batch_size={batch_size} must be divisible by group_size={group_size}"
        )
    num_env_groups = batch_size // group_size
    env_group_idx = torch.arange(batch_size) // group_size
    chunk_idx = torch.arange(n_chunk).unsqueeze(1)
    return chunk_idx * num_env_groups + env_group_idx.unsqueeze(0)


def build_dgpo_group_block_shuffle_id(
    num_rows: int, group_size: int, generator: torch.Generator
) -> torch.Tensor:
    """Permute flattened training rows by shuffling whole DGPO groups only."""
    if num_rows % group_size != 0:
        raise ValueError(
            f"num_rows={num_rows} must be divisible by group_size={group_size}"
        )
    num_groups = num_rows // group_size
    group_perm = torch.randperm(num_groups, generator=generator)
    within = torch.arange(group_size)
    return (group_perm.unsqueeze(1) * group_size + within).reshape(-1)


def dgpo_scatter_group_sum(values: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
    """Sum ``values`` over shared ``group_ids`` and broadcast sums back to each row."""
    flat_values = values.reshape(-1)
    flat_gids = group_ids.reshape(-1).long()
    _, inverse = torch.unique(flat_gids, sorted=True, return_inverse=True)
    num_groups = int(inverse.max().item()) + 1 if inverse.numel() else 0
    if num_groups == 0:
        return values
    sums = torch.zeros(num_groups, device=values.device, dtype=values.dtype)
    sums.scatter_add_(0, inverse, flat_values)
    return sums[inverse].reshape(values.shape)


def verify_dgpo_group_integrity(
    group_ids: torch.Tensor,
    group_size: int,
    *,
    advantages: torch.Tensor | None = None,
    adv_tol: float = 1e-3,
) -> dict[str, float]:
    """Assert each group id appears exactly ``group_size`` times; optional GRPO adv check."""
    flat_gids = group_ids.reshape(-1)
    _, counts = torch.unique(flat_gids, return_counts=True)
    min_c = int(counts.min().item()) if counts.numel() else 0
    max_c = int(counts.max().item()) if counts.numel() else 0
    all_complete = bool(torch.all(counts == group_size).item()) if counts.numel() else True
    if not all_complete:
        raise ValueError(
            f"DGPO group integrity failed: expected {group_size} rows/group, "
            f"got min={min_c} max={max_c}"
        )
    metrics = {
        "actor/dgpo_group_count_min": float(min_c),
        "actor/dgpo_group_count_max": float(max_c),
        "actor/dgpo_group_all_complete": float(all_complete),
    }
    if advantages is not None:
        flat_adv = advantages.reshape(-1)
        _, inv = torch.unique(flat_gids, sorted=True, return_inverse=True)
        num_groups = int(inv.max().item()) + 1
        group_sums = torch.zeros(num_groups, dtype=flat_adv.dtype)
        group_sums.scatter_add_(0, inv, flat_adv)
        max_abs = float(group_sums.abs().max().item()) if num_groups else 0.0
        metrics["actor/dgpo_group_adv_sum_max_abs"] = max_abs
        if max_abs > adv_tol:
            raise ValueError(
                f"DGPO GRPO advantage sum per group should be ~0, max_abs={max_abs}"
            )
    return metrics


def dgpo_mean_dsm_energy(weighted_sq: torch.Tensor) -> torch.Tensor:
    """Per-sample mean DSM energy over all non-batch dims (matches vendor DGPO)."""
    return weighted_sq.reshape(weighted_sq.shape[0], -1).mean(dim=1)


def compute_dgpo_group_weights(
    *,
    signed_adv: torch.Tensor,
    delta: torch.Tensor,
    group_ids: torch.Tensor,
    dpo_beta: float,
    group_size: int,
    verify: bool = True,
    loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute detached sigmoid group weights from true ``group_ids``."""
    if verify:
        verify_dgpo_group_integrity(group_ids, group_size)
    per = signed_adv * dpo_beta * delta / float(group_size)
    if loss_mask is not None:
        mask = loss_mask.reshape_as(per).to(dtype=per.dtype)
        per = per * mask
    gsum = dgpo_scatter_group_sum(per, group_ids)
    weight = torch.sigmoid(gsum).detach()
    metrics = {
        "actor/dgpo_group_weight_mean": weight.mean().item(),
        "actor/dgpo_group_weight_std": weight.std(unbiased=False).item()
        if weight.numel() > 1
        else 0.0,
        "actor/dgpo_group_weight_min": weight.min().item(),
        "actor/dgpo_group_weight_max": weight.max().item(),
    }
    return weight, metrics
