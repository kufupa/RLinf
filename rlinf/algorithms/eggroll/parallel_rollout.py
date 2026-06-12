from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np

from rlinf.algorithms.eggroll.population import EggrollMember


@dataclass(frozen=True)
class CommonSeedRolloutLayout:
    member_positions: np.ndarray
    reset_seeds: np.ndarray


def resolve_rollout_sizes(
    *,
    population_size: int | None,
    num_envs: int,
    envs_per_member: int,
) -> tuple[int, int]:
    if envs_per_member < 1:
        raise ValueError("envs_per_member must be >= 1")
    resolved_population = num_envs if population_size is None else population_size
    if resolved_population < 1:
        raise ValueError("population_size must be >= 1")
    return resolved_population, resolved_population * envs_per_member


def env_member_positions(*, population_size: int, envs_per_member: int) -> np.ndarray:
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    if envs_per_member < 1:
        raise ValueError("envs_per_member must be >= 1")
    return np.repeat(np.arange(population_size, dtype=np.int64), envs_per_member)


def common_seed_rollout_layout(
    *,
    population_size: int,
    eval_seeds_per_member: int,
    reset_seed_base: int,
) -> CommonSeedRolloutLayout:
    if population_size < 1:
        raise ValueError("population_size must be >= 1")
    if eval_seeds_per_member < 1:
        raise ValueError("eval_seeds_per_member must be >= 1")

    member_positions = np.tile(np.arange(population_size, dtype=np.int64), eval_seeds_per_member)
    reset_seeds = np.repeat(
        np.arange(
            int(reset_seed_base),
            int(reset_seed_base) + int(eval_seeds_per_member),
            dtype=np.int64,
        ),
        population_size,
    )
    return CommonSeedRolloutLayout(member_positions=member_positions, reset_seeds=reset_seeds)


def reset_seed_stride(
    *,
    envs_per_member: int,
    episodes_per_member: int,
    reset_seed_stride_override: int | None = None,
) -> int:
    if envs_per_member < 1:
        raise ValueError("envs_per_member must be >= 1")
    if episodes_per_member < 1:
        raise ValueError("episodes_per_member must be >= 1")
    if reset_seed_stride_override is not None:
        if reset_seed_stride_override < 1:
            raise ValueError("reset_seed_stride_override must be >= 1")
        return int(reset_seed_stride_override)
    return int(envs_per_member) * int(episodes_per_member)


def update_reset_seed_base(
    reset_seed_base: int,
    update_index: int,
    *,
    envs_per_member: int,
    episodes_per_member: int,
    reset_seed_mode: str = "per_update",
    reset_seed_stride_override: int | None = None,
) -> int:
    if update_index < 1:
        raise ValueError("update_index must be >= 1")
    mode = str(reset_seed_mode)
    if mode == "fixed":
        return int(reset_seed_base)
    if mode != "per_update":
        raise ValueError(f"reset_seed_mode must be 'per_update' or 'fixed', got {reset_seed_mode!r}")
    stride = reset_seed_stride(
        envs_per_member=envs_per_member,
        episodes_per_member=episodes_per_member,
        reset_seed_stride_override=reset_seed_stride_override,
    )
    return int(reset_seed_base) + (int(update_index) - 1) * stride


def aggregate_member_scores(
    members: list[EggrollMember],
    env_totals: np.ndarray,
    env_to_member: np.ndarray,
) -> list[EggrollMember]:
    env_totals = np.asarray(env_totals, dtype=np.float32).reshape(-1)
    env_to_member = np.asarray(env_to_member, dtype=np.int64).reshape(-1)
    if env_totals.shape != env_to_member.shape:
        raise ValueError(
            f"env totals shape {env_totals.shape} does not match member map {env_to_member.shape}"
        )
    if members and (env_to_member.min(initial=0) < 0 or env_to_member.max(initial=0) >= len(members)):
        raise ValueError("env_to_member contains an out-of-range member position")

    scored: list[EggrollMember] = []
    for position, member in enumerate(members):
        mask = env_to_member == position
        if not np.any(mask):
            raise ValueError(f"member position {position} has no environment replicas")
        scored.append(
            EggrollMember(
                index=member.index,
                seed=member.seed,
                delta=member.delta,
                score=float(env_totals[mask].mean()),
            )
        )
    return scored


def expand_members_for_envs(
    members: list[EggrollMember], env_to_member: np.ndarray
) -> list[EggrollMember]:
    env_to_member = np.asarray(env_to_member, dtype=np.int64).reshape(-1)
    if env_to_member.size == 0:
        raise ValueError("env_to_member must not be empty")
    if env_to_member.min() < 0 or env_to_member.max() >= len(members):
        raise ValueError("env_to_member contains an out-of-range member position")
    return [members[int(position)] for position in env_to_member]


def iter_chunk_lengths(*, total_steps: int, chunk_horizon: int) -> Iterator[int]:
    if total_steps < 1:
        raise ValueError("total_steps must be >= 1")
    if chunk_horizon < 1:
        raise ValueError("chunk_horizon must be >= 1")
    remaining = total_steps
    while remaining > 0:
        chunk = min(chunk_horizon, remaining)
        yield chunk
        remaining -= chunk
