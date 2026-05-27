from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rlinf.algorithms.eggroll.low_rank import LowRankDelta


@dataclass(frozen=True)
class EggrollPopulationConfig:
    population_size: int
    out_features: int
    in_features: int
    rank: int
    sigma: float
    seed: int


@dataclass(frozen=True)
class EggrollMember:
    index: int
    seed: int
    delta: LowRankDelta
    score: float | None = None


def _orthonormal_columns(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    sample = rng.standard_normal((rows, cols), dtype=np.float32)
    q, _ = np.linalg.qr(sample, mode="reduced")
    return q[:, :cols].astype(np.float32, copy=False)


def sample_population(config: EggrollPopulationConfig) -> list[EggrollMember]:
    if config.population_size < 1:
        raise ValueError("population_size must be >= 1")
    if config.rank < 1 or config.rank > min(config.out_features, config.in_features):
        raise ValueError(
            f"rank must be in [1, {min(config.out_features, config.in_features)}], got {config.rank}"
        )
    if config.sigma <= 0:
        raise ValueError("sigma must be positive")

    seed_rng = np.random.default_rng(config.seed)
    members: list[EggrollMember] = []
    for index in range(config.population_size):
        member_seed = int(seed_rng.integers(0, np.iinfo(np.int32).max))
        rng = np.random.default_rng(member_seed)
        left = _orthonormal_columns(rng, config.out_features, config.rank)
        right_basis = _orthonormal_columns(rng, config.in_features, config.rank)
        delta = LowRankDelta(left=left, right=right_basis.T, scale=config.sigma)
        members.append(EggrollMember(index=index, seed=member_seed, delta=delta))
    return members


def aggregate_weighted_delta(
    members: list[EggrollMember], *, learning_rate: float
) -> np.ndarray:
    if not members:
        raise ValueError("cannot aggregate an empty population")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")
    if any(member.score is None for member in members):
        raise ValueError("all members must have scores before aggregation")

    scores = np.asarray([member.score for member in members], dtype=np.float64)
    std = float(scores.std())
    first_delta = members[0].delta.materialize()
    if std == 0.0:
        return np.zeros_like(first_delta, dtype=np.float32)

    weights = (scores - float(scores.mean())) / std
    aggregate = np.zeros_like(first_delta, dtype=np.float64)
    for weight, member in zip(weights, members, strict=True):
        materialized = member.delta.materialize()
        if materialized.shape != first_delta.shape:
            raise ValueError(f"mixed delta shapes: {materialized.shape} and {first_delta.shape}")
        aggregate += float(weight) * materialized
    aggregate *= learning_rate / len(members)
    return aggregate.astype(np.float32)
