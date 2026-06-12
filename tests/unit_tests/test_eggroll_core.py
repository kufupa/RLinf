from __future__ import annotations

import numpy as np

from rlinf.algorithms.eggroll.low_rank import LowRankDelta
from rlinf.algorithms.eggroll.low_rank import temporary_low_rank_perturbation
from rlinf.algorithms.eggroll.parallel_rollout import aggregate_member_scores
from rlinf.algorithms.eggroll.parallel_rollout import env_member_positions
from rlinf.algorithms.eggroll.parallel_rollout import expand_members_for_envs
from rlinf.algorithms.eggroll.parallel_rollout import iter_chunk_lengths
from rlinf.algorithms.eggroll.parallel_rollout import resolve_rollout_sizes
from rlinf.algorithms.eggroll.population import EggrollMember
from rlinf.algorithms.eggroll.population import EggrollPopulationConfig
from rlinf.algorithms.eggroll.population import aggregate_weighted_delta
from rlinf.algorithms.eggroll.population import sample_population


class TinyWeight:
    def __init__(self, data: np.ndarray) -> None:
        self.data = data


class TinyModule:
    def __init__(self) -> None:
        self.weight = TinyWeight(np.arange(6, dtype=np.float32).reshape(2, 3))


def _member(index: int) -> EggrollMember:
    delta = LowRankDelta(
        left=np.ones((2, 1), dtype=np.float32),
        right=np.ones((1, 2), dtype=np.float32),
        scale=0.1,
    )
    return EggrollMember(index=index, seed=100 + index, delta=delta)


def test_low_rank_delta_materializes_outer_product() -> None:
    delta = LowRankDelta(
        left=np.array([[1.0], [2.0]], dtype=np.float32),
        right=np.array([[3.0, 4.0, 5.0]], dtype=np.float32),
        scale=0.5,
    )

    assert np.allclose(delta.materialize(), [[1.5, 2.0, 2.5], [3.0, 4.0, 5.0]])


def test_temporary_perturbation_restores_weights_after_exception() -> None:
    module = TinyModule()
    original = module.weight.data.copy()
    delta = LowRankDelta(
        left=np.ones((2, 1), dtype=np.float32),
        right=np.ones((1, 3), dtype=np.float32),
        scale=0.25,
    )

    try:
        with temporary_low_rank_perturbation(module, delta):
            assert np.allclose(module.weight.data, original + 0.25)
            raise RuntimeError("rollout failed")
    except RuntimeError:
        pass

    assert np.array_equal(module.weight.data, original)


def test_sample_population_is_deterministic_and_orthogonal_per_member() -> None:
    config = EggrollPopulationConfig(
        population_size=3,
        out_features=5,
        in_features=4,
        rank=2,
        sigma=0.1,
        seed=7,
    )

    first = sample_population(config)
    second = sample_population(config)

    assert len(first) == 3
    assert [member.seed for member in first] == [member.seed for member in second]
    assert np.allclose(first[0].delta.materialize(), second[0].delta.materialize())
    gram = first[0].delta.left.T @ first[0].delta.left
    assert np.allclose(gram, np.eye(2), atol=1e-6)


def test_sample_population_rejects_rank_larger_than_weight_shape() -> None:
    config = EggrollPopulationConfig(
        population_size=1,
        out_features=2,
        in_features=4,
        rank=3,
        sigma=0.1,
        seed=0,
    )

    try:
        sample_population(config)
    except ValueError as exc:
        assert "rank" in str(exc)
    else:
        raise AssertionError("expected rank validation failure")


def test_aggregate_weighted_delta_normalizes_scores_and_combines_materialized_deltas() -> None:
    members = [
        EggrollMember(
            index=0,
            seed=10,
            delta=sample_population(EggrollPopulationConfig(1, 2, 3, 1, 1.0, seed=10))[0].delta,
            score=1.0,
        ),
        EggrollMember(
            index=1,
            seed=11,
            delta=sample_population(EggrollPopulationConfig(1, 2, 3, 1, 1.0, seed=11))[0].delta,
            score=3.0,
        ),
    ]

    aggregate = aggregate_weighted_delta(members, learning_rate=0.5)

    scores = np.array([1.0, 3.0], dtype=np.float64)
    weights = (scores - scores.mean()) / scores.std()
    expected = 0.5 * (
        weights[0] * members[0].delta.materialize()
        + weights[1] * members[1].delta.materialize()
    ) / len(members)
    assert np.allclose(aggregate, expected.astype(np.float32))


def test_env_member_positions_repeats_each_member_for_env_replicas() -> None:
    positions = env_member_positions(population_size=3, envs_per_member=2)

    assert positions.tolist() == [0, 0, 1, 1, 2, 2]


def test_common_seed_rollout_layout_repeats_each_seed_for_all_members() -> None:
    from rlinf.algorithms.eggroll.parallel_rollout import common_seed_rollout_layout

    layout = common_seed_rollout_layout(
        population_size=3,
        eval_seeds_per_member=2,
        reset_seed_base=2000,
    )

    assert layout.member_positions.tolist() == [0, 1, 2, 0, 1, 2]
    assert layout.reset_seeds.tolist() == [2000, 2000, 2000, 2001, 2001, 2001]


def test_update_reset_seed_base_advances_each_update() -> None:
    from rlinf.algorithms.eggroll.parallel_rollout import update_reset_seed_base

    assert (
        update_reset_seed_base(
            2000,
            3,
            envs_per_member=4,
            episodes_per_member=1,
            reset_seed_mode="per_update",
        )
        == 2008
    )


def test_update_reset_seed_base_fixed_mode() -> None:
    from rlinf.algorithms.eggroll.parallel_rollout import update_reset_seed_base

    assert (
        update_reset_seed_base(
            2000,
            5,
            envs_per_member=4,
            episodes_per_member=1,
            reset_seed_mode="fixed",
        )
        == 2000
    )


def test_aggregate_member_scores_averages_replicas_per_member() -> None:
    members = [_member(10), _member(11), _member(12)]
    positions = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    env_totals = np.array([1.0, 3.0, 0.0, 2.0, -1.0, 5.0], dtype=np.float32)

    scored = aggregate_member_scores(members, env_totals, positions)

    assert [member.index for member in scored] == [10, 11, 12]
    assert [member.seed for member in scored] == [110, 111, 112]
    assert np.allclose([member.score for member in scored], [2.0, 1.0, 2.0])
    assert scored[0].delta is members[0].delta


def test_expand_members_for_envs_maps_each_env_slot_to_member_delta() -> None:
    members = [_member(0), _member(1)]
    positions = np.array([0, 1, 1, 0], dtype=np.int64)

    expanded = expand_members_for_envs(members, positions)

    assert [member.index for member in expanded] == [0, 1, 1, 0]
    assert expanded[1].delta is members[1].delta


def test_iter_chunk_lengths_caps_last_chunk_to_remaining_steps() -> None:
    assert list(iter_chunk_lengths(total_steps=12, chunk_horizon=5)) == [5, 5, 2]


def test_resolve_rollout_sizes_prefers_explicit_population_size() -> None:
    assert resolve_rollout_sizes(population_size=3, num_envs=99, envs_per_member=2) == (3, 6)
