from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from rlinf.algorithms.eggroll.parallel_rollout import env_member_positions
from rlinf.algorithms.eggroll.population import EggrollPopulationConfig
from scripts.run_smolvla_metaworld_eggroll import EggrollRunConfig
from scripts.run_smolvla_metaworld_eggroll import action_saturation_fraction
from scripts.run_smolvla_metaworld_eggroll import evaluate_eggroll_update
from scripts.run_smolvla_metaworld_eggroll import verify_batched_equivalence


class FakeModel(torch.nn.Module):
    def __init__(self, chunk_len: int = 3) -> None:
        super().__init__()
        self.target = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.target.weight.copy_(torch.tensor([[0.5, -0.25]], dtype=torch.float32))
        self.action_low = -1.0
        self.action_high = 1.0
        self.num_action_chunks = chunk_len

    @torch.no_grad()
    def predict_action_batch(self, obs, mode: str = "eval", compute_values: bool = False):
        del mode, compute_values
        state = obs["states"].float()
        action = self.target(state).reshape(-1, 1, 1).expand(-1, self.num_action_chunks, -1)
        return action, {"forward_inputs": {}}


@dataclass
class FakeEnv:
    num_envs: int
    chunk_len: int
    max_episode_steps: int

    def __post_init__(self) -> None:
        self.steps = 0
        self.reset_many_calls = []
        self.states = torch.stack(
            [
                torch.arange(1, self.num_envs + 1, dtype=torch.float32),
                torch.arange(self.num_envs, 0, -1, dtype=torch.float32),
            ],
            dim=1,
        )

    def reset(self):
        self.steps = 0
        return {"states": self.states.clone()}, {}

    def reset_many(self, reset_seeds):
        self.reset_many_calls.append([int(seed) for seed in reset_seeds])
        return self.reset()

    def chunk_step(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        chunk = actions.shape[1]
        rewards = torch.as_tensor(actions[:, :, 0], dtype=torch.float32)
        self.steps += chunk
        done = self.steps >= self.max_episode_steps
        terms = torch.zeros((self.num_envs, chunk), dtype=torch.bool)
        truncs = torch.full((self.num_envs, chunk), done, dtype=torch.bool)
        valid = torch.ones((self.num_envs, chunk), dtype=torch.bool)
        infos = [
            {"valid_action_mask": valid[:, i], "all_rows_terminal": bool(done)}
            for i in range(chunk)
        ]
        infos[-1]["valid_action_mask"] = valid
        infos[-1]["episode"] = {
            "success_once": torch.zeros(self.num_envs),
            "return": rewards.sum(dim=1),
            "episode_len": torch.full((self.num_envs,), self.steps),
        }
        return [{"states": self.states.clone()} for _ in range(chunk)], rewards, terms, truncs, infos

    def close(self):
        pass


def test_action_saturation_fraction_counts_near_bounds() -> None:
    actions = np.array([[[-1.0, 0.0], [0.5, 0.9999999]]], dtype=np.float32)

    assert action_saturation_fraction(actions, action_low=-1.0, action_high=1.0) == 0.5


def test_verify_batched_equivalence_matches_serial_member_patches() -> None:
    model = FakeModel(chunk_len=2)
    obs = {"states": torch.tensor([[1.0, 2.0], [2.0, 1.0]])}
    population_config = EggrollPopulationConfig(
        population_size=2,
        out_features=1,
        in_features=2,
        rank=1,
        sigma=0.1,
        seed=3,
    )

    payload = verify_batched_equivalence(
        model=model,
        obs=obs,
        target_module=model.target,
        population_config=population_config,
        env_to_member=env_member_positions(population_size=2, envs_per_member=1),
    )

    assert payload["ok"]
    assert payload["max_abs_diff"] <= payload["atol"]


def test_evaluate_eggroll_update_scores_members_and_updates_target_weight() -> None:
    model = FakeModel(chunk_len=2)
    env = FakeEnv(num_envs=2, chunk_len=2, max_episode_steps=4)
    before = model.target.weight.detach().clone()
    config = EggrollRunConfig(
        population_size=2,
        envs_per_member=1,
        episodes_per_member=1,
        steps_per_update=4,
        chunk_len=2,
        rank=1,
        sigma=0.1,
        learning_rate=0.5,
        seed=5,
        reset_seed_base=2000,
    )

    result = evaluate_eggroll_update(
        model=model,
        env=env,
        target_module=model.target,
        target_name="target",
        config=config,
        device=torch.device("cpu"),
    )

    assert result.metrics["population_size"] == 2
    assert result.metrics["member_episodes"] == 2
    assert result.metrics["scalar_env_steps"] == 8
    assert result.metrics["seconds_per_member_episode"] > 0.0
    assert len(result.member_scores) == 2
    assert not torch.equal(model.target.weight, before)


def test_evaluate_eggroll_update_uses_common_random_seed_rows() -> None:
    model = FakeModel(chunk_len=2)
    env = FakeEnv(num_envs=6, chunk_len=2, max_episode_steps=4)
    config = EggrollRunConfig(
        population_size=3,
        envs_per_member=2,
        episodes_per_member=1,
        steps_per_update=4,
        chunk_len=2,
        rank=1,
        sigma=0.1,
        learning_rate=0.5,
        seed=5,
        reset_seed_base=2000,
    )

    result = evaluate_eggroll_update(
        model=model,
        env=env,
        target_module=model.target,
        target_name="target",
        config=config,
        device=torch.device("cpu"),
    )

    assert env.reset_many_calls == [[2000, 2000, 2000, 2001, 2001, 2001]]
    assert result.metrics["fair_seed_layout"]["member_positions_head"] == [0, 1, 2, 0, 1, 2]
    assert result.metrics["fair_seed_layout"]["unique_reset_seeds"] == [2000, 2001]
