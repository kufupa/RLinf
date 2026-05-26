import math
import sys
import types
from dataclasses import dataclass
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import EmbodiedRolloutResult
from rlinf.envs import get_env_cls
from rlinf.models.embodiment.smolvla.smolvla_action_model import (
    SmolVLAForRLActionPrediction,
)
from scripts.run_smolvla_metaworld_direct_ppo import (
    compute_gae,
    compute_discounted_returns,
    masked_chunk_logprob,
    normalize_advantages,
    should_update_actor,
)


class IdentityPreprocessor:
    def __call__(self, obs):
        image = torch.as_tensor(obs["observation.image"], dtype=torch.float32)
        state = torch.as_tensor(obs["observation.state"], dtype=torch.float32)
        return {
            "observation": {
                "image": image,
                "state": state,
                "language": {
                    "tokens": torch.zeros((state.shape[0], 4), dtype=torch.long),
                    "attention_mask": torch.ones((state.shape[0], 4), dtype=torch.long),
                },
            }
        }


class IdentityPostprocessor:
    def __call__(self, action):
        return action


class FakeInnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.log_std = nn.Parameter(torch.zeros(1, 4))


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = FakeInnerModel()
        self.mean = nn.Parameter(torch.zeros(1, 4))
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1

    def select_action_distr_params(self, proc):
        bsz = proc["observation"]["state"].shape[0]
        return self.mean.expand(bsz, 4), self.model.log_std.expand(bsz, 4)


class FakeBundle:
    def __init__(self):
        self.policy = FakePolicy()
        self.preprocessor = IdentityPreprocessor()
        self.postprocessor = IdentityPostprocessor()
        self.obs_image_key = "observation.image"
        self.obs_state_key = "observation.state"


@dataclass
class FakeBatchStep:
    observation: dict
    reward: np.ndarray
    success: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    info: dict


class FakeMetaWorldRollout:
    def __init__(self, success_steps):
        self.success_steps = np.asarray(success_steps, dtype=np.int64)
        self.n_envs = int(self.success_steps.shape[0])
        self.steps = np.zeros(self.n_envs, dtype=np.int64)

    def reset_many(self, seeds):
        self.steps[:] = 0
        return {
            "pixels": np.zeros((self.n_envs, 8, 8, 3), dtype=np.uint8),
            "agent_pos": np.zeros((self.n_envs, 4), dtype=np.float32),
        }

    def step_batch(self, actions):
        self.steps += 1
        success = self.steps >= self.success_steps
        reward = success.astype(np.float32)
        return FakeBatchStep(
            observation={
                "pixels": np.zeros((self.n_envs, 8, 8, 3), dtype=np.uint8),
                "agent_pos": np.zeros((self.n_envs, 4), dtype=np.float32),
            },
            reward=reward,
            success=success,
            terminated=success,
            truncated=np.zeros(self.n_envs, dtype=bool),
            info={},
        )

    def close(self):
        pass


def _cfg(**kwargs):
    base = {
        "model_type": "smolvla",
        "model_path": "fake",
        "precision": None,
        "action_dim": 4,
        "num_action_chunks": 5,
        "add_value_head": True,
        "state_dim": 4,
        "freeze_all_but_ppo_trainables": False,
        "detach_critic_input": True,
    }
    base.update(kwargs)
    return OmegaConf.create(base)


def _patch_lerobot_preprocess(monkeypatch, preprocess_fn):
    lerobot_mod = types.ModuleType("lerobot")
    envs_mod = types.ModuleType("lerobot.envs")
    utils_mod = types.ModuleType("lerobot.envs.utils")
    utils_mod.preprocess_observation = preprocess_fn
    monkeypatch.setitem(sys.modules, "lerobot", lerobot_mod)
    monkeypatch.setitem(sys.modules, "lerobot.envs", envs_mod)
    monkeypatch.setitem(sys.modules, "lerobot.envs.utils", utils_mod)


def _patch_metaworld_rollout(monkeypatch, rollout):
    class Factory:
        def __new__(cls, *args, **kwargs):
            return rollout

    return patch(
        "rlinf.envs.metaworld.lerobot_adapter.OfficialLeRobotMetaWorldGRPORollout",
        Factory,
    )


def test_smolvla_env_type_dispatch():
    env_cls = get_env_cls("smolvla_metaworld")
    assert env_cls.__name__ == "SmolVLAMetaWorldEnv"


def test_smolvla_policy_returns_tensor_only_forward_inputs(monkeypatch):
    def fake_preprocess(observation):
        pixels = torch.as_tensor(observation["pixels"], dtype=torch.float32)
        state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)
        return {
            "observation.image": pixels.permute(0, 3, 1, 2) / 255.0,
            "observation.state": state,
        }

    _patch_lerobot_preprocess(monkeypatch, fake_preprocess)

    model = SmolVLAForRLActionPrediction(_cfg(), bundle=FakeBundle())
    env_obs = {
        "main_images": torch.zeros((2, 8, 8, 3), dtype=torch.uint8),
        "states": torch.zeros((2, 4), dtype=torch.float32),
        "task_descriptions": ["push", "push"],
    }
    actions, result = model.predict_action_batch(env_obs, mode="train")

    assert actions.shape == (2, 5, 4)
    assert result["prev_logprobs"].shape == (2, 5, 4)
    assert result["prev_values"].shape == (2, 1)
    assert "task" not in result["forward_inputs"]
    for value in result["forward_inputs"].values():
        assert isinstance(value, torch.Tensor)

    rollout = EmbodiedRolloutResult(max_episode_length=10)
    rollout.forward_inputs.append(result["forward_inputs"])
    trajectory = rollout.to_trajectory()
    assert trajectory.forward_inputs["smolvla_unsquashed_actions"].shape == (1, 2, 5, 4)


def test_smolvla_default_forward_logprob_parity(monkeypatch):
    def fake_preprocess(observation):
        pixels = torch.as_tensor(observation["pixels"], dtype=torch.float32)
        state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)
        return {
            "observation.image": pixels.permute(0, 3, 1, 2) / 255.0,
            "observation.state": state,
        }

    _patch_lerobot_preprocess(monkeypatch, fake_preprocess)

    torch.manual_seed(0)
    model = SmolVLAForRLActionPrediction(_cfg(), bundle=FakeBundle())
    env_obs = {
        "main_images": torch.zeros((3, 8, 8, 3), dtype=torch.uint8),
        "states": torch.zeros((3, 4), dtype=torch.float32),
        "task_descriptions": ["push"] * 3,
    }
    _, result = model.predict_action_batch(env_obs, mode="train")
    out = model.default_forward(result["forward_inputs"])
    torch.testing.assert_close(out["logprobs"], result["prev_logprobs"])
    assert out["logprobs"].shape == (3, 5, 4)
    assert torch.isfinite(out["values"]).all()


def test_smolvla_policy_reset_called_when_env_seed_changes(monkeypatch):
    def fake_preprocess(observation):
        pixels = torch.as_tensor(observation["pixels"], dtype=torch.float32)
        state = torch.as_tensor(observation["agent_pos"], dtype=torch.float32)
        return {
            "observation.image": pixels.permute(0, 3, 1, 2) / 255.0,
            "observation.state": state,
        }

    _patch_lerobot_preprocess(monkeypatch, fake_preprocess)

    bundle = FakeBundle()
    model = SmolVLAForRLActionPrediction(_cfg(), bundle=bundle)
    env_obs = {
        "main_images": torch.zeros((2, 8, 8, 3), dtype=torch.uint8),
        "states": torch.zeros((2, 4), dtype=torch.float32),
        "task_descriptions": ["push"] * 2,
        "reset_seeds": torch.tensor([1, 2], dtype=torch.long),
    }

    model.predict_action_batch(env_obs, mode="eval")
    model.predict_action_batch(env_obs, mode="eval")
    env_obs["reset_seeds"] = torch.tensor([3, 4], dtype=torch.long)
    model.predict_action_batch(env_obs, mode="eval")

    assert bundle.policy.reset_calls == 2


def test_smolvla_gaussian_log_prob_is_unsummed():
    mean = torch.zeros(2, 5, 4)
    log_std = torch.zeros_like(mean)
    sample = torch.zeros_like(mean)
    logp = SmolVLAForRLActionPrediction.gaussian_log_prob_per_dim(mean, log_std, sample)
    assert logp.shape == (2, 5, 4)
    expected = -0.5 * math.log(2 * math.pi)
    torch.testing.assert_close(logp, torch.full_like(logp, expected))


def test_smolvla_reshapes_long_jade_horizon():
    model = SmolVLAForRLActionPrediction(_cfg(), bundle=FakeBundle())
    mean = torch.arange(2 * 50 * 4, dtype=torch.float32).reshape(100, 4)
    log_std = torch.zeros_like(mean)
    mean_out, log_std_out = model._reshape_chunk_params_batch(
        mean,
        log_std,
        n_envs=2,
        chunk_len=5,
    )
    assert mean_out.shape == (2, 5, 4)
    assert log_std_out.shape == (2, 5, 4)
    torch.testing.assert_close(mean_out[1, 0], mean.reshape(2, 50, 4)[1, 0])


def test_direct_ppo_zero_reward_does_not_update_actor_signal():
    rewards = torch.zeros((2, 3), dtype=torch.float32)
    valid_mask = torch.ones((2, 3), dtype=torch.bool)
    values = torch.randn((2, 3), dtype=torch.float32)
    dones = torch.zeros((2, 3), dtype=torch.bool)

    advantages, _returns = compute_gae(
        rewards,
        values,
        dones,
        valid_mask,
        gamma=0.99,
        gae_lambda=0.95,
    )
    normalized, mean, std = normalize_advantages(advantages, valid_mask)

    assert should_update_actor(rewards, valid_mask) is False
    torch.testing.assert_close(normalized[valid_mask].mean(), torch.tensor(mean))
    assert math.isfinite(std)


def test_direct_ppo_mc_advantage_available_as_optional_fallback():
    rewards = torch.zeros((2, 3), dtype=torch.float32)
    dones = torch.zeros((2, 3), dtype=torch.bool)
    valid_mask = torch.ones((2, 3), dtype=torch.bool)
    random_values = torch.randn((2, 3), dtype=torch.float32)

    returns = compute_discounted_returns(rewards, dones, valid_mask, gamma=0.99)

    torch.testing.assert_close(returns, torch.zeros_like(returns))
    assert should_update_actor(rewards, valid_mask) is False
    assert not torch.equal(random_values, returns)


def test_masked_chunk_logprob_ignores_terminal_tail():
    logprobs = torch.ones((2, 5, 4), dtype=torch.float32)
    valid_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, True, True],
        ]
    )

    summed = masked_chunk_logprob(logprobs, valid_mask)

    torch.testing.assert_close(summed, torch.tensor([8.0, 20.0]))


def test_metaworld_chunk_step_masks_terminal_tail(monkeypatch):
    from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv

    rollout = FakeMetaWorldRollout(success_steps=[2, 5])
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "task_name": "push-v3",
            "task_description": "Push the puck to a goal",
            "reset_randomization_mode": "random_seeded",
            "max_episode_steps": 10,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "reward_mode": "sparse_success_delta",
            "reset_seed_base": 2000,
            "use_async_envs": False,
            "video_cfg": {"save_video": False},
        }
    )
    with _patch_metaworld_rollout(monkeypatch, rollout):
        env = SmolVLAMetaWorldEnv(
            cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None
        )
        try:
            env.reset()
            actions = np.zeros((2, 5, 4), dtype=np.float32)
            _obs_list, rewards, terms, truncs, infos_list = env.chunk_step(actions)
        finally:
            env.close()

    valid_mask = infos_list[-1]["valid_action_mask"]
    torch.testing.assert_close(
        valid_mask,
        torch.tensor(
            [
                [True, True, False, False, False],
                [True, True, True, True, True],
            ]
        ),
    )
    torch.testing.assert_close(rewards[0], torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]))
    assert bool(terms[0, 1])
    assert not bool(terms[0, 2:].any())
    assert not bool(truncs.any())


def test_metaworld_chunk_step_reports_all_rows_terminal(monkeypatch):
    from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv

    rollout = FakeMetaWorldRollout(success_steps=[1, 2])
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "task_name": "push-v3",
            "task_description": "Push the puck to a goal",
            "reset_randomization_mode": "random_seeded",
            "max_episode_steps": 10,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "reward_mode": "sparse_success_delta",
            "reset_seed_base": 2000,
            "use_async_envs": False,
            "video_cfg": {"save_video": False},
        }
    )
    with _patch_metaworld_rollout(monkeypatch, rollout):
        env = SmolVLAMetaWorldEnv(
            cfg, num_envs=2, seed_offset=0, total_num_processes=1, worker_info=None
        )
        try:
            env.reset()
            actions = np.zeros((2, 5, 4), dtype=np.float32)
            _obs_list, _rewards, _terms, _truncs, infos_list = env.chunk_step(actions)
        finally:
            env.close()

    assert infos_list[-1]["all_rows_terminal"] is True
    torch.testing.assert_close(
        infos_list[-1]["valid_action_mask"],
        torch.tensor(
            [
                [True, False, False, False, False],
                [True, True, False, False, False],
            ]
        ),
    )
