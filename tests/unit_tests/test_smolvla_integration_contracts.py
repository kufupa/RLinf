import math
import sys
import types

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
