"""Native MetaWorld + SmolVLA integration contracts.

Single-task push-v3 benchmark: ``test_metaworld_benchmark.py``.
PPO zero-reward / valid-mask RCA: ``test_smolvla_integration_contracts.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs import get_env_cls
from rlinf.models.embodiment.smolvla.smolvla_action_model import (
    SmolVLAForRLActionPrediction,
)
from test_smolvla_integration_contracts import (
    FakeBundle,
    _cfg,
    _patch_lerobot_preprocess,
)

# MetaWorldEnv.reset() runs RESET_STEP partial steps plus one full step before chunk.
_NATIVE_RESET_WARMUP_STEPS = 16


class FakeNativeVectorEnv:
    """Lightweight ReconfigureSubprocEnv stand-in for native MetaWorldEnv tests."""

    def __init__(
        self,
        num_envs: int,
        success_steps: list[int] | None = None,
    ):
        self.num_envs = num_envs
        self.success_steps = (
            np.asarray(success_steps, dtype=np.int64)
            if success_steps is not None
            else np.full(num_envs, np.iinfo(np.int64).max)
        )
        self._steps = np.zeros(num_envs, dtype=np.int64)
        self.chunk_step_calls = 0

    def render(self):
        return np.full((self.num_envs, 8, 8, 3), 128, dtype=np.uint8)

    def _step_env_ids(self, env_ids: np.ndarray):
        for eid in env_ids:
            self._steps[eid] += 1
        obs = np.zeros((len(env_ids), 39), dtype=np.float32)
        info_lists = [
            {"success": bool(self._steps[eid] >= self.success_steps[eid])}
            for eid in env_ids
        ]
        return (
            obs,
            np.zeros(len(env_ids), dtype=np.float32),
            np.zeros(len(env_ids), dtype=bool),
            np.zeros(len(env_ids), dtype=bool),
            info_lists,
        )

    def step(self, actions, id=None):
        if id is None:
            env_ids = np.arange(self.num_envs)
        else:
            env_ids = np.asarray(id)
        self.chunk_step_calls += 1
        return self._step_env_ids(env_ids)

    def reset(self, id=None):
        if id is None:
            self._steps[:] = 0
        else:
            self._steps[np.asarray(id)] = 0

    def reconfigure_env_fns(self, env_fn_params, env_idx):
        pass

    def begin_chunk(self):
        self.chunk_step_calls = 0


def _native_env_cfg(**kwargs):
    base = {
        "seed": 0,
        "task_suite_name": "metaworld_single",
        "task_names": ["push-v3"],
        "use_fixed_reset_state_ids": False,
        "use_ordered_reset_state_ids": False,
        "is_eval": False,
        "group_size": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        "use_rel_reward": True,
        "reward_coef": 1.0,
        "max_episode_steps": 10,
        "video_cfg": {"save_video": False},
    }
    base.update(kwargs)
    return OmegaConf.create(base)


def _patch_native_init_env(monkeypatch, fake_env: FakeNativeVectorEnv):
    from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

    def fake_init_env(self):
        self.env_names_all = self.task_suite.get_env_names()
        self.task_descriptions_all = self.task_suite.get_task_description()
        self.use_async_vector_env = False
        self.env = fake_env
        self.get_env_fn_params(np.arange(self.num_envs))

    monkeypatch.setattr(MetaWorldEnv, "_init_env", fake_init_env)


def _make_native_env(monkeypatch, num_envs=2, success_steps=None, **cfg_kwargs):
    from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

    fake_env = FakeNativeVectorEnv(num_envs, success_steps=success_steps)
    _patch_native_init_env(monkeypatch, fake_env)
    cfg = _native_env_cfg(**cfg_kwargs)
    env = MetaWorldEnv(
        cfg,
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    return env, fake_env


def test_metaworld_env_type_dispatch():
    env_cls = get_env_cls("metaworld")
    assert env_cls.__name__ == "MetaWorldEnv"


def test_native_metaworld_wrap_obs_contract():
    from rlinf.envs.metaworld.metaworld_env import MetaWorldEnv

    env = MetaWorldEnv.__new__(MetaWorldEnv)
    env.num_envs = 2
    env.task_descriptions = ["Push the puck to a goal"] * 2
    env.reset_state_ids = np.array([0, 1], dtype=np.int64)
    env.env = FakeNativeVectorEnv(num_envs=2)

    raw_obs = np.zeros((2, 39), dtype=np.float32)
    wrapped = env._wrap_obs(raw_obs)

    assert set(wrapped) == {
        "main_images",
        "states",
        "task_descriptions",
        "reset_seeds",
    }
    assert wrapped["main_images"].shape == (2, 8, 8, 3)
    assert wrapped["states"].shape == (2, 4)
    assert wrapped["task_descriptions"] == ["Push the puck to a goal"] * 2
    assert wrapped["reset_seeds"].shape == (2,)


def test_native_metaworld_reset_obs_matches_smolvla_bundle(monkeypatch):
    env, _fake_env = _make_native_env(monkeypatch, num_envs=2)
    obs, _infos = env.reset()

    bundle = FakeBundle()
    proc = bundle.preprocessor(
        {
            bundle.obs_image_key: obs["main_images"].permute(0, 3, 1, 2).float() / 255.0,
            bundle.obs_state_key: obs["states"],
        }
    )
    assert proc["observation"]["image"].shape == (2, 3, 8, 8)
    assert proc["observation"]["state"].shape == (2, 4)


def test_native_single_task_eval_repeats_reset_states_for_many_envs(monkeypatch):
    env, _fake_env = _make_native_env(monkeypatch, num_envs=25, is_eval=True)

    assert env.reset_state_ids.shape == (25,)
    assert env.task_ids.shape == (25,)
    assert env.trial_ids.shape == (25,)
    assert np.all(env.task_ids == 0)
    np.testing.assert_array_equal(env.trial_ids, np.tile(np.arange(10), 3)[:25])


def test_native_obs_smolvla_action_shape(monkeypatch):
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
        "task_descriptions": ["Push the puck to a goal"] * 2,
    }
    actions, result = model.predict_action_batch(env_obs, mode="train")

    assert actions.shape == (2, 5, 4)
    assert result["prev_logprobs"].shape == (2, 5, 4)
    assert result["prev_values"].shape == (2, 1)


def test_native_chunk_step_masks_past_terminal(monkeypatch):
    success_steps = [
        _NATIVE_RESET_WARMUP_STEPS + 2,
        _NATIVE_RESET_WARMUP_STEPS + 5,
    ]
    env, fake_env = _make_native_env(monkeypatch, num_envs=2, success_steps=success_steps)
    env.reset()
    fake_env.begin_chunk()
    actions = np.zeros((2, 5, 4), dtype=np.float32)
    _obs_list, rewards, terminations, _truncations, infos_list = env.chunk_step(actions)

    # Five chunk indices => five vector step calls; env 0 stops after index 1.
    assert fake_env.chunk_step_calls == 5
    assert fake_env._steps[0] == success_steps[0]
    assert fake_env._steps[1] == success_steps[1]
    valid_mask = infos_list[-1]["valid_action_mask"]
    assert valid_mask.shape == (2, 5)
    assert bool(valid_mask[0, 0])
    assert bool(valid_mask[0, 1])
    assert not bool(valid_mask[0, 2:].any())

    assert bool(terminations[0, 1])
    torch.testing.assert_close(
        rewards[0],
        torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0]),
    )
    assert infos_list[-1]["all_rows_terminal"] is True


def test_single_task_benchmark_covered_elsewhere():
    pytest.importorskip("rlinf.envs.metaworld")
    import test_metaworld_benchmark

    assert hasattr(test_metaworld_benchmark, "test_metaworld_single_push_v3")
