# Copyright 2025 The RLinf Authors.
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

"""Unit tests for oracle teacher-forced WM env wiring (mocked oracle + WM)."""

from pathlib import Path
from omegaconf import OmegaConf
from unittest.mock import MagicMock, patch

import numpy as np
import torch


def test_wm_flowsde_tf_config_contract():
    root = Path(__file__).resolve().parents[2]
    top = root / "examples/embodiment/config/metaworld_pushv3_wm_flowsde_tf_smolvla.yaml"
    env_cfg = root / "examples/embodiment/config/env/metaworld_smolvla_wm_pushv3.yaml"
    model_cfg = root / "examples/embodiment/config/model/smolvla.yaml"
    text = top.read_text() + env_cfg.read_text() + model_cfg.read_text()
    for needle in (
        "reward_mode: wm_latent_progress",
        "root_mode: oracle_teacher_forced",
        "noise_method: flow_sde",
        "adv_type: grpo",
        "logprob_type: chunk_level",
    ):
        assert needle in text


def _make_cfg(**overrides):
    base = {
        "seed": 0,
        "task_name": "push-v3",
        "task_description": "Push",
        "reset_randomization_mode": "random_seeded",
        "max_episode_steps": 1,
        "auto_reset": False,
        "ignore_terminations": False,
        "use_rel_reward": False,
        "reward_coef": 1.0,
        "reward_mode": "sparse_success_delta",
        "root_mode": "native",
        "group_size": 4,
        "num_action_chunks": 5,
        "reset_seed_base": 100,
        "use_async_envs": False,
        "video_cfg": {"save_video": False},
    }
    base.update(overrides)
    return OmegaConf.create(base)


def test_update_reset_state_ids_repeats_group_seeds():
    from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv

    cfg = _make_cfg(reward_mode="sparse_success_delta", group_size=4)
    with patch.object(SmolVLAMetaWorldEnv, "_make_rollout", return_value=MagicMock()):
        env = SmolVLAMetaWorldEnv(
            cfg=cfg,
            num_envs=8,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
    env.update_reset_state_ids()
    assert env.reset_state_ids.shape == (8,)
    assert len(np.unique(env.reset_state_ids[:4])) == 1
    assert len(np.unique(env.reset_state_ids[4:])) == 1


def test_wm_chunk_step_one_episode_no_sim_step():
    from rlinf.envs.metaworld.oracle_roots import OracleRootState
    from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv

    cfg = _make_cfg(
        reward_mode="wm_latent_progress",
        root_mode="oracle_teacher_forced",
        group_size=2,
        jepa_repo="/tmp/jepa",
        jepa_ckpt="/tmp/jepa.ckpt",
        wm_device="cpu",
        wm_proprio_alpha=0.1,
        goal_latent_mode="visual_proprio",
    )
    fake_root = OracleRootState(
        seed=1,
        frame_index_1based=5,
        policy_image=np.zeros((64, 64, 3), dtype=np.uint8),
        wm_image=np.zeros((64, 64, 3), dtype=np.uint8),
        proprio=np.zeros(4, dtype=np.float32),
        goal_latent={"visual": MagicMock()},
    )
    mock_rollout = MagicMock()
    with patch.object(SmolVLAMetaWorldEnv, "_make_rollout", return_value=mock_rollout):
        with patch(
            "rlinf.envs.metaworld.smolvla_metaworld_env._load_jepa_wm_bundle",
            return_value=MagicMock(),
        ):
            with patch(
                "rlinf.envs.metaworld.smolvla_metaworld_env._make_oracle_catalog"
            ) as catalog_factory:
                catalog_factory.return_value.next_root_for_group.return_value = (
                    fake_root
                )
                env = SmolVLAMetaWorldEnv(
                    cfg=cfg,
                    num_envs=2,
                    seed_offset=0,
                    total_num_processes=1,
                    worker_info=None,
                )
    env.reset_many(np.array([10, 10], dtype=np.int64))
    with patch.object(env, "_score_wm_latent_progress", return_value=np.array([0.1, 0.2])):
        obs_list, rewards, terms, truncs, infos = env.chunk_step(
            np.zeros((2, 5, 4), dtype=np.float32)
        )
    mock_rollout.step_batch.assert_not_called()
    assert rewards.shape == (2, 5)
    assert torch.allclose(rewards[:, -1], torch.tensor([0.1, 0.2]))
    assert terms[:, -1].all()
    assert obs_list[-1]["main_images"].shape[0] == 2
