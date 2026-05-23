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

from __future__ import annotations

import copy
from typing import Any, Optional, Union

import numpy as np
import torch

try:
    import gymnasium as gym
except ModuleNotFoundError:  # pragma: no cover - optional embodied dependency.

    class _Env:
        pass

    class _GymFallback:
        Env = _Env

    gym = _GymFallback()


def to_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    if isinstance(value, dict):
        return {key: to_tensor(item) for key, item in value.items()}
    return torch.as_tensor(value)


class SmolVLAMetaWorldEnv(gym.Env):
    """MetaWorld push-v3 env using SmolVLA's LeRobot image+proprio contract."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info

        self.seed = int(self.cfg.get("seed", 0)) + self.seed_offset
        self.task_name = str(self.cfg.get("task_name", "push-v3"))
        self.task_description = str(
            self.cfg.get("task_description", "Push the puck to a goal")
        )
        self.reset_randomization_mode = str(
            self.cfg.get("reset_randomization_mode", "random_seeded")
        )
        self.max_episode_steps = int(self.cfg.max_episode_steps)
        self.auto_reset = bool(self.cfg.auto_reset)
        self.ignore_terminations = bool(self.cfg.ignore_terminations)
        self.use_rel_reward = bool(self.cfg.get("use_rel_reward", True))
        self.reward_coef = float(self.cfg.get("reward_coef", 1.0))
        self.reward_mode = str(self.cfg.get("reward_mode", "sparse_success_delta"))

        self._is_start = True
        self._reset_counter = 0
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.reset_seeds = np.zeros(self.num_envs, dtype=np.int64)
        self.reset_state_ids = self.reset_seeds.copy()

        self.video_cfg = self.cfg.video_cfg
        self.env = self._make_rollout()
        self.last_obs = None

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _make_rollout(self):
        try:
            from smolvla_grpo.lerobot_metaworld_adapter import (
                OfficialLeRobotMetaWorldGRPORollout,
            )
        except Exception as exc:
            raise ImportError(
                "SmolVLAMetaWorldEnv requires Phase11 SmolVLA helpers on PYTHONPATH "
                "for the spike. Add /rds/general/user/aa6622/home/project/src."
            ) from exc

        return OfficialLeRobotMetaWorldGRPORollout(
            task=self.task_name,
            obs_type="pixels_agent_pos",
            n_envs=self.num_envs,
            use_async_envs=bool(self.cfg.get("use_async_envs", False)),
            reset_randomization_mode=self.reset_randomization_mode,
        )

    def _next_reset_seeds(self, n: int) -> np.ndarray:
        explicit = self.cfg.get("eval_seed_list", None)
        if explicit is not None and len(explicit) > 0:
            values = [
                int(explicit[(self._reset_counter + i) % len(explicit)])
                for i in range(n)
            ]
        else:
            base = int(self.cfg.get("reset_seed_base", self.seed))
            values = [base + self._reset_counter + i for i in range(n)]
        self._reset_counter += n
        return np.asarray(values, dtype=np.int64)

    def _wrap_obs(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        if "pixels" not in raw_obs or "agent_pos" not in raw_obs:
            raise KeyError("SmolVLA MetaWorld obs must contain 'pixels' and 'agent_pos'.")

        images = torch.as_tensor(np.asarray(raw_obs["pixels"]), dtype=torch.uint8)
        states = torch.as_tensor(np.asarray(raw_obs["agent_pos"]), dtype=torch.float32)
        if images.ndim == 3:
            images = images.unsqueeze(0)
        if states.ndim == 1:
            states = states.unsqueeze(0)
        if images.shape[0] != self.num_envs or states.shape[0] != self.num_envs:
            raise ValueError(
                f"Expected batch {self.num_envs}, got images={tuple(images.shape)} "
                f"states={tuple(states.shape)}"
            )

        return {
            "main_images": images.contiguous(),
            "states": states.contiguous(),
            "task_descriptions": [self.task_description for _ in range(self.num_envs)],
            "reset_seeds": torch.as_tensor(self.reset_seeds, dtype=torch.long),
        }

    def _reset_metrics(self, env_idx=None):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        self.prev_step_reward[env_idx] = 0.0
        self.success_once[env_idx] = False
        self.fail_once[env_idx] = False
        self.returns[env_idx] = 0.0
        self._elapsed_steps[env_idx] = 0

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is not None:
            env_idx_arr = np.asarray(env_idx)
            if env_idx_arr.size != self.num_envs or not np.array_equal(
                env_idx_arr, np.arange(self.num_envs)
            ):
                raise NotImplementedError(
                    "SmolVLAMetaWorldEnv currently resets the full vector batch. "
                    "Use auto_reset=False for PPO smoke."
                )

        seeds = self._next_reset_seeds(self.num_envs)
        self.reset_seeds = seeds
        self.reset_state_ids = seeds.copy()
        raw_obs = self.env.reset_many(seeds)
        self.last_obs = raw_obs
        self._reset_metrics()
        return self._wrap_obs(raw_obs), {}

    def update_reset_state_ids(self):
        """Compatibility hook for RLinf EnvWorker rollout finalization."""
        self.reset_state_ids = self.reset_seeds.copy()

    def _step_reward(self, env_reward: np.ndarray, success: np.ndarray) -> np.ndarray:
        if self.reward_mode == "dense_return":
            return self.reward_coef * env_reward.astype(np.float32)
        reward = self.reward_coef * success.astype(np.float32)
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff.astype(np.float32)
        return reward.astype(np.float32)

    def _record_metrics(self, step_reward, terminations, infos):
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info = {
            "success_once": self.success_once.copy(),
            "return": self.returns.copy(),
            "episode_len": self.elapsed_steps.copy(),
            "reward": self.returns / np.maximum(self.elapsed_steps, 1),
        }
        infos["episode"] = to_tensor(episode_info)
        return infos

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 4):
            raise ValueError(f"Expected action shape {(self.num_envs, 4)}, got {actions.shape}")

        self._elapsed_steps += 1
        batch_step = self.env.step_batch(actions)
        self.last_obs = batch_step.observation
        terminations = np.asarray(batch_step.success, dtype=bool).reshape(self.num_envs)
        truncations = self.elapsed_steps >= self.max_episode_steps
        step_reward = self._step_reward(np.asarray(batch_step.reward), terminations)

        obs = self._wrap_obs(batch_step.observation)
        infos = dict(batch_step.info) if isinstance(batch_step.info, dict) else {}
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        if auto_reset and self.auto_reset and dones.any():
            obs, infos = self._handle_auto_reset(dones, obs, infos)

        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []

        for i in range(chunk_size):
            obs, reward, terminations, truncations, infos = self.step(
                chunk_actions[:, i], auto_reset=False
            )
            obs_list.append(obs)
            infos_list.append(infos)
            chunk_rewards.append(reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)
        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()

        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, final_obs, infos):
        if np.asarray(dones).any() and np.asarray(dones).sum() != self.num_envs:
            raise NotImplementedError(
                "Partial auto reset is not supported for the initial SmolVLA PPO spike. "
                "Keep auto_reset=False or use episode-synchronous batches."
            )
        obs, reset_infos = self.reset()
        infos = copy.deepcopy(infos)
        final_infos = copy.deepcopy(infos)
        infos["final_observation"] = final_obs
        infos["final_info"] = final_infos
        infos["_final_info"] = to_tensor(dones)
        infos["_final_observation"] = to_tensor(dones)
        infos["_elapsed_steps"] = to_tensor(dones)
        infos.update(reset_infos)
        return obs, infos

    def close(self):
        self.env.close()
