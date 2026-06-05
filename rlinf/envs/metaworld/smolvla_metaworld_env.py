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


def _load_jepa_wm_bundle(cfg):
    jepa_repo = str(cfg.get("jepa_repo", "") or "")
    jepa_ckpt = str(cfg.get("jepa_ckpt", ""))
    if not jepa_repo or not jepa_ckpt:
        raise ValueError("wm_latent_progress requires jepa_repo and jepa_ckpt in env cfg")
    from rlinf.models.embodiment.world_model.jepa_wm import build_jepa_wm_bundle

    return build_jepa_wm_bundle(cfg)


def _make_oracle_catalog(cfg, wm_bundle):
    from rlinf.envs.metaworld.oracle_roots import OracleTeacherForcedCatalog

    return OracleTeacherForcedCatalog(cfg, wm_bundle)


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
        self.root_mode = str(self.cfg.get("root_mode", "native"))
        self.group_size = int(self.cfg.get("group_size", 1))
        self.num_group = max(1, self.num_envs // self.group_size)
        self.num_action_chunks = int(
            self.cfg.get("num_action_chunks", self.cfg.get("chunk_len", 5))
        )
        self._wm_teacher_forced = (
            self.reward_mode == "wm_latent_progress"
            and self.root_mode == "oracle_teacher_forced"
        )

        self._is_start = True
        self._reset_counter = 0
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.prev_step_reward = np.zeros(self.num_envs, dtype=np.float32)
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs, dtype=np.float32)
        self.reset_seeds = np.zeros(self.num_envs, dtype=np.int64)
        self.reset_state_ids = self.reset_seeds.copy()
        self._terminal_rows = np.zeros(self.num_envs, dtype=bool)

        self.video_cfg = self.cfg.video_cfg
        self.env = self._make_rollout()
        self.last_obs = None
        self._wm_bundle = None
        self._oracle_catalog = None
        self._oracle_roots: list[Any | None] = [None] * self.num_envs
        if self.reward_mode == "wm_latent_progress":
            self._wm_bundle = _load_jepa_wm_bundle(self.cfg)
        if self.root_mode == "oracle_teacher_forced":
            if self._wm_bundle is None:
                raise ValueError(
                    "root_mode=oracle_teacher_forced requires reward_mode=wm_latent_progress"
                )
            self._oracle_catalog = _make_oracle_catalog(self.cfg, self._wm_bundle)
        if self.group_size > 1 or self._wm_teacher_forced:
            self.update_reset_state_ids()

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
            from rlinf.envs.metaworld.lerobot_adapter import (
                OfficialLeRobotMetaWorldGRPORollout,
            )
        except Exception as exc:
            raise ImportError(
                "SmolVLAMetaWorldEnv requires LeRobot, MetaWorld, MuJoCo, and "
                "Gymnasium embodied dependencies."
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

        if reset_state_ids is not None:
            return self.reset_many(reset_state_ids)
        if self.group_size > 1 or self._wm_teacher_forced:
            return self.reset_many(self.reset_state_ids)
        return self.reset_many(self._next_reset_seeds(self.num_envs))

    def reset_many(self, reset_seeds):
        seeds = np.asarray([int(seed) for seed in reset_seeds], dtype=np.int64)
        if seeds.shape != (self.num_envs,):
            raise ValueError(f"reset_many expected {self.num_envs} seeds; got {seeds.shape[0]}")
        self.reset_seeds = seeds
        self.reset_state_ids = seeds.copy()
        if self._wm_teacher_forced:
            return self._reset_many_oracle_teacher_forced(seeds)
        raw_obs = self.env.reset_many(seeds)
        self.last_obs = raw_obs
        self._reset_metrics()
        self._terminal_rows[:] = False
        return self._wrap_obs(raw_obs), {}

    def _reset_many_oracle_teacher_forced(self, seeds: np.ndarray):
        assert self._oracle_catalog is not None
        images: list[np.ndarray] = []
        states: list[np.ndarray] = []
        root_by_seed: dict[int, Any] = {}
        for env_idx, seed in enumerate(seeds):
            seed_int = int(seed)
            if seed_int not in root_by_seed:
                root_by_seed[seed_int] = self._oracle_catalog.next_root_for_group(
                    seed_int
                )
            root = root_by_seed[seed_int]
            self._oracle_roots[env_idx] = root
            images.append(np.asarray(root.policy_image, dtype=np.uint8))
            states.append(np.asarray(root.proprio, dtype=np.float32))
        raw_obs = {
            "pixels": np.stack(images, axis=0),
            "agent_pos": np.stack(states, axis=0),
        }
        self.last_obs = raw_obs
        self._reset_metrics()
        self._terminal_rows[:] = False
        return self._wrap_obs(raw_obs), {}

    def update_reset_state_ids(self):
        """Compatibility hook for RLinf EnvWorker rollout finalization."""
        if self.group_size > 1 or self._wm_teacher_forced:
            group_seeds = self._next_reset_seeds(self.num_group)
            self.reset_state_ids = np.repeat(group_seeds, self.group_size).astype(
                np.int64
            )
            self.reset_seeds = self.reset_state_ids.copy()
            return
        self.reset_state_ids = self.reset_seeds.copy()

    def _score_wm_latent_progress(self, chunk_actions: np.ndarray) -> np.ndarray:
        from rlinf.models.embodiment.world_model.jepa_wm import score_wm_latent_progress

        assert self._wm_bundle is not None
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        proprio_alpha = float(self.cfg.get("wm_proprio_alpha", 0.1))
        goal_mode = str(self.cfg.get("goal_latent_mode", "visual_proprio"))
        for env_idx in range(self.num_envs):
            root = self._oracle_roots[env_idx]
            if root is None:
                raise RuntimeError(
                    f"missing oracle root for env {env_idx} in wm_latent_progress mode"
                )
            rewards[env_idx] = score_wm_latent_progress(
                self._wm_bundle,
                image=root.wm_image,
                proprio=root.proprio,
                chunk_actions=chunk_actions[env_idx],
                goal=root.goal_latent,
                candidate_index=env_idx % self.group_size,
                proprio_alpha=proprio_alpha,
                mode=goal_mode,
            )
        return self.reward_coef * rewards

    def _step_reward(
        self,
        env_reward: np.ndarray,
        success: np.ndarray,
        active_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        if active_mask is None:
            active_mask = np.ones(self.num_envs, dtype=bool)
        active_mask = np.asarray(active_mask, dtype=bool).reshape(self.num_envs)
        step_reward = np.zeros(self.num_envs, dtype=np.float32)
        env_reward = np.asarray(env_reward, dtype=np.float32).reshape(self.num_envs)
        success = np.asarray(success, dtype=bool).reshape(self.num_envs)
        if self.reward_mode == "dense_return":
            step_reward[active_mask] = self.reward_coef * env_reward[active_mask]
            return step_reward

        reward = self.reward_coef * success.astype(np.float32)
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            step_reward[active_mask] = reward_diff[active_mask]
            self.prev_step_reward[active_mask] = reward[active_mask]
            return step_reward
        step_reward[active_mask] = reward[active_mask]
        return step_reward

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

    def step(self, actions=None, auto_reset=True, active_mask=None):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, 4):
            raise ValueError(f"Expected action shape {(self.num_envs, 4)}, got {actions.shape}")

        if active_mask is None:
            active_mask = ~self._terminal_rows
        active_mask = np.asarray(active_mask, dtype=bool).reshape(self.num_envs)
        active_mask = active_mask & ~self._terminal_rows
        step_actions = actions.copy()
        step_actions[~active_mask] = 0.0

        self._elapsed_steps[active_mask] += 1
        batch_step = self.env.step_batch(step_actions)
        self.last_obs = batch_step.observation
        terminations = (
            np.asarray(batch_step.success, dtype=bool).reshape(self.num_envs)
            & active_mask
        )
        truncations = (self.elapsed_steps >= self.max_episode_steps) & active_mask
        step_reward = self._step_reward(
            np.asarray(batch_step.reward),
            terminations,
            active_mask=active_mask,
        )

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
        if self.reward_mode == "wm_latent_progress":
            return self._chunk_step_wm_latent_progress(chunk_actions)
        return self._chunk_step_sim(chunk_actions)

    def _chunk_step_wm_latent_progress(self, chunk_actions: np.ndarray):
        chunk_size = int(chunk_actions.shape[1])
        if chunk_size != self.num_action_chunks:
            raise ValueError(
                f"expected chunk size {self.num_action_chunks}, got {chunk_size}"
            )
        step_reward = self._score_wm_latent_progress(chunk_actions)
        obs = self._wrap_obs(self.last_obs)
        obs_list = [obs for _ in range(chunk_size)]
        chunk_rewards = torch.zeros(self.num_envs, chunk_size, dtype=torch.float32)
        chunk_rewards[:, -1] = torch.as_tensor(step_reward, dtype=torch.float32)
        chunk_terminations = torch.zeros(
            self.num_envs, chunk_size, dtype=torch.bool
        )
        chunk_terminations[:, -1] = True
        chunk_truncations = torch.zeros_like(chunk_terminations)
        self._elapsed_steps[:] = 1
        self._terminal_rows[:] = True
        infos = self._record_metrics(
            step_reward,
            chunk_terminations[:, -1].cpu().numpy(),
            {},
        )
        infos["valid_action_mask"] = torch.ones(
            self.num_envs, chunk_size, dtype=torch.bool
        )
        infos["executed_steps"] = torch.ones(self.num_envs, dtype=torch.long)
        infos["all_rows_terminal"] = True
        infos_list = [copy.deepcopy(infos) for _ in range(chunk_size)]
        if self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                np.ones(self.num_envs, dtype=bool), obs_list[-1], infos_list[-1]
            )
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _chunk_step_sim(self, chunk_actions):
        chunk_actions = np.asarray(chunk_actions, dtype=np.float32)
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        valid_action_masks = []
        executed_steps = np.zeros(self.num_envs, dtype=np.int64)
        terminal_rows = np.zeros(self.num_envs, dtype=bool)
        terminal_episode: dict[str, torch.Tensor] = {}

        for i in range(chunk_size):
            active = ~self._terminal_rows
            valid_action_masks.append(torch.as_tensor(active.copy(), dtype=torch.bool))
            if bool(active.any()):
                obs, reward, terminations, truncations, infos = self.step(
                    chunk_actions[:, i], auto_reset=False, active_mask=active
                )
            else:
                obs = self._wrap_obs(self.last_obs)
                reward = torch.zeros(self.num_envs, dtype=torch.float32)
                terminations = torch.zeros(self.num_envs, dtype=torch.bool)
                truncations = torch.zeros(self.num_envs, dtype=torch.bool)
                infos = self._record_metrics(
                    np.zeros(self.num_envs, dtype=np.float32),
                    np.zeros(self.num_envs, dtype=bool),
                    {},
                )
            infos = copy.deepcopy(infos)
            executed_steps += active.astype(np.int64)
            infos["valid_action_mask"] = to_tensor(active.copy())
            infos["executed_steps"] = to_tensor(executed_steps.copy())

            done_rows = torch.logical_or(terminations, truncations).detach().cpu().numpy()
            if np.asarray(done_rows).any():
                terminal_rows |= np.asarray(done_rows, dtype=bool)
                episode = infos.get("episode", {})
                for key, value in episode.items():
                    if not isinstance(value, torch.Tensor) or value.shape[:1] != (
                        self.num_envs,
                    ):
                        continue
                    if key not in terminal_episode:
                        terminal_episode[key] = torch.zeros_like(value)
                    terminal_episode[key][torch.as_tensor(done_rows, dtype=torch.bool)] = value[
                        torch.as_tensor(done_rows, dtype=torch.bool)
                    ]
                self._terminal_rows |= np.asarray(done_rows, dtype=bool)
                self._reset_metrics(np.asarray(done_rows, dtype=bool))

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
        valid_action_mask = torch.stack(valid_action_masks, dim=1)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        chunk_terminations = raw_chunk_terminations.clone()
        chunk_truncations = raw_chunk_truncations.clone()
        if infos_list:
            infos_list[-1] = copy.deepcopy(infos_list[-1])
            final_episode = infos_list[-1].get("episode", {})
            if terminal_episode:
                final_episode = copy.deepcopy(final_episode)
                done_tensor = torch.as_tensor(terminal_rows, dtype=torch.bool)
                for key, value in terminal_episode.items():
                    base = final_episode.get(key, torch.zeros_like(value))
                    if isinstance(base, torch.Tensor) and base.shape == value.shape:
                        base = base.clone()
                        base[done_tensor] = value[done_tensor]
                        final_episode[key] = base
                infos_list[-1]["episode"] = final_episode
            infos_list[-1]["valid_action_mask"] = valid_action_mask
            infos_list[-1]["executed_steps"] = torch.as_tensor(
                executed_steps, dtype=torch.long
            )
            infos_list[-1]["all_rows_terminal"] = bool(self._terminal_rows.all())

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
