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

import copy
import os

# Ensure MW envs only register once
import warnings
from typing import Optional, Union

import gymnasium as gym
import metaworld
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.metaworld import MetaWorldBenchmark
from rlinf.envs.metaworld.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor

if not getattr(metaworld, "_has_registered_mw_envs", False):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*Overriding environment.*already in registry.*"
        )
        metaworld.register_mw_envs()
    metaworld._has_registered_mw_envs = True


class MetaWorldEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)

        self.RESET_STEP = 15
        benchmark_kwargs = {}
        if OmegaConf.select(self.cfg, "task_names") is not None:
            task_names_raw = OmegaConf.to_container(self.cfg.task_names, resolve=True)
            benchmark_kwargs["task_names"] = (
                task_names_raw
                if isinstance(task_names_raw, list)
                else [task_names_raw]
            )
        if OmegaConf.select(self.cfg, "task_num_trials") is not None:
            benchmark_kwargs["task_num_trials"] = self.cfg.task_num_trials
        self.task_suite: MetaWorldBenchmark = MetaWorldBenchmark(
            self.cfg.task_suite_name,
            **benchmark_kwargs,
        )
        self.num_tasks = self.task_suite.get_num_tasks()
        self.task_num_trials = self.task_suite.get_task_num_trials()
        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._terminal_rows = np.zeros(self.num_envs, dtype=bool)
        self._last_raw_obs: Optional[np.ndarray] = None

        self.video_cfg = cfg.video_cfg

    def _init_env(self):
        # metaworld task and prompt description
        self.env_names_all = self.task_suite.get_env_names()
        self.task_descriptions_all = self.task_suite.get_task_description()
        env_fns = self.get_env_fns()
        self.use_async_vector_env = False
        if self.use_async_vector_env:
            assert not self.auto_reset, "AsyncVectorEnv does not support auto_reset."
            self.env = gym.vector.AsyncVectorEnv(env_fns)
        else:
            self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []
        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param):
                os.environ["MUJOCO_EGL_DEVICE_ID"] = str(self.seed_offset)
                env_name = param["env_name"]
                env = gym.make(
                    "Meta-World/MT1",
                    env_name=env_name,
                    render_mode="rgb_array",
                    camera_id=2,
                    disable_env_checker=True,
                )
                # Set camera position to align with sft
                env.env.env.env.env.env.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        task_descriptions = []
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        for env_id in range(self.num_envs):
            if env_id not in env_idx:
                task_descriptions.append(self.task_descriptions[env_id])
                continue
            env_name = self.env_names_all[self.task_ids[env_id]]
            task_description = self.task_descriptions_all[self.task_ids[env_id]]

            env_fn_params.append(
                {
                    "env_name": env_name,
                }
            )
            task_descriptions.append(task_description)
        self.task_descriptions = task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.num_tasks):
            self.trial_id_bins.append(self.task_num_trials)
            self.total_num_group_envs += self.task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def update_reset_state_ids(self):
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        reset_state_ids = self.reset_state_ids_all[self.seed_offset]
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        # get task id and trial id from reset state ids
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot

        return np.array(task_ids), np.array(trial_ids)

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

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        images = self.env.render()
        images = np.array(images)[:, ::-1, ::-1]
        state = obs[:, :4]
        return {
            "full_image": images,
            "state": state,
        }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        images = self.env.render()
        images = np.array(images)[:, ::-1, ::-1]
        state = obs_list[:, :4]
        for idx in range(self.num_envs):
            images_and_states = {
                "full_image": images[idx],
                "state": state[idx],
            }
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "states": states,
            "task_descriptions": self.task_descriptions,
            "reset_seeds": torch.as_tensor(self.reset_state_ids, dtype=torch.long),
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            if self.task_ids[env_id] != task_ids[j]:
                reconfig_env_idx.append(env_id)
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
        if self.use_async_vector_env:
            env_fns = self.get_env_fns()
            self.env = gym.vector.AsyncVectorEnv(env_fns)
            self.env.reset()
        else:
            if reconfig_env_idx:
                env_fn_params = self.get_env_fn_params(reconfig_env_idx)
                self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
            self.env.reset(id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)

        if self.use_async_vector_env:
            for _ in range(self.RESET_STEP):
                zero_action = np.zeros((self.num_envs, 4))
                raw_obs, _reward, _, _, _ = self.env.step(zero_action)
        else:
            for _ in range(self.RESET_STEP):
                zero_action = np.zeros((len(env_idx), 4))
                self.env.step(zero_action, id=env_idx)
            all_actions = np.zeros((self.num_envs, 4))
            raw_obs, _reward, _, _, _ = self.env.step(all_actions)

        self._last_raw_obs = raw_obs
        self._terminal_rows[np.asarray(env_idx, dtype=np.int64)] = False
        obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True, active_mask=None):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, dtype=np.float32)

        if active_mask is None:
            active_mask = ~self._terminal_rows
        active_mask = np.asarray(active_mask, dtype=bool).reshape(self.num_envs)
        active_mask = active_mask & ~self._terminal_rows
        active_ids = np.flatnonzero(active_mask)

        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)

        if active_ids.size > 0:
            self._elapsed_steps[active_mask] += 1
            step_actions = actions[active_mask]

            if self.use_async_vector_env:
                full_actions = np.zeros((self.num_envs, 4), dtype=np.float32)
                full_actions[active_mask] = step_actions
                raw_obs, _reward, _, _, infos = self.env.step(full_actions)
            else:
                raw_obs_part, _reward, _, _, info_lists = self.env.step(
                    step_actions, id=active_ids
                )
                infos = list_of_dict_to_dict_of_list(info_lists)
                if self._last_raw_obs is None:
                    self._last_raw_obs = np.zeros(
                        (self.num_envs, raw_obs_part.shape[1]), dtype=np.float32
                    )
                raw_obs = self._last_raw_obs.copy()
                raw_obs[active_ids] = raw_obs_part
            self._last_raw_obs = raw_obs

            active_terminations = np.asarray(infos["success"], dtype=bool)
            terminations[active_mask] = active_terminations
            truncations = (self.elapsed_steps >= self.cfg.max_episode_steps) & active_mask
        else:
            raw_obs = self._last_raw_obs
            infos = {}

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(terminations, active_mask=active_mask)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
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
                extracted_obs, step_reward, terminations, truncations, infos = self.step(
                    chunk_actions[:, i], auto_reset=False, active_mask=active
                )
            else:
                extracted_obs = self._wrap_obs(self._last_raw_obs)
                step_reward = torch.zeros(self.num_envs, dtype=torch.float32)
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
                    done_tensor = torch.as_tensor(done_rows, dtype=torch.bool)
                    terminal_episode[key][done_tensor] = value[done_tensor]
                self._terminal_rows |= np.asarray(done_rows, dtype=bool)
                self._reset_metrics(np.asarray(done_rows, dtype=bool))

            obs_list.append(extracted_obs)
            infos_list.append(infos)
            chunk_rewards.append(step_reward)
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

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
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

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations, active_mask=None):
        terminations = np.asarray(terminations, dtype=bool)
        if active_mask is None:
            active_mask = np.ones(self.num_envs, dtype=bool)
        active_mask = np.asarray(active_mask, dtype=bool).reshape(self.num_envs)
        step_reward = np.zeros(self.num_envs, dtype=np.float32)
        reward = self.cfg.reward_coef * terminations.astype(np.float32)
        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            step_reward[active_mask] = reward_diff[active_mask]
            self.prev_step_reward[active_mask] = reward[active_mask]
            return step_reward
        step_reward[active_mask] = reward[active_mask]
        return step_reward
