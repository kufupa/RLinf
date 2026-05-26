"""LeRobot MetaWorld vector adapter used by the SmolVLA MetaWorld env."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np

os.environ.setdefault("MUJOCO_GL", "osmesa")
os.environ.setdefault("PYOPENGL_PLATFORM", "osmesa")


class DeferredLeRobotMetaworldEnv(gym.Env):
    """Async-worker-safe LeRobot MetaWorld env construction."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 80}

    def __init__(
        self,
        *,
        task: str,
        obs_type: str = "pixels_agent_pos",
        render_mode: str = "rgb_array",
        camera_name: str = "corner2",
        observation_width: int = 480,
        observation_height: int = 480,
        reset_randomization_mode: str | None = None,
    ) -> None:
        from gymnasium import spaces
        from lerobot.envs import metaworld as lr_mw

        self.task = str(task).replace("metaworld-", "")
        self.obs_type = str(obs_type)
        self.render_mode = str(render_mode)
        self.camera_name = str(camera_name)
        self.observation_width = int(observation_width)
        self.observation_height = int(observation_height)
        self.reset_randomization_mode = str(
            reset_randomization_mode
            or os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "random_seeded")
        )
        if self.reset_randomization_mode not in {
            "fixed",
            "random_seeded",
            "random_unseeded",
            "lerobot_default",
        }:
            raise ValueError(
                "reset_randomization_mode must be fixed, random_seeded, "
                "random_unseeded, or lerobot_default"
            )
        self._env: Any | None = None
        self._last_raw_obs: np.ndarray | None = None
        self._max_episode_steps = 500
        self.task_description = lr_mw.TASK_DESCRIPTIONS[self.task]
        self.expert_policy = lr_mw.TASK_POLICY_MAPPING[self.task]()

        if self.obs_type == "pixels":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    )
                }
            )
        elif self.obs_type == "pixels_agent_pos":
            self.observation_space = spaces.Dict(
                {
                    "pixels": spaces.Box(
                        low=0,
                        high=255,
                        shape=(self.observation_height, self.observation_width, 3),
                        dtype=np.uint8,
                    ),
                    "agent_pos": spaces.Box(
                        low=-1000.0, high=1000.0, shape=(4,), dtype=np.float64
                    ),
                }
            )
        else:
            raise ValueError(f"Unsupported obs_type: {self.obs_type}")
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def _ensure_env(self) -> None:
        if self._env is not None:
            return
        import metaworld

        mt1 = metaworld.MT1(self.task, seed=42)
        env = mt1.train_classes[self.task](
            render_mode="rgb_array", camera_name=self.camera_name
        )
        env.set_task(mt1.train_tasks[0])
        if self.camera_name == "corner2":
            env.model.cam_pos[2] = [0.75, 0.075, 0.7]
        env._freeze_rand_vec = self.reset_randomization_mode == "fixed"
        if self.reset_randomization_mode == "lerobot_default":
            env._freeze_rand_vec = False
        env.reset()
        self._env = env
        self._sync_metaworld_reset_sites()
        self._max_episode_steps = int(env.max_path_length)

    def _sync_metaworld_reset_sites(self) -> None:
        """Flush MetaWorld reset-time model edits into MuJoCo data before rendering."""
        env = self._env
        if env is None:
            return

        target_site_config = getattr(env, "_target_site_config", ())
        set_pos_site = getattr(env, "_set_pos_site", None)
        if callable(set_pos_site):
            for site_args in target_site_config:
                set_pos_site(*site_args)

        import mujoco

        mujoco.mj_forward(env.model, env.data)

    def render(self) -> np.ndarray:
        self._ensure_env()
        raw_image = np.asarray(self._env.render())
        if self.camera_name == "corner2":
            return np.flip(raw_image, (0, 1))
        return raw_image

    def _format_raw_obs(self, raw_obs: np.ndarray) -> dict[str, Any]:
        image = self.render()
        if self.obs_type == "pixels":
            return {"pixels": image.copy()}
        if self.obs_type == "pixels_agent_pos":
            return {"pixels": image.copy(), "agent_pos": raw_obs[:4]}
        raise ValueError(f"Unsupported obs_type: {self.obs_type}")

    def reset(
        self, seed: int | None = None, **kwargs: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        self._ensure_env()
        super().reset(seed=seed)
        self._env._freeze_rand_vec = self.reset_randomization_mode == "fixed"
        if self.reset_randomization_mode == "lerobot_default":
            self._env._freeze_rand_vec = False
        if self.reset_randomization_mode == "random_seeded" and seed is not None:
            self._env.seed(int(seed))
            if hasattr(self._env, "seeded_rand_vec"):
                self._env.seeded_rand_vec = True
        elif hasattr(self._env, "seeded_rand_vec"):
            self._env.seeded_rand_vec = False
        raw_obs, _info = self._env.reset(seed=seed)
        self._sync_metaworld_reset_sites()
        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float64).copy()
        return self._format_raw_obs(raw_obs), {"is_success": False}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        self._ensure_env()
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim != 1:
            raise ValueError(f"Expected action shape (action_dim,), got {action_np.shape}")
        raw_obs, reward, done, truncated, info = self._env.step(action_np)
        self._last_raw_obs = np.asarray(raw_obs, dtype=np.float64).copy()
        is_success = bool(info.get("success", 0))
        terminated = bool(done or is_success)
        info.update({"task": self.task, "done": bool(done), "is_success": is_success})
        observation = self._format_raw_obs(raw_obs)
        if terminated:
            info["final_info"] = {
                "task": self.task,
                "done": bool(done),
                "is_success": is_success,
            }
        return observation, float(reward), terminated, bool(truncated), info

    def expert_action(self) -> np.ndarray:
        if self._last_raw_obs is None:
            raise RuntimeError("expert_action called before reset")
        return np.asarray(self.expert_policy.get_action(self._last_raw_obs), dtype=np.float32)

    def last_agent_pos(self) -> np.ndarray:
        if self._last_raw_obs is None:
            raise RuntimeError("last_agent_pos called before reset")
        return np.asarray(self._last_raw_obs[:4], dtype=np.float32)

    def last_raw_obs(self) -> np.ndarray:
        if self._last_raw_obs is None:
            raise RuntimeError("last_raw_obs called before reset")
        return np.asarray(self._last_raw_obs, dtype=np.float64).copy()

    def render_frame(self) -> np.ndarray:
        return self.render()

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None


def _coerce_success_value(value: Any) -> bool:
    success_arr = np.asarray(value).reshape(-1)
    return bool(success_arr[0]) if success_arr.size else False


def _first_success(value: Any) -> bool | None:
    if isinstance(value, dict):
        if "is_success" in value:
            return _coerce_success_value(value["is_success"])
        if "success" in value:
            return _coerce_success_value(value["success"])
        for item in value.values():
            found = _first_success(item)
            if found is not None:
                return found
        return None
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            for item in value.reshape(-1):
                found = _first_success(item)
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)):
        for item in value:
            found = _first_success(item)
            if found is not None:
                return found
    return None


def _successes_from_vector_info(info: dict[str, Any], n_envs: int) -> np.ndarray:
    """Per-env success flags from vector-env info, including final_info."""
    out = np.zeros((n_envs,), dtype=np.bool_)
    fin = info.get("final_info") if isinstance(info, dict) else None
    if fin is not None:
        if isinstance(fin, np.ndarray) and fin.dtype == object and fin.shape[0] == n_envs:
            for i in range(n_envs):
                out[i] = bool(_first_success(fin[i]) or False)
            return out
        if isinstance(fin, (list, tuple)) and len(fin) == n_envs:
            for i, item in enumerate(fin):
                out[i] = bool(_first_success(item) or False)
            return out
        if isinstance(fin, dict):
            stacked = any(
                isinstance(v, np.ndarray) and v.ndim >= 1 and int(v.shape[0]) == n_envs
                for v in fin.values()
            )
            if stacked:
                for i in range(n_envs):
                    part = {
                        k: (
                            v[i]
                            if isinstance(v, np.ndarray)
                            and v.ndim >= 1
                            and int(v.shape[0]) == n_envs
                            else v
                        )
                        for k, v in fin.items()
                    }
                    out[i] = bool(_first_success(part) or False)
                return out
            out[:] = bool(_first_success(fin) or False)
            return out
    for i in range(n_envs):
        chunk: dict[str, Any] = {}
        for key, value in info.items():
            if key == "final_info":
                continue
            if isinstance(value, np.ndarray) and value.shape[0] == n_envs:
                chunk[key] = value[i]
            elif isinstance(value, (list, tuple)) and len(value) == n_envs:
                chunk[key] = value[i]
        out[i] = bool(_first_success(chunk) or False)
    return out


def _restore_vector_final_obs(
    obs: dict[str, Any],
    info: dict[str, Any],
    terminal: np.ndarray,
    n_envs: int,
) -> dict[str, Any]:
    """Replace SAME_STEP autoreset observations with terminal observations if present."""
    final_obs = info.get("final_obs") if isinstance(info, dict) else None
    if final_obs is None or not isinstance(obs, dict):
        return obs

    out = {
        key: np.array(value, copy=True) if isinstance(value, np.ndarray) else value
        for key, value in obs.items()
    }
    for i in range(int(n_envs)):
        if not bool(terminal[i]):
            continue
        if isinstance(final_obs, dict):
            item = {
                key: (
                    value[i]
                    if isinstance(value, np.ndarray) and value.shape[0] == n_envs
                    else value
                )
                for key, value in final_obs.items()
            }
        else:
            item = final_obs[i]
        if not isinstance(item, dict):
            continue
        for key, value in item.items():
            if key in out and isinstance(out[key], np.ndarray):
                out[key][i] = value
    return out


@dataclass
class OfficialStep:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, Any]


@dataclass
class OfficialBatchStep:
    observation: dict[str, Any]
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    success: np.ndarray
    info: dict[str, Any]


class LazyForkserverAsyncVectorEnv:
    """Defer AsyncVectorEnv worker spawn until first use."""

    def __init__(
        self,
        env_fns: Sequence[Callable[[], Any]],
        *,
        mp_context: str = "forkserver",
        observation_space: Any | None = None,
        action_space: Any | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._env_fns = list(env_fns)
        self._mp_context = str(mp_context)
        self._impl: Any | None = None
        if observation_space is not None and action_space is not None and metadata is not None:
            self.observation_space = observation_space
            self.action_space = action_space
            self.metadata = metadata
        else:
            tmp = self._env_fns[0]()
            self.observation_space = tmp.observation_space
            self.action_space = tmp.action_space
            self.metadata = tmp.metadata
            tmp.close()
        self.single_observation_space = self.observation_space
        self.single_action_space = self.action_space

    @property
    def num_envs(self) -> int:
        return len(self._env_fns)

    def _ensure(self) -> None:
        if self._impl is not None:
            return
        from gymnasium.vector import AsyncVectorEnv
        from gymnasium.vector.vector_env import AutoresetMode

        ctx = os.environ.get("SMOLVLA_GRPO_ASYNC_MP_CONTEXT", self._mp_context)
        self._impl = AsyncVectorEnv(
            self._env_fns,
            context=ctx,
            shared_memory=True,
            autoreset_mode=AutoresetMode.SAME_STEP,
        )

    def reset(
        self, *, seed: Any = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        self._ensure()
        return self._impl.reset(seed=seed, options=options)

    def step(self, actions: Any) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
        self._ensure()
        return self._impl.step(actions)

    def close(self) -> None:
        if self._impl is not None:
            self._impl.close()
            self._impl = None

    def call(self, name: str, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        self._ensure()
        return self._impl.call(name, *args, **kwargs)

    def get_attr(self, name: str) -> tuple[Any, ...]:
        self._ensure()
        return self._impl.get_attr(name)

    @property
    def unwrapped(self) -> "LazyForkserverAsyncVectorEnv":
        return self

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        self._ensure()
        return getattr(self._impl, name)


class OfficialLeRobotMetaWorldGRPORollout:
    """GRPO rollout adapter matching LeRobot's official vector eval path."""

    def __init__(
        self,
        *,
        task: str,
        obs_type: str = "pixels_agent_pos",
        n_envs: int = 1,
        use_async_envs: bool = False,
        async_start_method: str = "forkserver",
        enable_expert_oracle: bool = False,
        reset_randomization_mode: str | None = None,
    ) -> None:
        from lerobot.envs.configs import MetaworldEnv

        self.task_group = task
        self.task_id = 0
        self.n_envs = int(n_envs)
        if self.n_envs < 1:
            raise ValueError("n_envs must be >= 1")
        self.use_async_envs = bool(use_async_envs)
        self.async_start_method = str(async_start_method)
        self.enable_expert_oracle = bool(enable_expert_oracle)
        self.reset_randomization_mode = str(
            reset_randomization_mode
            or os.environ.get("SMOLVLA_METAWORLD_RESET_MODE", "random_seeded")
        )
        if self.reset_randomization_mode not in {
            "fixed",
            "random_seeded",
            "random_unseeded",
            "lerobot_default",
        }:
            raise ValueError(
                "reset_randomization_mode must be fixed, random_seeded, "
                "random_unseeded, or lerobot_default"
            )

        self.env_cfg = MetaworldEnv(task=task, obs_type=obs_type)
        if self.reset_randomization_mode == "lerobot_default" and not self.enable_expert_oracle:
            from lerobot.envs.factory import make_env

            envs = make_env(
                self.env_cfg,
                n_envs=self.n_envs,
                use_async_envs=bool(self.use_async_envs),
                trust_remote_code=False,
            )
        else:
            gym_kwargs = dict(self.env_cfg.gym_kwargs)
            mode_for_deferred = (
                "random_unseeded"
                if self.reset_randomization_mode == "lerobot_default"
                else self.reset_randomization_mode
            )
            gym_kwargs["reset_randomization_mode"] = mode_for_deferred

        if self.enable_expert_oracle:
            if self.n_envs != 1:
                raise ValueError("enable_expert_oracle requires n_envs=1")
            from gymnasium.vector import SyncVectorEnv

            envs = {
                task: {
                    0: SyncVectorEnv(
                        [
                            lambda tn=task, kwargs=gym_kwargs: DeferredLeRobotMetaworldEnv(
                                task=tn, **kwargs
                            )
                        ]
                    )
                }
            }
        elif self.reset_randomization_mode != "lerobot_default" and self.use_async_envs:
            mp_ctx = self.async_start_method
            cached_obs_space: Any | None = None
            cached_act_space: Any | None = None
            cached_metadata: dict[str, Any] | None = None

            def _env_cls(fns: Sequence[Callable[[], Any]]) -> LazyForkserverAsyncVectorEnv:
                nonlocal cached_obs_space, cached_act_space, cached_metadata
                lazy = LazyForkserverAsyncVectorEnv(
                    fns,
                    mp_context=mp_ctx,
                    observation_space=cached_obs_space,
                    action_space=cached_act_space,
                    metadata=cached_metadata,
                )
                if cached_obs_space is None:
                    cached_obs_space = lazy.observation_space
                    cached_act_space = lazy.action_space
                    cached_metadata = lazy.metadata
                return lazy

            fns = [
                (
                    lambda tn=task, kwargs=gym_kwargs: DeferredLeRobotMetaworldEnv(
                        task=tn, **kwargs
                    )
                )
                for _ in range(self.n_envs)
            ]
            envs = {task: {0: _env_cls(fns)}}
        elif self.reset_randomization_mode != "lerobot_default":
            from gymnasium.vector import SyncVectorEnv

            envs = {
                task: {
                    0: SyncVectorEnv(
                        [
                            (
                                lambda tn=task, kwargs=gym_kwargs: DeferredLeRobotMetaworldEnv(
                                    task=tn, **kwargs
                                )
                            )
                            for _ in range(self.n_envs)
                        ]
                    )
                }
            }

        try:
            self.vec_env = envs[task][0]
        except KeyError as exc:
            available = ", ".join(str(k) for k in envs)
            raise KeyError(
                f"official_lerobot task {task!r} not found; available tasks: {available}"
            ) from exc

    @property
    def inner(self) -> Any:
        return self.vec_env

    @property
    def max_episode_steps(self) -> int:
        return int(self.vec_env.call("_max_episode_steps")[0])

    @property
    def action_dim(self) -> int:
        if hasattr(self.vec_env, "single_action_space"):
            shape = self.vec_env.single_action_space.shape
        else:
            shape = self.vec_env.action_space.shape
            if len(shape) > 1 and int(shape[0]) == self.n_envs:
                shape = shape[1:]
            elif len(shape) > 1 and int(shape[0]) == 1:
                shape = shape[1:]
        return int(np.prod(shape))

    def reset(self, reset_seed: int) -> dict[str, Any]:
        seeds = [int(reset_seed)] * self.n_envs
        obs, _info = self.vec_env.reset(seed=seeds)
        return obs

    def reset_many(self, reset_seeds: Sequence[int]) -> dict[str, Any]:
        seeds = [int(seed) for seed in reset_seeds]
        if len(seeds) != self.n_envs:
            raise ValueError(f"reset_many expected {self.n_envs} seeds; got {len(seeds)}")
        obs, _info = self.vec_env.reset(seed=seeds)
        return obs

    def _task_batch(self) -> list[str]:
        try:
            task_result = self.vec_env.call("task_description")
        except Exception:
            task_result = self.vec_env.call("task")
        if isinstance(task_result, tuple):
            task_result = list(task_result)
        if not isinstance(task_result, list):
            raise TypeError(f"Expected task call to return list/tuple, got {type(task_result)}")
        if not all(isinstance(item, str) for item in task_result):
            raise TypeError("All task items must be strings")
        return task_result

    def build_proc(self, observation: dict[str, Any], *, bundle: Any) -> Any:
        from lerobot.envs.utils import preprocess_observation

        obs = preprocess_observation(observation)
        obs["task"] = self._task_batch()
        return bundle.preprocessor(obs)

    def step_batch(self, action_batch: np.ndarray) -> OfficialBatchStep:
        action_np = np.asarray(action_batch, dtype=np.float32)
        expected = (self.n_envs, self.action_dim)
        if action_np.shape != expected:
            raise ValueError(
                f"official_lerobot vector action must have shape {expected}; got {action_np.shape}"
            )

        obs, reward, terminated, truncated, info = self.vec_env.step(action_np)
        info_d = info if isinstance(info, dict) else {}
        success = _successes_from_vector_info(info_d, self.n_envs)
        terminated_np = np.asarray(terminated, dtype=np.bool_).reshape(self.n_envs)
        truncated_np = np.asarray(truncated, dtype=np.bool_).reshape(self.n_envs)
        success_np = success.reshape(self.n_envs)
        terminal = np.logical_or(np.logical_or(terminated_np, truncated_np), success_np)
        obs = _restore_vector_final_obs(obs, info_d, terminal, self.n_envs)

        return OfficialBatchStep(
            observation=obs,
            reward=np.asarray(reward, dtype=np.float64).reshape(self.n_envs),
            terminated=terminated_np,
            truncated=truncated_np,
            success=success_np,
            info=info_d,
        )

    def step(self, action_batch: np.ndarray) -> OfficialStep:
        if self.n_envs != 1:
            raise ValueError("use step_batch when n_envs > 1")
        b = self.step_batch(action_batch)
        return OfficialStep(
            observation=b.observation,
            reward=float(b.reward[0]),
            terminated=bool(b.terminated[0]),
            truncated=bool(b.truncated[0]),
            success=bool(b.success[0]),
            info=b.info,
        )

    def close(self) -> None:
        self.vec_env.close()

    def expert_action(self) -> np.ndarray:
        return np.asarray(self.vec_env.call("expert_action")[0], dtype=np.float32)

    def last_agent_pos(self) -> np.ndarray:
        return np.asarray(self.vec_env.call("last_agent_pos")[0], dtype=np.float32)

    def last_raw_obs(self) -> np.ndarray:
        return np.asarray(self.vec_env.call("last_raw_obs")[0], dtype=np.float64)

    def render_frame(self) -> np.ndarray:
        return np.asarray(self.vec_env.call("render_frame")[0])


def resolve_lerobot_horizon(
    env: OfficialLeRobotMetaWorldGRPORollout,
    requested_max_steps: int | None,
) -> int:
    if requested_max_steps is None or int(requested_max_steps) <= 0:
        return env.max_episode_steps
    return int(requested_max_steps)
