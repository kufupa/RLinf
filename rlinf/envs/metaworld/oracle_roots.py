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

"""Oracle teacher-forced roots via runtime expert rollout (Phase12 parity)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class OracleRootState:
    """One teacher-forced segment root + goal latent."""

    seed: int
    frame_index_1based: int
    policy_image: np.ndarray
    wm_image: np.ndarray
    proprio: np.ndarray
    goal_latent: dict[str, Any]


def build_subgoal_schedule(
    *,
    max_frame_1based: int,
    chunk_len: int,
    success_frame_1based: int | None = None,
):
    """Re-export Phase12 subgoal schedule (pure helper)."""
    from smolvla_grpo.phase12_goals import build_subgoal_schedule as _build

    return _build(
        max_frame_1based=max_frame_1based,
        chunk_len=chunk_len,
        success_frame_1based=success_frame_1based,
    )


def _rollout_oracle(
    env_h: Any,
    *,
    seed: int,
    max_steps: int,
    cache_dir: Path,
) -> dict[str, Any]:
    """Expert oracle rollout (mirrors Phase12 ``_rollout_phase12_oracle``)."""
    from smolvla_grpo.phase12_pixels import (
        policy_rgb_from_obs,
        wm_rgb_from_policy_rgb_corner2,
    )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    obs = env_h.reset(int(seed))
    policy_frame = policy_rgb_from_obs(obs)
    frames = [policy_frame]
    wm_frames = [wm_rgb_from_policy_rgb_corner2(policy_frame)]
    proprios = [np.asarray(env_h.last_agent_pos(), dtype=np.float32)]
    raw_obs = [np.asarray(env_h.last_raw_obs(), dtype=np.float64)]
    successes: list[bool] = []
    success_frame_1based: int | None = None
    for step_idx in range(int(max_steps)):
        action = np.clip(env_h.expert_action(), -1.0, 1.0).reshape(1, -1).astype(
            np.float32
        )
        step = env_h.step(action)
        successes.append(bool(step.success))
        policy_frame = policy_rgb_from_obs(step.observation)
        frames.append(policy_frame)
        wm_frames.append(wm_rgb_from_policy_rgb_corner2(policy_frame))
        proprios.append(np.asarray(env_h.last_agent_pos(), dtype=np.float32))
        raw_obs.append(np.asarray(env_h.last_raw_obs(), dtype=np.float64))
        if bool(step.success) and success_frame_1based is None:
            success_frame_1based = int(step_idx + 2)
        if bool(step.success or step.terminated or step.truncated):
            break
    manifest = {
        "seed": int(seed),
        "max_steps": int(max_steps),
        "frame_count": len(frames),
        "success_frame_1based": success_frame_1based,
    }
    (cache_dir / "oracle_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return {
        "frames": frames,
        "wm_frames": wm_frames,
        "proprios": proprios,
        "raw_obs": raw_obs,
        "success_frame_1based": success_frame_1based,
    }


class OracleTeacherForcedCatalog:
    """Lazy oracle cache: one expert rollout per seed, many segment roots."""

    def __init__(self, cfg: Mapping[str, Any] | Any, wm_bundle: Any):
        self.cfg = cfg
        self.wm_bundle = wm_bundle
        self.task_name = str(cfg.get("task_name", "push-v3"))
        self.chunk_len = int(cfg.get("num_action_chunks", 5))
        self.max_steps = int(cfg.get("oracle_max_steps", 120))
        self.goal_latent_mode = str(cfg.get("goal_latent_mode", "visual_proprio"))
        self.cache_root = Path(
            cfg.get(
                "oracle_cache_dir",
                "/vol/bitbucket/aa6622/wm_flowsde_runs/oracle_cache",
            )
        )
        self._roots_by_seed: dict[int, list[OracleRootState]] = {}
        self._segment_cursor: dict[int, int] = {}

    def next_root_for_group(self, seed: int) -> OracleRootState:
        roots = self._roots_for_seed(int(seed))
        cursor = self._segment_cursor.get(int(seed), 0)
        root = roots[cursor % len(roots)]
        self._segment_cursor[int(seed)] = cursor + 1
        return root

    def _roots_for_seed(self, seed: int) -> list[OracleRootState]:
        if seed in self._roots_by_seed:
            return self._roots_by_seed[seed]
        from rlinf.envs.metaworld.lerobot_adapter import (
            OfficialLeRobotMetaWorldGRPORollout,
        )
        from smolvla_grpo.phase12_goals import build_local_transition_schedule
        from smolvla_grpo.phase12_root_cache import build_oracle_root_entry
        from smolvla_grpo.phase12_wm_reward import _encode_structured

        cache_dir = self.cache_root / f"seed_{seed}"
        oracle_env = OfficialLeRobotMetaWorldGRPORollout(
            task=self.task_name,
            n_envs=1,
            enable_expert_oracle=True,
            reset_randomization_mode="random_seeded",
        )
        try:
            oracle = _rollout_oracle(
                oracle_env,
                seed=seed,
                max_steps=self.max_steps,
                cache_dir=cache_dir,
            )
            transitions = build_local_transition_schedule(
                max_frame_1based=len(oracle["frames"]),
                chunk_len=self.chunk_len,
                success_frame_1based=oracle.get("success_frame_1based"),
            )
            roots: list[OracleRootState] = []
            for transition in transitions:
                root_idx = min(int(transition.root_frame_1based), len(oracle["frames"]))
                goal_idx = min(int(transition.goal_frame_1based), len(oracle["frames"]))
                entry = build_oracle_root_entry(
                    env_h=oracle_env,
                    bundle=None,
                    policy_frame=oracle["frames"][root_idx - 1],
                    wm_frame=oracle["wm_frames"][root_idx - 1],
                    raw_obs=oracle["raw_obs"][root_idx - 1],
                    proprio=oracle["proprios"][root_idx - 1],
                    frame_index_1based=root_idx,
                )
                goal_encoded = _encode_structured(
                    self.wm_bundle,
                    oracle["wm_frames"][goal_idx - 1],
                    oracle["proprios"][goal_idx - 1],
                    mode=self.goal_latent_mode,
                )
                roots.append(
                    OracleRootState(
                        seed=int(seed),
                        frame_index_1based=int(root_idx),
                        policy_image=entry.policy_image,
                        wm_image=entry.wm_image,
                        proprio=entry.proprio,
                        goal_latent=goal_encoded,
                    )
                )
        finally:
            oracle_env.close()
        if not roots:
            raise RuntimeError(f"oracle teacher-forced produced zero roots for seed={seed}")
        self._roots_by_seed[seed] = roots
        return roots
