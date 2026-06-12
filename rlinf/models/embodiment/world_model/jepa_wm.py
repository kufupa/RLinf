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

"""JEPA world-model reward adapter (Phase12 semantics via project monolith)."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def build_jepa_wm_bundle(cfg: Mapping[str, Any] | Any) -> Any:
    """Load frozen JEPA-WM bundle from Phase12 ``load_wm_bundle``."""
    jepa_repo = str(cfg.get("jepa_repo", "") or "")
    jepa_ckpt = str(cfg.get("jepa_ckpt", ""))
    device = str(cfg.get("wm_device", "cuda"))
    if not jepa_repo or not jepa_ckpt:
        raise ValueError("wm_latent_progress requires jepa_repo and jepa_ckpt in env cfg")

    import os

    os.environ.setdefault("JEPA_WM_DISABLE_IMAGE_HEAD", "1")
    from segment_grpo_loop import load_wm_bundle

    bundle = load_wm_bundle(jepa_repo, jepa_ckpt, device, required=True)
    bundle.model.eval()
    for param in bundle.model.parameters():
        param.requires_grad_(False)
    return bundle


def score_wm_latent_progress(
    wm_bundle: Any,
    *,
    image: np.ndarray,
    proprio: np.ndarray,
    chunk_actions: np.ndarray,
    goal: Mapping[str, Any],
    candidate_index: int = 0,
    proprio_alpha: float = 0.1,
    mode: str = "visual_proprio",
) -> float:
    """Return WM latent-progress reward (Phase12 ``wm_latent_progress`` key)."""
    from smolvla_grpo.phase12_wm_reward import score_phase12_chunk_with_wm

    score = score_phase12_chunk_with_wm(
        wm_bundle=wm_bundle,
        image=np.asarray(image),
        proprio=np.asarray(proprio, dtype=np.float32),
        chunk_actions=np.asarray(chunk_actions, dtype=np.float32),
        goal=goal,
        candidate_index=int(candidate_index),
        proprio_alpha=float(proprio_alpha),
        mode=str(mode),
    )
    return float(score.wm_latent_progress)
