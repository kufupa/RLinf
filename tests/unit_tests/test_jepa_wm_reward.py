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

"""Unit tests for JEPA WM reward adapter (mocked bundle)."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np


def test_score_wm_latent_progress_returns_scalar():
    fake_score = MagicMock()
    fake_score.wm_latent_progress = 0.42
    fake_wm_reward = MagicMock()
    fake_wm_reward.score_phase12_chunk_with_wm = MagicMock(return_value=fake_score)
    fake_smolvla_grpo = MagicMock()
    fake_smolvla_grpo.phase12_wm_reward = fake_wm_reward
    with patch.dict(
        sys.modules,
        {
            "smolvla_grpo": fake_smolvla_grpo,
            "smolvla_grpo.phase12_wm_reward": fake_wm_reward,
        },
    ):
        from rlinf.models.embodiment.world_model import jepa_wm

        reward = jepa_wm.score_wm_latent_progress(
            MagicMock(),
            image=np.zeros((64, 64, 3), dtype=np.uint8),
            proprio=np.zeros(4, dtype=np.float32),
            chunk_actions=np.zeros((5, 4), dtype=np.float32),
            goal={"visual": MagicMock()},
        )
    assert reward == 0.42


def test_build_jepa_wm_bundle_requires_paths():
    from rlinf.envs.metaworld.smolvla_metaworld_env import _load_jepa_wm_bundle
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"jepa_repo": "", "jepa_ckpt": ""})
    try:
        _load_jepa_wm_bundle(cfg)
        raise AssertionError("expected ValueError")
    except ValueError as exc:
        assert "jepa_repo" in str(exc)
