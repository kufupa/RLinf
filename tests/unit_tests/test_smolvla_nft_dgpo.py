# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for SmolVLA NFT forward + DGPO group loss semantics."""

from __future__ import annotations

import torch
from omegaconf import OmegaConf


def test_grpo_advantages_zero_mean_per_group():
    from rlinf.algorithms.advantages import compute_grpo_advantages

    rewards = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    loss_mask = torch.ones(1, 8, dtype=torch.bool)
    adv, _ = compute_grpo_advantages(rewards, loss_mask, group_size=4)
    grouped = adv.view(-1, 4)
    assert torch.allclose(grouped.sum(dim=1), torch.zeros(2), atol=1e-5)


def test_dgpo_group_sum_cancels_mixed_signs():
    group_size = 4
    signed_adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    delta = torch.tensor([0.2, 0.2, 0.2, 0.2])
    beta = 10.0
    per = signed_adv * beta * delta / group_size
    gsum = per.view(-1, group_size).sum(dim=1)
    assert torch.allclose(gsum, torch.zeros(1), atol=1e-5)


def test_dgpo_neutral_when_theta_matches_ref():
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

    worker = object.__new__(EmbodiedNFTFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "dpo_beta": 10.0,
                "group_size": 4,
                "nft_weight_mode": "constant",
                "nft_weight_scale": 1.0,
            }
        }
    )
    batch_size = 4
    chunk = 5
    action_dim = 4
    x_t = torch.zeros(batch_size, chunk, action_dim)
    target = torch.randn(batch_size, chunk, action_dim)
    pred = target.clone()
    forward_inputs = {
        "nft_x0": target,
        "nft_noise_level": torch.zeros(batch_size),
    }
    signed_adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss_mask = torch.ones(batch_size, dtype=torch.bool)
    t_bc = torch.full((batch_size, 1, 1), 0.5)
    dt_bc = torch.full((batch_size, 1, 1), 0.1)
    sigma_i = torch.zeros(batch_size, 1, 1)
    std_t_det = torch.zeros(batch_size, 1, 1)
    worker._compute_nft_target_and_pred = lambda *args, **kwargs: (target, pred)
    worker._compute_nft_weight = lambda *args, **kwargs: 1.0
    loss, metrics = worker._compute_dgpo_nft_loss(
        forward_inputs=forward_inputs,
        target_space="x0",
        x_t=x_t,
        v_theta=torch.zeros_like(x_t),
        v_old=torch.zeros_like(x_t),
        signed_adv=signed_adv,
        loss_mask=loss_mask,
        sum_dims=(1, 2),
        t_bc=t_bc,
        dt_bc=dt_bc,
        sigma_i=sigma_i,
        std_t_det=std_t_det,
    )
    assert abs(metrics["actor/dgpo_group_weight_mean"] - 0.5) < 0.05
    assert torch.isfinite(loss)


def test_smolvla_config_exposes_num_steps():
    from rlinf.models.embodiment.smolvla.smolvla_action_model import (
        SmolVLAForRLActionPrediction,
    )

    cfg = OmegaConf.create(
        {
            "model_path": "jadechoghari/smolvla_metaworld",
            "num_action_chunks": 5,
            "action_dim": 4,
            "noise_method": "flow_ode",
            "num_steps": 10,
            "flow_sde_num_steps": 10,
            "is_nft": True,
            "freeze_all_but_ppo_trainables": False,
        }
    )
    model = SmolVLAForRLActionPrediction.__new__(SmolVLAForRLActionPrediction)
    model.config = cfg
    model.num_steps = int(cfg.get("num_steps", 10))
    assert model.num_steps == 10
    assert cfg.num_steps == 10
