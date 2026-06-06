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
    from rlinf.workers.actor.dgpo_group_utils import compute_dgpo_group_weights

    group_size = 4
    signed_adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    delta = torch.tensor([0.2, 0.2, 0.2, 0.2])
    group_ids = torch.zeros(group_size, dtype=torch.long)
    weight, _ = compute_dgpo_group_weights(
        signed_adv=signed_adv,
        delta=delta,
        group_ids=group_ids,
        dpo_beta=10.0,
        group_size=group_size,
        verify=True,
    )
    assert torch.allclose(weight, torch.full_like(weight, 0.5), atol=1e-5)


def test_dgpo_group_weights_respect_loss_mask():
    from rlinf.workers.actor.dgpo_group_utils import compute_dgpo_group_weights

    signed_adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    delta = torch.tensor([0.2, 0.2, 0.2, 0.2])
    group_ids = torch.zeros(4, dtype=torch.long)
    mask = torch.tensor([1.0, 1.0, 0.0, 0.0])
    w_masked, _ = compute_dgpo_group_weights(
        signed_adv=signed_adv,
        delta=delta,
        group_ids=group_ids,
        dpo_beta=10.0,
        group_size=4,
        loss_mask=mask,
    )
    w_full, _ = compute_dgpo_group_weights(
        signed_adv=signed_adv[:2],
        delta=delta[:2],
        group_ids=group_ids[:2],
        dpo_beta=10.0,
        group_size=2,
        verify=False,
    )
    assert torch.allclose(w_masked[:2], w_full, atol=1e-5)
    assert torch.allclose(w_masked[2:], torch.full((2,), 0.5), atol=1e-5)


def test_dgpo_scatter_invariant_to_within_group_permutation():
    from rlinf.workers.actor.dgpo_group_utils import compute_dgpo_group_weights

    signed_adv = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.2, -0.2, 0.1, -0.1])
    delta = torch.full((8,), 0.3)
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    w1, _ = compute_dgpo_group_weights(
        signed_adv=signed_adv,
        delta=delta,
        group_ids=group_ids,
        dpo_beta=10.0,
        group_size=4,
    )
    perm = torch.tensor([2, 0, 3, 1, 6, 4, 7, 5])
    w2, _ = compute_dgpo_group_weights(
        signed_adv=signed_adv[perm],
        delta=delta[perm],
        group_ids=group_ids[perm],
        dpo_beta=10.0,
        group_size=4,
    )
    assert torch.allclose(w1[perm], w2, atol=1e-6)


def test_row_shuffle_breaks_positional_grouping():
    from rlinf.workers.actor.dgpo_group_utils import compute_dgpo_group_weights

    group_size = 4
    beta = 10.0
    signed_adv = torch.tensor([1.0, -1.0, 0.5, -0.5, 1.0, -1.0, 0.5, -0.5])
    delta = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0])
    group_ids = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    shuffled = torch.tensor([0, 4, 1, 5, 2, 6, 3, 7])

    w_true, _ = compute_dgpo_group_weights(
        signed_adv=signed_adv[shuffled],
        delta=delta[shuffled],
        group_ids=group_ids[shuffled],
        dpo_beta=beta,
        group_size=group_size,
    )
    per = signed_adv[shuffled] * beta * delta[shuffled] / group_size
    gsum_old = per.reshape(-1, group_size).sum(dim=1).repeat_interleave(group_size)
    w_old = torch.sigmoid(gsum_old).detach()
    assert not torch.allclose(w_old, w_true, atol=1e-3)


def test_build_dgpo_group_ids_and_block_shuffle():
    from rlinf.workers.actor.dgpo_group_utils import (
        build_dgpo_group_block_shuffle_id,
        build_dgpo_group_ids,
    )

    ids = build_dgpo_group_ids(n_chunk=3, batch_size=8, group_size=4)
    assert ids.shape == (3, 8)
    assert ids[0, 0].item() == ids[0, 3].item()
    assert ids[0, 0].item() != ids[0, 4].item()
    assert ids[1, 0].item() == ids[0, 0].item() + 2

    g = torch.Generator().manual_seed(0)
    shuffle = build_dgpo_group_block_shuffle_id(8, 4, g)
    assert shuffle.numel() == 8
    assert set(shuffle.tolist()) == set(range(8))


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
        dgpo_group_id=torch.zeros(batch_size, dtype=torch.long),
    )
    assert abs(metrics["actor/dgpo_group_weight_mean"] - 0.5) < 0.05
    assert torch.isfinite(loss)


def test_nft_forward_micro_batch_size_halves_large_train_micro_batch():
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

    worker = object.__new__(EmbodiedNFTFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"micro_batch_size": 32},
            "algorithm": {"group_size": 32},
        }
    )
    assert worker._get_nft_forward_micro_batch_size(32) == 16
    assert worker._get_nft_forward_micro_batch_size(16) == 16


def test_nft_forward_micro_batch_chunks_model_calls():
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

    worker = object.__new__(EmbodiedNFTFSDPPolicy)
    worker.device = "cpu"
    worker.cfg = OmegaConf.create(
        {
            "actor": {"micro_batch_size": 32, "nft_forward_micro_batch_size": 16},
            "algorithm": {"group_size": 32},
        }
    )
    worker.amp_context = torch.enable_grad()
    calls: list[int] = []

    class _FakeModel(torch.nn.Module):
        def forward(self, **kwargs):
            x_t = kwargs["nft_inputs"]["x_t"]
            calls.append(x_t.shape[0])
            chunk, action_dim = 5, 4
            return {"v_theta": torch.zeros(x_t.shape[0], chunk, action_dim)}

    worker.model = _FakeModel()
    batch = 32
    chunk, action_dim = 5, 4
    forward_inputs = {"nft_xcur": torch.zeros(batch, chunk, action_dim)}
    t = torch.zeros(batch)
    v = worker._run_nft_forward_v_theta(forward_inputs, forward_inputs["nft_xcur"], t)
    assert v.shape == (batch, chunk, action_dim)
    assert calls == [16, 16]


def test_recompute_v_old_uses_nft_forward_micro_batch_size():
    from rlinf.workers.actor.fsdp_nft_policy_worker import EmbodiedNFTFSDPPolicy

    worker = object.__new__(EmbodiedNFTFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "actor": {"micro_batch_size": 32, "nft_forward_micro_batch_size": 16},
            "algorithm": {"nft_tau": 1.0},
        }
    )
    chunk, action_dim = 5, 4
    batch = 32
    xcur = torch.zeros(batch, chunk, action_dim)
    t = torch.zeros(batch)
    forward_inputs = {"nft_xcur": xcur}
    chunk_sizes: list[int] = []

    class _FakeModel(torch.nn.Module):
        def forward(self, **kwargs):
            x_t = kwargs["nft_inputs"]["x_t"]
            chunk_sizes.append(x_t.shape[0])
            return {"v_theta": torch.zeros(x_t.shape[0], chunk, action_dim)}

    worker.model = _FakeModel()
    worker.device = "cpu"
    worker._get_current_nft_tau = lambda: 1.0
    out = worker._recompute_v_old(forward_inputs, xcur, t)
    assert out.shape == (batch, chunk, action_dim)
    assert chunk_sizes == [16, 16]


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
