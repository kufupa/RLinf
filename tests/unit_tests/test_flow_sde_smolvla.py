# Copyright 2025 The RLinf Authors.
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

"""Unit tests for the stochastic-flow (SDE) chunk log-probability helpers."""

import math

import torch


def test_sde_step_logprob_matches_gaussian_closed_form():
    from rlinf.algorithms.flow_sde import sde_step_logprob

    mean = torch.zeros(2, 4)
    std = torch.full((2, 4), 0.5)
    sample = torch.full((2, 4), 0.1)
    lp = sde_step_logprob(sample, mean, std)  # summed over action dim -> [2]
    var = 0.25
    expect = -0.5 * ((0.1**2) / var + math.log(2 * math.pi * var)) * 4
    assert lp.shape == (2,)
    assert torch.allclose(lp, torch.full((2,), expect), atol=1e-5)


def test_sde_step_params_euler_mean_std():
    from rlinf.algorithms.flow_sde import sde_step_params

    x_t = torch.zeros(2, 4)
    velocity = torch.ones(2, 4)
    mean, std = sde_step_params(x_t, velocity, dt=0.5, noise_level=1.0)
    assert torch.allclose(mean, torch.full((2, 4), 0.5))
    assert torch.allclose(std, torch.full((2, 4), math.sqrt(0.5)))


def test_sde_step_logprob_zero_std_is_finite():
    """std==0 (deterministic ODE limit) must not produce NaN/inf."""
    from rlinf.algorithms.flow_sde import sde_step_logprob

    lp = sde_step_logprob(torch.zeros(1, 4), torch.zeros(1, 4), torch.zeros(1, 4))
    assert torch.isfinite(lp).all()
    assert torch.allclose(lp, torch.zeros(1))
