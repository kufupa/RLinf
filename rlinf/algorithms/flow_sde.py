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

"""Per-step log-probabilities for stochastic (SDE) flow-matching sampling.

A flow-matching policy denoises an action chunk over several Euler steps. When
the sampler injects Gaussian noise at each step (the "flow_sde" noise method),
the policy distribution factorises into a product of per-step Gaussian
transition kernels. These helpers score a single step so that both rollout
sampling and the actor-side log-probability recompute share identical math.

The helpers are intentionally free of any policy/model coupling: they take the
denoise state, the predicted transition mean/std, and return log-probabilities.
This mirrors the in-repo ``openpi`` and ``lingbotvla`` flow-SDE
implementations while keeping the math reusable and unit-testable on CPU.
"""

from __future__ import annotations

import math

import torch


def sde_step_logprob_per_dim(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Diagonal-Gaussian log-prob of one denoise step, per action dimension.

    Dimensions with ``std == 0`` (the deterministic ODE limit) contribute zero
    log-probability instead of ``-inf``/``NaN``, matching the masking used by
    the ``lingbotvla`` flow-SDE implementation.

    Args:
        sample: Realised next denoise state, shape ``(..., action_dim)``.
        mean: Transition-kernel mean, broadcastable to ``sample``.
        std: Transition-kernel standard deviation, broadcastable to ``sample``.

    Returns:
        Per-dimension log-probabilities with the same shape as ``sample``.
    """
    sample = sample.to(torch.float32)
    mean = mean.to(torch.float32)
    std = std.to(torch.float32)

    zero_std = std == 0
    std_safe = torch.where(zero_std, torch.ones_like(std), std)
    log_prob = -0.5 * (
        ((sample - mean) / std_safe) ** 2 + torch.log(2 * math.pi * std_safe**2)
    )
    return torch.where(zero_std, torch.zeros_like(log_prob), log_prob)


def sde_step_logprob(
    sample: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Per-step log-prob summed over the trailing action dimension.

    Args:
        sample: Realised next denoise state, shape ``(..., action_dim)``.
        mean: Transition-kernel mean, broadcastable to ``sample``.
        std: Transition-kernel standard deviation, broadcastable to ``sample``.

    Returns:
        Log-probabilities reduced over the last dimension, shape ``sample.shape[:-1]``.
    """
    return sde_step_logprob_per_dim(sample, mean, std).sum(dim=-1)


def sde_step_params(
    x_t: torch.Tensor,
    velocity: torch.Tensor,
    dt: float,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Euler-Maruyama transition mean/std for ``x_{t+dt}`` given the velocity field.

    The deterministic Euler update is ``x_t + velocity * dt``; the stochastic
    term adds isotropic Gaussian noise with standard deviation
    ``noise_level * sqrt(|dt|)``. Both the sampler and the log-prob recompute
    call this so the kernel they agree on is defined in exactly one place.

    Args:
        x_t: Current denoise state.
        velocity: Predicted flow velocity at ``x_t``.
        dt: Euler step size (the SmolVLA flow sampler uses a negative ``dt`` as
            it integrates time from 1 to 0; only the magnitude affects the std).
        noise_level: SDE diffusion scale. ``0`` recovers the deterministic ODE.

    Returns:
        Tuple ``(mean, std)`` broadcastable to ``x_t``.
    """
    mean = x_t + velocity * dt
    std = torch.full_like(mean, float(noise_level) * math.sqrt(abs(float(dt))))
    return mean, std
