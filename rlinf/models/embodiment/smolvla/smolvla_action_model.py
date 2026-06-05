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

import json
import math
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from rlinf.algorithms.flow_sde import sde_step_logprob_per_dim, sde_step_params
from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.value_head import ValueHead


def _to_device(value: Any, device: torch.device, dtype: torch.dtype | None = None) -> Any:
    if isinstance(value, torch.Tensor):
        if dtype is not None and value.is_floating_point():
            return value.to(device=device, dtype=dtype)
        return value.to(device=device)
    if isinstance(value, dict):
        return {key: _to_device(item, device, dtype=dtype) for key, item in value.items()}
    return value


def _flatten_tensor_tree(
    value: Any, path: tuple[str, ...] = ()
) -> dict[str, torch.Tensor]:
    if isinstance(value, torch.Tensor):
        key = "smolvla_proc::" + json.dumps(list(path), separators=(",", ":"))
        return {key: value.detach().cpu().contiguous()}
    if isinstance(value, dict):
        out: dict[str, torch.Tensor] = {}
        for key, item in value.items():
            out.update(_flatten_tensor_tree(item, (*path, str(key))))
        return out
    return {}


def _unflatten_tensor_tree(flat: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    prefix = "smolvla_proc::"
    for key, value in flat.items():
        if not key.startswith(prefix):
            continue
        path = json.loads(key[len(prefix) :])
        cur = out
        for part in path[:-1]:
            cur = cur.setdefault(part, {})
        cur[path[-1]] = value
    return out


def _dtype_from_name(name: str | torch.dtype | None) -> torch.dtype | None:
    if isinstance(name, torch.dtype):
        return name
    if name in (None, "", "null", "none"):
        return None
    key = str(name).lower()
    if key in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if key in {"fp16", "float16", "half"}:
        return torch.float16
    if key in {"fp32", "float32", "float"}:
        return torch.float32
    raise ValueError(f"Unsupported SmolVLA dtype: {name}")


def _first_tensor_device(value: Any) -> torch.device | None:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, dict):
        for item in value.values():
            device = _first_tensor_device(item)
            if device is not None:
                return device
    return None


def _get_proc_value(proc: dict[str, Any], key: str) -> Any:
    if key in proc:
        return proc[key]
    cur: Any = proc
    for part in str(key).split("."):
        if not isinstance(cur, dict) or part not in cur:
            raise KeyError(key)
        cur = cur[part]
    return cur


class SmolVLAForRLActionPrediction(nn.Module, BasePolicy):
    """RLinf adapter for SmolVLA's Gaussian continuous-action hooks."""

    def __init__(self, cfg, torch_dtype=None, bundle=None):
        super().__init__()
        self.config = cfg
        self.action_dim = int(cfg.get("action_dim", 4))
        self.num_action_chunks = int(cfg.get("num_action_chunks", 5))
        self.action_low = float(cfg.get("action_low", -1.0))
        self.action_high = float(cfg.get("action_high", 1.0))
        self.detach_critic_input = bool(cfg.get("detach_critic_input", True))
        self.noise_method = str(cfg.get("noise_method", "gaussian"))
        self.flow_sde_num_steps = int(cfg.get("flow_sde_num_steps", 10))
        self.flow_sde_noise_level = float(cfg.get("flow_sde_noise_level", 1.0))
        self._bundle = bundle if bundle is not None else self._load_bundle(cfg)
        self.policy = self._bundle.policy
        self.preprocessor = self._bundle.preprocessor
        self.postprocessor = self._bundle.postprocessor
        self.obs_image_key = self._bundle.obs_image_key
        self.obs_state_key = self._bundle.obs_state_key
        self._last_reset_seeds: torch.Tensor | None = None

        if cfg.get("add_value_head", False):
            self.value_head = ValueHead(
                input_dim=int(cfg.get("state_dim", 4)),
                hidden_sizes=tuple(cfg.get("value_head_hidden_sizes", (128, 128))),
                output_dim=1,
                activation=str(cfg.get("value_head_activation", "relu")),
                bias_last=True,
            )
            ref_param = next(self.policy.parameters(), None)
            if ref_param is not None:
                self.value_head.to(device=ref_param.device, dtype=ref_param.dtype)

        uniform_dtype = _dtype_from_name(
            cfg.get("fsdp_uniform_dtype", None) or torch_dtype
        )
        if uniform_dtype is not None:
            self.to(dtype=uniform_dtype)

        self.assert_smolvla_api()
        self.freeze_for_ppo()

    def _load_bundle(self, cfg):
        try:
            from smolvla_pipeline.evaluator import _load_smolvla_bundle
        except Exception as exc:
            raise ImportError(
                "SmolVLA adapter requires Phase11 SmolVLA helpers on PYTHONPATH "
                "for the spike. Add /rds/general/user/aa6622/home/project/src."
            ) from exc

        model_path = str(cfg.get("model_path", "jadechoghari/smolvla_metaworld"))
        return _load_smolvla_bundle(
            model_path,
            n_action_steps=int(cfg.get("n_action_steps", self.num_action_chunks)),
        )

    def assert_smolvla_api(self):
        if not hasattr(self.policy, "select_action_distr_params"):
            raise RuntimeError("SmolVLA policy missing select_action_distr_params hook.")
        model = getattr(self.policy, "model", None)
        if not hasattr(model, "log_std"):
            raise RuntimeError("SmolVLA policy.model missing log_std parameter.")

    def freeze_for_ppo(self):
        if not bool(self.config.get("freeze_all_but_ppo_trainables", True)):
            return
        for param in self.policy.parameters():
            param.requires_grad = False

        lm_expert = getattr(
            getattr(getattr(self.policy, "model", None), "vlm_with_expert", None),
            "lm_expert",
            None,
        )
        if lm_expert is not None:
            for param in lm_expert.parameters():
                param.requires_grad = True

        log_std = getattr(getattr(self.policy, "model", None), "log_std", None)
        if isinstance(log_std, nn.Parameter):
            log_std.requires_grad = True

        if hasattr(self, "value_head"):
            for param in self.value_head.parameters():
                param.requires_grad = True

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        return super().forward(forward_type=forward_type, **kwargs)

    def _build_raw_batch(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        images = env_obs["main_images"]
        states = env_obs["states"]
        if isinstance(images, torch.Tensor):
            images_np = images.detach().cpu().numpy()
        else:
            images_np = np.asarray(images)
        if isinstance(states, torch.Tensor):
            states_np = states.detach().cpu().numpy()
        else:
            states_np = np.asarray(states)

        if images_np.dtype != np.uint8:
            images_np = np.clip(images_np, 0, 255).astype(np.uint8)
        task_descriptions = env_obs.get("task_descriptions", [""] * len(states_np))
        return {
            "pixels": images_np,
            "agent_pos": states_np.astype(np.float32),
            "task": list(task_descriptions),
        }

    def _reset_policy_state_if_needed(self, env_obs: dict[str, Any]) -> None:
        reset_seeds = env_obs.get("reset_seeds")
        if reset_seeds is None:
            return
        seeds = torch.as_tensor(reset_seeds, dtype=torch.long).detach().cpu()
        if (
            self._last_reset_seeds is not None
            and self._last_reset_seeds.shape == seeds.shape
            and torch.equal(self._last_reset_seeds, seeds)
        ):
            return
        policy_reset = getattr(self.policy, "reset", None)
        if callable(policy_reset):
            policy_reset()
        self._last_reset_seeds = seeds.clone()

    def obs_to_proc(self, env_obs: dict[str, Any]) -> dict[str, Any]:
        try:
            from lerobot.envs.utils import preprocess_observation
        except Exception as exc:
            raise ImportError("SmolVLA preprocessing requires lerobot.") from exc

        raw = self._build_raw_batch(env_obs)
        obs = preprocess_observation(
            {"pixels": raw["pixels"], "agent_pos": raw["agent_pos"]}
        )
        obs["task"] = raw["task"]
        proc = self.preprocessor(obs)
        try:
            _get_proc_value(proc, self.obs_image_key)
            _get_proc_value(proc, self.obs_state_key)
        except KeyError as exc:
            raise KeyError(
                f"SmolVLA preprocessor must produce {self.obs_image_key!r} and "
                f"{self.obs_state_key!r}; got {sorted(proc.keys())}"
            ) from exc
        return proc

    def _clear_policy_forward_state(self) -> None:
        policy_reset = getattr(self.policy, "reset", None)
        if callable(policy_reset):
            policy_reset()

    def _get_distr_params_chunk_batch(
        self,
        proc: dict[str, Any],
        *,
        n_envs: int,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._clear_policy_forward_state()
        device = _first_tensor_device(proc)
        param_dtype = self._floating_param_dtype()
        if (
            device is not None
            and device.type == "cuda"
            and param_dtype in {torch.bfloat16, torch.float16}
        ):
            autocast_context = torch.autocast(device_type="cuda", dtype=param_dtype)
        else:
            autocast_context = nullcontext()

        with autocast_context:
            policy_hook = getattr(self.policy, "_get_distr_params_chunk", None)
            if callable(policy_hook):
                try:
                    mean, log_std = policy_hook(proc)
                except TypeError:
                    mean, log_std = policy_hook(proc, chunk_len=int(chunk_len))
            else:
                model = getattr(self.policy, "model", None)
                model_hook = getattr(model, "_get_distr_params_chunk", None)
                if callable(model_hook):
                    try:
                        mean, log_std = model_hook(proc)
                    except TypeError:
                        mean, log_std = model_hook(proc, chunk_len=int(chunk_len))
                else:
                    mean, log_std = self.policy.select_action_distr_params(proc)

        return self._reshape_chunk_params_batch(
            mean,
            log_std,
            n_envs=int(n_envs),
            chunk_len=int(chunk_len),
        )

    def _reshape_chunk_params_batch(
        self,
        mean: torch.Tensor,
        log_std: torch.Tensor,
        *,
        n_envs: int,
        chunk_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean = mean.reshape(-1, self.action_dim)
        log_std = log_std.reshape(-1, self.action_dim)
        total = n_envs * chunk_len
        if mean.shape[0] == n_envs:
            mean = mean[:, None, :].expand(n_envs, chunk_len, self.action_dim)
            log_std = log_std[:, None, :].expand(n_envs, chunk_len, self.action_dim)
        elif mean.shape[0] == total:
            mean = mean.reshape(n_envs, chunk_len, self.action_dim)
            log_std = log_std.reshape(n_envs, chunk_len, self.action_dim)
        elif mean.shape[0] % n_envs == 0 and mean.shape[0] >= total:
            # Jade can emit a longer internal horizon; PPO trains on first RLinf chunk.
            horizon = mean.shape[0] // n_envs
            mean = mean.reshape(n_envs, horizon, self.action_dim)[:, :chunk_len, :]
            log_std = log_std.reshape(n_envs, horizon, self.action_dim)[
                :, :chunk_len, :
            ]
        elif mean.shape[0] == 1:
            mean = mean.expand(total, self.action_dim).reshape(
                n_envs, chunk_len, self.action_dim
            )
            log_std = log_std.expand(total, self.action_dim).reshape(
                n_envs, chunk_len, self.action_dim
            )
        else:
            raise RuntimeError(
                f"Cannot reshape SmolVLA params from {tuple(mean.shape)} to "
                f"({n_envs}, {chunk_len}, {self.action_dim})"
            )
        return mean, log_std

    @staticmethod
    def gaussian_log_prob_per_dim(
        mean: torch.Tensor,
        log_std: torch.Tensor,
        sample: torch.Tensor,
    ) -> torch.Tensor:
        mean = mean.float()
        log_std = log_std.float()
        sample = sample.float()
        std = torch.exp(log_std)
        var = std * std
        return -0.5 * (
            ((sample - mean) ** 2) / var + 2 * log_std + math.log(2 * math.pi)
        )

    def _postprocess_actions(self, policy_tensor: torch.Tensor) -> torch.Tensor:
        rows = []
        flat = policy_tensor.reshape(-1, self.action_dim)
        for row in flat:
            out = self.postprocessor(row.reshape(1, -1))
            if hasattr(out, "detach"):
                arr = out.detach().float().cpu().numpy().reshape(-1)
            else:
                arr = np.asarray(out, dtype=np.float32).reshape(-1)
            if arr.size != self.action_dim:
                raise RuntimeError(
                    f"SmolVLA action dim mismatch: expected {self.action_dim}, got {arr.size}"
                )
            rows.append(np.clip(arr, self.action_low, self.action_high))
        actions = torch.as_tensor(
            np.stack(rows, axis=0),
            dtype=torch.float32,
            device=policy_tensor.device,
        )
        return actions.reshape_as(policy_tensor)

    def _values_from_proc(self, proc: dict[str, Any], compute_values: bool) -> torch.Tensor:
        state = _get_proc_value(proc, self.obs_state_key).float()
        if not compute_values or not hasattr(self, "value_head"):
            return torch.zeros((state.shape[0], 1), dtype=state.dtype, device=state.device)
        value_input = state.detach() if self.detach_critic_input else state
        value_param = next(self.value_head.parameters(), None)
        if value_param is not None:
            value_input = value_input.to(dtype=value_param.dtype, device=value_param.device)
        return self.value_head(value_input).float()

    def _floating_param_dtype(self) -> torch.dtype | None:
        for param in self.parameters():
            if param.is_floating_point():
                return param.dtype
        return None

    def _proc_to_flow_batch(self, proc: dict[str, Any], device: torch.device):
        batch = _to_device(proc, device, dtype=self._floating_param_dtype())
        if hasattr(self.policy, "_prepare_batch"):
            batch = self.policy._prepare_batch(batch)
        images, img_masks = self.policy.prepare_images(batch)
        state = self.policy.prepare_state(batch)
        from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

        lang_tokens = batch[OBS_LANGUAGE_TOKENS]
        lang_masks = batch[OBS_LANGUAGE_ATTENTION_MASK]
        return images, img_masks, lang_tokens, lang_masks, state

    def _build_smolvla_flow_cache(
        self, proc: dict[str, Any], device: torch.device
    ) -> tuple[Any, Any, torch.Tensor]:
        from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

        images, img_masks, lang_tokens, lang_masks, state = self._proc_to_flow_batch(
            proc, device
        )
        model = self.policy.model
        param_dtype = self._floating_param_dtype()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=param_dtype)
            if device.type == "cuda"
            and param_dtype in {torch.bfloat16, torch.float16}
            else nullcontext()
        )
        with autocast_context:
            prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
                images, img_masks, lang_tokens, lang_masks, state=state
            )
            prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
            prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
            _, past_key_values = model.vlm_with_expert.forward(
                attention_mask=prefix_att_2d_masks,
                position_ids=prefix_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, None],
                use_cache=model.config.use_cache,
                fill_kv_cache=True,
            )
        return prefix_pad_masks, past_key_values, state

    @torch.no_grad()
    def _flow_sde_sample(
        self,
        proc: dict[str, Any],
        *,
        n_envs: int,
        mode: str,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = next(self.parameters()).device
        prefix_pad_masks, past_key_values, state = self._build_smolvla_flow_cache(
            proc, device
        )
        model = self.policy.model
        num_steps = int(self.flow_sde_num_steps)
        dt = -1.0 / float(num_steps)
        bsize = int(state.shape[0])
        action_dim = int(self.policy.config.action_feature.shape[0])
        model_chunk = int(model.config.chunk_size)
        chunk_len = int(self.num_action_chunks)
        max_action_dim = int(model.config.max_action_dim)
        noise = torch.randn(
            (bsize, model_chunk, max_action_dim),
            device=device,
            dtype=torch.float32,
        )
        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        chain_steps = [x_t.detach().cpu()]
        param_dtype = self._floating_param_dtype()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=param_dtype)
            if device.type == "cuda"
            and param_dtype in {torch.bfloat16, torch.float16}
            else nullcontext()
        )
        with autocast_context:
            while time.item() >= (-dt / 2.0) - 1e-6:
                expanded_time = time.expand(bsize)
                v_t = model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=expanded_time,
                )
                mean, std = sde_step_params(
                    x_t.float(), v_t.float(), dt, self.flow_sde_noise_level
                )
                if mode == "train":
                    x_t = mean + std * torch.randn_like(mean)
                else:
                    x_t = mean
                chain_steps.append(x_t.detach().cpu())
                time = time + dt
        chains = torch.stack(chain_steps, dim=1)
        final = chains[:, -1, :chunk_len, :action_dim].to(
            device=device, dtype=torch.float32
        )
        denoise_inds = (
            torch.arange(num_steps, device=device, dtype=torch.long)[None]
            .expand(bsize, -1)
            .contiguous()
        )
        return final, chains, denoise_inds

    def _flow_sde_logprob_from_chains(
        self,
        proc: dict[str, Any],
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        chains = chains.to(device=device, dtype=torch.float32)
        denoise_inds = denoise_inds.to(device=device, dtype=torch.long)
        prefix_pad_masks, past_key_values, state = self._build_smolvla_flow_cache(
            proc, device
        )
        model = self.policy.model
        num_steps = int(self.flow_sde_num_steps)
        dt = -1.0 / float(num_steps)
        bsize = int(chains.shape[0])
        action_dim = int(self.policy.config.action_feature.shape[0])
        chunk_len = int(self.num_action_chunks)
        total_lp = torch.zeros(
            bsize,
            chunk_len,
            action_dim,
            device=device,
            dtype=torch.float32,
        )
        param_dtype = self._floating_param_dtype()
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=param_dtype)
            if device.type == "cuda"
            and param_dtype in {torch.bfloat16, torch.float16}
            else nullcontext()
        )
        with autocast_context:
            for step_idx in range(num_steps):
                time = torch.tensor(
                    1.0 + step_idx * dt, dtype=torch.float32, device=device
                )
                expanded_time = time.expand(bsize)
                x_t = chains[:, step_idx]
                x_next = chains[:, step_idx + 1]
                v_t = model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=expanded_time,
                )
                mean, std = sde_step_params(
                    x_t.float(), v_t.float(), dt, self.flow_sde_noise_level
                )
                total_lp += sde_step_logprob_per_dim(
                    x_next[:, :chunk_len, :action_dim],
                    mean[:, :chunk_len, :action_dim],
                    std[:, :chunk_len, :action_dim],
                )
        return total_lp

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs,
        mode: str = "train",
        compute_values: bool = True,
        **kwargs,
    ):
        self._reset_policy_state_if_needed(env_obs)
        proc = self.obs_to_proc(env_obs)
        device = next(self.parameters()).device
        proc = _to_device(proc, device, dtype=self._floating_param_dtype())
        batch_size = int(_get_proc_value(proc, self.obs_state_key).shape[0])

        if self.noise_method == "flow_sde":
            unsquashed, chains, denoise_inds = self._flow_sde_sample(
                proc, n_envs=batch_size, mode=mode
            )
            actions = self._postprocess_actions(unsquashed)
            prev_logprobs = self._flow_sde_logprob_from_chains(proc, chains, denoise_inds)
            prev_values = self._values_from_proc(proc, compute_values=compute_values)
            forward_inputs = _flatten_tensor_tree(proc)
            forward_inputs["chains"] = chains.detach().cpu().contiguous()
            forward_inputs["denoise_inds"] = denoise_inds.detach().cpu().contiguous()
            forward_inputs["smolvla_unsquashed_actions"] = (
                unsquashed.detach().cpu().contiguous()
            )
            forward_inputs["action"] = (
                actions.detach().reshape(batch_size, -1).cpu().contiguous()
            )
            return actions, {
                "prev_logprobs": prev_logprobs.detach(),
                "prev_values": prev_values.detach(),
                "forward_inputs": forward_inputs,
            }

        mean, log_std = self._get_distr_params_chunk_batch(
            proc,
            n_envs=batch_size,
            chunk_len=self.num_action_chunks,
        )
        if mode == "train":
            noise = torch.randn_like(mean)
            unsquashed = mean + torch.exp(log_std) * noise
        elif mode == "eval":
            unsquashed = mean.clone()
        else:
            raise NotImplementedError(f"{mode=}")

        actions = self._postprocess_actions(unsquashed)
        prev_logprobs = self.gaussian_log_prob_per_dim(mean, log_std, unsquashed)
        prev_values = self._values_from_proc(proc, compute_values=compute_values)
        forward_inputs = _flatten_tensor_tree(proc)
        forward_inputs["smolvla_unsquashed_actions"] = (
            unsquashed.detach().cpu().contiguous()
        )
        forward_inputs["action"] = actions.detach().reshape(batch_size, -1).cpu().contiguous()

        return actions, {
            "prev_logprobs": prev_logprobs.detach(),
            "prev_values": prev_values.detach(),
            "forward_inputs": forward_inputs,
        }

    def default_forward(
        self,
        forward_inputs,
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = True,
        **kwargs,
    ):
        device = next(self.parameters()).device
        proc = _unflatten_tensor_tree(forward_inputs)
        proc = _to_device(proc, device, dtype=self._floating_param_dtype())

        if self.noise_method == "flow_sde" and "chains" in forward_inputs:
            chains = forward_inputs["chains"].to(device)
            denoise_inds = forward_inputs["denoise_inds"].to(device)
            output_dict: dict[str, Any] = {}
            if compute_logprobs:
                output_dict["logprobs"] = self._flow_sde_logprob_from_chains(
                    proc, chains, denoise_inds
                )
            if compute_entropy:
                output_dict["entropy"] = torch.zeros(
                    (chains.shape[0], self.num_action_chunks, self.action_dim),
                    device=device,
                    dtype=torch.float32,
                )
            if compute_values:
                output_dict["values"] = self._values_from_proc(
                    proc, compute_values=True
                )
            return output_dict

        unsquashed = forward_inputs["smolvla_unsquashed_actions"].to(device)
        if unsquashed.ndim == 3:
            n_envs, chunk_len, action_dim = unsquashed.shape
            if int(action_dim) != self.action_dim:
                raise RuntimeError(
                    f"smolvla_unsquashed_actions action_dim={action_dim} "
                    f"!= {self.action_dim}"
                )
        elif unsquashed.ndim == 2:
            n_envs = int(unsquashed.shape[0])
            chunk_len = self.num_action_chunks
            unsquashed = unsquashed.reshape(n_envs, chunk_len, self.action_dim)
        else:
            raise RuntimeError(
                f"smolvla_unsquashed_actions must be 2D or 3D, got {tuple(unsquashed.shape)}"
            )
        mean, log_std = self._get_distr_params_chunk_batch(
            proc,
            n_envs=int(n_envs),
            chunk_len=int(chunk_len),
        )

        output_dict = {}
        output_dict["mean"] = mean
        output_dict["log_std"] = log_std
        if compute_logprobs:
            output_dict["logprobs"] = self.gaussian_log_prob_per_dim(
                mean, log_std, unsquashed
            )
        if compute_entropy:
            output_dict["entropy"] = 0.5 * torch.log(
                2 * math.pi * math.e * torch.exp(2 * log_std.float())
            )
        if compute_values:
            output_dict["values"] = self._values_from_proc(proc, compute_values=True)
        return output_dict
