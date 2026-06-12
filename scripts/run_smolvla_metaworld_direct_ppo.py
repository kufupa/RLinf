from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from torch_load_mmap import torch_load_mmap_default

from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv
from rlinf.models.embodiment.smolvla import get_model


DEFAULT_CHECKPOINT = (
    "jadechoghari/smolvla_metaworld"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct SmolVLA MetaWorld PPO runner.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--task-name", default="push-v3")
    parser.add_argument("--task-description", default="Push the puck to a goal")
    parser.add_argument("--model-path", default=os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--total-updates", type=int, default=100000)
    parser.add_argument("--steps-per-update", type=int, default=120)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--max-runtime-s", type=float, default=15.5 * 3600)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-seed-base", type=int, default=2000)
    parser.add_argument("--eval-seed-base", type=int, default=50000)
    parser.add_argument("--reward-mode", default="sparse_success_delta")
    parser.add_argument("--use-rel-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--noop-policy", action="store_true")
    parser.add_argument("--force-zero-reward", action="store_true")
    parser.add_argument("--save-initial-checkpoint", action="store_true")
    parser.add_argument("--resume-checkpoint", default="")
    parser.add_argument("--advantage-mode", choices=("gae", "mc"), default="gae")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-envs", type=int, default=2)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--save-interval", type=int, default=10)
    parser.add_argument("--log-interval", type=int, default=1)
    return parser.parse_args()


class TimingStats:
    def __init__(self, device: torch.device):
        self.device = device
        self.totals: dict[str, float] = defaultdict(float)
        self.counts: dict[str, int] = defaultdict(int)

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def time(self, name: str):
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] += time.perf_counter() - start
            self.counts[name] += 1

    def payload(self) -> dict[str, Any]:
        total = sum(self.totals.values())
        return {
            "total_timed_s": round(total, 6),
            "totals_s": {key: round(value, 6) for key, value in sorted(self.totals.items())},
            "counts": {key: int(value) for key, value in sorted(self.counts.items())},
            "avg_s": {
                key: round(self.totals[key] / max(self.counts[key], 1), 6)
                for key in sorted(self.totals)
            },
        }


def make_env(args: argparse.Namespace) -> SmolVLAMetaWorldEnv:
    cfg = OmegaConf.create(
        {
            "seed": args.seed,
            "task_name": args.task_name,
            "task_description": args.task_description,
            "reset_randomization_mode": "random_seeded",
            "max_episode_steps": args.max_episode_steps,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": bool(args.use_rel_reward),
            "reward_coef": 1.0,
            "reward_mode": args.reward_mode,
            "reset_seed_base": args.reset_seed_base,
            "use_async_envs": False,
            "video_cfg": {"save_video": False},
        }
    )
    return SmolVLAMetaWorldEnv(
        cfg,
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )


def make_model(args: argparse.Namespace, device: torch.device):
    cfg = OmegaConf.create(
        {
            "model_type": "smolvla",
            "model_path": args.model_path,
            "precision": None,
            "load_to_device": False,
            "action_dim": 4,
            "num_action_chunks": args.chunk_len,
            "state_dim": 4,
            "n_action_steps": args.chunk_len,
            "add_value_head": True,
            "detach_critic_input": True,
            "freeze_all_but_ppo_trainables": True,
            "action_low": -1.0,
            "action_high": 1.0,
            "is_lora": False,
        }
    )
    model = get_model(cfg, None).to(device)
    model.train()
    return model


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(num_envs, device=rewards.device)
    next_value = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        valid = valid_mask[:, t].float()
        mask = (1.0 - dones[:, t].float()) * valid
        delta = rewards[:, t] + gamma * next_value * mask - values[:, t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[:, t] = last_adv * valid
        next_value = values[:, t]
    returns = advantages + values
    return advantages, returns


def compute_discounted_returns(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    valid_mask: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    num_envs = rewards.shape[0]
    returns = torch.zeros_like(rewards)
    next_return = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        valid = valid_mask[:, t].float()
        mask = (1.0 - dones[:, t].float()) * valid
        next_return = (rewards[:, t] + gamma * next_return * mask) * valid
        returns[:, t] = next_return
    return returns


def normalize_advantages(advantages: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    valid = valid_mask.bool()
    normalized = torch.zeros_like(advantages)
    if not bool(valid.any()):
        return normalized, 0.0, 0.0
    valid_adv = advantages[valid]
    mean = valid_adv.mean()
    std = valid_adv.std(unbiased=False)
    if float(std.detach().cpu()) > 1e-8:
        normalized[valid] = (valid_adv - mean) / (std + 1e-8)
    else:
        normalized[valid] = valid_adv - mean
    valid_norm = normalized[valid]
    return normalized, float(valid_norm.mean().detach().cpu()), float(valid_norm.std(unbiased=False).detach().cpu())


def should_update_actor(
    rewards: torch.Tensor,
    valid_mask: torch.Tensor,
    eps: float = 1e-8,
) -> bool:
    if not bool(valid_mask.any()):
        return False
    reward_sum = (rewards * valid_mask.to(rewards.device).float()).abs().sum()
    return float(reward_sum.detach().cpu()) > eps


def masked_chunk_logprob(logprobs: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    valid = valid_mask.to(device=logprobs.device, dtype=logprobs.dtype).unsqueeze(-1)
    return (logprobs * valid).sum(dim=(1, 2))


def _trainable_param_norm(model: torch.nn.Module, needle: str | None = "all") -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_value = "value_head" in name
        if needle == "value_head" and not is_value:
            continue
        if needle is None and is_value:
            continue
        value = float(param.detach().float().norm().cpu())
        total += value * value
    return total ** 0.5


def _grad_norm_by_name(model: torch.nn.Module, needle: str | None) -> float:
    total = 0.0
    for name, param in model.named_parameters():
        if not param.requires_grad or param.grad is None:
            continue
        is_value = "value_head" in name
        if needle == "value_head" and not is_value:
            continue
        if needle is None and is_value:
            continue
        value = float(param.grad.detach().float().norm().cpu())
        total += value * value
    return total ** 0.5


def tensorize_forward_inputs(batch: list[dict[str, torch.Tensor]], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.cat([item[key] for item in batch], dim=0)
        out[key] = stacked[indices.cpu()]
    return out


def save_checkpoint(
    output_dir: Path,
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    update: int,
    metrics: dict[str, Any],
) -> None:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"update_{update:06d}.pt"
    trainable_state = {
        name: param.detach().cpu()
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    torch.save(
        {
            "update": update,
            "checkpoint_type": "trainable_delta",
            "trainable_model": trainable_state,
            "optimizer": optim.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    latest = ckpt_dir / "latest.pt"
    tmp = ckpt_dir / "latest.tmp"
    torch.save({"path": str(path), "update": update, "metrics": metrics}, tmp)
    tmp.replace(latest)


def load_trainable_checkpoint(
    model: torch.nn.Module,
    optim: torch.optim.Optimizer,
    checkpoint_path: str,
    device: torch.device,
) -> tuple[int, dict[str, Any]]:
    checkpoint = torch_load_mmap_default(checkpoint_path, map_location="cpu")
    trainable_state = checkpoint.get("trainable_model")
    if not isinstance(trainable_state, dict):
        raise KeyError(f"{checkpoint_path} missing trainable_model")

    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in trainable_state.items():
            if name not in named_params:
                raise KeyError(f"{checkpoint_path} has unknown param {name}")
            param = named_params[name]
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))

    optimizer_state = checkpoint.get("optimizer")
    if isinstance(optimizer_state, dict):
        optim.load_state_dict(optimizer_state)
        for state in optim.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device=device)

    return int(checkpoint.get("update", 0)), checkpoint.get("metrics", {})


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"direct_ppo: run={args.run_name} device={device} output={output_dir}", flush=True)
    model = make_model(args, device)
    trainable = [param for param in model.parameters() if param.requires_grad]
    optim = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "value_head" not in n], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "value_head" in n], "lr": args.value_lr},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    env = make_env(args)
    start_time = time.time()

    try:
        steps_per_rollout = max(args.chunk_len, args.steps_per_update)
        chunks_per_rollout = max(1, steps_per_rollout // args.chunk_len)
        init_timers = TimingStats(device)
        with init_timers.time("env_reset"):
            obs, _ = env.reset()
        metrics: dict[str, Any] = {
            "time_s": 0.0,
            "run_name": args.run_name,
            "update": 0,
            "env_steps": 0,
            "reward_mean": 0.0,
            "return_mean": 0.0,
            "raw_reward_sum": 0.0,
            "actor_update_skipped": True,
        }
        if args.save_initial_checkpoint:
            save_checkpoint(output_dir, model, optim, 0, metrics)
        start_update = 0
        if args.resume_checkpoint:
            start_update, resume_metrics = load_trainable_checkpoint(
                model, optim, args.resume_checkpoint, device
            )
            metrics.update(
                {
                    "resume_checkpoint": args.resume_checkpoint,
                    "resume_from_update": start_update,
                    "source_metrics": resume_metrics,
                }
            )
            print(
                "direct_ppo: resumed "
                + json.dumps(
                    {
                        "checkpoint": args.resume_checkpoint,
                        "start_update": start_update,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            save_checkpoint(output_dir, model, optim, start_update, metrics)

        for update in range(start_update + 1, args.total_updates + 1):
            rollout_inputs: list[dict[str, torch.Tensor]] = []
            rollout_old_logp: list[torch.Tensor] = []
            rollout_values: list[torch.Tensor] = []
            rollout_rewards: list[torch.Tensor] = []
            rollout_dones: list[torch.Tensor] = []
            rollout_valid_masks: list[torch.Tensor] = []
            rollout_entropy: list[torch.Tensor] = []
            successes = []
            ep_lens = []
            action_clip_fracs = []
            done_any_count = 0
            done_all_count = 0
            param_norm_before = _trainable_param_norm(model, "all")
            actor_param_norm_before = _trainable_param_norm(model, None)
            critic_param_norm_before = _trainable_param_norm(model, "value_head")
            timers = TimingStats(device)
            if update == start_update + 1:
                timers.totals.update(init_timers.totals)
                timers.counts.update(init_timers.counts)

            for _ in range(chunks_per_rollout):
                with timers.time("policy_predict_action_batch"):
                    actions, policy_out = model.predict_action_batch(obs, mode="train")
                with timers.time("rollout_old_logprob_value_forward"):
                    out = model.default_forward(policy_out["forward_inputs"])
                with timers.time("action_cpu_transfer"):
                    chunk_actions = actions.detach().cpu().numpy()
                with timers.time("env_chunk_step"):
                    obs_list, rewards, terms, truncs, infos_list = env.chunk_step(chunk_actions)
                if args.force_zero_reward:
                    rewards = torch.zeros_like(rewards)
                with timers.time("rollout_storage"):
                    valid_mask = infos_list[-1].get(
                        "valid_action_mask",
                        torch.ones_like(rewards, dtype=torch.bool),
                    )
                    valid_mask_cpu = valid_mask.detach().cpu().bool()
                    rollout_inputs.append(policy_out["forward_inputs"])
                    rollout_old_logp.append(policy_out["prev_logprobs"].detach().cpu())
                    rollout_values.append(out["values"].detach().cpu())
                    rollout_rewards.append(rewards.detach().cpu())
                    rollout_dones.append((terms | truncs).detach().cpu())
                    rollout_valid_masks.append(valid_mask_cpu)
                    rollout_entropy.append(out["entropy"].detach().cpu())
                    done_chunk = (terms | truncs).detach().cpu()
                    done_any_count += int(done_chunk.any().item())
                    done_all_count += int(done_chunk.all(dim=0).sum().item())
                    action_clip_fracs.append(
                        torch.as_tensor(
                            np.isclose(chunk_actions, -1.0) | np.isclose(chunk_actions, 1.0),
                            dtype=torch.float32,
                        )
                        .mean(dim=2)
                        .cpu()
                    )
                    final_info = infos_list[-1].get("episode", {})
                    if "success_once" in final_info:
                        successes.extend(final_info["success_once"].detach().cpu().numpy().astype(float).tolist())
                    if "episode_len" in final_info:
                        ep_lens.extend(final_info["episode_len"].detach().cpu().numpy().astype(float).tolist())
                    obs = obs_list[-1]
                if bool(infos_list[-1].get("all_rows_terminal", False)) or bool((terms | truncs).all()):
                    with timers.time("env_reset"):
                        obs, _ = env.reset()

            with timers.time("rollout_collate"):
                old_logp = torch.cat(rollout_old_logp, dim=1).to(device)
                values_chunk = torch.cat(rollout_values, dim=0).reshape(chunks_per_rollout, args.num_envs).T
                rewards = torch.cat(rollout_rewards, dim=1).to(device)
                dones = torch.cat(rollout_dones, dim=1).to(device)
                valid_mask = torch.cat(rollout_valid_masks, dim=1).to(device).bool()
            if not bool(valid_mask.any()):
                with timers.time("env_reset"):
                    obs, _ = env.reset()
                raise RuntimeError(
                    f"rollout at update {update} has zero valid steps after terminal masking"
                )
            with timers.time("advantage_compute"):
                chunk_valid = valid_mask.reshape(args.num_envs, chunks_per_rollout, args.chunk_len)
                rewards_chunk = (
                    rewards.reshape(args.num_envs, chunks_per_rollout, args.chunk_len)
                    * chunk_valid.float()
                ).sum(dim=2)
                dones_chunk = dones.reshape(args.num_envs, chunks_per_rollout, args.chunk_len).any(dim=2)
                values = values_chunk.to(device)
                if args.advantage_mode == "gae":
                    advantages_raw, returns = compute_gae(
                        rewards_chunk,
                        values,
                        dones_chunk,
                        chunk_valid.any(dim=2),
                        args.gamma ** args.chunk_len,
                        args.gae_lambda,
                    )
                else:
                    returns = compute_discounted_returns(
                        rewards_chunk,
                        dones_chunk,
                        chunk_valid.any(dim=2),
                        args.gamma ** args.chunk_len,
                    )
                    advantages_raw = returns.detach()
                train_actor = should_update_actor(rewards, valid_mask)
                if train_actor:
                    advantages, adv_norm_mean, adv_norm_std = normalize_advantages(
                        advantages_raw,
                        chunk_valid.any(dim=2),
                    )
                else:
                    advantages = torch.zeros_like(advantages_raw)
                    adv_norm_mean = 0.0
                    adv_norm_std = 0.0

            with timers.time("ppo_batch_prepare"):
                flat_count = args.num_envs * chunks_per_rollout
                flat_indices = torch.arange(flat_count)
                env_ids = flat_indices % args.num_envs
                chunk_ids = flat_indices // args.num_envs
                old_logp_chunk = old_logp.reshape(args.num_envs, chunks_per_rollout, args.chunk_len, 4)
                old_valid_chunk = chunk_valid.reshape(args.num_envs, chunks_per_rollout, args.chunk_len, 1)
                old_logp_flat = masked_chunk_logprob(
                    old_logp_chunk[env_ids, chunk_ids],
                    old_valid_chunk[env_ids, chunk_ids].squeeze(-1),
                ).to(device)
                valid_chunk_flat = chunk_valid.any(dim=2)[env_ids, chunk_ids].to(device)
                valid_action_flat = chunk_valid[env_ids, chunk_ids].to(device)
                adv_flat = advantages[env_ids, chunk_ids].to(device)
                ret_flat = returns[env_ids, chunk_ids].to(device)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_grad_norm = 0.0
            total_actor_grad_norm = 0.0
            total_critic_grad_norm = 0.0
            ratio_values = []
            approx_kl_values = []
            clip_values = []
            log_std_values = []
            total_mb = 0
            batch_size = max(1, min(args.minibatch_envs, flat_count))
            for _epoch in range(args.update_epochs):
                perm = torch.randperm(flat_count)
                for start in range(0, flat_count, batch_size):
                    idx = perm[start : start + batch_size]
                    valid_idx = valid_chunk_flat[idx.to(device)]
                    if not bool(valid_idx.any()):
                        continue
                    idx_device = idx.to(device)
                    with timers.time("ppo_minibatch_collate"):
                        forward_inputs = tensorize_forward_inputs(rollout_inputs, idx)
                    with timers.time("ppo_minibatch_forward"):
                        out = model.default_forward(forward_inputs)
                        mb_valid = valid_action_flat[idx_device].unsqueeze(-1).float()
                        logp = masked_chunk_logprob(out["logprobs"], valid_action_flat[idx_device])
                        values_new = out["values"].reshape(-1)
                        entropy = (out["entropy"] * mb_valid).sum() / mb_valid.sum().clamp_min(1.0)
                        ratio = torch.exp(logp - old_logp_flat[idx_device])
                        effective = valid_idx.float()
                        if train_actor and not args.noop_policy:
                            pg1 = ratio * adv_flat[idx_device]
                            pg2 = (
                                torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
                                * adv_flat[idx_device]
                            )
                            policy_loss = -(
                                torch.min(pg1, pg2) * effective
                            ).sum() / effective.sum().clamp_min(1.0)
                            entropy_loss_term = args.entropy_coef * entropy
                        else:
                            policy_loss = torch.zeros((), device=device)
                            entropy_loss_term = torch.zeros((), device=device)
                        value_loss = torch.nn.functional.mse_loss(
                            values_new[valid_idx].float(), ret_flat[idx_device][valid_idx].float()
                        )
                        loss = policy_loss + args.value_coef * value_loss - entropy_loss_term
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"non-finite loss at update {update}: {loss}")

                    with timers.time("optimizer_zero_grad"):
                        optim.zero_grad(set_to_none=True)
                    with timers.time("ppo_backward"):
                        loss.backward()
                    with timers.time("grad_norms"):
                        actor_grad_norm = _grad_norm_by_name(model, None)
                        critic_grad_norm = _grad_norm_by_name(model, "value_head")
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    with timers.time("optimizer_step"):
                        optim.step()

                    with timers.time("ppo_metric_extract"):
                        total_policy_loss += float(policy_loss.detach().cpu())
                        total_value_loss += float(value_loss.detach().cpu())
                        total_entropy += float(entropy.detach().cpu())
                        total_grad_norm += float(grad_norm.detach().cpu())
                        total_actor_grad_norm += actor_grad_norm
                        total_critic_grad_norm += critic_grad_norm
                        ratio_det = ratio.detach()[valid_idx].float().cpu()
                        if ratio_det.numel() > 0:
                            ratio_values.append(ratio_det)
                            old_logp_det = old_logp_flat[idx_device].detach()[valid_idx].float().cpu()
                            logp_det = logp.detach()[valid_idx].float().cpu()
                            approx_kl_values.append(old_logp_det - logp_det)
                            clip_values.append(
                                ((ratio_det - 1.0).abs() > args.clip_ratio).float()
                            )
                        log_std_values.append(out["log_std"].detach().float().cpu().reshape(-1))
                        total_mb += 1

            elapsed = time.time() - start_time
            param_norm_after = _trainable_param_norm(model, "all")
            actor_param_norm_after = _trainable_param_norm(model, None)
            critic_param_norm_after = _trainable_param_norm(model, "value_head")
            valid_chunk_bool = chunk_valid.any(dim=2)
            raw_adv_valid = advantages_raw[valid_chunk_bool]
            ratio_cat = torch.cat(ratio_values) if ratio_values else torch.ones(1)
            approx_kl_cat = torch.cat(approx_kl_values) if approx_kl_values else torch.zeros(1)
            clip_cat = torch.cat(clip_values) if clip_values else torch.zeros(1)
            log_std_cat = torch.cat(log_std_values) if log_std_values else torch.zeros(1)
            action_clip = torch.cat(action_clip_fracs, dim=1) if action_clip_fracs else torch.zeros(1)
            metrics = {
                "time_s": round(elapsed, 3),
                "run_name": args.run_name,
                "update": update,
                "env_steps": update * args.num_envs * chunks_per_rollout * args.chunk_len,
                "reward_mean": float(rewards.mean().detach().cpu()),
                "return_mean": float(rewards.sum(dim=1).mean().detach().cpu()),
                "raw_reward_sum": float(rewards.sum().detach().cpu()),
                "success_count": int(np.sum(successes)) if successes else 0,
                "done_any_count": done_any_count,
                "done_all_count": done_all_count,
                "valid_step_frac": float(valid_mask.float().mean().detach().cpu()),
                "adv_raw_mean": float(raw_adv_valid.mean().detach().cpu()) if raw_adv_valid.numel() else 0.0,
                "adv_raw_std": float(raw_adv_valid.std(unbiased=False).detach().cpu()) if raw_adv_valid.numel() else 0.0,
                "adv_mean": adv_norm_mean,
                "adv_std": adv_norm_std,
                "adv_norm_mean": adv_norm_mean,
                "adv_norm_std": adv_norm_std,
                "success_once_mean": float(np.mean(successes)) if successes else 0.0,
                "episode_len_mean": float(np.mean(ep_lens)) if ep_lens else 0.0,
                "policy_loss": total_policy_loss / max(total_mb, 1),
                "value_loss": total_value_loss / max(total_mb, 1),
                "entropy": total_entropy / max(total_mb, 1),
                "grad_norm": total_grad_norm / max(total_mb, 1),
                "actor_grad_norm": total_actor_grad_norm / max(total_mb, 1),
                "critic_grad_norm": total_critic_grad_norm / max(total_mb, 1),
                "ratio_mean": float(ratio_cat.mean().item()),
                "ratio_min": float(ratio_cat.min().item()),
                "ratio_max": float(ratio_cat.max().item()),
                "approx_kl": float(approx_kl_cat.mean().item()),
                "clip_fraction": float(clip_cat.mean().item()),
                "log_std_mean": float(log_std_cat.mean().item()),
                "log_std_std": float(log_std_cat.std(unbiased=False).item()),
                "log_std_min": float(log_std_cat.min().item()),
                "log_std_max": float(log_std_cat.max().item()),
                "action_clip_fraction": float(action_clip.mean().item()),
                "param_delta_norm": float(abs(param_norm_after - param_norm_before)),
                "actor_param_delta_norm": float(abs(actor_param_norm_after - actor_param_norm_before)),
                "critic_param_delta_norm": float(abs(critic_param_norm_after - critic_param_norm_before)),
                "actor_update_skipped": bool((not train_actor) or args.noop_policy),
                "timing": timers.payload(),
                "call_counts": {
                    "predict_action_batch": chunks_per_rollout,
                    "rollout_old_logprob_value_forward": chunks_per_rollout,
                    "ppo_forward": total_mb,
                    "backward": total_mb,
                    "optimizer_step": total_mb,
                    "env_chunk_step": chunks_per_rollout,
                    "env_step": chunks_per_rollout * args.chunk_len,
                    "chunk_steps": chunks_per_rollout,
                    "minibatches": total_mb,
                },
                "shapes": {
                    "num_envs": args.num_envs,
                    "steps_per_update": args.steps_per_update,
                    "chunk_len": args.chunk_len,
                    "chunks_per_rollout": chunks_per_rollout,
                    "action_shape": [args.num_envs, args.chunk_len, 4],
                    "ppo_minibatch_envs": batch_size,
                },
            }
            with timers.time("metrics_jsonl_write"):
                append_jsonl(metrics_path, metrics)
            if update % args.log_interval == 0:
                with timers.time("stdout_log"):
                    metrics["timing"] = timers.payload()
                    print("DIRECT_SMOLVLA_PPO_METRIC " + json.dumps(metrics, sort_keys=True), flush=True)
            if update == 1 or update % args.save_interval == 0:
                with timers.time("checkpoint_save"):
                    metrics["timing"] = timers.payload()
                    save_checkpoint(output_dir, model, optim, update, metrics)
            if elapsed >= args.max_runtime_s:
                print(f"direct_ppo: max runtime reached at update={update}", flush=True)
                break

        save_checkpoint(output_dir, model, optim, update, metrics)
        print("DIRECT_SMOLVLA_PPO_RUN_OK " + json.dumps(metrics, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
