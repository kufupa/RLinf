from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from scripts.run_smolvla_metaworld_direct_ppo import (
    DEFAULT_CHECKPOINT,
    TimingStats,
    compute_gae,
    make_env,
    make_model,
    masked_chunk_logprob,
    normalize_advantages,
    should_update_actor,
    tensorize_forward_inputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SmolVLA MetaWorld component microbenchmarks.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task-name", default="push-v3")
    parser.add_argument("--task-description", default="Push the puck to a goal")
    parser.add_argument("--batch-sizes", default="1,4,8,16,32")
    parser.add_argument("--policy-iters", type=int, default=100)
    parser.add_argument("--env-iters", type=int, default=100)
    parser.add_argument("--rollout-iters", type=int, default=25)
    parser.add_argument("--ppo-updates", type=int, default=1)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps-per-update", type=int, default=120)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-seed-base", type=int, default=2000)
    parser.add_argument("--reward-mode", default="sparse_success_delta")
    parser.add_argument("--use-rel-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--value-lr", type=float, default=1e-4)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--minibatch-envs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--modes", default="policy,env,rollout,ppo")
    return parser.parse_args()


def emit(path: Path, payload: dict[str, Any]) -> None:
    line = json.dumps(payload, sort_keys=True)
    print("SMOLVLA_COMPONENT_BENCH " + line, flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def tensor_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    if isinstance(value, dict):
        return {str(key): tensor_summary(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [tensor_summary(item) for item in value[:4]]
    return str(type(value).__name__)


def repeat_obs(obs: dict[str, Any], batch_size: int) -> dict[str, Any]:
    base = {}
    for key, value in obs.items():
        if isinstance(value, torch.Tensor):
            reps = [batch_size] + [1] * (value.ndim - 1)
            base[key] = value[:1].repeat(*reps).contiguous()
        elif isinstance(value, list):
            base[key] = [value[0] if value else ""] * batch_size
        else:
            arr = np.asarray(value)
            reps = [batch_size] + [1] * (arr.ndim - 1)
            base[key] = np.repeat(arr[:1], reps[0], axis=0)
    return base


def cuda_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        payload.update(
            {
                "gpu_name": torch.cuda.get_device_name(device),
                "device_count": torch.cuda.device_count(),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device),
                "max_memory_reserved": torch.cuda.max_memory_reserved(device),
            }
        )
    try:
        query = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,name,utilization.gpu,utilization.memory,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        )
        payload["nvidia_smi"] = [line.strip() for line in query.splitlines() if line.strip()]
    except Exception as exc:
        payload["nvidia_smi_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def make_direct_args(args: argparse.Namespace, num_envs: int) -> argparse.Namespace:
    values = vars(args).copy()
    values["num_envs"] = int(num_envs)
    values["model_path"] = args.model_path or DEFAULT_CHECKPOINT
    return argparse.Namespace(**values)


def bench_policy(args: argparse.Namespace, model, obs, device, out_path: Path) -> None:
    for batch_size in [int(item) for item in args.batch_sizes.split(",") if item.strip()]:
        cached_obs = repeat_obs(obs, batch_size)
        timers = TimingStats(device)
        torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
        first_shapes = None
        for i in range(args.policy_iters):
            with timers.time("policy_predict_action_batch"):
                actions, policy_out = model.predict_action_batch(cached_obs, mode="train")
            if i == 0:
                first_shapes = {
                    "obs": tensor_summary(cached_obs),
                    "actions": tensor_summary(actions),
                    "forward_inputs": tensor_summary(policy_out["forward_inputs"]),
                    "prev_logprobs": tensor_summary(policy_out["prev_logprobs"]),
                }
        emit(
            out_path,
            {
                "mode": "policy_only",
                "batch_size": batch_size,
                "iters": args.policy_iters,
                "timing": timers.payload(),
                "shapes": first_shapes,
                "hardware": cuda_payload(device),
            },
        )


def bench_env(args: argparse.Namespace, device, out_path: Path) -> None:
    env_args = make_direct_args(args, args.num_envs)
    env = make_env(env_args)
    timers = TimingStats(device)
    try:
        with timers.time("env_reset"):
            obs, _ = env.reset()
        action = np.zeros((args.num_envs, args.chunk_len, 4), dtype=np.float32)
        first_shapes = {"obs": tensor_summary(obs), "action": {"shape": list(action.shape), "dtype": str(action.dtype)}}
        for _ in range(args.env_iters):
            with timers.time("env_chunk_step"):
                obs_list, rewards, terms, truncs, infos = env.chunk_step(action)
            if bool(infos[-1].get("all_rows_terminal", False)) or bool((terms | truncs).all()):
                with timers.time("env_reset"):
                    env.reset()
        emit(
            out_path,
            {
                "mode": "env_only",
                "num_envs": args.num_envs,
                "chunk_len": args.chunk_len,
                "iters": args.env_iters,
                "env_step_calls": args.env_iters * args.chunk_len,
                "timing": timers.payload(),
                "shapes": first_shapes,
                "hardware": cuda_payload(device),
            },
        )
    finally:
        env.close()


def collect_rollout(args: argparse.Namespace, model, env, obs, device):
    chunks_per_rollout = max(1, max(args.chunk_len, args.steps_per_update) // args.chunk_len)
    rollout_inputs = []
    rollout_old_logp = []
    rollout_values = []
    rollout_rewards = []
    rollout_dones = []
    rollout_valid_masks = []
    timers = TimingStats(device)
    for _ in range(chunks_per_rollout):
        with timers.time("policy_predict_action_batch"):
            actions, policy_out = model.predict_action_batch(obs, mode="train")
        with timers.time("rollout_old_logprob_value_forward"):
            out = model.default_forward(policy_out["forward_inputs"])
        with timers.time("action_cpu_transfer"):
            chunk_actions = actions.detach().cpu().numpy()
        with timers.time("env_chunk_step"):
            obs_list, rewards, terms, truncs, infos_list = env.chunk_step(chunk_actions)
        with timers.time("rollout_storage"):
            valid_mask = infos_list[-1].get("valid_action_mask", torch.ones_like(rewards, dtype=torch.bool))
            rollout_inputs.append(policy_out["forward_inputs"])
            rollout_old_logp.append(policy_out["prev_logprobs"].detach().cpu())
            rollout_values.append(out["values"].detach().cpu())
            rollout_rewards.append(rewards.detach().cpu())
            rollout_dones.append((terms | truncs).detach().cpu())
            rollout_valid_masks.append(valid_mask.detach().cpu().bool())
        obs = obs_list[-1]
        if bool(infos_list[-1].get("all_rows_terminal", False)) or bool((terms | truncs).all()):
            with timers.time("env_reset"):
                obs, _ = env.reset()
    return obs, rollout_inputs, rollout_old_logp, rollout_values, rollout_rewards, rollout_dones, rollout_valid_masks, timers


def prepare_ppo_batch(args, device, rollout_old_logp, rollout_values, rollout_rewards, rollout_dones, rollout_valid_masks):
    chunks_per_rollout = max(1, max(args.chunk_len, args.steps_per_update) // args.chunk_len)
    old_logp = torch.cat(rollout_old_logp, dim=1).to(device)
    values_chunk = torch.cat(rollout_values, dim=0).reshape(chunks_per_rollout, args.num_envs).T
    rewards = torch.cat(rollout_rewards, dim=1).to(device)
    dones = torch.cat(rollout_dones, dim=1).to(device)
    valid_mask = torch.cat(rollout_valid_masks, dim=1).to(device).bool()
    chunk_valid = valid_mask.reshape(args.num_envs, chunks_per_rollout, args.chunk_len)
    rewards_chunk = (
        rewards.reshape(args.num_envs, chunks_per_rollout, args.chunk_len) * chunk_valid.float()
    ).sum(dim=2)
    dones_chunk = dones.reshape(args.num_envs, chunks_per_rollout, args.chunk_len).any(dim=2)
    advantages_raw, returns = compute_gae(
        rewards_chunk,
        values_chunk.to(device),
        dones_chunk,
        chunk_valid.any(dim=2),
        args.gamma ** args.chunk_len,
        args.gae_lambda,
    )
    train_actor = should_update_actor(rewards, valid_mask)
    advantages = (
        normalize_advantages(advantages_raw, chunk_valid.any(dim=2))[0]
        if train_actor
        else torch.zeros_like(advantages_raw)
    )
    flat_count = args.num_envs * chunks_per_rollout
    flat_indices = torch.arange(flat_count)
    env_ids = flat_indices % args.num_envs
    chunk_ids = flat_indices // args.num_envs
    old_logp_chunk = old_logp.reshape(args.num_envs, chunks_per_rollout, args.chunk_len, 4)
    old_valid_chunk = chunk_valid.reshape(args.num_envs, chunks_per_rollout, args.chunk_len, 1)
    return {
        "chunks_per_rollout": chunks_per_rollout,
        "flat_count": flat_count,
        "old_logp_flat": masked_chunk_logprob(
            old_logp_chunk[env_ids, chunk_ids],
            old_valid_chunk[env_ids, chunk_ids].squeeze(-1),
        ).to(device),
        "valid_chunk_flat": chunk_valid.any(dim=2)[env_ids, chunk_ids].to(device),
        "valid_action_flat": chunk_valid[env_ids, chunk_ids].to(device),
        "adv_flat": advantages[env_ids, chunk_ids].to(device),
        "ret_flat": returns[env_ids, chunk_ids].to(device),
        "train_actor": train_actor,
    }


def bench_rollout_and_ppo(args: argparse.Namespace, model, env, obs, device, out_path: Path) -> None:
    optim = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "value_head" not in n], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters() if p.requires_grad and "value_head" in n], "lr": args.value_lr},
        ],
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.01,
    )
    trainable = [param for param in model.parameters() if param.requires_grad]
    torch.cuda.reset_peak_memory_stats(device) if device.type == "cuda" else None
    last_rollout = None
    rollout_timers = TimingStats(device)
    for _ in range(args.rollout_iters):
        result = collect_rollout(args, model, env, obs, device)
        obs = result[0]
        last_rollout = result
        for key, value in result[-1].totals.items():
            rollout_timers.totals[key] += value
        for key, value in result[-1].counts.items():
            rollout_timers.counts[key] += value
    emit(
        out_path,
        {
            "mode": "rollout_only",
            "num_envs": args.num_envs,
            "rollout_iters": args.rollout_iters,
            "timing": rollout_timers.payload(),
            "call_counts": {
                "predict_action_batch": rollout_timers.counts.get("policy_predict_action_batch", 0),
                "env_chunk_step": rollout_timers.counts.get("env_chunk_step", 0),
                "env_step": rollout_timers.counts.get("env_chunk_step", 0) * args.chunk_len,
            },
            "hardware": cuda_payload(device),
        },
    )
    if last_rollout is None:
        return

    _, rollout_inputs, rollout_old_logp, rollout_values, rollout_rewards, rollout_dones, rollout_valid_masks, _ = last_rollout
    ppo_batch = prepare_ppo_batch(
        args,
        device,
        rollout_old_logp,
        rollout_values,
        rollout_rewards,
        rollout_dones,
        rollout_valid_masks,
    )
    timers = TimingStats(device)
    total_mb = 0
    for _ in range(args.ppo_updates):
        for _epoch in range(args.update_epochs):
            perm = torch.randperm(ppo_batch["flat_count"])
            for start in range(0, ppo_batch["flat_count"], args.minibatch_envs):
                idx = perm[start : start + args.minibatch_envs]
                valid_idx = ppo_batch["valid_chunk_flat"][idx.to(device)]
                if not bool(valid_idx.any()):
                    continue
                idx_device = idx.to(device)
                with timers.time("ppo_minibatch_collate"):
                    forward_inputs = tensorize_forward_inputs(rollout_inputs, idx)
                with timers.time("ppo_minibatch_forward"):
                    out = model.default_forward(forward_inputs)
                    logp = masked_chunk_logprob(out["logprobs"], ppo_batch["valid_action_flat"][idx_device])
                    values_new = out["values"].reshape(-1)
                    ratio = torch.exp(logp - ppo_batch["old_logp_flat"][idx_device])
                    if ppo_batch["train_actor"]:
                        pg1 = ratio * ppo_batch["adv_flat"][idx_device]
                        pg2 = (
                            torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
                            * ppo_batch["adv_flat"][idx_device]
                        )
                        policy_loss = -(
                            torch.min(pg1, pg2) * valid_idx.float()
                        ).sum() / valid_idx.float().sum().clamp_min(1.0)
                    else:
                        policy_loss = torch.zeros((), device=device)
                    value_loss = torch.nn.functional.mse_loss(
                        values_new[valid_idx].float(),
                        ppo_batch["ret_flat"][idx_device][valid_idx].float(),
                    )
                    loss = policy_loss + args.value_coef * value_loss
                with timers.time("optimizer_zero_grad"):
                    optim.zero_grad(set_to_none=True)
                with timers.time("ppo_backward"):
                    loss.backward()
                with timers.time("grad_norms"):
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                with timers.time("optimizer_step"):
                    optim.step()
                total_mb += 1
    emit(
        out_path,
        {
            "mode": "ppo_update_only",
            "num_envs": args.num_envs,
            "ppo_updates": args.ppo_updates,
            "minibatches": total_mb,
            "timing": timers.payload(),
            "call_counts": {
                "ppo_forward": timers.counts.get("ppo_minibatch_forward", 0),
                "backward": timers.counts.get("ppo_backward", 0),
                "optimizer_step": timers.counts.get("optimizer_step", 0),
            },
            "hardware": cuda_payload(device),
        },
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "component_benchmark.jsonl"
    if out_path.exists():
        out_path.unlink()
    if args.model_path:
        os.environ["SMOLVLA_CHECKPOINT"] = args.model_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    direct_args = make_direct_args(args, args.num_envs)
    model = make_model(direct_args, device)
    env = make_env(direct_args)
    try:
        reset_timers = TimingStats(device)
        with reset_timers.time("env_reset"):
            obs, _ = env.reset()
        emit(
            out_path,
            {
                "mode": "identity",
                "run_name": args.run_name,
                "device": str(device),
                "model_path": direct_args.model_path,
                "pid": os.getpid(),
                "cpu_count": os.cpu_count(),
                "reset_timing": reset_timers.payload(),
                "obs": tensor_summary(obs),
                "hardware": cuda_payload(device),
            },
        )
        modes = {item.strip() for item in args.modes.split(",") if item.strip()}
        if "policy" in modes:
            bench_policy(args, model, obs, device, out_path)
        if "env" in modes:
            bench_env(args, device, out_path)
        if "rollout" in modes or "ppo" in modes:
            bench_rollout_and_ppo(args, model, env, obs, device, out_path)
    finally:
        env.close()


if __name__ == "__main__":
    main()
