from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv
from rlinf.models.embodiment.smolvla import get_model


DEFAULT_CHECKPOINT = (
    "/rds/general/user/aa6622/home/.cache/huggingface/hub/"
    "models--jadechoghari--smolvla_metaworld/snapshots/"
    "ef3089ecb84eeeb7d33fedab24f6c76180a68900"
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
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "reward_mode": "sparse_success_delta",
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
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_envs = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros(num_envs, device=rewards.device)
    next_value = torch.zeros(num_envs, device=rewards.device)
    for t in reversed(range(rewards.shape[1])):
        mask = 1.0 - dones[:, t].float()
        delta = rewards[:, t] + gamma * next_value * mask - values[:, t]
        last_adv = delta + gamma * gae_lambda * mask * last_adv
        advantages[:, t] = last_adv
        next_value = values[:, t]
    returns = advantages + values
    return advantages, returns


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
        obs, _ = env.reset()
        steps_per_rollout = max(args.chunk_len, args.steps_per_update)
        chunks_per_rollout = max(1, steps_per_rollout // args.chunk_len)
        for update in range(1, args.total_updates + 1):
            rollout_inputs: list[dict[str, torch.Tensor]] = []
            rollout_old_logp: list[torch.Tensor] = []
            rollout_values: list[torch.Tensor] = []
            rollout_rewards: list[torch.Tensor] = []
            rollout_dones: list[torch.Tensor] = []
            rollout_entropy: list[torch.Tensor] = []
            successes = []
            ep_lens = []

            for _ in range(chunks_per_rollout):
                actions, policy_out = model.predict_action_batch(obs, mode="train")
                out = model.default_forward(policy_out["forward_inputs"])
                chunk_actions = actions.detach().cpu().numpy()
                obs_list, rewards, terms, truncs, infos_list = env.chunk_step(chunk_actions)
                rollout_inputs.append(policy_out["forward_inputs"])
                rollout_old_logp.append(policy_out["prev_logprobs"].detach().cpu())
                rollout_values.append(out["values"].detach().cpu())
                rollout_rewards.append(rewards.detach().cpu())
                rollout_dones.append((terms | truncs).detach().cpu())
                rollout_entropy.append(out["entropy"].detach().cpu())
                final_info = infos_list[-1].get("episode", {})
                if "success_once" in final_info:
                    successes.extend(final_info["success_once"].detach().cpu().numpy().astype(float).tolist())
                if "episode_len" in final_info:
                    ep_lens.extend(final_info["episode_len"].detach().cpu().numpy().astype(float).tolist())
                obs = obs_list[-1]
                if bool((terms | truncs).any()):
                    obs, _ = env.reset()

            old_logp = torch.cat(rollout_old_logp, dim=1).to(device)
            values_chunk = torch.cat(rollout_values, dim=0).reshape(chunks_per_rollout, args.num_envs).T
            rewards = torch.cat(rollout_rewards, dim=1).to(device)
            dones = torch.cat(rollout_dones, dim=1).to(device)
            values = values_chunk.repeat_interleave(args.chunk_len, dim=1).to(device)
            advantages, returns = compute_gae(rewards, values, dones, args.gamma, args.gae_lambda)
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

            flat_count = args.num_envs * chunks_per_rollout
            flat_indices = torch.arange(flat_count)
            env_ids = flat_indices % args.num_envs
            chunk_ids = flat_indices // args.num_envs
            old_logp_chunk = old_logp.reshape(args.num_envs, chunks_per_rollout, args.chunk_len, 4)
            old_logp_flat = old_logp_chunk[env_ids, chunk_ids].sum(dim=(1, 2)).to(device)
            adv_flat = advantages.reshape(args.num_envs, chunks_per_rollout, args.chunk_len)[
                env_ids, chunk_ids
            ].sum(dim=1).to(device)
            ret_flat = returns.reshape(args.num_envs, chunks_per_rollout, args.chunk_len)[
                env_ids, chunk_ids
            ].mean(dim=1).to(device)

            total_policy_loss = 0.0
            total_value_loss = 0.0
            total_entropy = 0.0
            total_grad_norm = 0.0
            total_mb = 0
            batch_size = max(1, min(args.minibatch_envs, flat_count))
            for _epoch in range(args.update_epochs):
                perm = torch.randperm(flat_count)
                for start in range(0, flat_count, batch_size):
                    idx = perm[start : start + batch_size]
                    idx_device = idx.to(device)
                    forward_inputs = tensorize_forward_inputs(rollout_inputs, idx)
                    out = model.default_forward(forward_inputs)
                    logp = out["logprobs"].sum(dim=(1, 2))
                    values_new = out["values"].reshape(-1)
                    entropy = out["entropy"].mean()
                    ratio = torch.exp(logp - old_logp_flat[idx_device])
                    pg1 = ratio * adv_flat[idx_device]
                    pg2 = (
                        torch.clamp(ratio, 1 - args.clip_ratio, 1 + args.clip_ratio)
                        * adv_flat[idx_device]
                    )
                    policy_loss = -torch.min(pg1, pg2).mean()
                    value_loss = torch.nn.functional.mse_loss(
                        values_new.float(), ret_flat[idx_device].float()
                    )
                    loss = policy_loss + args.value_coef * value_loss - args.entropy_coef * entropy
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"non-finite loss at update {update}: {loss}")

                    optim.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                    optim.step()

                    total_policy_loss += float(policy_loss.detach().cpu())
                    total_value_loss += float(value_loss.detach().cpu())
                    total_entropy += float(entropy.detach().cpu())
                    total_grad_norm += float(grad_norm.detach().cpu())
                    total_mb += 1

            elapsed = time.time() - start_time
            metrics = {
                "time_s": round(elapsed, 3),
                "run_name": args.run_name,
                "update": update,
                "env_steps": update * args.num_envs * chunks_per_rollout * args.chunk_len,
                "reward_mean": float(rewards.mean().detach().cpu()),
                "return_mean": float(rewards.sum(dim=1).mean().detach().cpu()),
                "adv_mean": float(advantages.mean().detach().cpu()),
                "adv_std": float(advantages.std(unbiased=False).detach().cpu()),
                "success_once_mean": float(np.mean(successes)) if successes else 0.0,
                "episode_len_mean": float(np.mean(ep_lens)) if ep_lens else 0.0,
                "policy_loss": total_policy_loss / max(total_mb, 1),
                "value_loss": total_value_loss / max(total_mb, 1),
                "entropy": total_entropy / max(total_mb, 1),
                "grad_norm": total_grad_norm / max(total_mb, 1),
            }
            append_jsonl(metrics_path, metrics)
            if update % args.log_interval == 0:
                print("DIRECT_SMOLVLA_PPO_METRIC " + json.dumps(metrics, sort_keys=True), flush=True)
            if update == 1 or update % args.save_interval == 0:
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
