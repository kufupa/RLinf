from __future__ import annotations

import argparse
import json
import os
import re
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
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA PPO checkpoints.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task-name", default="push-v3")
    parser.add_argument("--task-description", default="Push the puck to a goal")
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=50000)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--checkpoint-stride", type=int, default=1)
    parser.add_argument("--max-checkpoints", type=int, default=0)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--only-updates", default="")
    return parser.parse_args()


def checkpoint_update(path: Path) -> int:
    match = re.search(r"update_(\d+)\.pt$", path.name)
    if not match:
        return -1
    return int(match.group(1))


def list_checkpoints(path: Path, stride: int, max_checkpoints: int) -> list[Path]:
    ckpts = sorted(path.glob("update_*.pt"), key=checkpoint_update)
    if stride > 1:
        ckpts = [ckpt for idx, ckpt in enumerate(ckpts) if idx % stride == 0]
    if max_checkpoints > 0:
        ckpts = ckpts[:max_checkpoints]
    return ckpts


def filter_checkpoints_by_update(ckpts: list[Path], only_updates: str) -> list[Path]:
    if not only_updates:
        return ckpts
    wanted = {int(part.strip()) for part in only_updates.split(",") if part.strip()}
    return [ckpt for ckpt in ckpts if checkpoint_update(ckpt) in wanted]


def make_env(args: argparse.Namespace) -> SmolVLAMetaWorldEnv:
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "task_name": args.task_name,
            "task_description": args.task_description,
            "reset_randomization_mode": "random_seeded",
            "max_episode_steps": args.max_episode_steps,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "reward_coef": 1.0,
            "reward_mode": "sparse_success_delta",
            "reset_seed_base": args.seed_base,
            "eval_seed_list": [
                args.seed_base + idx for idx in range(args.num_envs)
            ],
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
    model.eval()
    return model


def load_trainable_delta(model: torch.nn.Module, ckpt_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    trainable = checkpoint.get("trainable_model")
    if not isinstance(trainable, dict):
        raise KeyError(f"{ckpt_path} missing trainable_model")
    named_params = dict(model.named_parameters())
    with torch.no_grad():
        for name, tensor in trainable.items():
            if name not in named_params:
                raise KeyError(f"{ckpt_path} has unknown param {name}")
            param = named_params[name]
            param.copy_(tensor.to(device=param.device, dtype=param.dtype))
    return checkpoint


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def eval_seed_list(args: argparse.Namespace) -> list[int]:
    return [args.seed_base + idx for idx in range(args.num_envs)]


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    env: SmolVLAMetaWorldEnv,
    args: argparse.Namespace,
) -> dict[str, Any]:
    seeds = eval_seed_list(args)
    env._reset_counter = 0
    obs, _ = env.reset()
    reward_sum = torch.zeros(args.num_envs, dtype=torch.float32)
    success = torch.zeros(args.num_envs, dtype=torch.bool)
    episode_len = torch.zeros(args.num_envs, dtype=torch.float32)
    chunks = args.max_episode_steps // args.chunk_len
    for _ in range(chunks):
        actions, _policy_out = model.predict_action_batch(
            obs,
            mode="eval",
            compute_values=False,
        )
        obs_list, rewards, terms, truncs, infos_list = env.chunk_step(
            actions.detach().cpu().numpy()
        )
        reward_sum += rewards.sum(dim=1).float()
        done = terms.any(dim=1) | truncs.any(dim=1)
        success |= terms.any(dim=1)
        episode_len += (~done | (episode_len == 0)).float() * args.chunk_len
        final_info = infos_list[-1].get("episode", {})
        if "success_once" in final_info:
            success |= final_info["success_once"].bool()
        obs = obs_list[-1]
    per_env_success = success.cpu().tolist()
    return {
        "success_rate": float(success.float().mean().item()),
        "success_count": int(success.sum().item()),
        "return_mean": float(reward_sum.mean().item()),
        "return_max": float(reward_sum.max().item()),
        "episode_len_mean": float(episode_len.mean().item()),
        "eval_seeds": seeds,
        "per_env_success": per_env_success,
        "success_seeds": [seed for seed, ok in zip(seeds, per_env_success) if ok],
    }


def main() -> None:
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    config_path.write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")

    ckpts = list_checkpoints(
        checkpoint_dir,
        stride=max(1, args.checkpoint_stride),
        max_checkpoints=args.max_checkpoints,
    )
    ckpts = filter_checkpoints_by_update(ckpts, args.only_updates)
    if not ckpts and not args.include_baseline:
        raise FileNotFoundError(f"No update_*.pt checkpoints in {checkpoint_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"eval_sweep: run={args.run_name} device={device} ckpts={len(ckpts)} "
        f"nenvs={args.num_envs} out={output_dir}",
        flush=True,
    )
    model = make_model(args, device)
    env = make_env(args)
    rows: list[dict[str, Any]] = []
    start_time = time.time()
    try:
        if args.include_baseline:
            t0 = time.time()
            metrics = evaluate_checkpoint(model, env, args)
            row = {
                "run_name": args.run_name,
                "checkpoint": "baseline",
                "update": 0,
                "source_train_metrics": {},
                "eval_wall_s": round(time.time() - t0, 3),
                **metrics,
            }
            append_jsonl(results_path, row)
            rows.append(row)
            print("SMOLVLA_EVAL_RESULT " + json.dumps(row, sort_keys=True), flush=True)
        for ckpt in ckpts:
            t0 = time.time()
            checkpoint = load_trainable_delta(model, ckpt)
            metrics = evaluate_checkpoint(model, env, args)
            row = {
                "run_name": args.run_name,
                "checkpoint": str(ckpt),
                "update": int(checkpoint.get("update", checkpoint_update(ckpt))),
                "source_train_metrics": checkpoint.get("metrics", {}),
                "eval_wall_s": round(time.time() - t0, 3),
                **metrics,
            }
            append_jsonl(results_path, row)
            rows.append(row)
            print("SMOLVLA_EVAL_RESULT " + json.dumps(row, sort_keys=True), flush=True)

        best = max(rows, key=lambda item: (item["success_rate"], item["return_mean"]))
        summary = {
            "run_name": args.run_name,
            "num_checkpoints": len(rows),
            "total_wall_s": round(time.time() - start_time, 3),
            "best_update": best["update"],
            "best_success_rate": best["success_rate"],
            "best_return_mean": best["return_mean"],
            "last_update": rows[-1]["update"],
            "last_success_rate": rows[-1]["success_rate"],
            "last_return_mean": rows[-1]["return_mean"],
            "results_path": str(results_path),
        }
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print("SMOLVLA_EVAL_SWEEP_OK " + json.dumps(summary, sort_keys=True), flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
