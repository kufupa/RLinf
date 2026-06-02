from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from torch_load_mmap import torch_load_mmap_with_mode

from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv
from rlinf.models.embodiment.smolvla import get_model


DEFAULT_CHECKPOINT = (
    "jadechoghari/smolvla_metaworld"
)


# region agent log
_AGENT_DEBUG_LOG_PATH = Path("/vol/bitbucket/aa6622/.cursor/debug-9943fa.log")
_AGENT_DEBUG_SESSION_ID = "9943fa"


def _agent_proc_io() -> dict[str, int | str]:
    try:
        values: dict[str, int | str] = {}
        for line in Path("/proc/self/io").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            if key in {"rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes"}:
                values[key] = int(raw.strip())
        return values
    except Exception as exc:  # pragma: no cover - debug-only best effort
        return {"error": type(exc).__name__}


def _agent_tensor_summary(obj: Any) -> dict[str, int]:
    tensors = 0
    bytes_ = 0
    stack = [obj]
    while stack:
        item = stack.pop()
        if torch.is_tensor(item):
            tensors += 1
            bytes_ += item.numel() * item.element_size()
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
    return {"tensors": tensors, "bytes": bytes_}


def _agent_log(hypothesis_id: str, location: str, message: str, data: dict[str, Any]) -> None:
    payload = {
        "sessionId": _AGENT_DEBUG_SESSION_ID,
        "id": f"log_{time.time_ns()}",
        "timestamp": int(time.time() * 1000),
        "runId": os.environ.get("AGENT_DEBUG_RUN_ID", os.environ.get("SLURM_JOB_ID", "local")),
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
    }
    try:
        with _AGENT_DEBUG_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
    except Exception:
        pass
# endregion


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SmolVLA PPO checkpoints.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task-name", default="push-v3")
    parser.add_argument("--task-description", default="Push the puck to a goal")
    parser.add_argument("--num-envs", type=int, default=25)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=0,
        help="Total held-out episodes (must divide num-envs). 0 => one wave of num-envs.",
    )
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


def resolve_checkpoint_dir(checkpoint_dir: Path) -> Path:
    checkpoint_dir = checkpoint_dir.resolve()
    if checkpoint_dir.name == "checkpoints":
        sibling_eval = checkpoint_dir.parent / "checkpoints_eval"
        if sibling_eval.is_dir():
            return sibling_eval
    nested_eval = checkpoint_dir / "checkpoints_eval"
    if nested_eval.is_dir():
        return nested_eval
    return checkpoint_dir


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
    # region agent log
    load_t0 = time.perf_counter()
    io_before = _agent_proc_io()
    try:
        stat = ckpt_path.stat()
        ckpt_info = {
            "checkpoint": str(ckpt_path),
            "size_bytes": stat.st_size,
            "mtime": stat.st_mtime,
            "io_before": io_before,
        }
    except OSError as exc:
        ckpt_info = {"checkpoint": str(ckpt_path), "stat_error": type(exc).__name__, "io_before": io_before}
    _agent_log(
        "H1,H2",
        "eval_smolvla_metaworld_ckpt_sweep.py:load_trainable_delta:start",
        "checkpoint load start",
        ckpt_info,
    )
    # endregion
    checkpoint, load_mode = torch_load_mmap_with_mode(
        ckpt_path,
        map_location="cpu",
        weights_only=False,
    )
    # region agent log
    load_elapsed_s = time.perf_counter() - load_t0
    policy_state = checkpoint.get("policy_state_dict")
    optimizer_state = checkpoint.get("optimizer_state_dict")
    trainable_state = checkpoint.get("trainable_model")
    _agent_log(
        "H1,H2",
        "eval_smolvla_metaworld_ckpt_sweep.py:load_trainable_delta:torch_load_done",
        "torch.load finished",
        {
            "checkpoint": str(ckpt_path),
            "elapsed_s": load_elapsed_s,
            "load_mode": load_mode,
            "payload_keys": sorted(str(key) for key in checkpoint.keys()),
            "policy_state": _agent_tensor_summary(policy_state),
            "optimizer_state": _agent_tensor_summary(optimizer_state),
            "trainable_model": _agent_tensor_summary(trainable_state),
            "io_after": _agent_proc_io(),
        },
    )
    copy_t0 = time.perf_counter()
    # endregion
    trainable = checkpoint.get("trainable_model")
    if isinstance(trainable, dict):
        named_params = dict(model.named_parameters())
        with torch.no_grad():
            for name, tensor in trainable.items():
                if name not in named_params:
                    raise KeyError(f"{ckpt_path} has unknown param {name}")
                param = named_params[name]
                param.copy_(tensor.to(device=param.device, dtype=param.dtype))
        policy = getattr(model, "policy", model)
        policy_reset = getattr(policy, "reset", None)
        if callable(policy_reset):
            policy_reset()
        # region agent log
        _agent_log(
            "H3",
            "eval_smolvla_metaworld_ckpt_sweep.py:load_trainable_delta:trainable_copy_done",
            "trainable state copied into model",
            {"checkpoint": str(ckpt_path), "elapsed_s": time.perf_counter() - copy_t0},
        )
        # endregion
        return checkpoint

    policy_state = checkpoint.get("policy_state_dict")
    if isinstance(policy_state, dict):
        policy = getattr(model, "policy", model)
        policy.load_state_dict(policy_state, strict=False)
        policy_reset = getattr(policy, "reset", None)
        if callable(policy_reset):
            policy_reset()
        # region agent log
        _agent_log(
            "H3",
            "eval_smolvla_metaworld_ckpt_sweep.py:load_trainable_delta:policy_copy_done",
            "policy state copied into model",
            {"checkpoint": str(ckpt_path), "elapsed_s": time.perf_counter() - copy_t0},
        )
        # endregion
        return checkpoint

    raise KeyError(f"{ckpt_path} missing trainable_model or policy_state_dict (GRPO export)")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def eval_seed_list(args: argparse.Namespace) -> list[int]:
    return [args.seed_base + idx for idx in range(args.num_envs)]


@torch.no_grad()
def evaluate_many_episodes(
    model: torch.nn.Module,
    env: SmolVLAMetaWorldEnv,
    args: argparse.Namespace,
    num_episodes: int,
) -> dict[str, Any]:
    if num_episodes < 1:
        raise ValueError("num_episodes must be >= 1")
    num_envs = int(args.num_envs)
    if num_episodes % num_envs != 0:
        raise ValueError(
            f"num_episodes={num_episodes} must be divisible by num_envs={num_envs}"
        )
    waves = num_episodes // num_envs
    chunks = args.max_episode_steps // args.chunk_len
    per_episode_success: list[bool] = []
    per_episode_return: list[float] = []
    all_seeds: list[int] = []

    for wave in range(waves):
        seeds = [args.seed_base + wave * num_envs + idx for idx in range(num_envs)]
        all_seeds.extend(seeds)
        obs, _ = env.reset_many(seeds)
        reward_sum = torch.zeros(num_envs, dtype=torch.float32)
        success = torch.zeros(num_envs, dtype=torch.bool)
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
            success |= terms.any(dim=1)
            final_info = infos_list[-1].get("episode", {})
            if "success_once" in final_info:
                success |= final_info["success_once"].bool()
            obs = obs_list[-1]
        per_episode_success.extend(success.cpu().tolist())
        per_episode_return.extend(reward_sum.cpu().tolist())

    success_arr = np.asarray(per_episode_success, dtype=np.float32)
    return_arr = np.asarray(per_episode_return, dtype=np.float32)
    return {
        "success_rate": float(success_arr.mean()),
        "success_count": int(success_arr.sum()),
        "return_mean": float(return_arr.mean()),
        "return_max": float(return_arr.max()),
        "episode_len_mean": float(args.max_episode_steps),
        "num_episodes": num_episodes,
        "eval_seeds": all_seeds,
        "per_env_success": per_episode_success,
        "success_seeds": [seed for seed, ok in zip(all_seeds, per_episode_success) if ok],
    }


def run_eval_for_checkpoint(
    model: torch.nn.Module,
    env: SmolVLAMetaWorldEnv,
    args: argparse.Namespace,
) -> dict[str, Any]:
    num_episodes = int(args.num_episodes) if int(args.num_episodes) > 0 else int(args.num_envs)
    if num_episodes > int(args.num_envs):
        return evaluate_many_episodes(model, env, args, num_episodes)
    return evaluate_checkpoint(model, env, args)


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
    # region agent log
    os.environ["AGENT_DEBUG_RUN_ID"] = f"{args.run_name}:{os.environ.get('SLURM_JOB_ID', 'local')}:{os.getpid()}"
    # endregion
    requested_checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir = resolve_checkpoint_dir(requested_checkpoint_dir)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    config_path = output_dir / "config.json"
    summary_path = output_dir / "summary.json"
    config = vars(args).copy()
    config["resolved_checkpoint_dir"] = str(checkpoint_dir)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    ckpts = list_checkpoints(
        checkpoint_dir,
        stride=max(1, args.checkpoint_stride),
        max_checkpoints=args.max_checkpoints,
    )
    ckpts = filter_checkpoints_by_update(ckpts, args.only_updates)
    if not ckpts and not args.include_baseline:
        raise FileNotFoundError(f"No update_*.pt checkpoints in {checkpoint_dir}")

    num_episodes = int(args.num_episodes) if int(args.num_episodes) > 0 else int(args.num_envs)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # region agent log
    _agent_log(
        "H2,H4,H5",
        "eval_smolvla_metaworld_ckpt_sweep.py:main:start",
        "eval sweep process start",
        {
            "run_name": args.run_name,
            "requested_checkpoint_dir": str(requested_checkpoint_dir),
            "resolved_checkpoint_dir": str(checkpoint_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "output_dir": str(output_dir),
            "num_checkpoints": len(ckpts),
            "include_baseline": bool(args.include_baseline),
            "only_updates": args.only_updates,
            "num_envs": int(args.num_envs),
            "num_episodes": num_episodes,
            "device": str(device),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "hostname": os.uname().nodename,
        },
    )
    model_t0 = time.perf_counter()
    # endregion
    print(
        f"eval_sweep: run={args.run_name} device={device} ckpts={len(ckpts)} "
        f"nenvs={args.num_envs} num_episodes={num_episodes} seed_base={args.seed_base} "
        f"out={output_dir}",
        flush=True,
    )
    model = make_model(args, device)
    # region agent log
    _agent_log(
        "H5",
        "eval_smolvla_metaworld_ckpt_sweep.py:main:model_ready",
        "base model loaded",
        {"elapsed_s": time.perf_counter() - model_t0, "io_after": _agent_proc_io()},
    )
    env_t0 = time.perf_counter()
    # endregion
    env = make_env(args)
    # region agent log
    _agent_log(
        "H4",
        "eval_smolvla_metaworld_ckpt_sweep.py:main:env_ready",
        "eval env constructed",
        {"elapsed_s": time.perf_counter() - env_t0},
    )
    # endregion
    rows: list[dict[str, Any]] = []
    start_time = time.time()
    try:
        if args.include_baseline:
            t0 = time.time()
            # region agent log
            _agent_log(
                "H4",
                "eval_smolvla_metaworld_ckpt_sweep.py:main:baseline_eval_start",
                "baseline rollout start",
                {"run_name": args.run_name, "num_episodes": num_episodes},
            )
            # endregion
            metrics = run_eval_for_checkpoint(model, env, args)
            # region agent log
            _agent_log(
                "H4",
                "eval_smolvla_metaworld_ckpt_sweep.py:main:baseline_eval_done",
                "baseline rollout done",
                {"elapsed_s": time.time() - t0, "success_rate": metrics.get("success_rate")},
            )
            # endregion
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
            # region agent log
            eval_t0 = time.perf_counter()
            _agent_log(
                "H4",
                "eval_smolvla_metaworld_ckpt_sweep.py:main:checkpoint_eval_start",
                "checkpoint rollout start",
                {"checkpoint": str(ckpt), "update": int(checkpoint.get("update", checkpoint_update(ckpt)))},
            )
            # endregion
            metrics = run_eval_for_checkpoint(model, env, args)
            # region agent log
            _agent_log(
                "H4",
                "eval_smolvla_metaworld_ckpt_sweep.py:main:checkpoint_eval_done",
                "checkpoint rollout done",
                {
                    "checkpoint": str(ckpt),
                    "elapsed_s": time.perf_counter() - eval_t0,
                    "success_rate": metrics.get("success_rate"),
                },
            )
            # endregion
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
