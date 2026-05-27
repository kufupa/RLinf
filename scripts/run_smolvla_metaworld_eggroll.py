from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rlinf.algorithms.eggroll.batched_low_rank import batched_low_rank_module_patch
from rlinf.algorithms.eggroll.parallel_rollout import aggregate_member_scores
from rlinf.algorithms.eggroll.parallel_rollout import env_member_positions
from rlinf.algorithms.eggroll.parallel_rollout import iter_chunk_lengths
from rlinf.algorithms.eggroll.population import EggrollMember
from rlinf.algorithms.eggroll.population import EggrollPopulationConfig
from rlinf.algorithms.eggroll.population import aggregate_weighted_delta
from rlinf.algorithms.eggroll.population import sample_population
from rlinf.algorithms.eggroll.targets import find_eggroll_targets
from scripts.run_smolvla_metaworld_direct_ppo import DEFAULT_CHECKPOINT
from scripts.run_smolvla_metaworld_direct_ppo import TimingStats
from scripts.run_smolvla_metaworld_direct_ppo import make_env
from scripts.run_smolvla_metaworld_direct_ppo import make_model


@dataclass(frozen=True)
class EggrollRunConfig:
    population_size: int
    envs_per_member: int
    episodes_per_member: int
    steps_per_update: int
    chunk_len: int
    rank: int
    sigma: float
    learning_rate: float
    seed: int


@dataclass(frozen=True)
class EggrollUpdateResult:
    metrics: dict[str, Any]
    member_scores: list[EggrollMember]
    dense_update: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Standalone SmolVLA MetaWorld EGGROLL runner.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task-name", default="push-v3")
    parser.add_argument("--task-description", default="Push the puck to a goal")
    parser.add_argument("--target-name", default="")
    parser.add_argument("--population-size", type=int, default=2)
    parser.add_argument("--envs-per-member", type=int, default=1)
    parser.add_argument("--episodes-per-member", type=int, default=1)
    parser.add_argument("--total-updates", type=int, default=1)
    parser.add_argument("--steps-per-update", type=int, default=120)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1e-3)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-seed-base", type=int, default=2000)
    parser.add_argument("--reward-mode", default="sparse_success_delta")
    parser.add_argument("--use-rel-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--verify-batched-equivalence", action="store_true")
    return parser.parse_args()


def emit(path: Path, marker: str, payload: dict[str, Any]) -> None:
    line = json.dumps({"marker": marker, **payload}, sort_keys=True)
    print(f"{marker} {line}", flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def action_saturation_fraction(
    actions: Any,
    *,
    action_low: float,
    action_high: float,
    eps: float = 1e-6,
) -> float:
    if hasattr(actions, "detach"):
        arr = actions.detach().float().cpu().numpy()
    else:
        arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    return float(((arr <= action_low + eps) | (arr >= action_high - eps)).mean())


def cuda_payload(device: torch.device) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if device.type == "cuda":
        payload.update(
            {
                "device_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(device),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device),
                "max_memory_reserved": torch.cuda.max_memory_reserved(device),
            }
        )
    return payload


def reset_policy_replay_state(model: Any, *, seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model._last_reset_seeds = None
    policy_reset = getattr(getattr(model, "policy", None), "reset", None)
    if callable(policy_reset):
        policy_reset()


def verify_batched_equivalence(
    *,
    model: Any,
    obs: dict[str, Any],
    target_module: Any,
    population_config: EggrollPopulationConfig,
    env_to_member: np.ndarray,
    atol: float = 1e-5,
) -> dict[str, Any]:
    members = sample_population(population_config)
    with torch.no_grad():
        reset_policy_replay_state(model, seed=population_config.seed)
        with batched_low_rank_module_patch(
            target_module,
            deltas=[member.delta for member in members],
            member_positions=env_to_member,
            allow_flattened_batch=True,
        ):
            batched_actions, _ = model.predict_action_batch(obs, mode="eval", compute_values=False)

        serial_actions = torch.empty_like(batched_actions)
        for member_position, member in enumerate(members):
            rows = torch.as_tensor(env_to_member == member_position, dtype=torch.bool)
            if not bool(rows.any()):
                continue
            reset_policy_replay_state(model, seed=population_config.seed)
            with batched_low_rank_module_patch(
                target_module,
                deltas=[member.delta],
                member_positions=np.zeros_like(env_to_member, dtype=np.int64),
                allow_flattened_batch=True,
            ):
                member_actions, _ = model.predict_action_batch(obs, mode="eval", compute_values=False)
            serial_actions[rows] = member_actions[rows]

    diff = (serial_actions - batched_actions).abs()
    max_abs_diff = float(diff.max().detach().cpu()) if diff.numel() else 0.0
    return {
        "ok": bool(max_abs_diff <= atol),
        "max_abs_diff": max_abs_diff,
        "atol": float(atol),
        "shape": list(batched_actions.shape),
    }


def evaluate_eggroll_update(
    *,
    model: Any,
    env: Any,
    target_module: Any,
    target_name: str,
    config: EggrollRunConfig,
    device: torch.device,
) -> EggrollUpdateResult:
    expected_envs = config.population_size * config.envs_per_member
    if int(env.num_envs) != expected_envs:
        raise ValueError(f"env.num_envs={env.num_envs} does not match expected {expected_envs}")

    weight_shape = tuple(int(dim) for dim in target_module.weight.shape)
    population_config = EggrollPopulationConfig(
        population_size=config.population_size,
        out_features=weight_shape[0],
        in_features=weight_shape[1],
        rank=config.rank,
        sigma=config.sigma,
        seed=config.seed,
    )
    members = sample_population(population_config)
    env_to_member = env_member_positions(
        population_size=config.population_size,
        envs_per_member=config.envs_per_member,
    )
    timers = TimingStats(device)
    started = time.perf_counter()
    env_totals = np.zeros(expected_envs, dtype=np.float32)
    scalar_env_steps = 0
    saturation_values: list[float] = []

    for _episode in range(config.episodes_per_member):
        with timers.time("env_reset"):
            obs, _ = env.reset()
        for chunk_steps in iter_chunk_lengths(
            total_steps=config.steps_per_update,
            chunk_horizon=config.chunk_len,
        ):
            with timers.time("policy_predict_action_batch"):
                with torch.no_grad():
                    with batched_low_rank_module_patch(
                        target_module,
                        deltas=[member.delta for member in members],
                        member_positions=env_to_member,
                        allow_flattened_batch=True,
                    ):
                        actions, _ = model.predict_action_batch(
                            obs,
                            mode="eval",
                            compute_values=False,
                        )
            actions = actions[:, :chunk_steps, :]
            saturation_values.append(
                action_saturation_fraction(
                    actions,
                    action_low=model.action_low,
                    action_high=model.action_high,
                )
            )
            with timers.time("action_cpu_transfer"):
                chunk_actions = actions.detach().cpu().numpy()
            with timers.time("env_chunk_step"):
                obs_list, rewards, terms, truncs, infos_list = env.chunk_step(chunk_actions)

            final_info = infos_list[-1] if infos_list else {}
            valid_mask = final_info.get(
                "valid_action_mask",
                torch.ones_like(rewards, dtype=torch.bool),
            )
            valid_mask = valid_mask[:, :chunk_steps].detach().cpu().bool()
            reward_chunk = rewards[:, :chunk_steps].detach().cpu().float()
            env_totals += (reward_chunk * valid_mask.float()).sum(dim=1).numpy()
            scalar_env_steps += int(valid_mask.sum().item())
            obs = obs_list[chunk_steps - 1]
            if bool(final_info.get("all_rows_terminal", False)) or bool((terms | truncs).all()):
                break

    scored_members = aggregate_member_scores(
        members,
        env_totals / max(config.episodes_per_member, 1),
        env_to_member,
    )
    dense_update = aggregate_weighted_delta(scored_members, learning_rate=config.learning_rate)
    with torch.no_grad():
        target_module.weight.add_(
            torch.as_tensor(dense_update, dtype=target_module.weight.dtype, device=target_module.weight.device)
        )

    elapsed_s = time.perf_counter() - started
    member_episodes = config.population_size * config.envs_per_member * config.episodes_per_member
    metrics = {
        "target_name": target_name,
        "population_size": config.population_size,
        "envs_per_member": config.envs_per_member,
        "episodes_per_member": config.episodes_per_member,
        "population_seed": config.seed,
        "member_episodes": member_episodes,
        "steps_per_update": config.steps_per_update,
        "chunk_len": config.chunk_len,
        "scalar_env_steps": scalar_env_steps,
        "elapsed_s": round(elapsed_s, 6),
        "seconds_per_member_episode": elapsed_s / max(member_episodes, 1),
        "env_steps_per_second": scalar_env_steps / max(elapsed_s, 1e-12),
        "action_saturation_fraction": float(np.mean(saturation_values)) if saturation_values else 0.0,
        "timing": timers.payload(),
        "hardware": cuda_payload(device),
    }
    return EggrollUpdateResult(metrics=metrics, member_scores=scored_members, dense_update=dense_update)


def _resolve_target(model: Any, target_name: str):
    policy_model = getattr(model.policy, "model", model.policy)
    modules = dict(policy_model.named_modules())
    if target_name:
        if target_name not in modules:
            raise ValueError(f"target {target_name!r} not found")
        return target_name, modules[target_name]
    target = find_eggroll_targets(policy_model)[0]
    return target.name, target.module


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8")
    args.num_envs = int(args.population_size) * int(args.envs_per_member)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = None
    try:
        model = make_model(args, device)
        model.eval()
        env = make_env(args)
        target_name, target_module = _resolve_target(model, args.target_name)
        run_config = EggrollRunConfig(
            population_size=args.population_size,
            envs_per_member=args.envs_per_member,
            episodes_per_member=args.episodes_per_member,
            steps_per_update=args.steps_per_update,
            chunk_len=args.chunk_len,
            rank=args.rank,
            sigma=args.sigma,
            learning_rate=args.learning_rate,
            seed=args.seed,
        )
        if args.verify_batched_equivalence:
            obs, _ = env.reset()
            payload = verify_batched_equivalence(
                model=model,
                obs=obs,
                target_module=target_module,
                population_config=EggrollPopulationConfig(
                    population_size=args.population_size,
                    out_features=int(target_module.weight.shape[0]),
                    in_features=int(target_module.weight.shape[1]),
                    rank=args.rank,
                    sigma=args.sigma,
                    seed=args.seed,
                ),
                env_to_member=env_member_positions(
                    population_size=args.population_size,
                    envs_per_member=args.envs_per_member,
                ),
            )
            emit(metrics_path, "SMOLVLA_EGGROLL_EQUIV_OK" if payload["ok"] else "SMOLVLA_EGGROLL_EQUIV_FAILED", payload)
            if not payload["ok"]:
                return 3

        for update in range(1, args.total_updates + 1):
            update_config = replace(run_config, seed=run_config.seed + update - 1)
            result = evaluate_eggroll_update(
                model=model,
                env=env,
                target_module=target_module,
                target_name=target_name,
                config=update_config,
                device=device,
            )
            np.save(output_dir / f"dense_update_{update:06d}.npy", result.dense_update)
            emit(
                metrics_path,
                "SMOLVLA_EGGROLL_UPDATE",
                {
                    "run_name": args.run_name,
                    "update": update,
                    **result.metrics,
                    "member_scores": [
                        {"index": member.index, "seed": member.seed, "score": member.score}
                        for member in result.member_scores
                    ],
                },
            )
        np.save(output_dir / "target_weight_final.npy", target_module.weight.detach().float().cpu().numpy())
        emit(
            metrics_path,
            "SMOLVLA_EGGROLL_TARGET_WEIGHT_SAVED",
            {
                "path": str(output_dir / "target_weight_final.npy"),
                "target_name": target_name,
            },
        )
        emit(metrics_path, "SMOLVLA_EGGROLL_RUN_OK", {"run_name": args.run_name, "updates": args.total_updates})
        return 0
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
