from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from rlinf.algorithms.eggroll.low_rank import temporary_low_rank_perturbation
from rlinf.algorithms.eggroll.targets import TargetProbeResult
from rlinf.algorithms.eggroll.targets import action_effect_summary
from rlinf.algorithms.eggroll.targets import find_eggroll_targets
from rlinf.algorithms.eggroll.targets import make_probe_delta
from rlinf.algorithms.eggroll.targets import record_module_io
from rlinf.algorithms.eggroll.targets import select_best_probe_result
from scripts.run_smolvla_metaworld_direct_ppo import DEFAULT_CHECKPOINT
from scripts.run_smolvla_metaworld_direct_ppo import TimingStats
from scripts.run_smolvla_metaworld_direct_ppo import make_env
from scripts.run_smolvla_metaworld_direct_ppo import make_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe SmolVLA modules for EGGROLL targets.")
    parser.add_argument("--run-name", default="smolvla_eggroll_target_probe")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-path", default=os.environ.get("SMOLVLA_CHECKPOINT", DEFAULT_CHECKPOINT))
    parser.add_argument("--task-name", default="push-v3")
    parser.add_argument("--task-description", default="Push the puck to a goal")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--max-episode-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reset-seed-base", type=int, default=2000)
    parser.add_argument("--reward-mode", default="sparse_success_delta")
    parser.add_argument("--use-rel-reward", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--scale", type=float, default=1e-3)
    parser.add_argument("--max-targets", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def emit(path: Path, marker: str, payload: dict[str, Any]) -> None:
    line = json.dumps({"marker": marker, **payload}, sort_keys=True)
    print(f"{marker} {line}", flush=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "target_probe.jsonl"
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    timers = TimingStats(device)
    started = time.perf_counter()
    env = None

    emit(
        out_path,
        "SMOLVLA_EGGROLL_TARGET_PROBE_START",
        {
            "run_name": args.run_name,
            "args": vars(args),
            "hardware": cuda_payload(device),
        },
    )
    try:
        with timers.time("make_env"):
            env = make_env(args)
        with timers.time("make_model"):
            model = make_model(args, device)
        model.eval()
        with timers.time("env_reset"):
            obs, _ = env.reset()
        with timers.time("base_predict_action_batch"):
            base_actions, _ = model.predict_action_batch(obs, mode="eval", compute_values=False)

        policy_model = getattr(model.policy, "model", model.policy)
        targets = find_eggroll_targets(policy_model)[: args.max_targets]
        emit(
            out_path,
            "SMOLVLA_EGGROLL_TARGET_CANDIDATES",
            {
                "count": len(targets),
                "targets": [
                    {"name": target.name, "shape": list(target.shape), "param_count": target.param_count}
                    for target in targets
                ],
            },
        )

        results: list[TargetProbeResult] = []
        for index, target in enumerate(targets):
            delta = make_probe_delta(
                target,
                rank=args.rank,
                scale=args.scale,
                seed=args.seed + 1009 * (index + 1),
            )
            with record_module_io(target.module) as telemetry:
                with temporary_low_rank_perturbation(target.module, delta):
                    with timers.time("perturbed_predict_action_batch"):
                        perturbed_actions, _ = model.predict_action_batch(
                            obs,
                            mode="eval",
                            compute_values=False,
                        )
            summary = action_effect_summary(
                base_actions,
                perturbed_actions,
                action_low=model.action_low,
                action_high=model.action_high,
            )
            result = TargetProbeResult(
                name=target.name,
                shape=target.shape,
                max_abs_diff=float(summary["max_abs_diff"]),
                mean_abs_diff=float(summary["mean_abs_diff"]),
                finite=bool(summary["finite"]),
                saturation_fraction=float(summary["saturation_fraction"]),
                call_count=int(telemetry.call_count),
                input_shapes=list(telemetry.input_shapes),
                output_shapes=list(telemetry.output_shapes),
            )
            results.append(result)
            emit(
                out_path,
                "SMOLVLA_EGGROLL_TARGET_CANDIDATE",
                {
                    **result.payload(),
                    "per_horizon_max_abs_diff": summary["per_horizon_max_abs_diff"],
                    "telemetry": telemetry.payload(),
                },
            )

        selected = select_best_probe_result(results)
        emit(
            out_path,
            "SMOLVLA_EGGROLL_TARGET_SELECTED",
            {
                **selected.payload(),
                "elapsed_s": round(time.perf_counter() - started, 6),
                "timing": timers.payload(),
                "hardware": cuda_payload(device),
            },
        )
        return 0
    except Exception as exc:
        emit(
            out_path,
            "SMOLVLA_EGGROLL_TARGET_PROBE_FAILED",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": round(time.perf_counter() - started, 6),
                "timing": timers.payload(),
                "hardware": cuda_payload(device),
            },
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    raise SystemExit(main())
