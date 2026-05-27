from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PI05_SECONDS_PER_MEMBER_EPISODE = 10.18


@dataclass(frozen=True)
class WorkerConfig:
    population_size: int
    envs_per_member: int
    steps_per_update: int
    chunk_len: int
    target_name: str
    output_dir: Path
    run_name: str
    gpu_slot: str = "0"
    episodes_per_member: int = 1
    rank: int = 1
    sigma: float = 1e-4
    learning_rate: float = 1e-3
    seed: int = 0
    verify_batched_equivalence: bool = False


@dataclass(frozen=True)
class WorkerResult:
    population_size: int
    envs_per_member: int
    output_dir: str
    ok: bool
    seconds_per_member_episode: float | None
    peak_vram_gb: float | None
    failure_kind: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Adaptive two-A30 SmolVLA EGGROLL campaign controller.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--target-name", required=True)
    parser.add_argument("--gpu-slots", default="0,1")
    parser.add_argument("--max-population-soft", type=int, default=96)
    parser.add_argument("--steps-per-update", type=int, default=120)
    parser.add_argument("--chunk-len", type=int, default=5)
    parser.add_argument("--envs-per-member", type=int, default=1)
    parser.add_argument("--episodes-per-member", type=int, default=1)
    parser.add_argument("--max-runtime-s", type=float, default=6 * 3600)
    parser.add_argument("--cpus-per-worker", type=int, default=6)
    return parser.parse_args()


def classify_failure(text: str) -> str:
    lower = text.lower()
    if "out of memory" in lower or "cuda oom" in lower:
        return "cuda_oom"
    if "smolvla_eggroll_equiv_failed" in lower or "max_abs_diff" in lower and "equiv" in lower:
        return "equivalence"
    if "no usable eggroll target" in lower or "target_selected" in lower and "missing" in lower:
        return "target_probe"
    if "modulenotfounderror" in lower or "importerror" in lower:
        return "import_env"
    if "slurm" in lower or "srun:" in lower:
        return "slurm"
    if "mujoco" in lower or "metaworld" in lower or "env crashed" in lower:
        return "env"
    return "unknown"


def next_population_candidates(
    history: list[WorkerResult],
    *,
    max_population_soft: int,
) -> list[int]:
    tried = {result.population_size for result in history}
    failures = [result.population_size for result in history if not result.ok]
    goods = [result for result in history if result.ok]
    if not goods:
        return [pop for pop in [4, 8, 16] if pop <= max_population_soft and pop not in tried]

    best = min(goods, key=lambda result: result.seconds_per_member_episode or float("inf"))
    if failures:
        cap = min(failures)
        candidates = [pop for pop in [best.population_size + 4, (best.population_size + cap) // 2] if pop < cap]
    elif (best.peak_vram_gb or 0.0) < 10.0:
        candidates = [16, 32, 48, 64, 80, 96]
    elif (best.peak_vram_gb or 0.0) < 18.0:
        candidates = [best.population_size * 2, best.population_size + 16, best.population_size + 32]
    else:
        candidates = [best.population_size + 8, best.population_size + 16]
    return sorted({pop for pop in candidates if pop > 0 and pop <= max_population_soft and pop not in tried})


def build_worker_command(config: WorkerConfig, *, cpus_per_worker: int) -> list[str]:
    python_bin = os.environ.get("PYTHON_BIN", "python")
    return [
        python_bin,
        "scripts/run_smolvla_metaworld_eggroll.py",
        "--run-name",
        config.run_name,
        "--output-dir",
        str(config.output_dir),
        "--target-name",
        config.target_name,
        "--population-size",
        str(config.population_size),
        "--envs-per-member",
        str(config.envs_per_member),
        "--episodes-per-member",
        str(config.episodes_per_member),
        "--total-updates",
        "1",
        "--steps-per-update",
        str(config.steps_per_update),
        "--chunk-len",
        str(config.chunk_len),
        "--max-episode-steps",
        "120",
        "--rank",
        str(config.rank),
        "--sigma",
        str(config.sigma),
        "--learning-rate",
        str(config.learning_rate),
        "--seed",
        str(config.seed),
    ]


def add_verify_flag(command: list[str], config: WorkerConfig) -> list[str]:
    if config.verify_batched_equivalence:
        return [*command, "--verify-batched-equivalence"]
    return command


def worker_environment(config: WorkerConfig, *, cpus_per_worker: int) -> dict[str, str]:
    env = os.environ.copy()
    root = env.get("RLINF_ROOT", str(Path(__file__).resolve().parents[1]))
    project_src = env.get("PROJECT_SRC")
    pythonpath_parts = [root]
    if project_src:
        pythonpath_parts.append(project_src)
    if env.get("PYTHONPATH"):
        pythonpath_parts.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = ":".join(pythonpath_parts)
    env["CUDA_VISIBLE_DEVICES"] = config.gpu_slot
    env["OMP_NUM_THREADS"] = str(max(1, cpus_per_worker))
    env["MKL_NUM_THREADS"] = str(max(1, cpus_per_worker))
    env["OPENBLAS_NUM_THREADS"] = str(max(1, cpus_per_worker))
    return env


def parse_worker_metrics(run_dir: Path) -> WorkerResult:
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return WorkerResult(0, 0, str(run_dir), False, None, None, "missing_metrics")
    update: dict[str, Any] | None = None
    run_ok = False
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("marker") == "SMOLVLA_EGGROLL_UPDATE":
            update = payload
        if payload.get("marker") == "SMOLVLA_EGGROLL_RUN_OK":
            run_ok = True
    if update is None or not run_ok:
        return WorkerResult(0, 0, str(run_dir), False, None, None, "missing_success_marker")
    hardware = update.get("hardware", {})
    peak_bytes = max(
        int(hardware.get("max_memory_allocated", 0) or 0),
        int(hardware.get("max_memory_reserved", 0) or 0),
    )
    return WorkerResult(
        population_size=int(update["population_size"]),
        envs_per_member=int(update["envs_per_member"]),
        output_dir=str(run_dir),
        ok=True,
        seconds_per_member_episode=float(update["seconds_per_member_episode"]),
        peak_vram_gb=round(peak_bytes / 1024**3, 3) if peak_bytes else None,
        failure_kind=None,
    )


def append_handoff(path: Path, line: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def run_worker(config: WorkerConfig, *, cpus_per_worker: int) -> WorkerResult:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    command = add_verify_flag(build_worker_command(config, cpus_per_worker=cpus_per_worker), config)
    stdout_path = config.output_dir / "worker_stdout.log"
    with stdout_path.open("w", encoding="utf-8") as stdout:
        process = subprocess.run(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
            cwd=os.environ.get("RLINF_ROOT"),
            env=worker_environment(config, cpus_per_worker=cpus_per_worker),
        )
    if process.returncode != 0:
        text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
        return WorkerResult(
            population_size=config.population_size,
            envs_per_member=config.envs_per_member,
            output_dir=str(config.output_dir),
            ok=False,
            seconds_per_member_episode=None,
            peak_vram_gb=None,
            failure_kind=classify_failure(text),
        )
    return parse_worker_metrics(config.output_dir)


def run_worker_wave(configs: list[WorkerConfig], *, cpus_per_worker: int) -> list[WorkerResult]:
    launched = []
    for config in configs:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        command = add_verify_flag(build_worker_command(config, cpus_per_worker=cpus_per_worker), config)
        stdout_path = config.output_dir / "worker_stdout.log"
        stdout = stdout_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            stdout=stdout,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=os.environ.get("RLINF_ROOT"),
            env=worker_environment(config, cpus_per_worker=cpus_per_worker),
        )
        launched.append((config, process, stdout, stdout_path))

    results: list[WorkerResult] = []
    for config, process, stdout, stdout_path in launched:
        returncode = process.wait()
        stdout.close()
        if returncode != 0:
            text = stdout_path.read_text(encoding="utf-8", errors="replace") if stdout_path.exists() else ""
            results.append(
                WorkerResult(
                    population_size=config.population_size,
                    envs_per_member=config.envs_per_member,
                    output_dir=str(config.output_dir),
                    ok=False,
                    seconds_per_member_episode=None,
                    peak_vram_gb=None,
                    failure_kind=classify_failure(text),
                )
            )
        else:
            results.append(parse_worker_metrics(config.output_dir))
    return results


def main() -> int:
    args = parse_args()
    started = time.time()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    handoff = out_dir / "overnight_handoff.md"
    summary_path = out_dir / "campaign_summary.json"
    history: list[WorkerResult] = []
    pending = [4, 16, 32, 48, 64, 80, 96]
    pending = [pop for pop in pending if pop <= args.max_population_soft]
    gpu_slots = [slot.strip() for slot in args.gpu_slots.split(",") if slot.strip()]
    if not gpu_slots:
        gpu_slots = ["0"]
    append_handoff(handoff, f"# SmolVLA EGGROLL Campaign\n\nTarget: `{args.target_name}`\n")

    while pending and time.time() - started < args.max_runtime_s:
        wave = pending[:2]
        pending = pending[2:]
        append_handoff(handoff, f"## Wave populations {wave}")
        configs = [
            WorkerConfig(
                population_size=pop,
                envs_per_member=args.envs_per_member,
                episodes_per_member=args.episodes_per_member,
                steps_per_update=args.steps_per_update,
                chunk_len=args.chunk_len,
                target_name=args.target_name,
                output_dir=out_dir / f"pop_{pop:04d}",
                run_name=f"smolvla_eggroll_pop_{pop:04d}",
                gpu_slot=gpu_slots[index % len(gpu_slots)],
                seed=1000 + pop,
                verify_batched_equivalence=(not history and index == 0),
            )
            for index, pop in enumerate(wave)
        ]
        wave_results = run_worker_wave(configs, cpus_per_worker=args.cpus_per_worker)
        for result in wave_results:
            history.append(result)
            append_handoff(
                handoff,
                f"- `{result.population_size}`: `{json.dumps(asdict(result), sort_keys=True)}`",
            )
        if any(result.ok for result in wave_results):
            pending = next_population_candidates(history, max_population_soft=args.max_population_soft)
        elif any(result.failure_kind == "cuda_oom" for result in wave_results):
            pending = next_population_candidates(history, max_population_soft=args.max_population_soft)
        if not pending:
            break

    ok_results = [result for result in history if result.ok and result.seconds_per_member_episode is not None]
    best = min(ok_results, key=lambda result: result.seconds_per_member_episode) if ok_results else None
    summary = {
        "history": [asdict(result) for result in history],
        "best": asdict(best) if best is not None else None,
        "pi05_seconds_per_member_episode": PI05_SECONDS_PER_MEMBER_EPISODE,
        "beats_pi05": bool(best and best.seconds_per_member_episode < PI05_SECONDS_PER_MEMBER_EPISODE),
        "elapsed_s": round(time.time() - started, 6),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    append_handoff(handoff, f"## Summary\n`{json.dumps(summary, sort_keys=True)}`")
    print("SMOLVLA_EGGROLL_CAMPAIGN_OK " + json.dumps(summary, sort_keys=True), flush=True)
    return 0 if best is not None else 2


if __name__ == "__main__":
    raise SystemExit(main())
