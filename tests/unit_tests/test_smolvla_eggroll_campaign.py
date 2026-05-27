from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_campaign():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_smolvla_eggroll_campaign.py"
    spec = importlib.util.spec_from_file_location("run_smolvla_eggroll_campaign", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load campaign script from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_classify_failure_detects_common_root_causes() -> None:
    module = _load_campaign()

    assert module.classify_failure("CUDA out of memory while allocating") == "cuda_oom"
    assert module.classify_failure("SMOLVLA_EGGROLL_EQUIV_FAILED max_abs_diff") == "equivalence"
    assert module.classify_failure("No usable EGGROLL target survived probe") == "target_probe"
    assert module.classify_failure("mujoco.FatalError env crashed") == "env"
    assert module.classify_failure("ModuleNotFoundError: No module named ray") == "import_env"


def test_brave_population_plan_jumps_when_memory_headroom_is_large() -> None:
    module = _load_campaign()
    history = [
        module.WorkerResult(
            population_size=2,
            envs_per_member=1,
            output_dir="/tmp/pop2",
            ok=True,
            seconds_per_member_episode=2.5,
            peak_vram_gb=4.0,
            failure_kind=None,
        )
    ]

    candidates = module.next_population_candidates(history, max_population_soft=96)

    assert candidates[:3] == [16, 32, 48]
    assert 96 in candidates


def test_build_worker_command_uses_srun_not_nested_sbatch(tmp_path: Path) -> None:
    module = _load_campaign()
    config = module.WorkerConfig(
        population_size=16,
        envs_per_member=1,
        steps_per_update=120,
        chunk_len=5,
        target_name="vlm_with_expert.lm_expert.layers.1.mlp.down_proj",
        output_dir=tmp_path / "worker",
        run_name="worker_pop16",
    )

    command = module.build_worker_command(config, cpus_per_worker=14)

    assert command[:5] == ["srun", "--exclusive", "--gres=gpu:1", "--cpus-per-task=14", "--ntasks=1"]
    assert "sbatch" not in command
    assert "--population-size" in command
    assert "16" in command
    assert "--target-name" in command
    assert config.target_name in command


def test_parse_worker_metrics_reads_required_markers(tmp_path: Path) -> None:
    module = _load_campaign()
    run_dir = tmp_path / "worker"
    run_dir.mkdir()
    (run_dir / "metrics.jsonl").write_text(
        json.dumps(
            {
                "marker": "SMOLVLA_EGGROLL_UPDATE",
                "population_size": 16,
                "envs_per_member": 1,
                "seconds_per_member_episode": 1.2,
                "hardware": {"max_memory_allocated": 8 * 1024**3},
            }
        )
        + "\n"
        + json.dumps({"marker": "SMOLVLA_EGGROLL_RUN_OK"})
        + "\n",
        encoding="utf-8",
    )

    result = module.parse_worker_metrics(run_dir)

    assert result.ok
    assert result.population_size == 16
    assert result.seconds_per_member_episode == 1.2
    assert result.peak_vram_gb == 8.0
