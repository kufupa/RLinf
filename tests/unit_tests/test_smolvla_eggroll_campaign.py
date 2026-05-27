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
    assert module.classify_failure("run_smolvla_metaworld_eggroll.py ModuleNotFoundError") == "import_env"


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


def test_parse_population_list_uses_explicit_order() -> None:
    module = _load_campaign()

    assert module.parse_population_list("128,160,192,256") == [128, 160, 192, 256]
    assert module.parse_population_list("") == []


def test_build_worker_command_uses_one_allocation_not_nested_sbatch(tmp_path: Path) -> None:
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

    command = module.build_worker_command(config, cpus_per_worker=6)

    assert "scripts/run_smolvla_metaworld_eggroll.py" in command
    assert "sbatch" not in command
    assert "srun" not in command
    assert "--population-size" in command
    assert "16" in command
    assert "--target-name" in command
    assert config.target_name in command


def test_worker_environment_pins_gpu_and_repo_path(tmp_path: Path, monkeypatch) -> None:
    module = _load_campaign()
    monkeypatch.setenv("RLINF_ROOT", "/repo")
    config = module.WorkerConfig(
        population_size=4,
        envs_per_member=1,
        steps_per_update=120,
        chunk_len=5,
        target_name="target",
        output_dir=tmp_path / "worker",
        run_name="worker_pop4",
        gpu_slot="1",
    )

    env = module.worker_environment(config, cpus_per_worker=6)

    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["PYTHONPATH"].split(":")[0] == "/repo"
    assert env["OMP_NUM_THREADS"] == "6"


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
                "resources": {
                    "gpu_memory_used_mb_max": 6144.0,
                    "gpu_util_percent_mean": 23.5,
                    "rss_mb_max": 8192.0,
                },
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
    assert result.peak_gpu_memory_used_gb == 6.0
    assert result.gpu_util_percent_mean == 23.5
    assert result.rss_gb_max == 8.0
