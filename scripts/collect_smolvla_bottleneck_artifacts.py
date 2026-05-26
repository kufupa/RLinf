from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs" / "pbs"


RUNS = {
    "direct_ppo_4env_sparse": {
        "job_id": "2835191.pbs-7",
        "log": "smolvla-direct-s2-sparse-2835191.pbs-7.driver.log",
        "pbs": "scripts/pbs/smolvla_metaworld_direct_ppo_stage2_sparse.pbs",
    },
    "direct_ppo_4env_dense": {
        "job_id": "2835192.pbs-7",
        "log": "smolvla-direct-s2-dense-2835192.pbs-7.driver.log",
        "pbs": "scripts/pbs/smolvla_metaworld_direct_ppo_stage2_dense.pbs",
    },
    "direct_ppo_8env_sparse": {
        "job_id": "2835766.pbs-7",
        "log": "smolvla-direct-s3-sparse-2835766.pbs-7.driver.log",
        "pbs": "scripts/pbs/smolvla_metaworld_direct_ppo_stage3_sparse_scale.pbs",
    },
    "direct_ppo_4env_lr3e7": {
        "job_id": "2836701.pbs-7",
        "log": "smolvla-direct-s3b-lr3e7-2836701.pbs-7.driver.log",
        "pbs": "scripts/pbs/smolvla_metaworld_direct_ppo_stage3b_sparse_lr3e7.pbs",
    },
    "native_rlinf_4env_ppo": {
        "job_id": "2842447.pbs-7",
        "log": "native-metaworld-learning-4env-2842447.pbs-7.driver.log",
        "pbs": "scripts/pbs/native_metaworld_pushv3_learning_4env.pbs",
    },
    "native_rlinf_8env_ppo": {
        "job_id": "2842187.pbs-7",
        "log": "native-metaworld-scaling-8env-2842187.pbs-7.driver.log",
        "pbs": "scripts/pbs/native_metaworld_pushv3_scaling_8env.pbs",
    },
    "native_rlinf_12env_ppo": {
        "job_id": "2842188.pbs-7",
        "log": "native-metaworld-scaling-12env-2842188.pbs-7.driver.log",
        "pbs": "scripts/pbs/native_metaworld_pushv3_scaling_12env.pbs",
    },
    "native_osmesa_4env": {
        "job_id": "2841687.pbs-7",
        "log": "native-metaworld-osmesa-microbench-2841687.pbs-7.driver.log",
        "pbs": "scripts/pbs/native_metaworld_pushv3_osmesa_microbench.pbs",
    },
    "native_egl_4env": {
        "job_id": "2841982.pbs-7",
        "log": "native-metaworld-egl-microbench-2841982.pbs-7.driver.log",
        "pbs": "scripts/pbs/native_metaworld_pushv3_egl_microbench.pbs",
    },
    "native_grpo_4env": {
        "job_id": "2842501.pbs-7",
        "log": "native-metaworld-grpo-learning-4env-2842501.pbs-7.driver.log",
        "pbs": "scripts/pbs/native_metaworld_pushv3_grpo_learning_4env.pbs",
    },
}


PREFIXES = (
    "DIRECT_SMOLVLA_PPO_METRIC ",
    "DIRECT_SMOLVLA_PPO_RUN_OK ",
    "SMOLVLA_EVAL_RESULT ",
    "SMOLVLA_EVAL_SWEEP_OK ",
    "SMOLVLA_COMPONENT_BENCH ",
)


def run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(
            cmd,
            cwd=ROOT,
            text=True,
            stderr=subprocess.STDOUT,
            timeout=30,
        ).strip()
    except Exception as exc:
        return f"ERROR {type(exc).__name__}: {exc}"


def package_version(name: str) -> str:
    try:
        mod = __import__(name)
        return str(getattr(mod, "__version__", "unknown"))
    except Exception as exc:
        return f"IMPORT_ERROR {type(exc).__name__}: {exc}"


def capture_versions() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "python": sys.version,
        "executable": sys.executable,
        "cwd": str(ROOT),
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MUJOCO_GL": os.environ.get("MUJOCO_GL"),
            "PYOPENGL_PLATFORM": os.environ.get("PYOPENGL_PLATFORM"),
            "SMOLVLA_CHECKPOINT": os.environ.get("SMOLVLA_CHECKPOINT"),
        },
        "git": {
            "head": run(["git", "rev-parse", "HEAD"]),
            "status_short": run(["git", "status", "--short", "--branch"]),
            "log_recent": run(["git", "log", "--oneline", "-n", "25"]),
        },
        "packages": {
            "torch": package_version("torch"),
            "rlinf": package_version("rlinf"),
            "lerobot": package_version("lerobot"),
            "metaworld": package_version("metaworld"),
            "ray": package_version("ray"),
        },
        "nvidia_smi": run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]),
        "qstat_user": run(["qstat", "-u", os.environ.get("USER", "")]),
    }
    try:
        import torch

        payload["torch_cuda"] = {
            "version": torch.version.cuda,
            "available": torch.cuda.is_available(),
            "device_count": torch.cuda.device_count(),
            "device_names": [
                torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
            ]
            if torch.cuda.is_available()
            else [],
        }
    except Exception as exc:
        payload["torch_cuda_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def extract_json_lines(src: Path, dst: Path) -> int:
    count = 0
    with src.open("r", encoding="utf-8", errors="ignore") as inp, dst.open(
        "w", encoding="utf-8"
    ) as out:
        for line in inp:
            for prefix in PREFIXES:
                if prefix in line:
                    out.write(line.split(prefix, 1)[1].strip() + "\n")
                    count += 1
                    break
    return count


def main() -> None:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "logs" / "artifacts" / f"smolvla_bottleneck_{stamp}"
    raw_dir = out_dir / "raw_logs"
    json_dir = out_dir / "json_lines"
    script_dir = out_dir / "pbs_scripts"
    raw_dir.mkdir(parents=True, exist_ok=True)
    json_dir.mkdir(parents=True, exist_ok=True)
    script_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "created_utc": stamp,
        "root": str(ROOT),
        "versions": capture_versions(),
        "runs": {},
        "new_timing_artifacts": {
            "direct_runner": "scripts/run_smolvla_metaworld_direct_ppo.py",
            "component_microbench": "scripts/benchmark_smolvla_metaworld_components.py",
            "component_pbs": "scripts/pbs/smolvla_metaworld_component_benchmark.pbs",
            "submitted_job": "2855402.pbs-7",
        },
    }

    for name, meta in RUNS.items():
        log_src = LOG_DIR / meta["log"]
        pbs_src = ROOT / meta["pbs"]
        run_payload = dict(meta)
        if log_src.exists():
            log_dst = raw_dir / log_src.name
            shutil.copy2(log_src, log_dst)
            json_dst = json_dir / f"{name}.jsonl"
            count = extract_json_lines(log_src, json_dst)
            run_payload.update(
                {
                    "raw_log_copy": str(log_dst),
                    "raw_log_original": str(log_src),
                    "extracted_jsonl": str(json_dst),
                    "extracted_json_line_count": count,
                    "raw_log_size_bytes": log_src.stat().st_size,
                }
            )
        else:
            run_payload["raw_log_missing"] = str(log_src)
        if pbs_src.exists():
            pbs_dst = script_dir / pbs_src.name
            shutil.copy2(pbs_src, pbs_dst)
            run_payload["pbs_script_copy"] = str(pbs_dst)
            run_payload["pbs_script_original"] = str(pbs_src)
        else:
            run_payload["pbs_script_missing"] = str(pbs_src)
        manifest["runs"][name] = run_payload

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print("SMOLVLA_BOTTLENECK_ARTIFACTS " + json.dumps({"dir": str(out_dir), "manifest": str(manifest_path)}, sort_keys=True))


if __name__ == "__main__":
    main()
