from __future__ import annotations

import os
import resource
import subprocess
import time
from dataclasses import dataclass
from dataclasses import field
from typing import Any


def parse_nvidia_smi_csv(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        rows.append(
            {
                "index": parts[0],
                "name": parts[1],
                "gpu_util_percent": float(parts[2]),
                "memory_util_percent": float(parts[3]),
                "memory_used_mb": float(parts[4]),
                "memory_total_mb": float(parts[5]),
            }
        )
    return rows


def sample_nvidia_smi() -> list[dict[str, Any]]:
    query = "index,name,utilization.gpu,utilization.memory,memory.used,memory.total"
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode != 0:
        return []
    return parse_nvidia_smi_csv(result.stdout)


def sample_process_resources() -> dict[str, Any]:
    payload: dict[str, Any] = {"pid": os.getpid()}
    try:
        import psutil
    except ModuleNotFoundError:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        payload.update({"rss_mb": float(usage.ru_maxrss) / 1024.0, "source": "resource"})
        return payload

    proc = psutil.Process(os.getpid())
    children = proc.children(recursive=True)
    rss = 0
    cpu = 0.0
    live_children = 0
    for process in [proc, *children]:
        try:
            rss += process.memory_info().rss
            cpu += float(process.cpu_percent(interval=None))
            if process.pid != proc.pid:
                live_children += 1
        except psutil.Error:
            continue
    vm = psutil.virtual_memory()
    payload.update(
        {
            "rss_mb": rss / 1024**2,
            "cpu_percent": cpu,
            "system_memory_used_percent": float(vm.percent),
            "system_memory_available_mb": float(vm.available) / 1024**2,
            "child_processes": live_children,
            "source": "psutil",
        }
    )
    return payload


@dataclass
class ResourceMonitor:
    samples: list[dict[str, Any]] = field(default_factory=list)

    def sample(self) -> None:
        self.samples.append(
            {
                "time_s": time.time(),
                "process": sample_process_resources(),
                "gpus": sample_nvidia_smi(),
            }
        )

    def summary(self) -> dict[str, Any]:
        if not self.samples:
            return {"samples": 0}
        rss_values = [sample["process"].get("rss_mb", 0.0) for sample in self.samples]
        gpu_mem = [
            gpu.get("memory_used_mb", 0.0)
            for sample in self.samples
            for gpu in sample.get("gpus", [])
        ]
        gpu_util = [
            gpu.get("gpu_util_percent", 0.0)
            for sample in self.samples
            for gpu in sample.get("gpus", [])
        ]
        return {
            "samples": len(self.samples),
            "rss_mb_max": max(rss_values) if rss_values else 0.0,
            "gpu_memory_used_mb_max": max(gpu_mem) if gpu_mem else 0.0,
            "gpu_util_percent_mean": sum(gpu_util) / len(gpu_util) if gpu_util else 0.0,
            "first": self.samples[0],
            "last": self.samples[-1],
        }
