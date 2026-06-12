#!/usr/bin/env python3
"""Parse native MetaWorld RLinf benchmark driver logs for throughput metrics."""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

METRIC_PAIR_RE = re.compile(r"([A-Za-z0-9_/]+)=([\d.eE+-]+)")
TABLE_START_RE = re.compile(r"^╭─")
TABLE_END_RE = re.compile(r"^╰─")


@dataclass(frozen=True)
class StepMetrics:
    step: int | None
    time_step_s: float | None
    time_env: dict[str, float]
    time_rollout: dict[str, float]
    env_steps_per_s: float | None


def parse_metric_tables(text: str) -> list[dict[str, float]]:
    """Return metric dicts for each printed metric table block."""
    lines = text.splitlines()
    tables: list[dict[str, float]] = []
    in_table = False
    current: dict[str, float] = {}

    for line in lines:
        if TABLE_START_RE.match(line):
            in_table = True
            current = {}
            continue
        if in_table and TABLE_END_RE.match(line):
            if current:
                tables.append(current)
            in_table = False
            current = {}
            continue
        if not in_table or not line.startswith("│"):
            continue
        for name, raw_value in METRIC_PAIR_RE.findall(line):
            current[name] = float(raw_value)
    return tables


def extract_time_metrics(table: dict[str, float]) -> tuple[float | None, dict[str, float], dict[str, float]]:
    time_step_s = table.get("step")
    time_env: dict[str, float] = {}
    time_rollout: dict[str, float] = {}
    for key, value in table.items():
        if key.startswith("env/"):
            time_env[key.removeprefix("env/")] = value
        elif key.startswith("rollout/"):
            time_rollout[key.removeprefix("rollout/")] = value
    return time_step_s, time_env, time_rollout


def compute_env_steps_per_s(
    time_step_s: float | None,
    num_envs: int,
    max_steps_per_rollout_epoch: int,
    rollout_epoch: int,
) -> float | None:
    if time_step_s is None or time_step_s <= 0:
        return None
    env_steps = num_envs * max_steps_per_rollout_epoch * rollout_epoch
    return env_steps / time_step_s


def summarize_log(
    text: str,
    *,
    num_envs: int,
    max_steps_per_rollout_epoch: int,
    rollout_epoch: int,
) -> dict[str, object]:
    tables = parse_metric_tables(text)
    step_rows: list[StepMetrics] = []
    for idx, table in enumerate(tables):
        time_step_s, time_env, time_rollout = extract_time_metrics(table)
        step_rows.append(
            StepMetrics(
                step=idx,
                time_step_s=time_step_s,
                time_env=time_env,
                time_rollout=time_rollout,
                env_steps_per_s=compute_env_steps_per_s(
                    time_step_s,
                    num_envs,
                    max_steps_per_rollout_epoch,
                    rollout_epoch,
                ),
            )
        )

    valid_step_times = [row.time_step_s for row in step_rows if row.time_step_s is not None]
    valid_throughput = [row.env_steps_per_s for row in step_rows if row.env_steps_per_s is not None]

    last = step_rows[-1] if step_rows else None
    summary: dict[str, object] = {
        "num_tables": len(tables),
        "num_envs": num_envs,
        "max_steps_per_rollout_epoch": max_steps_per_rollout_epoch,
        "rollout_epoch": rollout_epoch,
        "last_time_step_s": last.time_step_s if last else None,
        "last_env_steps_per_s": last.env_steps_per_s if last else None,
        "mean_time_step_s": statistics.fmean(valid_step_times) if valid_step_times else None,
        "mean_env_steps_per_s": statistics.fmean(valid_throughput) if valid_throughput else None,
        "steps": [asdict(row) for row in step_rows],
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract time/env, time/rollout, and env-steps/s from native MetaWorld driver logs."
    )
    parser.add_argument("logs", nargs="+", type=Path, help="Driver log file(s) to parse")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps-per-rollout-epoch", type=int, default=120)
    parser.add_argument("--rollout-epoch", type=int, default=2)
    parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results: dict[str, dict[str, object]] = {}

    for log_path in args.logs:
        if not log_path.is_file():
            print(f"error: missing log file: {log_path}", file=sys.stderr)
            return 2
        summary = summarize_log(
            log_path.read_text(encoding="utf-8", errors="replace"),
            num_envs=args.num_envs,
            max_steps_per_rollout_epoch=args.max_steps_per_rollout_epoch,
            rollout_epoch=args.rollout_epoch,
        )
        results[str(log_path)] = summary

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for log_path, summary in results.items():
        print(f"log: {log_path}")
        print(f"  tables: {summary['num_tables']}")
        print(f"  last time/step: {summary['last_time_step_s']}")
        print(f"  mean time/step: {summary['mean_time_step_s']}")
        print(f"  last env-steps/s: {summary['last_env_steps_per_s']}")
        print(f"  mean env-steps/s: {summary['mean_env_steps_per_s']}")
        last_steps = summary["steps"]
        if last_steps:
            last_row = last_steps[-1]
            if last_row["time_env"]:
                env_bits = ", ".join(
                    f"{name}={value:.3f}s" for name, value in sorted(last_row["time_env"].items())
                )
                print(f"  last time/env: {env_bits}")
            if last_row["time_rollout"]:
                rollout_bits = ", ".join(
                    f"{name}={value:.3f}s"
                    for name, value in sorted(last_row["time_rollout"].items())
                )
                print(f"  last time/rollout: {rollout_bits}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
