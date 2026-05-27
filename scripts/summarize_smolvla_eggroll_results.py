from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

PI05_SECONDS_PER_MEMBER_EPISODE = 10.18


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize SmolVLA MetaWorld EGGROLL result artifacts.")
    parser.add_argument("--campaign-dir", required=True)
    parser.add_argument("--production-dir", default="")
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def production_runs(path: Path) -> list[dict[str, Any]]:
    runs = []
    if not path:
        return runs
    for metrics_path in sorted(path.glob("seed_*/metrics.jsonl")):
        updates = [row for row in read_jsonl(metrics_path) if row.get("marker") == "SMOLVLA_EGGROLL_UPDATE"]
        if not updates:
            continue
        runs.append(
            {
                "run_dir": str(metrics_path.parent),
                "updates": len(updates),
                "last_seconds_per_member_episode": updates[-1]["seconds_per_member_episode"],
                "last_score_sum": sum(member["score"] for member in updates[-1].get("member_scores", [])),
            }
        )
    return runs


def main() -> int:
    args = parse_args()
    campaign_dir = Path(args.campaign_dir)
    summary = json.loads((campaign_dir / "campaign_summary.json").read_text(encoding="utf-8"))
    best = summary["best"]
    payload = {
        "marker": "SMOLVLA_EGGROLL_SUMMARY",
        "campaign_dir": str(campaign_dir),
        "best_population_size": best["population_size"],
        "best_seconds_per_member_episode": best["seconds_per_member_episode"],
        "best_peak_vram_gb": best["peak_vram_gb"],
        "pi05_seconds_per_member_episode": PI05_SECONDS_PER_MEMBER_EPISODE,
        "speedup_vs_pi05": PI05_SECONDS_PER_MEMBER_EPISODE / best["seconds_per_member_episode"],
        "production_runs": production_runs(Path(args.production_dir)) if args.production_dir else [],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
