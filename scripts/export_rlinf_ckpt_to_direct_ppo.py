#!/usr/bin/env python3
"""Export RLinf FSDP full_weights.pt into direct-PPO eval checkpoints (update_*.pt)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))
from torch_load_mmap import torch_load_mmap_default

from rlinf.models.embodiment.smolvla import get_model


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--rlinf-full-weights",
        default="",
        help=".../model_state_dict/full_weights.pt (single export)",
    )
    p.add_argument("--output-dir", required=True, help="Directory for update_*.pt files")
    p.add_argument("--updates", default="20", help="Comma-separated global steps to export")
    p.add_argument("--rlinf-ckpt-root", default="", help=".../checkpoints (parent of global_step_*)")
    p.add_argument("--model-path", default="")
    return p.parse_args()


def trainable_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: param.detach().cpu().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def export_one(model: torch.nn.Module, full_weights: Path, out_path: Path, update: int) -> None:
    state = torch_load_mmap_default(full_weights, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(state, strict=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "update": update,
        "trainable_model": trainable_state(model),
        "metrics": {"source": str(full_weights), "missing_keys": len(missing), "unexpected_keys": len(unexpected)},
    }
    torch.save(payload, out_path)
    print(
        f"RLINF_EXPORT_PPO_CKPT_OK update={update} out={out_path} "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        flush=True,
    )


def main() -> int:
    args = parse_args()
    if not args.rlinf_ckpt_root and not args.rlinf_full_weights:
        raise SystemExit("Pass --rlinf-ckpt-root or --rlinf-full-weights")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = OmegaConf.create(
        {
            "model_type": "smolvla",
            "model_path": args.model_path or __import__("os").environ.get("SMOLVLA_CHECKPOINT", "jadechoghari/smolvla_metaworld"),
            "precision": None,
            "load_to_device": False,
            "action_dim": 4,
            "num_action_chunks": 5,
            "state_dim": 4,
            "n_action_steps": 5,
            "add_value_head": True,
            "detach_critic_input": True,
            "freeze_all_but_ppo_trainables": True,
        }
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg, None).to(device)
    model.eval()

    updates = [int(x.strip()) for x in args.updates.split(",") if x.strip()]
    ckpt_root = Path(args.rlinf_ckpt_root) if args.rlinf_ckpt_root else None

    for step in updates:
        if ckpt_root is not None:
            full = (
                ckpt_root
                / f"global_step_{step}"
                / "actor"
                / "model_state_dict"
                / "full_weights.pt"
            )
        else:
            full = Path(args.rlinf_full_weights)
            if len(updates) > 1:
                raise ValueError("Pass --rlinf-ckpt-root when exporting multiple updates")
        if not full.is_file():
            raise FileNotFoundError(full)
        export_one(model, full, out_dir / f"update_{step:06d}.pt", step)

    (out_dir / "export_manifest.json").write_text(
        json.dumps({"updates": updates, "output_dir": str(out_dir)}, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
