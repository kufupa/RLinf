from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.metaworld.smolvla_metaworld_env import SmolVLAMetaWorldEnv
from rlinf.models.embodiment.smolvla import get_model


def _checkpoint() -> str:
    return os.environ.get(
        "SMOLVLA_CHECKPOINT",
        "jadechoghari/smolvla_metaworld",
    )


def main() -> None:
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"smoke: device={device}", flush=True)

    env_cfg = OmegaConf.create(
        {
            "seed": 0,
            "task_name": "push-v3",
            "task_description": "Push the puck to a goal",
            "reset_randomization_mode": "random_seeded",
            "max_episode_steps": 10,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": True,
            "reward_coef": 1.0,
            "reward_mode": "sparse_success_delta",
            "reset_seed_base": 2000,
            "use_async_envs": False,
            "video_cfg": {"save_video": False},
        }
    )
    model_cfg = OmegaConf.create(
        {
            "model_type": "smolvla",
            "model_path": _checkpoint(),
            "precision": None,
            "load_to_device": False,
            "action_dim": 4,
            "num_action_chunks": 5,
            "state_dim": 4,
            "n_action_steps": 5,
            "add_value_head": True,
            "detach_critic_input": True,
            "freeze_all_but_ppo_trainables": True,
            "action_low": -1.0,
            "action_high": 1.0,
            "is_lora": False,
        }
    )

    print("smoke: load model begin", flush=True)
    model = get_model(model_cfg, None).to(device)
    model.train()
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise RuntimeError("no trainable SmolVLA PPO params")
    optim = torch.optim.AdamW(trainable, lr=1e-6)
    print(
        f"smoke: load model done trainable_params={sum(p.numel() for p in trainable)}",
        flush=True,
    )

    env = SmolVLAMetaWorldEnv(
        env_cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, _ = env.reset()
        print(
            f"smoke: reset image={tuple(obs['main_images'].shape)} "
            f"state={tuple(obs['states'].shape)} seed={obs['reset_seeds'].tolist()}",
            flush=True,
        )
        actions, rollout = model.predict_action_batch(obs, mode="train")
        _obs_list, rewards, terms, truncs, _infos = env.chunk_step(
            actions.detach().cpu().numpy()
        )
        print(
            f"smoke: rollout actions={tuple(actions.shape)} rewards={rewards.tolist()} "
            f"terms={terms.tolist()} truncs={truncs.tolist()}",
            flush=True,
        )

        old_logp = rollout["prev_logprobs"].to(device).sum(dim=(1, 2))
        out = model.default_forward(rollout["forward_inputs"])
        new_logp = out["logprobs"].sum(dim=(1, 2))
        values = out["values"].reshape(-1)
        returns = rewards.to(device).sum(dim=1).float()
        advantages = returns - values.detach()
        ratio = torch.exp(new_logp - old_logp)
        clipped = torch.clamp(ratio, 0.8, 1.2)
        policy_loss = -torch.min(ratio * advantages, clipped * advantages).mean()
        value_loss = torch.nn.functional.mse_loss(values, returns)
        entropy = out["entropy"].mean()
        loss = policy_loss + 0.5 * value_loss - 0.0 * entropy
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite PPO loss: {loss}")

        optim.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optim.step()
        print(
            "SMOLVLA_METAWORLD_PPO_SMOKE_OK "
            f"loss={float(loss.detach().cpu()):.6f} "
            f"policy_loss={float(policy_loss.detach().cpu()):.6f} "
            f"value_loss={float(value_loss.detach().cpu()):.6f} "
            f"ratio={float(ratio.detach().cpu().mean()):.6f} "
            f"grad_norm={float(grad_norm.detach().cpu()):.6f}",
            flush=True,
        )
        out_dir = Path(os.environ.get("SMOLVLA_PPO_SMOKE_OUT", "logs/smolvla_ppo_smoke"))
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "SUCCESS").write_text("ok\n", encoding="utf-8")
    finally:
        env.close()


if __name__ == "__main__":
    main()
