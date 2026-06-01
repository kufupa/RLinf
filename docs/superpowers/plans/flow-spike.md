# SmolVLA flow logprob spike (Phase B)

**Venv:** `/vol/bitbucket/aa6622/.envs/lerobot_mw_py310/lib/python3.12/site-packages/lerobot/policies/smolvla/modeling_smolvla.py`

- Denoise loop: `num_steps` iterations, `dt = -1/num_steps`, euler integration on `v_t`.
- `sample_actions` returns terminal `x_t` + expanded `log_std` only — **no** intermediate `A^tau` export today.
- `euler_step_noise_std` adds noise to velocity at each step (L925–926).

**Hook for `flow_sde`:** record one stochastic step index per chunk `(tau_idx, A_tau, mu_tau, sigma_tau, A_next)` inside denoise loop; replay same noise seed on recompute.

**Reference:** `RLinf/rlinf/models/embodiment/openpi/openpi_action_model.py` `sample_mean_var_val` (flow_sde branch), `get_logprob_norm`.

**Status:** math in `project/src/smolvla_grpo/flow_logprob.py`; venv hook **pending** — trainer rejects `--logprob-mode flow_sde` until hook lands.
