# SmolVLA MetaWorld Environment Build

This is the environment used for SmolVLA + MetaWorld PPO/GRPO on Imperial PBS.

## Imperial Module Setup

The working GPU jobs used the PBS common bootstrap:

```bash
module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load Mesa/24.1.3-GCCcore-13.3.0
```

Then jobs invoked:

```bash
/rds/general/user/aa6622/home/.envs/lerobot_mw_py312/bin/python
```

Important: that venv is module-dependent. Running the venv Python directly from
a shell without the matching `Python/3.12.3-GCCcore-13.3.0` module can fail with:

```text
libpython3.12.so.1.0: cannot open shared object file
```

That does not mean the GPU job env is broken. It means the venv needs the module
environment that supplies the shared Python library.

## Portable Variables

`scripts/pbs/smolvla_metaworld_common.sh` accepts these overrides:

```bash
export RLINF_ROOT=/path/to/RLinf
export RLINF_TOOLS_MODULE=tools/prod
export RLINF_PYTHON_MODULE=Python/3.12.3-GCCcore-13.3.0
export RLINF_MESA_MODULE=Mesa/24.1.3-GCCcore-13.3.0
export PYTHON_BIN=/path/to/venv/bin/python
export HF_HOME=/path/to/hf-cache
export SMOLVLA_CHECKPOINT=/path/or/model-id
export RLINF_EPHEMERAL_BASE=/scratch/$USER/rlinf_ray_runs
```

`GRPO_PYTHON` is still accepted as an alias for `PYTHON_BIN`.

## Recommended Rebuild On Another Imperial Cluster

```bash
git clone https://github.com/kufupa/RLinf.git
cd RLinf
git checkout smolvla-metaworld-ppo-grpo

module purge
module load tools/prod
module load Python/3.12.3-GCCcore-13.3.0
module load Mesa/24.1.3-GCCcore-13.3.0

python -m venv ~/.envs/lerobot_mw_py312
source ~/.envs/lerobot_mw_py312/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements/smolvla-metaworld-py312-imperial.txt
```

If `pip install -e .` rejects Python 3.12 because of `requires-python`, run from
source with:

```bash
export PYTHONPATH=$PWD:${PYTHONPATH:-}
```

That is how the PBS jobs execute the code path.

## Model Checkpoint

Default model id:

```text
jadechoghari/smolvla_metaworld
```

For offline cluster execution, point to the already-downloaded snapshot:

```bash
export SMOLVLA_CHECKPOINT=/path/to/smolvla_metaworld/snapshot
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

The user reported the other LLM already has the SmolVLA checkpoint, so handoff
should pass that local path through `SMOLVLA_CHECKPOINT`.

## Rendering

Default headless rendering:

```bash
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LIBGL_ALWAYS_SOFTWARE=1
```

EGL can be tested by overriding `MUJOCO_GL=egl`, but OSMesa was the stable default
for the runs documented here.

## Known Fixes

- PBS queue mismatch: generic GPU requests on Imperial may need explicit
  `gpu_type` and `Qlist` fields matching the queue/node.
- Python 3.12 shared library error: load the same Python module used to create
  the venv before invoking venv Python.
- Parent repo helper dependency: the LeRobot MetaWorld adapter is now vendored
  into RLinf under `rlinf/envs/metaworld/lerobot_adapter.py`, so
  `project/src/smolvla_grpo` is no longer needed for this path.
- Resume checkpoints: resume scripts require `RESUME_CKPT=/path/to/update_*.pt`
  instead of hard-coded ephemeral paths.
