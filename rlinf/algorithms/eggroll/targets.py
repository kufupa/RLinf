from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from fnmatch import fnmatchcase
from typing import Any, Iterator

import numpy as np

from rlinf.algorithms.eggroll.low_rank import LowRankDelta
from rlinf.algorithms.eggroll.population import EggrollPopulationConfig
from rlinf.algorithms.eggroll.population import sample_population


EGGROLL_TARGET_PATTERNS = (
    "action_in_proj",
    "action_out_proj",
    "time_mlp_in",
    "time_mlp_out",
    "*.action_in_proj",
    "*.action_out_proj",
    "*.time_mlp_in",
    "*.time_mlp_out",
    "vlm_with_expert.lm_expert.layers.*",
    "*.vlm_with_expert.lm_expert.layers.*",
    "paligemma_with_expert.gemma_expert.model.layers.*.self_attn.q_proj",
    "paligemma_with_expert.gemma_expert.model.layers.*.self_attn.k_proj",
    "paligemma_with_expert.gemma_expert.model.layers.*.self_attn.v_proj",
    "paligemma_with_expert.gemma_expert.model.layers.*.self_attn.o_proj",
    "paligemma_with_expert.gemma_expert.model.layers.*.mlp.gate_proj",
    "paligemma_with_expert.gemma_expert.model.layers.*.mlp.up_proj",
    "paligemma_with_expert.gemma_expert.model.layers.*.mlp.down_proj",
)


@dataclass(frozen=True)
class EggrollTarget:
    name: str
    module: Any
    shape: tuple[int, int]

    @property
    def param_count(self) -> int:
        return self.shape[0] * self.shape[1]


@dataclass
class ModuleIOTelemetry:
    call_count: int = 0
    input_shapes: list[tuple[int, ...]] | None = None
    output_shapes: list[tuple[int, ...]] | None = None
    input_dtypes: list[str] | None = None
    input_devices: list[str] | None = None

    def __post_init__(self) -> None:
        self.input_shapes = [] if self.input_shapes is None else self.input_shapes
        self.output_shapes = [] if self.output_shapes is None else self.output_shapes
        self.input_dtypes = [] if self.input_dtypes is None else self.input_dtypes
        self.input_devices = [] if self.input_devices is None else self.input_devices

    def payload(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TargetProbeResult:
    name: str
    shape: tuple[int, int]
    max_abs_diff: float
    mean_abs_diff: float
    finite: bool
    saturation_fraction: float
    call_count: int
    input_shapes: list[tuple[int, ...]]
    output_shapes: list[tuple[int, ...]]

    def payload(self) -> dict[str, Any]:
        return asdict(self)


def find_eggroll_targets(
    model: Any,
    *,
    patterns: tuple[str, ...] = EGGROLL_TARGET_PATTERNS,
) -> list[EggrollTarget]:
    targets: list[EggrollTarget] = []
    for name, module in model.named_modules():
        if not any(fnmatchcase(name, pattern) for pattern in patterns):
            continue
        shape = _weight_shape(module)
        if shape is None:
            continue
        targets.append(EggrollTarget(name=name, module=module, shape=shape))

    if not targets:
        raise ValueError("No EGGROLL target modules found")
    return targets


def make_probe_delta(
    target: EggrollTarget,
    *,
    rank: int,
    scale: float,
    seed: int,
) -> LowRankDelta:
    rank = min(rank, target.shape[0], target.shape[1])
    return sample_population(
        EggrollPopulationConfig(
            population_size=1,
            out_features=target.shape[0],
            in_features=target.shape[1],
            rank=rank,
            sigma=scale,
            seed=seed,
        )
    )[0].delta


@contextmanager
def record_module_io(module: Any) -> Iterator[ModuleIOTelemetry]:
    telemetry = ModuleIOTelemetry()

    def hook(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        import torch

        input_tensor = next((item for item in inputs if isinstance(item, torch.Tensor)), None)
        if input_tensor is not None:
            telemetry.input_shapes.append(tuple(int(dim) for dim in input_tensor.shape))
            telemetry.input_dtypes.append(str(input_tensor.dtype))
            telemetry.input_devices.append(str(input_tensor.device))
        if isinstance(output, torch.Tensor):
            telemetry.output_shapes.append(tuple(int(dim) for dim in output.shape))
        telemetry.call_count += 1

    handle = module.register_forward_hook(hook)
    try:
        yield telemetry
    finally:
        handle.remove()


def action_effect_summary(
    base_actions: Any,
    perturbed_actions: Any,
    *,
    action_low: float,
    action_high: float,
    saturation_eps: float = 1e-6,
) -> dict[str, Any]:
    base = _to_float_numpy(base_actions)
    perturbed = _to_float_numpy(perturbed_actions)
    if base.shape != perturbed.shape:
        raise ValueError(f"action shapes differ: {base.shape} vs {perturbed.shape}")
    diff = np.abs(perturbed - base)
    finite = bool(np.isfinite(base).all() and np.isfinite(perturbed).all())
    saturated = (perturbed <= action_low + saturation_eps) | (perturbed >= action_high - saturation_eps)
    if diff.ndim >= 3:
        horizon_axes = tuple(axis for axis in range(diff.ndim) if axis != 1)
        per_horizon = diff.max(axis=horizon_axes).astype(float).tolist()
    else:
        per_horizon = [float(diff.max(initial=0.0))]
    return {
        "finite": finite,
        "max_abs_diff": float(diff.max(initial=0.0)),
        "mean_abs_diff": float(diff.mean() if diff.size else 0.0),
        "per_horizon_max_abs_diff": per_horizon,
        "saturation_fraction": float(saturated.mean() if saturated.size else 0.0),
    }


def select_best_probe_result(
    results: list[TargetProbeResult],
    *,
    min_action_effect: float = 1e-8,
    max_saturation_fraction: float = 0.9,
) -> TargetProbeResult:
    accepted = [
        result
        for result in results
        if result.finite
        and result.call_count > 0
        and result.max_abs_diff > min_action_effect
        and result.saturation_fraction < max_saturation_fraction
    ]
    if not accepted:
        raise ValueError("No usable EGGROLL target survived probe")
    return max(accepted, key=lambda result: (result.mean_abs_diff, result.max_abs_diff))


def _weight_shape(module: Any) -> tuple[int, int] | None:
    weight = getattr(module, "weight", None)
    shape = getattr(weight, "shape", None)
    if shape is None:
        data = getattr(weight, "data", None)
        shape = getattr(data, "shape", None)
    if shape is None or len(shape) != 2:
        return None
    return (int(shape[0]), int(shape[1]))


def _to_float_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)
