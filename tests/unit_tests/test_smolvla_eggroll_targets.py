from __future__ import annotations

import torch

from rlinf.algorithms.eggroll.targets import TargetProbeResult
from rlinf.algorithms.eggroll.targets import action_effect_summary
from rlinf.algorithms.eggroll.targets import find_eggroll_targets
from rlinf.algorithms.eggroll.targets import make_probe_delta
from rlinf.algorithms.eggroll.targets import record_module_io
from rlinf.algorithms.eggroll.targets import select_best_probe_result


class TinySmolVLAModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_in_proj = torch.nn.Linear(4, 8)
        self.vlm_with_expert = torch.nn.Module()
        self.vlm_with_expert.lm_expert = torch.nn.Module()
        self.vlm_with_expert.lm_expert.layers = torch.nn.ModuleList(
            [torch.nn.Sequential(torch.nn.Linear(8, 8))]
        )
        self.vision_tower = torch.nn.Sequential(torch.nn.Linear(4, 8))
        self.not_a_target = torch.nn.Linear(4, 8)


def test_find_eggroll_targets_prefers_action_and_expert_linears() -> None:
    targets = find_eggroll_targets(TinySmolVLAModel())

    names = [target.name for target in targets]
    assert "action_in_proj" in names
    assert "vlm_with_expert.lm_expert.layers.0.0" in names
    assert "vision_tower.0" not in names
    assert "not_a_target" not in names
    assert targets[0].shape == (8, 4)


def test_find_eggroll_targets_rejects_empty_target_set() -> None:
    with torch.no_grad():
        model = torch.nn.Sequential(torch.nn.Linear(2, 2))

    try:
        find_eggroll_targets(model)
    except ValueError as exc:
        assert "No EGGROLL" in str(exc)
    else:
        raise AssertionError("expected no-target validation failure")


def test_record_module_io_captures_call_shapes_dtype_and_device() -> None:
    module = torch.nn.Linear(4, 3)
    with record_module_io(module) as telemetry:
        module(torch.zeros(2, 5, 4))

    assert telemetry.call_count == 1
    assert telemetry.input_shapes == [(2, 5, 4)]
    assert telemetry.output_shapes == [(2, 5, 3)]
    assert telemetry.input_dtypes == ["torch.float32"]
    assert telemetry.input_devices == ["cpu"]


def test_make_probe_delta_matches_target_weight_shape() -> None:
    target = find_eggroll_targets(TinySmolVLAModel())[0]

    delta = make_probe_delta(target, rank=2, scale=0.01, seed=123)

    assert delta.left.shape == (target.shape[0], 2)
    assert delta.right.shape == (2, target.shape[1])
    assert delta.materialize().shape == target.shape


def test_action_effect_summary_uses_full_horizon_and_saturation() -> None:
    base = torch.zeros(2, 3, 4)
    perturbed = base.clone()
    perturbed[1, 2, 3] = 0.5
    perturbed[0, 0, 0] = 1.0

    summary = action_effect_summary(base, perturbed, action_low=-1.0, action_high=1.0)

    assert summary["finite"]
    assert summary["max_abs_diff"] == 1.0
    assert summary["mean_abs_diff"] > 0.0
    assert summary["per_horizon_max_abs_diff"] == [1.0, 0.0, 0.5]
    assert summary["saturation_fraction"] == 1 / 24


def test_select_best_probe_result_rejects_zero_nonfinite_and_saturated_targets() -> None:
    results = [
        TargetProbeResult(
            name="zero",
            shape=(8, 4),
            max_abs_diff=0.0,
            mean_abs_diff=0.0,
            finite=True,
            saturation_fraction=0.0,
            call_count=1,
            input_shapes=[(2, 5, 4)],
            output_shapes=[(2, 5, 8)],
        ),
        TargetProbeResult(
            name="bad",
            shape=(8, 4),
            max_abs_diff=2.0,
            mean_abs_diff=1.0,
            finite=False,
            saturation_fraction=0.0,
            call_count=1,
            input_shapes=[(2, 5, 4)],
            output_shapes=[(2, 5, 8)],
        ),
        TargetProbeResult(
            name="saturated",
            shape=(8, 4),
            max_abs_diff=3.0,
            mean_abs_diff=1.0,
            finite=True,
            saturation_fraction=0.95,
            call_count=1,
            input_shapes=[(2, 5, 4)],
            output_shapes=[(2, 5, 8)],
        ),
        TargetProbeResult(
            name="safe",
            shape=(8, 4),
            max_abs_diff=0.25,
            mean_abs_diff=0.05,
            finite=True,
            saturation_fraction=0.1,
            call_count=1,
            input_shapes=[(2, 5, 4)],
            output_shapes=[(2, 5, 8)],
        ),
    ]

    selected = select_best_probe_result(results)

    assert selected.name == "safe"
