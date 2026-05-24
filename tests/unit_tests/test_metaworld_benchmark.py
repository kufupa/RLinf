import pytest

from rlinf.envs.metaworld import MetaWorldBenchmark


def test_metaworld_single_push_v3():
    bench = MetaWorldBenchmark("metaworld_single", task_names=["push-v3"])
    assert bench.get_num_tasks() == 1
    assert bench.get_env_names() == ["push-v3"]
    assert bench.get_task_description() == ["Push the puck to a goal"]
    assert bench.get_task_num_trials() == 10


def test_metaworld_single_configurable_trials():
    bench = MetaWorldBenchmark(
        "metaworld_single",
        task_names=["push-v3"],
        task_num_trials=25,
    )
    assert bench.get_task_num_trials() == 25


def test_metaworld_single_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown MetaWorld task names"):
        MetaWorldBenchmark("metaworld_single", task_names=["not-a-task"])


def test_metaworld_single_requires_task_names():
    with pytest.raises(ValueError, match="metaworld_single requires task_names"):
        MetaWorldBenchmark("metaworld_single")


def test_metaworld_50_still_works():
    bench = MetaWorldBenchmark("metaworld_50")
    assert bench.get_num_tasks() == 50
    assert bench.get_task_num_trials() == 10
    assert len(bench.get_env_names()) == 50
    assert len(bench.get_task_description()) == 50
