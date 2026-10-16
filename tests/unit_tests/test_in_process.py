# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from nemo_gym import in_process
from nemo_gym.config_types import BaseServerConfig, ConfigError
from nemo_gym.global_config import maybe_get_global_config_dict
from nemo_gym.in_process import TaskRunner, plan_benchmark
from nemo_gym.rollout_collection import RolloutCollectionHelper


HEAD = BaseServerConfig(host="127.0.0.1", port=4242)


def _row(task_id: str, prompt: str = "solve it") -> dict:
    return {
        "task_id": task_id,
        "agent_ref": {"name": "agent"},
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
    }


class _FakeRuns:
    """Stands in for `run_examples`: records what was sent and returns a reward per row."""

    def __init__(self, reward: float = 1.0) -> None:
        self.reward = reward
        self.sent: list[list[dict]] = []

    def __call__(self, rows, head_server_config):
        self.sent.append(rows)
        assert head_server_config == HEAD

        async def completed(row):
            return row, {"reward": self.reward, "row_task_id": row["task_id"]}

        return [completed(row) for row in rows]


class TestTaskRunner:
    def test_indexes_each_task_once(self) -> None:
        runner = TaskRunner([_row("a"), _row("a"), _row("b")], head_server=HEAD)

        assert runner.task_ids() == ["a", "b"]

    def test_refuses_rows_without_ids_and_conflicting_rows(self) -> None:
        with pytest.raises(ConfigError, match="no task id"):
            TaskRunner([_row("a"), {"responses_create_params": {"input": []}}], head_server=HEAD)
        with pytest.raises(ConfigError, match="share a task id"):
            TaskRunner([_row("a", "x"), _row("a", "y")], head_server=HEAD)

    async def test_runs_one_task_with_repeats_and_rollout_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a"), _row("b")], head_server=HEAD)
        runs = _FakeRuns(reward=1.0)
        monkeypatch.setattr(
            RolloutCollectionHelper,
            "run_examples",
            lambda self, rows, head_server_config: runs(rows, head_server_config),
        )

        result = await runner.run_task("b", num_repeats=2, rollout_id="trial-7")

        assert result.task_id == "b"
        assert result.rewards == [1.0, 1.0]
        assert [row["task_id"] for row in runs.sent[0]] == ["b", "b"]
        assert [row["_ng_rollout_id"] for row in runs.sent[0]] == ["trial-7-r0", "trial-7-r1"]

    async def test_a_single_rollout_keeps_the_rollout_id_as_given(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a")], head_server=HEAD)
        runs = _FakeRuns()
        monkeypatch.setattr(
            RolloutCollectionHelper,
            "run_examples",
            lambda self, rows, head_server_config: runs(rows, head_server_config),
        )

        await runner.run_task("a", rollout_id="trial-7")

        assert runs.sent[0][0]["_ng_rollout_id"] == "trial-7"

    async def test_unknown_task_is_a_config_error(self) -> None:
        with pytest.raises(ConfigError, match="No input rows for task ids"):
            await TaskRunner([_row("a")], head_server=HEAD).run_task("z")


class TestPlanBenchmark:
    def test_plans_and_collates_without_touching_process_config(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = {}

        def fake_config(config_path, overrides):
            seen["config_path"], seen["overrides"] = config_path, overrides
            return {"config_paths": [config_path]}

        def fake_plan(global_config_dict, prepare_config, task_ids):
            seen["task_ids"] = task_ids
            return "the plan"

        monkeypatch.setattr(in_process, "_benchmark_config", fake_config)
        monkeypatch.setattr("nemo_gym.cli.eval.plan_for_config", fake_plan)
        monkeypatch.setattr("nemo_gym.cli.eval.PrepareBenchmarkConfig.model_validate", lambda config: "prepare")
        monkeypatch.setattr(
            "nemo_gym.cli.main._asset_config_path", lambda flag, name: f"benchmarks/{name}/config.yaml"
        )
        monkeypatch.setattr(in_process, "_collated_rows", lambda config, directory: [_row("a"), _row("b"), _row("a")])

        planned = plan_benchmark("bench", task_ids=["a"], overrides={"x": 1}, work_dir=Path("/unused"))

        assert seen == {"config_path": "benchmarks/bench/config.yaml", "overrides": {"x": 1}, "task_ids": ["a"]}
        assert planned.plan == "the plan"
        assert [row["task_id"] for row in planned.rows] == ["a", "a"]
        assert maybe_get_global_config_dict() is None
