# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from pathlib import Path

import pytest

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig, ConfigError
from nemo_gym.config.global_config import maybe_get_global_config_dict
from nemo_gym.environment import in_process
from nemo_gym.environment.in_process import PlannedBenchmark, TaskRunner, plan_benchmark, serve_benchmark
from nemo_gym.eval.eval_plan import PlanBenchmark, PlanGym, build_eval_plan
from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper


HEAD = BaseServerConfig(host="127.0.0.1", port=4242)


def _row(task_id: str, prompt: str = "solve it") -> dict:
    return {
        "task_id": task_id,
        "agent_ref": {"name": "agent"},
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
    }


def _planned(*task_ids: str, prompt: str = "solve it", commit: str | None = "abc") -> PlannedBenchmark:
    rows = [_row(task_id, prompt) for task_id in task_ids]
    plan = build_eval_plan(
        rows,
        benchmark=PlanBenchmark(name="bench", config_paths=["benchmarks/bench/config.yaml"]),
        taskset="bench",
        gym=PlanGym(version="0.0.0", commit=commit),
    )
    return PlannedBenchmark(plan=plan, rows=rows)


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
        runner = TaskRunner([_row("a"), _row("a"), _row("b")], registry_server=HEAD)

        assert runner.task_ids() == ["a", "b"]

    def test_refuses_rows_without_ids_and_conflicting_rows(self) -> None:
        with pytest.raises(ConfigError, match="no task id"):
            TaskRunner([_row("a"), {"responses_create_params": {"input": []}}], registry_server=HEAD)
        with pytest.raises(ConfigError, match="share a task id"):
            TaskRunner([_row("a", "x"), _row("a", "y")], registry_server=HEAD)

    async def test_runs_one_task_with_repeats_and_rollout_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a"), _row("b")], registry_server=HEAD)
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
        runner = TaskRunner([_row("a")], registry_server=HEAD)
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
            await TaskRunner([_row("a")], registry_server=HEAD).run_task("z")


class TestPlannedBenchmark:
    def test_task_digests_map_each_task_id_to_the_digest_of_its_input(self) -> None:
        planned = _planned("a", "b")

        digests = planned.task_digests()

        assert list(digests) == planned.task_ids() == ["a", "b"]
        assert digests == {task.task_id.task_id: task.digest for task in planned.plan.tasks}

    def test_a_changed_input_changes_the_digest(self) -> None:
        assert _planned("a", prompt="one").task_digests() != _planned("a", prompt="two").task_digests()

    def test_gym_commit_is_the_commit_the_plan_was_made_with(self) -> None:
        assert _planned("a", commit="abc").gym_commit == "abc"
        assert _planned("a", commit=None).gym_commit is None

    def test_json_carries_a_plan_to_another_process_unchanged(self) -> None:
        planned = _planned("a", "b")

        restored = PlannedBenchmark.from_json(planned.to_json())

        assert restored == planned
        assert restored.task_digests() == planned.task_digests()

    def test_json_refuses_a_row_it_cannot_represent(self) -> None:
        planned = _planned("a")
        planned.rows[0]["score"] = float("nan")

        with pytest.raises(ValueError):
            planned.to_json()


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
        monkeypatch.setattr("nemo_gym.environment.in_process.plan_for_config", fake_plan)
        monkeypatch.setattr(
            "nemo_gym.environment.in_process.PrepareBenchmarkConfig.model_validate", lambda config: "prepare"
        )
        monkeypatch.setattr(
            "nemo_gym.environment.in_process.asset_config_path", lambda flag, name: f"benchmarks/{name}/config.yaml"
        )
        monkeypatch.setattr(in_process, "_collated_rows", lambda config, directory: [_row("a"), _row("b"), _row("a")])

        planned = plan_benchmark("bench", task_ids=["a"], overrides={"x": 1}, work_dir=Path("/unused"))

        assert seen == {"config_path": "benchmarks/bench/config.yaml", "overrides": {"x": 1}, "task_ids": ["a"]}
        assert planned.plan == "the plan"
        assert [row["task_id"] for row in planned.rows] == ["a", "a"]
        assert maybe_get_global_config_dict() is None


DOTTED_OVERRIDES = [
    {"judge.model": "x"},
    {"judge": {"a.b": 1}},
    {"items": [{"a.b": 1}]},
]


class TestDottedOverrides:
    @pytest.mark.parametrize("overrides", DOTTED_OVERRIDES)
    def test_planning_refuses_a_dotted_key(self, monkeypatch: pytest.MonkeyPatch, overrides: dict) -> None:
        monkeypatch.setattr(
            "nemo_gym.environment.in_process.asset_config_path", lambda flag, name: f"{flag}/{name}.yaml"
        )

        with pytest.raises(ConfigError, match="dotted"):
            plan_benchmark("bench", overrides=overrides)

    def test_nested_keys_are_accepted(self) -> None:
        assert in_process._checked_overrides({"judge": {"model": "x"}, "items": [{"a": 1}]}) == {
            "judge": {"model": "x"},
            "items": [{"a": 1}],
        }

    def test_no_overrides_are_an_empty_dict(self) -> None:
        assert in_process._checked_overrides(None) == {}


class _FakeRunHelper:
    """Stands in for `RunHelper`: records the config it is started with and whether it was shut down."""

    started_with = None
    shut_down = 0
    start_error = None

    def start(self, parser_config) -> None:
        type(self).started_with = parser_config
        if type(self).start_error is not None:
            raise type(self).start_error

    def shutdown(self) -> None:
        type(self).shut_down += 1


class TestServeBenchmark:
    @pytest.fixture(autouse=True)
    def fakes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeRunHelper.started_with, _FakeRunHelper.shut_down, _FakeRunHelper.start_error = None, 0, None
        monkeypatch.setattr("nemo_gym.environment.in_process.RunHelper", _FakeRunHelper)
        monkeypatch.setattr(
            "nemo_gym.environment.in_process.asset_config_path",
            lambda flag, name: f"{'benchmarks' if flag == 'benchmark' else 'responses_api_models'}/{name}.yaml",
        )
        monkeypatch.setattr(ServerClient, "load_head_server_config", classmethod(lambda cls: HEAD))

    def test_serves_from_a_config_dict_and_stops_on_exit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["gym", "unrelated=1"])
        overrides = {"judge": {"model": "x"}, "policy_base_url": "http://model/v1"}

        with serve_benchmark("bench", overrides=overrides) as registry_server:
            assert registry_server == HEAD
            assert _FakeRunHelper.shut_down == 0

        parser_config = _FakeRunHelper.started_with
        assert parser_config.skip_load_from_cli
        assert parser_config.initial_global_config_dict == {
            "config_paths": ["benchmarks/bench.yaml", "responses_api_models/openai_model.yaml"],
            "judge": {"model": "x"},
            "policy_base_url": "http://model/v1",
        }
        assert _FakeRunHelper.shut_down == 1
        assert sys.argv == ["gym", "unrelated=1"]

    def test_the_registry_port_is_set_in_the_config(self) -> None:
        with serve_benchmark("bench", registry_port=5555):
            pass

        assert _FakeRunHelper.started_with.initial_global_config_dict["head_server"] == {"port": 5555}

    def test_the_registry_port_keeps_the_rest_of_the_registry_server_config(self) -> None:
        with serve_benchmark("bench", overrides={"head_server": {"host": "0.0.0.0"}}, registry_port=5555):
            pass

        assert _FakeRunHelper.started_with.initial_global_config_dict["head_server"] == {
            "host": "0.0.0.0",
            "port": 5555,
        }

    def test_gym_chooses_the_registry_port_by_default(self) -> None:
        with serve_benchmark("bench"):
            pass

        assert "head_server" not in _FakeRunHelper.started_with.initial_global_config_dict

    @pytest.mark.parametrize("port", [0, -1, 65536, 70000])
    def test_a_registry_port_that_is_not_a_port_is_refused_before_anything_starts(self, port: int) -> None:
        with pytest.raises(ConfigError, match="registry_port"):
            with serve_benchmark("bench", registry_port=port):
                pass

        assert _FakeRunHelper.started_with is None

    def test_the_model_type_is_a_parameter(self) -> None:
        with serve_benchmark("bench", model_type="vllm_model"):
            pass

        assert _FakeRunHelper.started_with.initial_global_config_dict["config_paths"][1] == (
            "responses_api_models/vllm_model.yaml"
        )

    def test_overrides_may_replace_the_config_paths(self) -> None:
        with serve_benchmark("bench", overrides={"config_paths": ["mine.yaml"]}):
            pass

        assert _FakeRunHelper.started_with.initial_global_config_dict["config_paths"] == ["mine.yaml"]

    @pytest.mark.parametrize("overrides", DOTTED_OVERRIDES)
    def test_serving_refuses_a_dotted_key_before_starting_anything(self, overrides: dict) -> None:
        with pytest.raises(ConfigError, match="dotted"):
            with serve_benchmark("bench", overrides=overrides):
                pytest.fail("the body must not run")

        assert _FakeRunHelper.started_with is None

    def test_stops_the_servers_when_the_caller_fails(self) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with serve_benchmark("bench"):
                raise RuntimeError("boom")

        assert _FakeRunHelper.shut_down == 1

    def test_a_failed_start_raises_and_leaves_the_clean_up_to_start(self) -> None:
        _FakeRunHelper.start_error = ConfigError("model endpoint never answered")

        with pytest.raises(ConfigError, match="never answered"):
            with serve_benchmark("bench"):
                pytest.fail("the body must not run")

        assert _FakeRunHelper.shut_down == 0
