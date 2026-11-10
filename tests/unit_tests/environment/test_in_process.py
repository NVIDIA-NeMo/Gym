# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import sys
from pathlib import Path

import pytest

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import BaseServerConfig, ConfigError
from nemo_gym.config.global_config import maybe_get_global_config_dict, scoped_global_config_dict
from nemo_gym.core.episode_types import EpisodeId
from nemo_gym.core.failure_records import EpisodeFailure
from nemo_gym.environment import in_process
from nemo_gym.environment.in_process import (
    PlannedBenchmark,
    RolloutOutcome,
    TaskFailure,
    TaskRunner,
    plan_benchmark,
    serve_benchmark,
)
from nemo_gym.eval.eval_plan import PlanBenchmark, PlanGym, build_eval_plan
from nemo_gym.rollouts.rollout_collection import RolloutCollectionHelper
from nemo_gym.rollouts.rollout_outcomes import RolloutFailureRecord
from nemo_gym.server_runtime.supervisor.server_supervisor import ServerSpinupTimeoutError, ServerStartFailedError


HEAD = BaseServerConfig(host="127.0.0.1", port=4242)


def _row(task_id: str, prompt: str = "solve it") -> dict:
    return {
        "task_id": task_id,
        "agent_ref": {"name": "agent"},
        "responses_create_params": {"input": [{"role": "user", "content": prompt}]},
    }


def _collated_row(task_id: str) -> dict:
    """A row as plan_benchmark collates it: it names a task source, and Gym resolves its agent when it dispatches."""
    return {**{key: value for key, value in _row(task_id).items() if key != "agent_ref"}, "task_source": "agent"}


def _planned(*task_ids: str, prompt: str = "solve it", commit: str | None = "abc") -> PlannedBenchmark:
    rows = [_row(task_id, prompt) for task_id in task_ids]
    plan = build_eval_plan(
        rows,
        benchmark=PlanBenchmark(name="bench", config_paths=["benchmarks/bench/config.yaml"]),
        taskset="bench",
        gym=PlanGym(version="0.0.0", commit=commit),
    )
    return PlannedBenchmark(plan=plan, rows=rows)


class _Completions:
    """The completion iterator `run_outcomes` returns: awaitables to iterate, and an `aclose` that records the call."""

    def __init__(self, completions: list) -> None:
        self._completions = completions
        self.closed = False

    def __iter__(self):
        return iter(self._completions)

    async def aclose(self) -> None:
        self.closed = True


class _FakeRuns:
    """Stands in for `run_outcomes`: records what was sent and returns a reward per row."""

    def __init__(self, reward: float = 1.0, fail: bool = False) -> None:
        self.reward = reward
        self.fail = fail
        self.sent: list[list[dict]] = []
        self.bounds: dict = {}
        self.completions: _Completions | None = None

    def __call__(self, rows, head_server_config, **bounds):
        self.sent.append(rows)
        self.bounds = bounds
        assert head_server_config == HEAD

        async def completed(row):
            if self.fail:
                raise RuntimeError("rollout failed")
            return {**row, "agent_ref": {"name": "agent"}}, {"reward": self.reward, "row_task_id": row["task_id"]}

        self.completions = _Completions([completed(row) for row in rows])
        return self.completions


class _FakeResults:
    """Stands in for `run_outcomes`: returns the given `/run` result for the one row."""

    def __init__(self, result) -> None:
        self.result = result

    def __call__(self, rows, head_server_config, **bounds):
        async def completed(row):
            return {**row, "agent_ref": {"name": "agent"}}, self.result

        return _Completions([completed(row) for row in rows])


class TestOutcomes:
    @pytest.mark.parametrize(
        "raw, outcome",
        [
            (
                {
                    "reward": 1.0,
                    "responses_create_params": {"input": [{"role": "user", "content": "solve it"}]},
                    "response": {"output": [{"type": "message", "content": "done"}]},
                },
                RolloutOutcome(
                    reward=1.0,
                    prompt='[{"role": "user", "content": "solve it"}]',
                    output='[{"type": "message", "content": "done"}]',
                ),
            ),
            (
                {"reward": 0.0, "failure_kind": "judge_failed", "failure_reason": "no verdict"},
                RolloutOutcome(
                    reward=0.0,
                    failure=TaskFailure(
                        kind="judge_failed", source="environment", delivery="delivered", reason="no verdict"
                    ),
                ),
            ),
            (
                {"reward": 0.0, "failure_kind": "session_lost"},
                RolloutOutcome(
                    reward=0.0,
                    failure=TaskFailure(kind="session_lost", source="environment", delivery="delivered"),
                ),
            ),
            ({"reward": 0.0, "mask_sample": True}, RolloutOutcome(reward=0.0, masked=True)),
            (
                {"reward": 0.0, "mask_sample": True, "failure_kind": "session_lost"},
                RolloutOutcome(
                    reward=0.0,
                    masked=True,
                    failure=TaskFailure(kind="session_lost", source="environment", delivery="delivered"),
                ),
            ),
        ],
    )
    async def test_the_outcome_is_typed_and_the_raw_result_stays_beside_it(
        self, monkeypatch: pytest.MonkeyPatch, raw: dict, outcome: RolloutOutcome
    ) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", _FakeResults(raw))

        result = await runner.run_task("a")

        assert result == outcome
        assert result.raw == raw


def _record(
    *,
    source: str = "collector",
    delivery: str = "delivered",
    kind: str | None = None,
    stage: str | None = None,
    terminal: bool = False,
    http_status: int | None = 500,
    exception_type: str | None = "ClientResponseError",
) -> RolloutFailureRecord:
    return RolloutFailureRecord(
        episode_id=EpisodeId(rollout_id="r-r0"),
        run_id="run",
        source=source,
        delivery=delivery,
        failure=EpisodeFailure(failure_reason="boom", terminal=terminal, failure_kind=kind, stage=stage),
        http_status=http_status,
        exception_type=exception_type,
    )


class TestTaskFailure:
    @pytest.mark.parametrize(
        "kind, domain",
        [
            ("judge_failed", "judge"),
            ("agent_timeout", "agent"),
            ("provider_unavailable", "provider"),
            ("model_context_limit", "model"),
            ("myenv:bad_input", "myenv"),
            ("cancelled", "cancelled"),
        ],
    )
    def test_the_domain_is_the_part_of_the_kind_before_the_first_underscore_or_colon(
        self, kind: str, domain: str
    ) -> None:
        assert TaskFailure(kind=kind, source="environment", delivery="delivered").domain == domain

    @pytest.mark.parametrize(
        "kind, evaluation_side",
        [
            ("model_request_error", False),
            ("agent_timeout", False),
            ("model_transport_error", True),
            ("agent_run_error", True),
            ("agent_request_failed", True),
            ("myenv:bad_input", None),
            ("not_a_kind", None),
        ],
    )
    def test_evaluation_side_tells_a_model_outcome_from_an_evaluation_fault(
        self, kind: str, evaluation_side: bool | None
    ) -> None:
        assert TaskFailure(kind=kind, source="environment", delivery="delivered").evaluation_side is evaluation_side


class TestFailedRuns:
    async def test_a_failed_run_comes_back_as_an_outcome(self, monkeypatch: pytest.MonkeyPatch) -> None:
        record = _record(kind="model_request_error", stage="agent", terminal=True)
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", _FakeResults(record))

        result = await runner.run_task("a")

        assert result.raw == record.model_dump(mode="json")
        assert result == RolloutOutcome(
            reward=None,
            failure=TaskFailure(
                kind="model_request_error",
                source="collector",
                delivery="delivered",
                stage="agent",
                terminal=True,
                reason="boom",
                http_status=500,
                exception_type="ClientResponseError",
            ),
        )

    @pytest.mark.parametrize(
        "source, delivery, kind",
        [
            ("environment", "delivered", "environment_server_failed"),
            ("collector", "delivered", "agent_request_failed"),
            ("collector", "possibly_delivered", "agent_request_failed"),
            ("collector", "not_sent", "agent_request_failed"),
        ],
    )
    async def test_a_failure_that_names_no_cause_gets_the_kind_for_who_reported_it(
        self, monkeypatch: pytest.MonkeyPatch, source: str, delivery: str, kind: str
    ) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        record = _record(source=source, delivery=delivery, http_status=None, exception_type=None)
        monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", _FakeResults(record))

        failure = (await runner.run_task("a")).failure

        assert failure is not None
        assert (failure.kind, failure.source, failure.delivery) == (kind, source, delivery)
        assert failure.http_status is None
        assert failure.exception_type is None


class TestRolloutMetrics:
    @pytest.fixture
    def recorded(self, monkeypatch: pytest.MonkeyPatch) -> list[tuple]:
        calls: list[tuple] = []

        def completed(*args, **kwargs):
            calls.append(("completed", args, kwargs))

        def rewarded(*args, **kwargs):
            calls.append(("reward", args, kwargs))

        monkeypatch.setattr(in_process, "record_rollout_completed", completed)
        monkeypatch.setattr(in_process, "record_rollout_reward", rewarded)
        return calls

    @pytest.mark.parametrize(
        "raw, expected",
        [
            (
                {"reward": 1.0},
                [
                    ("completed", ("scored",), {"dispatch_name": "agent"}),
                    ("reward", (1.0,), {"dispatch_name": "agent", "masked": False}),
                ],
            ),
            (
                {"reward": 0.0, "mask_sample": True},
                [
                    ("completed", ("scored",), {"dispatch_name": "agent"}),
                    ("reward", (0.0,), {"dispatch_name": "agent", "masked": True}),
                ],
            ),
            (
                _record(delivery="possibly_delivered", http_status=502, exception_type="ClientResponseError"),
                [
                    (
                        "completed",
                        ("failed",),
                        {
                            "dispatch_name": "agent",
                            "failure_kind": "agent_request_failed",
                            "failure_type": "ClientResponseError",
                        },
                    ),
                ],
            ),
            (
                {"reward": 0.0, "mask_sample": True, "failure_kind": "judge_failed"},
                [
                    (
                        "completed",
                        ("failed",),
                        {"dispatch_name": "agent", "failure_kind": "judge_failed", "failure_type": None},
                    ),
                ],
            ),
            (
                {"response": {"output": []}, "execute_only": True},
                [("completed", ("unscored",), {"dispatch_name": "agent"})],
            ),
        ],
    )
    async def test_a_failure_makes_the_rollout_failed_and_a_reward_without_one_makes_it_scored(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[tuple], raw, expected: list[tuple]
    ) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", _FakeResults(raw))

        await runner.run_task("a")

        assert recorded == expected

    async def test_the_dispatch_name_is_that_of_the_row_gym_resolved(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[tuple]
    ) -> None:
        runner = TaskRunner([_collated_row("a")], registry_server=HEAD)
        monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", _FakeResults({"reward": 1.0}))

        await runner.run_task("a")

        assert recorded[0] == ("completed", ("scored",), {"dispatch_name": "agent"})

    async def test_a_scored_rollout_never_carries_a_failure_label(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[tuple]
    ) -> None:
        raws = [
            {"reward": 1.0},
            {"reward": 0.0, "mask_sample": True},
            {"reward": 0.0, "failure_kind": "judge_failed"},
            {"response": {"output": []}, "execute_only": True},
            _record(delivery="possibly_delivered", http_status=502, exception_type="ClientResponseError"),
        ]
        for raw in raws:
            runner = TaskRunner([_row("a")], registry_server=HEAD)
            monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", _FakeResults(raw))
            await runner.run_task("a")

        counted = [call for call in recorded if call[0] == "completed"]
        assert [call[1][0] for call in counted] == ["scored", "scored", "failed", "unscored", "failed"]
        for _, (outcome,), labels in counted:
            assert outcome == "failed" or set(labels) == {"dispatch_name"}

    async def test_a_failed_rollout_adds_nothing_to_the_reward_count(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[tuple]
    ) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        monkeypatch.setattr(
            RolloutCollectionHelper,
            "run_outcomes",
            _FakeResults({"reward": 0.0, "mask_sample": True, "failure_kind": "judge_failed"}),
        )

        await runner.run_task("a")

        assert [call[0] for call in recorded] == ["completed"]

    async def test_a_row_without_an_agent_or_environment_server_is_counted_under_unknown(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[tuple]
    ) -> None:
        async def completed(row):
            return row, {"reward": 1.0}

        runner = TaskRunner([_collated_row("a")], registry_server=HEAD)
        monkeypatch.setattr(
            RolloutCollectionHelper,
            "run_outcomes",
            lambda self, rows, head_server_config, **bounds: _Completions([completed(row) for row in rows]),
        )

        outcome = await runner.run_task("a")

        assert outcome.reward == 1.0
        assert recorded[0] == ("completed", ("scored",), {"dispatch_name": "unknown"})

    async def test_a_reward_that_is_not_finite_is_counted_but_not_ranged(
        self, monkeypatch: pytest.MonkeyPatch, recorded: list[tuple]
    ) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", _FakeResults({"reward": float("nan")}))

        await runner.run_task("a")

        assert recorded == [("completed", ("scored",), {"dispatch_name": "agent"})]


class TestTaskRunner:
    def test_indexes_each_task_once(self) -> None:
        runner = TaskRunner([_row("a"), _row("a"), _row("b")], registry_server=HEAD)

        assert runner.task_ids() == ["a", "b"]

    def test_refuses_rows_without_ids_and_conflicting_rows(self) -> None:
        with pytest.raises(ConfigError, match="no task id"):
            TaskRunner([_row("a"), {"responses_create_params": {"input": []}}], registry_server=HEAD)
        with pytest.raises(ConfigError, match="share a task id"):
            TaskRunner([_row("a", "x"), _row("a", "y")], registry_server=HEAD)

    async def test_runs_one_task_under_the_given_rollout_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a"), _row("b")], registry_server=HEAD)
        runs = _FakeRuns(reward=1.0)
        monkeypatch.setattr(
            RolloutCollectionHelper,
            "run_outcomes",
            lambda self, rows, head_server_config, **bounds: runs(rows, head_server_config, **bounds),
        )

        result = await runner.run_task("b", rollout_id="trial-7")

        assert result.reward == 1.0
        assert [row["task_id"] for row in runs.sent[0]] == ["b"]
        assert [row["_ng_rollout_id"] for row in runs.sent[0]] == ["trial-7"]

    @staticmethod
    def _serve(monkeypatch: pytest.MonkeyPatch, runs: _FakeRuns) -> None:
        monkeypatch.setattr(
            RolloutCollectionHelper,
            "run_outcomes",
            lambda self, rows, head_server_config, **bounds: runs(rows, head_server_config, **bounds),
        )

    async def test_bounds_the_connection_retries_as_gym_eval_run_does(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        runs = _FakeRuns()
        self._serve(monkeypatch, runs)

        await runner.run_task("a")

        assert runs.bounds == {"max_num_tries": 2, "connect_grace_s": 60.0}

    async def test_calls_without_a_rollout_id_get_different_ones(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a"), _row("b")], registry_server=HEAD)
        runs = _FakeRuns()
        self._serve(monkeypatch, runs)

        await asyncio.gather(runner.run_task("a"), runner.run_task("b"), runner.run_task("a"))

        ids = [row["_ng_rollout_id"] for sent in runs.sent for row in sent]
        assert len(ids) == 3
        assert len(set(ids)) == 3

    async def test_closes_the_completions_when_a_rollout_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        runs = _FakeRuns(fail=True)
        self._serve(monkeypatch, runs)

        with pytest.raises(RuntimeError, match="rollout failed"):
            await runner.run_task("a")

        assert runs.completions.closed

    async def test_closes_the_completions_when_the_task_is_cancelled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        release = asyncio.Event()
        hung = []

        async def never_finishes():
            await release.wait()

        def run_outcomes(self, rows, head_server_config, **bounds):
            hung.append(_Completions([never_finishes()]))
            return hung[0]

        monkeypatch.setattr(RolloutCollectionHelper, "run_outcomes", run_outcomes)
        task = asyncio.ensure_future(runner.run_task("a"))
        await asyncio.sleep(0)
        task.cancel()

        with pytest.raises(asyncio.CancelledError):
            await task

        assert hung[0].closed

    async def test_ignores_the_command_line_of_the_calling_program(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["trainer", "--help"])
        runner = TaskRunner([_row("a")], registry_server=HEAD)
        self._serve(monkeypatch, _FakeRuns())

        result = await runner.run_task("a")

        assert result.reward == 1.0

    def test_refuses_a_materialized_task_row(self) -> None:
        materialized = {
            "task_id": {"taskset": "bench", "task_id": "a"},
            "task_input": {"responses_create_params": {"input": []}, "task_data": {}},
        }

        with pytest.raises(ConfigError, match="materialized"):
            TaskRunner([materialized], registry_server=HEAD)

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

    def test_the_prepare_step_sees_the_given_config_and_the_previous_one_comes_back(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_gym.config.global_config import get_global_config_dict

        monkeypatch.setattr(sys, "argv", ["trainer", "--help", "++host_only=1"])
        seen = {}

        def fake_plan(global_config_dict, prepare_config, task_ids):
            seen["config"] = get_global_config_dict()
            return "the plan"

        monkeypatch.setattr(in_process, "_benchmark_config", lambda path, overrides: {"hf_token": "given"})
        monkeypatch.setattr("nemo_gym.environment.in_process.plan_for_config", fake_plan)
        monkeypatch.setattr(
            "nemo_gym.environment.in_process.PrepareBenchmarkConfig.model_validate", lambda config: "prepare"
        )
        monkeypatch.setattr("nemo_gym.environment.in_process.asset_config_path", lambda flag, name: "bench.yaml")
        monkeypatch.setattr(in_process, "_collated_rows", lambda config, directory: [])

        plan_benchmark("bench")

        assert seen["config"] == {"hf_token": "given"}
        assert maybe_get_global_config_dict() is None

    def test_the_benchmark_config_leaves_the_active_exporters_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parsed = []

        def parse(self, parser_config):
            parsed.append(parser_config)
            return {}

        monkeypatch.setattr(in_process.GlobalConfigDictParser, "parse", parse)

        in_process._benchmark_config("benchmarks/bench/config.yaml", None)

        assert parsed[0].skip_exporters


class TestScopedGlobalConfig:
    def test_restores_the_previous_config_even_when_the_block_fails(self) -> None:
        with scoped_global_config_dict({"outer": 1}):
            with pytest.raises(RuntimeError):
                with scoped_global_config_dict({"inner": 2}):
                    assert maybe_get_global_config_dict() == {"inner": 2}
                    raise RuntimeError("boom")

            assert maybe_get_global_config_dict() == {"outer": 1}

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
    registry_port = None

    def start(self, parser_config) -> None:
        type(self).started_with = parser_config
        head = parser_config.initial_global_config_dict.get("head_server") or {}
        type(self).registry_port = head.get("port")
        if type(self).start_error is not None:
            raise type(self).start_error

    def shutdown(self) -> None:
        type(self).shut_down += 1


class TestServeBenchmark:
    @pytest.fixture(autouse=True)
    def fakes(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _FakeRunHelper.started_with, _FakeRunHelper.shut_down, _FakeRunHelper.start_error = None, 0, None
        _FakeRunHelper.registry_port = None
        monkeypatch.setattr("nemo_gym.environment.in_process.RunHelper", _FakeRunHelper)
        monkeypatch.setattr(
            "nemo_gym.environment.in_process.asset_config_path",
            lambda flag, name: f"{'benchmarks' if flag == 'benchmark' else 'responses_api_models'}/{name}.yaml",
        )
        monkeypatch.setattr(
            ServerClient,
            "load_head_server_config",
            classmethod(lambda cls: BaseServerConfig(host=HEAD.host, port=_FakeRunHelper.registry_port or HEAD.port)),
        )

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

    def test_a_registry_port_in_use_stops_the_servers_and_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            ServerClient,
            "load_head_server_config",
            classmethod(lambda cls: BaseServerConfig(host="127.0.0.1", port=6000)),
        )

        with pytest.raises(ConfigError, match="registry_port 5555 is in use"):
            with serve_benchmark("bench", registry_port=5555):
                pytest.fail("the body must not run")

        assert _FakeRunHelper.shut_down == 1

    @pytest.mark.parametrize(
        "error",
        [ServerSpinupTimeoutError("too slow"), ServerStartFailedError("it exited", server_name="agent", errno=None)],
    )
    def test_a_server_that_does_not_start_is_a_config_error_with_the_cause(self, error: RuntimeError) -> None:
        _FakeRunHelper.start_error = error

        with pytest.raises(ConfigError, match="did not start") as raised:
            with serve_benchmark("bench"):
                pytest.fail("the body must not run")

        assert raised.value.__cause__ is error

    def test_ignores_the_command_line_of_the_calling_program(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(sys, "argv", ["trainer", "--help"])

        with serve_benchmark("bench"):
            pass

        assert _FakeRunHelper.started_with.skip_exporters
