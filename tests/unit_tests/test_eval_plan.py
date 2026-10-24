# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf
from yaml import safe_load

import nemo_gym.global_config
from nemo_gym.cli.eval import plan_benchmark
from nemo_gym.config_types import ConfigError
from nemo_gym.eval_plan import (
    EvalPlan,
    PlanBenchmark,
    PlanGym,
    build_eval_plan,
    canonical_json,
    eval_plan_json,
    eval_plan_json_schema,
)


def _row(task_id: str, prompt: str = "solve it", **extra: object) -> dict:
    return {"task_id": task_id, "responses_create_params": {"input": [{"role": "user", "content": prompt}]}, **extra}


def _plan(rows: list[dict], **kwargs: object) -> EvalPlan:
    return build_eval_plan(
        rows,
        benchmark=PlanBenchmark(name="bench", config_paths=["benchmarks/bench/config.yaml"]),
        taskset=kwargs.pop("taskset", "bench"),
        gym=PlanGym(version="0.0.0", commit="abc"),
        **kwargs,
    )


class TestBuildEvalPlan:
    def test_tasks_are_sorted_by_id_with_taskset_and_digest(self) -> None:
        plan = _plan([_row("b"), _row("a")])

        assert [task.task_id.task_id for task in plan.tasks] == ["a", "b"]
        assert {task.task_id.taskset for task in plan.tasks} == {"bench"}
        assert plan.tasks[0].digest == "sha256:" + hashlib.sha256(canonical_json(_row("a"))).hexdigest()
        assert plan.selected_task_ids is None

    def test_plan_records_ids_and_digests_but_not_task_inputs(self) -> None:
        document = json.loads(eval_plan_json(_plan([_row("a")])))

        assert set(document["tasks"][0]) == {"task_id", "digest"}

    def test_task_ids_use_the_given_taskset(self) -> None:
        plan = _plan([_row("a")], taskset="declared_taskset")

        assert plan.tasks[0].task_id.taskset == "declared_taskset"

    def test_routing_and_bookkeeping_fields_are_not_part_of_the_task(self) -> None:
        plain = _plan([_row("a")]).tasks[0]
        routed = _plan([_row("a", task_source="agent_1", agent_ref={"name": "agent_1"}, _ng_task_index=7)]).tasks[0]

        assert routed.digest == plain.digest

    def test_digest_changes_with_task_content(self) -> None:
        assert _plan([_row("a", "one")]).tasks[0].digest != _plan([_row("a", "two")]).tasks[0].digest

    def test_problem_id_and_instance_id_are_accepted(self) -> None:
        rows = [
            {"problem_id": "p1", "responses_create_params": {"input": []}},
            {"instance_id": "repo__repo-1", "responses_create_params": {"input": []}},
        ]

        assert [task.task_id.task_id for task in _plan(rows).tasks] == ["p1", "repo__repo-1"]

    def test_rows_without_an_id_are_refused(self) -> None:
        rows = [_row("a"), {"task_name": "terminal-bench/x", "responses_create_params": {"input": []}}]

        with pytest.raises(ConfigError, match=r"1 of 2 tasks have no task id .*rows \[1\]"):
            _plan(rows)

    def test_already_materialized_rows_are_refused(self) -> None:
        row = {"task_id": {"taskset": "bench", "task_id": "a"}, "task_input": {"prompt": "solve it"}}

        with pytest.raises(ConfigError, match=r"Benchmark 'bench': Expected a flat dataset row"):
            _plan([row])

    def test_duplicate_ids_are_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"not unique: \['a'\]"):
            _plan([_row("a", "one"), _row("a", "two"), _row("b")])

    def test_selected_task_ids_limit_the_plan(self) -> None:
        plan = _plan([_row("a"), _row("b"), _row("c")], selected_task_ids=["c", "a"])

        assert [task.task_id.task_id for task in plan.tasks] == ["a", "c"]
        assert plan.selected_task_ids == ["a", "c"]

    def test_unknown_selected_task_ids_are_refused(self) -> None:
        with pytest.raises(ConfigError, match=r"no tasks with ids \['z'\]"):
            _plan([_row("a")], selected_task_ids=["a", "z"])


class TestEvalPlanJson:
    def test_row_order_does_not_change_the_file(self) -> None:
        rows = [_row("a", "one"), _row("b", "two"), _row("c", "three")]

        assert eval_plan_json(_plan(rows)) == eval_plan_json(_plan(list(reversed(rows))))

    def test_file_round_trips_with_schema_label(self) -> None:
        plan = _plan([_row("a")])
        document = json.loads(eval_plan_json(plan))

        assert document["schema"] == "nemo-gym.eval-plan.v1"
        assert EvalPlan.model_validate(document) == plan

    def test_json_schema_describes_the_file(self) -> None:
        schema = json.loads(eval_plan_json_schema())

        assert schema["properties"]["schema"]["const"] == "nemo-gym.eval-plan.v1"
        assert set(schema["required"]) >= {"gym", "benchmark", "tasks"}


@pytest.fixture(autouse=True)
def _mock_port_allocation(monkeypatch):
    monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", lambda **_: 12345)


class TestPlanBenchmarkCommand:
    def _config(self, tmp_path: Path, *, prompt_config: str | None = None, **plan_args: object) -> dict:
        prepare_script = tmp_path / "prepare.py"
        prepare_script.write_text("")
        dataset = {
            "name": "fake_bench",
            "type": "benchmark",
            "jsonl_fpath": str(tmp_path / "benchmark.jsonl"),
            "prepare_script": str(prepare_script),
        }
        if prompt_config is not None:
            dataset["prompt_config"] = prompt_config
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            json.dumps({"fake_agent": {"responses_api_agents": {"simple_agent": {"datasets": [dataset]}}}})
        )
        return {"config_paths": [str(config_path)], **safe_load(config_path.read_text()), **plan_args}

    def _prepare_module(self, tmp_path: Path, rows: list[dict]) -> MagicMock:
        def prepare() -> Path:
            output = tmp_path / "benchmark.jsonl"
            output.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return output

        module = MagicMock()
        module.prepare.side_effect = prepare
        return module

    def _run(self, config: dict, module: MagicMock | None = None) -> None:
        with (
            patch("nemo_gym.cli.eval.get_global_config_dict", return_value=OmegaConf.create(config)),
            patch("nemo_gym.cli.eval.importlib.import_module", return_value=module or MagicMock()),
        ):
            plan_benchmark()

    def test_writes_the_plan_file(self, tmp_path: Path) -> None:
        out = tmp_path / "plans" / "plan.json"
        rows = [_row("b"), _row("a")]

        self._run(self._config(tmp_path, plan_output_fpath=str(out)), self._prepare_module(tmp_path, rows))

        plan = EvalPlan.model_validate_json(out.read_text())
        assert [task.task_id.task_id for task in plan.tasks] == ["a", "b"]
        assert plan.benchmark.name == "fake_bench"

    def test_task_ids_select_a_subset(self, tmp_path: Path) -> None:
        out = tmp_path / "plan.json"
        config = self._config(tmp_path, plan_output_fpath=str(out), plan_task_ids=["b"])

        self._run(config, self._prepare_module(tmp_path, [_row("a"), _row("b")]))

        assert [task["task_id"]["task_id"] for task in json.loads(out.read_text())["tasks"]] == ["b"]

    def test_prompt_templates_are_refused(self, tmp_path: Path, capsys) -> None:
        config = self._config(
            tmp_path, prompt_config="benchmarks/prompts/generic/math.yaml", plan_output_fpath=str(tmp_path / "p")
        )

        with pytest.raises(SystemExit):
            self._run(config)

        assert "plans do not support prompt templates yet" in " ".join(capsys.readouterr().out.split())
        assert not (tmp_path / "p").exists()

    def test_several_benchmarks_are_refused(self, tmp_path: Path, capsys) -> None:
        config = self._config(tmp_path, plan_output_fpath=str(tmp_path / "p"))
        second = json.loads(json.dumps(config["fake_agent"]))
        second["responses_api_agents"]["simple_agent"]["datasets"][0]["name"] = "other_bench"
        config["other_agent"] = second

        with pytest.raises(SystemExit):
            self._run(config)

        assert "A plan covers exactly one benchmark, but the config declares 2" in " ".join(
            capsys.readouterr().out.split()
        )
        assert not (tmp_path / "p").exists()

    def test_out_is_required(self, tmp_path: Path, capsys) -> None:
        with pytest.raises(SystemExit):
            self._run(self._config(tmp_path))

        assert "--out" in capsys.readouterr().out

    def test_schema_flag_prints_the_schema(self, tmp_path: Path, capsys) -> None:
        self._run({"print_plan_schema": True})

        assert json.loads(capsys.readouterr().out)["title"] == "EvalPlan"
