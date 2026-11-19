# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Collation checks each row against the request model of the server that receives it."""

import json
from pathlib import Path
from typing import Any

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_gym.config.global_config import GlobalConfigDictParser
from nemo_gym.data.dataset_preparation import TrainDataProcessor


_GOOD_REQUEST = {"input": "What is 2+2?"}
# A Chat Completions tool: name, description and parameters sit under "function" instead of on the tool itself.
_CHAT_COMPLETIONS_REQUEST = {
    "input": "What is 2+2?",
    "tools": [{"type": "function", "function": {"name": "add", "parameters": {"type": "object", "properties": {}}}}],
}
_SWE_TOGETHER_TASK_DATA = {
    "task_id": "task-1",
    "image": "example.com/task-1:tag",
    "image_digest": "sha256:" + "0" * 64,
}


def _write(tmp_path: Path, rows: list[dict], name: str = "rows") -> Path:
    path = tmp_path / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _dataset(path: Path, **extra: Any) -> dict:
    dataset = {"name": path.stem, "type": "train", "jsonl_fpath": str(path), "license": "Apache 2.0"}
    if extra.get("type") == "benchmark":
        dataset["prepare_script"] = "prepare.py"
    return dataset | extra


def _config(datasets: list[dict], *environment_servers: tuple[str, str], **top_level: Any) -> DictConfig:
    """Agent ``agent`` declares ``datasets``; each ``(instance, implementation)`` environment server fronts it."""
    config: dict[str, Any] = {
        "agent": {
            "responses_api_agents": {
                "simple_agent": {
                    "entrypoint": "app.py",
                    "datasets": datasets,
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                }
            }
        }
    }
    for instance, implementation in environment_servers:
        config[instance] = {
            "environment_servers": {
                implementation: {
                    "entrypoint": "app.py",
                    "agent_server": {"type": "responses_api_agents", "name": "agent"},
                }
            }
        }
    return OmegaConf.create(config | top_level)


def _collate_metrics(config: DictConfig, mode: str, *, with_config: bool = True) -> dict:
    processor = TrainDataProcessor()
    if with_config:
        processor._global_config_dict = config
    configs = GlobalConfigDictParser().filter_for_server_instance_configs(config)
    return processor.validate_samples_and_aggregate_metrics(
        [c for c in configs if c.datasets], overwrite_metrics_conflicts=False, task_data_validation=mode
    )


def _two_agents(
    servers: dict[str, str], *, datasets: list[dict] | None = None, shared_resources_datasets: list[dict] | None = None
) -> dict[str, Any]:
    """Agents ``agent_a`` and ``agent_b``; ``servers`` maps each agent to the implementation of the server fronting it.

    ``agent_a`` declares ``datasets``. With ``shared_resources_datasets``, both agents reference resources server
    ``shared_rs``, which declares those datasets.
    """
    config: dict[str, Any] = {}
    for agent in ("agent_a", "agent_b"):
        inner: dict[str, Any] = {
            "entrypoint": "app.py",
            "model_server": {"type": "responses_api_models", "name": "policy_model"},
        }
        if agent == "agent_a" and datasets:
            inner["datasets"] = datasets
        if shared_resources_datasets is not None:
            inner["resources_server"] = {"type": "resources_servers", "name": "shared_rs"}
        config[agent] = {"responses_api_agents": {"simple_agent": inner}}
        if agent in servers:
            config[f"{agent}_env"] = {
                "environment_servers": {
                    servers[agent]: {
                        "entrypoint": "app.py",
                        "agent_server": {"type": "responses_api_agents", "name": agent},
                    }
                }
            }
    if shared_resources_datasets is not None:
        config["shared_rs"] = {
            "resources_servers": {
                "example_single_tool_call": {
                    "entrypoint": "app.py",
                    "domain": "other",
                    "datasets": shared_resources_datasets,
                }
            }
        }
    return config


_BAD_CONTEXT_ROW = {"responses_create_params": _GOOD_REQUEST, "agent_context": {"user": {"values": {}}}}


def _single_dataset_state(config: DictConfig):
    """The validator state of the one dataset in ``config``, checked as collation checks it."""
    processor = TrainDataProcessor()
    processor._global_config_dict = config
    (c,) = [c for c in GlobalConfigDictParser().filter_for_server_instance_configs(config) if c.datasets]
    (d,) = c.datasets
    return processor._validate_samples_and_aggregate_metrics_single_dataset(
        d, request_check=processor._request_check_for(c, d, task_data_validation="error")
    )


class TestTaskRows:
    def test_a_taskset_row_is_checked_against_its_environment_servers_task_input(self, tmp_path):
        rows = [{"responses_create_params": _CHAT_COMPLETIONS_REQUEST}] * 6 + [
            {"responses_create_params": _GOOD_REQUEST}
        ]
        path = _write(tmp_path, rows)
        config = _config(
            [_dataset(path, taskset="math")],
            ("env", "single_agent_turn"),
            environment_server_routes={"math": "env"},
        )

        with pytest.raises(ValueError) as error:
            _collate_metrics(config, "error")

        message = str(error.value)
        assert f"{path}: 6/7 rows failed request validation" in message
        assert "rows go to: env (single_agent_turn) (7 rows)" in message
        assert "row 0: env (single_agent_turn): task_input.responses_create_params.tools.0." in message
        assert "row 4:" in message and "row 5:" not in message  # five reasons, then a count
        assert "... and 1 more rows" in message
        assert not (tmp_path / "rows_metrics.json").exists()

    def test_a_task_row_is_routed_by_its_own_taskset(self, tmp_path):
        good = {"task_id": {"taskset": "swe", "task_id": "1"}, "task_input": {"task_data": _SWE_TOGETHER_TASK_DATA}}
        extra_key = {
            "task_id": {"taskset": "swe", "task_id": "2"},
            "task_input": {"task_data": _SWE_TOGETHER_TASK_DATA, "x": 1},
        }
        bad_digest = {
            "task_id": {"taskset": "swe", "task_id": "3"},
            "task_input": {"task_data": _SWE_TOGETHER_TASK_DATA | {"image_digest": "sha256:short"}},
        }
        # The entry sets no taskset: each task row names its own.
        path = _write(tmp_path, [good, extra_key, bad_digest])
        config = _config([_dataset(path)], ("swe_env", "swe_together"), environment_server_routes={"swe": "swe_env"})

        with pytest.raises(ValueError) as error:
            _collate_metrics(config, "error")

        message = str(error.value)
        assert "2/3 rows failed" in message
        assert "row 1: swe_env (swe_together): task_input.x: Extra inputs are not permitted" in message
        assert "row 2: swe_env (swe_together): task_input.task_data.image_digest:" in message

    def test_a_task_row_without_a_route_is_rejected(self, tmp_path):
        rows = [{"task_id": {"taskset": f"orphan{index}", "task_id": "1"}, "task_input": {}} for index in range(7)]
        path = _write(tmp_path, rows)
        config = _config([_dataset(path)], ("env", "single_agent_turn"), environment_server_routes={"math": "env"})

        with pytest.raises(ValueError) as error:
            _collate_metrics(config, "error")

        message = str(error.value)
        assert "No environment server route is configured for taskset 'orphan0'" in message
        # One receiver entry for every unrouted taskset, so the report stays bounded.
        assert "rows go to: no environment_server_routes entry (7 rows)" in message

    def test_routes_are_read_once_per_dataset_however_many_rows_it_has(self, tmp_path, monkeypatch):
        rows = [{"responses_create_params": _GOOD_REQUEST}] * 50
        path = _write(tmp_path, rows)
        routes = {f"other{index}": "env" for index in range(500)} | {"math": "env"}
        config = _config(
            [_dataset(path, taskset="math")], ("env", "single_agent_turn"), environment_server_routes=routes
        )
        calls = []
        original = TrainDataProcessor._routes
        monkeypatch.setattr(TrainDataProcessor, "_routes", lambda self: calls.append(1) or original(self))

        _collate_metrics(config, "error")

        assert len(calls) == 1

    def test_a_server_without_a_task_input_module_leaves_task_rows_unchecked(self, tmp_path, monkeypatch, capsys):
        (tmp_path / "environment_servers" / "untyped_env").mkdir(parents=True)
        # A source marker makes the folder a server folder, not just a folder of that name.
        (tmp_path / "environment_servers" / "untyped_env" / "requirements.txt").write_text("")
        monkeypatch.chdir(tmp_path)
        task_row = {"task_id": {"taskset": "math", "task_id": "1"}, "task_input": {"anything": 1}}
        task_rows = _write(tmp_path, [task_row], "tasks")
        flat_rows = _write(tmp_path, [{"responses_create_params": _CHAT_COMPLETIONS_REQUEST}], "flat")
        config = _config(
            [_dataset(task_rows), _dataset(flat_rows, type="example")],
            ("env", "untyped_env"),
            environment_server_routes={"math": "env"},
        )

        with pytest.raises(ValueError) as error:
            _collate_metrics(config, "error")

        # Task rows are not checked, as a server without task_data.py skips that check.
        printed = capsys.readouterr().out
        assert "tasks.jsonl: none of 1 rows failed request validation, but some were not fully checked." in printed
        assert "1 rows not checked: untyped_env ships no task_input.py" in printed
        # Flat rows still need to be run requests, which every server that takes flat rows requires.
        message = str(error.value)
        assert "row 0: env (untyped_env): responses_create_params.tools.0." in message
        assert "1 rows checked only as run requests: untyped_env ships no task_input.py" in message

    def test_a_task_row_routed_to_a_relay_is_rejected(self, tmp_path):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config(
            [_dataset(path, taskset="math")], ("env", "legacy_agent"), environment_server_routes={"math": "env"}
        )

        with pytest.raises(
            ValueError, match=r"env \(legacy_agent\) is a relay \(its task_input.py sets TaskInput = None\)"
        ):
            _collate_metrics(config, "error")

    def test_a_prompt_templated_taskset_row_passes_without_a_source_request(self, tmp_path):
        prompt = tmp_path / "prompt.yaml"
        prompt.write_text("user: 'Q: {question}'\n")
        path = _write(tmp_path, [{"question": "What is 2+2?"}])
        config = _config(
            [_dataset(path, type="benchmark", taskset="math", prompt_config=str(prompt))],
            ("env", "single_agent_turn"),
            environment_server_routes={"math": "env"},
        )

        assert _collate_metrics(config, "error")["benchmark"].number_of_examples == 1


class TestFlatRows:
    def test_a_flat_row_is_checked_against_single_agent_turn_legacys_task_input(self, tmp_path):
        bad_context = {"responses_create_params": _GOOD_REQUEST, "agent_context": {"user": {"values": {"a": "b"}}}}
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}, bad_context])
        config = _config([_dataset(path)], ("env", "single_agent_turn_legacy"))

        with pytest.raises(ValueError) as error:
            _collate_metrics(config, "error")

        message = str(error.value)
        assert "1/2 rows failed" in message
        assert "row 1: env (single_agent_turn_legacy): task_input.agent_context:" in message
        assert "single_agent_turn has one role, 'agent'" in message

    def test_a_typed_server_without_flat_rows_rejects_them(self, tmp_path):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config([_dataset(path)], ("env", "single_agent_turn"))

        with pytest.raises(ValueError, match=r"env \(single_agent_turn\) accepts only task rows"):
            _collate_metrics(config, "error")

    def test_a_relay_passes_flat_rows_to_the_agents_run_request(self, tmp_path):
        path = _write(
            tmp_path,
            [{"responses_create_params": _GOOD_REQUEST}, {"responses_create_params": _CHAT_COMPLETIONS_REQUEST}],
        )
        config = _config([_dataset(path)], ("env", "legacy_agent"))

        with pytest.raises(ValueError, match=r"row 1: env \(legacy_agent\): responses_create_params\.tools\.0\."):
            _collate_metrics(config, "error")

    def test_an_agent_fronted_by_several_environment_servers_rejects_flat_rows(self, tmp_path):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config([_dataset(path)], ("one", "single_agent_turn_legacy"), ("two", "legacy_agent"))

        with pytest.raises(ValueError, match="fronted by several environment servers"):
            _collate_metrics(config, "error")

    def test_rows_of_a_resources_server_without_an_agent_are_checked_as_run_requests(self, tmp_path):
        # The agent comes from another config at run time, so the only receiver known is "some server's /run".
        path = _write(
            tmp_path,
            [{"responses_create_params": _GOOD_REQUEST}, {"responses_create_params": _CHAT_COMPLETIONS_REQUEST}],
        )
        config = OmegaConf.create(
            {
                "resources": {
                    "resources_servers": {
                        "example_single_tool_call": {
                            "entrypoint": "app.py",
                            "domain": "other",
                            "datasets": [_dataset(path)],
                        }
                    }
                }
            }
        )

        with pytest.raises(ValueError) as error:
            _collate_metrics(config, "error")

        message = str(error.value)
        assert "1/2 rows failed" in message
        assert "row 1: resources (no single agent): responses_create_params.tools.0." in message
        # The configuration error that left the dataset without an agent stays in the report.
        assert "2 rows checked only as run requests: no single agent runs the rows of resources:" in message
        assert "no agent in the running config references resources server 'resources'" in message

    def test_taskset_routing_mode_rejects_flat_rows(self, tmp_path):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config([_dataset(path)], environment_routing_mode="taskset")

        with pytest.raises(
            ValueError, match="taskset routing requires rows containing task_id.taskset and task_input"
        ):
            _collate_metrics(config, "error")


class TestFlatRowRouting:
    """A flat row is checked against every agent rollout collection sends it to."""

    def test_agent_map_reroutes_rows_before_the_receiver_is_chosen(self, tmp_path):
        # agent_a's server takes only task rows; the map sends the rows to agent_b's relay instead.
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _two_agents(
            {"agent_a": "single_agent_turn", "agent_b": "legacy_agent"}, datasets=[_dataset(path)]
        ) | {"agent_map": {"agent_a": "agent_b"}}

        state = _single_dataset_state(OmegaConf.create(config))

        assert state.number_of_rejected_source_rows == 0
        assert dict(state.receivers) == {"agent_b_env (legacy_agent)": 1}

    def test_agent_map_default_applies_when_the_source_has_no_entry(self, tmp_path):
        path = _write(tmp_path, [_BAD_CONTEXT_ROW])
        config = _two_agents(
            {"agent_a": "legacy_agent", "agent_b": "single_agent_turn_legacy"}, datasets=[_dataset(path)]
        ) | {"agent_map": {"_default": "agent_b"}}

        with pytest.raises(ValueError, match=r"agent_b_env \(single_agent_turn_legacy\): task_input\.agent_context"):
            _collate_metrics(OmegaConf.create(config), "error")

    def test_agent_name_is_the_default_agent(self, tmp_path):
        # +agent_name is the documented spelling of agent_map._default.
        good = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}], name="good")
        config = _two_agents(
            {"agent_a": "single_agent_turn", "agent_b": "legacy_agent"}, datasets=[_dataset(good)]
        ) | {"agent_name": "agent_b"}
        state = _single_dataset_state(OmegaConf.create(config))
        assert state.number_of_rejected_source_rows == 0
        assert dict(state.receivers) == {"agent_b_env (legacy_agent)": 1}

        bad = _write(tmp_path, [_BAD_CONTEXT_ROW], name="bad")
        config = _two_agents(
            {"agent_a": "legacy_agent", "agent_b": "single_agent_turn_legacy"}, datasets=[_dataset(bad)]
        ) | {"agent_name": "agent_b"}
        with pytest.raises(ValueError, match=r"agent_b_env \(single_agent_turn_legacy\): task_input\.agent_context"):
            _collate_metrics(OmegaConf.create(config), "error")

    def test_fan_out_checks_every_destination(self, tmp_path):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}, _BAD_CONTEXT_ROW])
        config = _two_agents(
            {"agent_a": "legacy_agent", "agent_b": "single_agent_turn_legacy"}, datasets=[_dataset(path)]
        ) | {"fan_out": {"agent_a": ["agent_a", "agent_b"]}}

        with pytest.raises(ValueError) as error:
            _collate_metrics(OmegaConf.create(config), "error")

        message = str(error.value)
        assert "1/2 rows failed" in message
        assert (
            "rows go to: agent_a_env (legacy_agent) (2 rows), agent_b_env (single_agent_turn_legacy) (2 rows)"
            in message
        )
        assert "row 1: agent_b_env (single_agent_turn_legacy): task_input.agent_context" in message

    def test_a_pin_on_one_dataset_routes_every_flat_row_of_its_source(self, tmp_path):
        pinned = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}], name="pinned")
        unpinned = _write(tmp_path, [_BAD_CONTEXT_ROW], name="unpinned")
        config = _two_agents(
            {"agent_a": "legacy_agent", "agent_b": "single_agent_turn_legacy"},
            shared_resources_datasets=[
                _dataset(pinned, type="benchmark", agent="agent_b"),
                _dataset(unpinned, type="benchmark"),
            ],
        )

        with pytest.raises(ValueError) as error:
            _collate_metrics(OmegaConf.create(config), "error")

        assert f"{unpinned}: 1/1 rows failed" in str(error.value)
        assert "agent_b_env (single_agent_turn_legacy): task_input.agent_context" in str(error.value)

    def test_conflicting_pins_on_one_source_are_refused(self, tmp_path):
        first = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}], name="first")
        second = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}], name="second")
        config = _two_agents(
            {"agent_a": "legacy_agent", "agent_b": "legacy_agent"},
            shared_resources_datasets=[
                _dataset(first, type="benchmark", agent="agent_a"),
                _dataset(second, type="benchmark", agent="agent_b"),
            ],
        )

        with pytest.raises(ValueError, match=r"'shared_rs': its datasets pin conflicting agents"):
            _collate_metrics(OmegaConf.create(config), "error")

    @pytest.mark.parametrize("environment_server", ["legacy_agent", None])
    def test_a_row_without_a_request_is_counted_as_not_checked(self, tmp_path, environment_server):
        # toolsandbox_agent and aviary_agent fill in the request; the relay and the agent's /run alike pass it on.
        path = _write(tmp_path, [{"task_idx": 0}, {"responses_create_params": _GOOD_REQUEST}])
        servers = (("env", environment_server),) if environment_server else ()
        state = _single_dataset_state(_config([_dataset(path)], *servers))

        assert state.number_of_rejected_source_rows == 0
        assert state.not_checked_source_rows == {
            "no responses_create_params, which some agents (toolsandbox_agent, aviary_agent) fill in and others refuse": 1
        }
        # The row stays out of the metrics, as it did before collation checked requests.
        assert state.metrics.number_of_examples == 1

    def test_a_row_without_a_request_is_rejected_by_a_server_that_requires_one(self, tmp_path):
        path = _write(tmp_path, [{"task_idx": 0}])
        config = _config([_dataset(path)], ("env", "single_agent_turn_legacy"))

        with pytest.raises(ValueError, match="1/1 rows failed"):
            _collate_metrics(config, "error")

    @pytest.mark.parametrize(
        ("controls", "rejected"),
        [
            ({"_ng_skip_token_capture": "yes"}, True),
            ({"_ng_skip_token_capture": True}, False),
            ({"_ng_skip_token_capture": False}, False),
            ({"_ng_skip_token_capture": None}, False),
            ({}, False),
        ],
    )
    def test_flat_row_controls_are_checked_before_materialization_drops_them(self, tmp_path, controls, rejected):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST} | controls])
        state = _single_dataset_state(_config([_dataset(path)], ("env", "single_agent_turn_legacy")))

        assert state.number_of_rejected_source_rows == int(rejected)

    @pytest.mark.parametrize(
        "routing, message",
        [
            ({"environment_routing_mode": "legacy"}, "environment_server_name is required"),
            (
                {"environment_routing_mode": "Legacy", "environment_server_name": "env"},
                "Unknown environment_routing_mode 'Legacy'",
            ),
        ],
    )
    def test_a_routing_mode_rollout_collection_refuses_is_reported(self, tmp_path, routing, message):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config([_dataset(path)], ("env", "single_agent_turn_legacy"), **routing)

        with pytest.raises(ValueError, match=message):
            _collate_metrics(config, "error")


class TestReportsAndMetrics:
    def test_metrics_come_from_the_source_rows(self, tmp_path):
        """A valid flat row in a taskset dataset and a prompt-templated row keep the metrics they had before."""
        prompt = tmp_path / "prompt.yaml"
        prompt.write_text("user: 'Q: {question}'\n")
        flat = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST | {"temperature": 0.5}}], "flat")
        templated = _write(tmp_path, [{"question": "q"}, {"question": "a longer question"}], "tpl")
        datasets = [
            _dataset(flat, taskset="math"),
            _dataset(templated, type="benchmark", taskset="math", prompt_config=str(prompt)),
        ]
        config = _config(datasets, ("env", "single_agent_turn"), environment_server_routes={"math": "env"})

        checked = _collate_metrics(config, "error")
        unchecked = _collate_metrics(config, "error", with_config=False)

        for dataset_type in ("train", "benchmark"):
            assert checked[dataset_type].aggregate() == unchecked[dataset_type].aggregate()
        assert checked["train"].aggregate().temperature.average == 0.5

    def test_warn_mode_prints_the_report_and_drops_rejected_rows_from_metrics(self, tmp_path, capsys):
        path = _write(
            tmp_path,
            [{"responses_create_params": _GOOD_REQUEST}, {"responses_create_params": _CHAT_COMPLETIONS_REQUEST}],
        )
        config = _config(
            [_dataset(path, taskset="math")], ("env", "single_agent_turn"), environment_server_routes={"math": "env"}
        )

        metrics = _collate_metrics(config, "warn")

        assert "1/2 rows failed request validation" in capsys.readouterr().out
        assert metrics["train"].number_of_examples == 1

    def test_off_mode_reports_nothing_and_still_drops_rejected_rows(self, tmp_path, capsys):
        path = _write(
            tmp_path,
            [{"responses_create_params": _GOOD_REQUEST}, {"responses_create_params": _CHAT_COMPLETIONS_REQUEST}],
        )
        config = _config(
            [_dataset(path, taskset="math")], ("env", "single_agent_turn"), environment_server_routes={"math": "env"}
        )

        metrics = _collate_metrics(config, "off")

        assert "request validation" not in capsys.readouterr().out
        assert metrics["train"].number_of_examples == 1

    def test_a_task_input_module_that_does_not_load_fails_error_mode_and_is_counted_in_warn_mode(
        self, tmp_path, monkeypatch, capsys
    ):
        server_dir = tmp_path / "environment_servers" / "broken_env"
        server_dir.mkdir(parents=True)
        (server_dir / "task_input.py").write_text("import does_not_exist_anywhere\n")
        monkeypatch.chdir(tmp_path)
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config(
            [_dataset(path, taskset="math")], ("env", "broken_env"), environment_server_routes={"math": "env"}
        )

        from nemo_gym.data.task_input import TaskInputModuleError

        with pytest.raises(TaskInputModuleError, match="Failed to import"):
            _collate_metrics(config, "error")
        with pytest.warns(UserWarning, match="Not checking rows sent to broken_env"):
            metrics = _collate_metrics(config, "warn")
        assert "1 rows not checked: the task input module of broken_env does not load" in capsys.readouterr().out
        assert metrics["train"].number_of_examples == 1
