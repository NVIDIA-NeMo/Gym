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
        path = _write(tmp_path, [{"task_id": {"taskset": "orphan", "task_id": "1"}, "task_input": {}}])
        config = _config([_dataset(path)], ("env", "single_agent_turn"), environment_server_routes={"math": "env"})

        with pytest.raises(ValueError, match="no environment_server_routes entry for taskset 'orphan'"):
            _collate_metrics(config, "error")

    def test_a_task_row_routed_to_a_relay_is_rejected(self, tmp_path):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config(
            [_dataset(path, taskset="math")], ("env", "legacy_agent"), environment_server_routes={"math": "env"}
        )

        with pytest.raises(ValueError, match=r"env \(legacy_agent\) relays to an agent's /run, which takes flat rows"):
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

    def test_taskset_routing_mode_rejects_flat_rows(self, tmp_path):
        path = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST}])
        config = _config([_dataset(path)], environment_routing_mode="taskset")

        with pytest.raises(ValueError, match="environment_routing_mode=taskset refuses flat rows"):
            _collate_metrics(config, "error")


class TestReportsAndMetrics:
    def test_metrics_come_from_the_source_rows(self, tmp_path):
        """A valid flat row in a taskset dataset and a prompt-templated row keep the metrics they had before."""
        prompt = tmp_path / "prompt.yaml"
        prompt.write_text("user: 'Q: {question}'\n")
        flat = _write(tmp_path, [{"responses_create_params": _GOOD_REQUEST | {"temperature": 0.5}}], "flat")
        templated = _write(
            tmp_path, [{"question": "q"}, {"question": "r", "responses_create_params": _GOOD_REQUEST}], "tpl"
        )
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
