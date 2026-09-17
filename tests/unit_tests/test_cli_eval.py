# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import sys
from pathlib import Path

import pytest
import requests
from omegaconf import DictConfig
from pytest import MonkeyPatch

import nemo_gym.global_config
import nemo_gym.server_utils
from nemo_gym.cli.eval import _validate_prepared_split_file_exists, _validate_split_datasets_declared
from nemo_gym.cli.main import main
from nemo_gym.config_types import ConfigError, ResponsesAPIAgentServerInstanceConfig


def _make_agent_instance_config(name: str, dataset_specs: list) -> ResponsesAPIAgentServerInstanceConfig:
    server_type_config_dict = {
        "responses_api_agents": {
            "simple_agent": {
                "host": "127.0.0.1",
                "port": 12345,
                "entrypoint": "app.py",
                "datasets": [
                    {
                        "name": d["name"],
                        "type": d["type"],
                        "jsonl_fpath": d.get("jsonl_fpath", f"path/{d['name']}.jsonl"),
                        "license": None if d["type"] == "example" else "Apache 2.0",
                    }
                    for d in dataset_specs
                ],
                "resources_server": {
                    "type": "resources_servers",
                    "name": f"{name}_resources_server",
                },
                "model_server": {
                    "type": "responses_api_models",
                    "name": "policy_model",
                },
            }
        }
    }
    return ResponsesAPIAgentServerInstanceConfig(
        name=name,
        server_type_config_dict=DictConfig(server_type_config_dict),
        responses_api_agents=server_type_config_dict["responses_api_agents"],
    )


class TestValidateSplitDatasetsDeclared:
    def test_passes_when_a_dataset_of_the_split_type_is_declared(self) -> None:
        configs = [_make_agent_instance_config("my_agent", [{"name": "train_data", "type": "train"}])]
        _validate_split_datasets_declared("train", configs)

    def test_fails_fast_when_only_example_data_is_declared(self) -> None:
        configs = [
            _make_agent_instance_config(
                "example_agent",
                [{"name": "example", "type": "example", "jsonl_fpath": "resources_servers/x/data/example.jsonl"}],
            )
        ]
        with pytest.raises(ConfigError) as exc_info:
            _validate_split_datasets_declared("train", configs)
        message = str(exc_info.value)
        # The error must name the requested split, list what is declared, and give the
        # copy-pasteable --no-serve recipe for the example file.
        assert "No dataset of type `train`" in message
        assert "example_agent: example (type: example)" in message
        assert "--no-serve --input resources_servers/x/data/example.jsonl" in message

    def test_fails_when_no_datasets_are_declared_at_all(self) -> None:
        configs = [_make_agent_instance_config("bare_agent", [])]
        with pytest.raises(ConfigError, match=r"- \(none\)"):
            _validate_split_datasets_declared("validation", configs)

    def test_mismatched_split_lists_declared_types(self) -> None:
        configs = [_make_agent_instance_config("val_agent", [{"name": "val_data", "type": "validation"}])]
        with pytest.raises(ConfigError, match=r"val_agent: val_data \(type: validation\)"):
            _validate_split_datasets_declared("train", configs)


class TestValidatePreparedSplitFileExists:
    def test_passes_when_the_file_exists(self, tmp_path: Path) -> None:
        fpath = tmp_path / "train.jsonl"
        fpath.write_text("{}\n")
        _validate_prepared_split_file_exists(fpath, "train", tmp_path)

    def test_fails_with_the_split_and_the_files_actually_prepared(self, tmp_path: Path) -> None:
        (tmp_path / "validation.jsonl").write_text("{}\n")
        with pytest.raises(ConfigError, match=r"split `train`.*\['validation.jsonl'\]"):
            _validate_prepared_split_file_exists(tmp_path / "train.jsonl", "train", tmp_path)

    def test_fails_with_none_when_the_output_dir_is_missing(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(ConfigError, match=r"none"):
            _validate_prepared_split_file_exists(missing_dir / "train.jsonl", "train", missing_dir)


class TestEvalRunNoServeWithoutHeadServer:
    """`gym eval run --no-serve` collects against servers that are already running, so nothing listening
    on the head server port is the user's most likely mistake (#2687). Driven through the real `main()`
    so the whole path is exercised: the config is parsed, the rows are materialized, and the head server
    fetch is the only thing faked (it refuses the connection, exactly as a closed port does)."""

    def _arrange_no_serve_run_against_a_closed_port(self, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
        input_path = tmp_path / "input.jsonl"
        input_path.write_text(json.dumps({"responses_create_params": {"input": "hi"}}) + "\n")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gym",
                "eval",
                "run",
                "--no-serve",
                "--agent",
                "simple_agent",
                "-i",
                str(input_path),
                "-o",
                str(tmp_path / "out.jsonl"),
            ],
        )
        # A clean cwd so no repo-local env.yaml is merged in, and a fresh parse of the argv above.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(nemo_gym.global_config, "_GLOBAL_CONFIG_DICT", None)
        # Rich soft-wraps at 80 columns when stdout is not a TTY, which would split the message mid-sentence.
        monkeypatch.setenv("COLUMNS", "1000")

        # `requests.exceptions.ConnectionError` is what `ServerClient.load_from_global_config` catches; it is
        # unrelated to the builtin `ConnectionError`, which would sail straight through.
        def refuse(*args, **kwargs):
            raise requests.exceptions.ConnectionError("[Errno 61] Connection refused")

        monkeypatch.setattr(nemo_gym.server_utils.requests, "get", refuse)

    def test_exits_one_with_a_single_error_line_and_no_traceback(
        self, tmp_path: Path, monkeypatch: MonkeyPatch, capsys
    ) -> None:
        self._arrange_no_serve_run_against_a_closed_port(tmp_path, monkeypatch)

        with pytest.raises(SystemExit) as exit_info:
            main()

        assert exit_info.value.code == 1
        captured = capsys.readouterr()
        assert "Error: Could not connect to the head server at http://127.0.0.1:11000." in captured.out
        assert "Start it with: `gym env start`." in captured.out
        assert "Traceback" not in captured.out + captured.err
        assert "ValueError" not in captured.out + captured.err
