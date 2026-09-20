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
import subprocess
import sys
from pathlib import Path
from textwrap import dedent

import pytest
from omegaconf import DictConfig

from nemo_gym.cli.eval import _validate_prepared_split_file_exists, _validate_split_datasets_declared
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


def test_aggregate_cli_exits_nonzero_after_saving_failed_agent_entry(tmp_path: Path) -> None:
    """A single-agent HTTP 500 must fail the real CLI process after saving error details."""
    shard = tmp_path / "shard.jsonl"
    shard.write_text(
        json.dumps({"agent_ref": {"name": "agent_a"}, "_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 1.0})
        + "\n"
    )
    output = tmp_path / "rollouts.jsonl"
    script = dedent("""\
        from types import SimpleNamespace
        from unittest.mock import AsyncMock, patch

        from aiohttp import ClientResponseError
        from nemo_gym.cli.main import main
        from nemo_gym.rollout_collection import RolloutCollectionHelper

        error = ClientResponseError(
            request_info=SimpleNamespace(real_url="http://agent/aggregate_metrics"),
            history=(), status=500, message="aggregation unavailable",
        )
        client = SimpleNamespace(post=AsyncMock(side_effect=error))
        with patch.object(RolloutCollectionHelper, "setup_server_client", return_value=client):
            main()
        """)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            "eval",
            "aggregate",
            "--input",
            str(shard),
            "--output",
            str(output),
            "--no-health-check",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert completed.returncode != 0, completed.stdout + completed.stderr
    assert "Aggregation failed for agents: agent_a" in completed.stdout + completed.stderr
    metrics = json.loads((tmp_path / "rollouts_aggregate_metrics.json").read_text())
    assert metrics[0]["aggregation_error"]["http_status"] == 500
    assert json.loads(output.read_text()) == json.loads(shard.read_text())
