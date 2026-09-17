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
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from omegaconf import DictConfig

import nemo_gym.cli.eval as cli_eval
import nemo_gym.rollout_collection as rollout_collection
from nemo_gym import NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, PARENT_DIR
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


@pytest.mark.parametrize("split", ["example", "train"])
def test_e2e_prepares_only_requested_data_and_manages_servers(monkeypatch, tmp_path, split):
    datasets = []
    for kind in ("example", "train"):
        path = tmp_path / f"{kind}.jsonl"
        path.write_text(json.dumps({"responses_create_params": {"input": kind}}) + "\n")
        datasets.append({"name": kind, "type": kind, "jsonl_fpath": str(path)})
    agent = _make_agent_instance_config("sample_agent", datasets)
    config = DictConfig(
        {
            "sample_agent": agent.server_type_config_dict,
            "split": split,
            "output_jsonl_fpath": str(tmp_path / "results/rollouts.jsonl"),
            "limit": 1,
            "disable_aggregation": True,
        }
    )
    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: config)
    servers = Mock()
    monkeypatch.setattr(cli_eval, "RunHelper", Mock(return_value=servers))
    collector = Mock(run_from_config=AsyncMock())
    monkeypatch.setattr(rollout_collection, "RolloutCollectionHelper", Mock(return_value=collector))

    cli_eval.e2e_rollout_collection()

    prepared = tmp_path / f"results/rollouts/preprocessed_datasets/{split}.jsonl"
    row = json.loads(prepared.read_text())
    assert row["responses_create_params"]["input"] == split
    assert row["task_source"] == "sample_agent"
    assert len(list(prepared.parent.glob("*.jsonl"))) == 1
    collected_config = collector.run_from_config.call_args.args[0]
    assert collected_config.input_jsonl_fpath == str(prepared)
    assert collected_config.limit == 1
    servers.start.assert_called_once_with(None)
    servers.shutdown.assert_called_once_with()
    assert len(config.sample_agent.responses_api_agents.simple_agent.datasets) == 2


def test_package_example_eval_ignores_conflicting_checkout_sidecars(monkeypatch, tmp_path):
    relative = Path("environments/workplace_assistant/data/example.jsonl")
    packaged_data = (PARENT_DIR / relative).read_bytes()
    original_metrics = (PARENT_DIR / relative.with_name("example_metrics.json")).read_bytes()
    checkout = tmp_path / "checkout"
    package = tmp_path / "package"
    for root in (checkout, package):
        (root / relative).parent.mkdir(parents=True)
    (package / relative).write_bytes(packaged_data)
    package_metrics = package / relative.with_name("example_metrics.json")
    package_metrics.write_bytes(original_metrics)
    (checkout / relative).write_text('{"responses_create_params":{"input":"wrong checkout sample"}}\n')
    checkout_metrics = checkout / relative.with_name("example_metrics.json")
    checkout_metrics.write_bytes(original_metrics)
    checkout_prepared = checkout / relative.with_name("example_prepare.jsonl")
    checkout_prepared.write_text("keep checkout output")
    monkeypatch.chdir(checkout)
    monkeypatch.setenv(NEMO_GYM_EXTRA_ROOTS_ENV_VAR_NAME, str(package))
    agent = _make_agent_instance_config(
        "sample_agent",
        [{"name": name, "type": "example", "jsonl_fpath": str(relative)} for name in ("example", "example_again")],
    )
    config = DictConfig(
        {
            "sample_agent": agent.server_type_config_dict,
            "second_agent": agent.server_type_config_dict,
            "split": "example",
            "output_jsonl_fpath": "results/workplace_assistant.jsonl",
            "limit": 1,
            "disable_aggregation": True,
        }
    )
    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(cli_eval, "RunHelper", Mock(return_value=Mock()))
    collector = Mock(run_from_config=AsyncMock())
    monkeypatch.setattr(rollout_collection, "RolloutCollectionHelper", Mock(return_value=collector))

    cli_eval.e2e_rollout_collection()

    collected = collector.run_from_config.call_args.args[0]
    prepared = Path(collected.input_jsonl_fpath)
    rows = [json.loads(line) for line in prepared.read_text().splitlines()]
    assert len(rows) == 4 * len(packaged_data.splitlines())
    assert rows[0]["responses_create_params"] == json.loads(packaged_data.splitlines()[0])["responses_create_params"]
    assert {row["task_source"] for row in rows} == {"sample_agent", "second_agent"}
    assert len(list(prepared.parent.glob("inputs/*/*/example_metrics.json"))) == 4
    assert checkout_metrics.read_bytes() == original_metrics
    assert checkout_prepared.read_text() == "keep checkout output"
    assert (package / relative).read_bytes() == packaged_data
    assert package_metrics.read_bytes() == original_metrics
    assert set(package.rglob("*.json*")) == {package / relative, package_metrics}
    assert config.sample_agent.responses_api_agents.simple_agent.datasets[0].jsonl_fpath == str(relative)


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
        # supported example split.
        assert "No dataset of type `train`" in message
        assert "example_agent: example (type: example)" in message
        assert "--split example" in message

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
