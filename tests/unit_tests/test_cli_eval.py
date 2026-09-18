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
                        **({"prepare_script": "unused_prepare.py"} if d["type"] == "benchmark" else {}),
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
    assert len(rows) == 1
    assert rows[0]["responses_create_params"] == json.loads(packaged_data.splitlines()[0])["responses_create_params"]
    assert {row["task_source"] for row in rows} == {"sample_agent"}
    assert len(list(prepared.parent.glob("inputs/*/*/example_metrics.json"))) == 1
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


@pytest.mark.parametrize("split", ["example", "train", "validation", "benchmark"])
@pytest.mark.parametrize("limit", [1, 5, 8, None, 0])
def test_e2e_limit_preserves_order_split_and_both_repeat_levels(monkeypatch, tmp_path, split, limit):
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"responses_create_params": {"input": str(i)}}) + "\n" for i in range(2)))
    agents = {}
    for name, repeats in [("first", 2), ("second", 3)]:
        dataset_source = tmp_path / f"{name}.jsonl"
        dataset_source.write_bytes(source.read_bytes())
        instance = _make_agent_instance_config(
            name, [{"name": name, "type": split, "jsonl_fpath": str(dataset_source)}]
        )
        dataset = instance.server_type_config_dict.responses_api_agents.simple_agent.datasets[0]
        dataset.num_repeats = repeats
        agents[name] = instance.server_type_config_dict
    config = DictConfig(
        {
            **agents,
            "split": split,
            "output_jsonl_fpath": str(tmp_path / "out/rows.jsonl"),
            "limit": limit,
            "num_repeats": 2,
            "disable_aggregation": True,
        }
    )
    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(cli_eval, "RunHelper", Mock(return_value=Mock()))
    collect = AsyncMock()
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "run_from_config", collect)
    cli_eval.e2e_rollout_collection()
    collected_config = collect.call_args.args[0]
    rows = rollout_collection.RolloutCollectionHelper()._preprocess_rows_from_config(collected_config)
    original = [
        (name, str(i)) for name, repeats in [("first", 2), ("second", 3)] for i in range(2) for _ in range(repeats)
    ]
    expected = original[:limit] if limit else original
    assert [(r["task_source"], r["responses_create_params"]["input"]) for r in rows] == [
        r for r in expected for _ in range(2)
    ]
    assert [config[name].responses_api_agents.simple_agent.datasets[0].num_repeats for name in agents] == [2, 3]
    if limit:
        assert not list(tmp_path.glob("*_prepare.jsonl"))


def test_e2e_limit_does_not_process_large_tail_or_unselected_datasets(monkeypatch, tmp_path):
    source = tmp_path / "large.jsonl"
    prefix = "".join(json.dumps({"responses_create_params": {"input": str(i)}}) + "\n" for i in range(5))
    source.write_text(prefix + "malformed tail must not be parsed\n" * 100_000)
    datasets = [
        {"name": "selected", "type": "train", "jsonl_fpath": str(source)},
        {"name": "unselected", "type": "train", "jsonl_fpath": str(tmp_path / "missing.jsonl")},
        {"name": "other_split", "type": "validation", "jsonl_fpath": str(tmp_path / "missing2.jsonl")},
    ]
    agent = _make_agent_instance_config("sample_agent", datasets)
    config = DictConfig(
        {
            "sample_agent": agent.server_type_config_dict,
            "split": "train",
            "limit": 5,
            "output_jsonl_fpath": str(tmp_path / "out/rows.jsonl"),
            "disable_aggregation": True,
        }
    )
    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(cli_eval, "RunHelper", Mock(return_value=Mock()))
    collect = AsyncMock()
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "run_from_config", collect)
    cli_eval.e2e_rollout_collection()
    prepared = Path(collect.call_args.args[0].input_jsonl_fpath)
    assert len(prepared.read_text().splitlines()) == 5
    staged = list(prepared.parent.glob("inputs/*/*/*.jsonl"))
    assert sum(p.stat().st_size for p in staged) < 5000
    assert not source.with_name("large_prepare.jsonl").exists()
    assert len(config.sample_agent.responses_api_agents.simple_agent.datasets) == 3


@pytest.mark.parametrize("split", ["train", "benchmark"])
def test_e2e_limit_downloads_only_selected_source_or_reports_missing_benchmark(monkeypatch, tmp_path, split):
    import nemo_gym.train_data_utils as data_utils

    source = tmp_path / "selected.jsonl"
    unselected = tmp_path / "unselected.jsonl"
    agent = _make_agent_instance_config(
        "sample_agent",
        [
            {"name": name, "type": split, "jsonl_fpath": str(path)}
            for name, path in [("selected", source), ("unselected", unselected)]
        ],
    )
    if split == "train":
        for dataset in agent.server_type_config_dict.responses_api_agents.simple_agent.datasets:
            dataset.source = {"type": "huggingface", "repo_id": f"test/{dataset.name}"}
    config = DictConfig(
        {
            "sample_agent": agent.server_type_config_dict,
            "split": split,
            "limit": 2,
            "output_jsonl_fpath": str(tmp_path / "out/rows.jsonl"),
            "disable_aggregation": True,
        }
    )
    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(data_utils, "get_global_config_dict", lambda: config)
    monkeypatch.setattr(data_utils, "validate_backend_credentials", lambda backend: (True, None))
    monkeypatch.setattr(cli_eval, "RunHelper", Mock(return_value=Mock()))
    collect = AsyncMock()
    monkeypatch.setattr(rollout_collection.RolloutCollectionHelper, "run_from_config", collect)

    def download(download_config):
        Path(download_config.output_fpath).write_text(
            "".join(json.dumps({"responses_create_params": {"input": str(i)}}) + "\n" for i in range(10))
        )

    fetch = Mock(side_effect=download)
    monkeypatch.setattr(data_utils, "download_hf_dataset_as_jsonl", fetch)
    if split == "benchmark":
        with pytest.raises(ValueError, match="gym eval prepare"):
            cli_eval.e2e_rollout_collection()
        fetch.assert_not_called()
        collect.assert_not_called()
    else:
        cli_eval.e2e_rollout_collection()
        fetch.assert_called_once()
        assert fetch.call_args.args[0].repo_id == "test/selected"
        prepared = Path(collect.call_args.args[0].input_jsonl_fpath)
        rows = [json.loads(line) for line in prepared.read_text().splitlines()]
        assert [row["responses_create_params"]["input"] for row in rows] == ["0", "1"]
        assert len(source.read_text().splitlines()) == 10
        assert not unselected.exists()
