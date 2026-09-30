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
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

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


class TestE2EInputSelection:
    @pytest.fixture
    def runtime(self, monkeypatch, tmp_path):
        from nemo_gym.cli import eval as cli_eval

        config = DictConfig({"output_jsonl_fpath": str(tmp_path / "out.jsonl"), "disable_health_check": True})
        monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: config)
        servers = MagicMock()
        monkeypatch.setattr(cli_eval, "RunHelper", lambda: servers)
        collector = MagicMock(run_from_config=AsyncMock())
        monkeypatch.setattr("nemo_gym.rollout_collection.RolloutCollectionHelper", lambda: collector)
        processor = MagicMock()
        monkeypatch.setattr("nemo_gym.train_data_utils.TrainDataProcessor", lambda: processor)
        return config, servers, collector, processor

    @pytest.mark.parametrize("split", [None, "benchmark"])
    @pytest.mark.parametrize("relative", [False, True])
    def test_explicit_input_skips_preparation(self, runtime, monkeypatch, tmp_path, capsys, split, relative):
        from nemo_gym.cli.eval import e2e_rollout_collection

        config, servers, collector, processor = runtime
        input_path = tmp_path / "subset.jsonl"
        input_path.write_text('{"task_id": "selected"}\n')
        monkeypatch.chdir(tmp_path)
        config.input_jsonl_fpath = input_path.name if relative else str(input_path)
        if split is not None:
            config.split = split
        config.reuse_existing_data_preparation = True
        config.limit = 1
        config.num_repeats = 2
        e2e_rollout_collection()
        processor.run.assert_not_called()
        servers.start.assert_called_once_with(None)
        servers.shutdown.assert_called_once_with()
        collector.run_from_config.assert_awaited_once()
        collected = collector.run_from_config.call_args.args[0]
        assert Path(collected.input_jsonl_fpath) == input_path
        assert collected.limit == 1
        assert collected.num_repeats == 2
        assert not (tmp_path / "out" / "preprocessed_datasets").exists()
        assert "skipped (--input)" in capsys.readouterr().out

    @pytest.mark.parametrize("directory", [False, True])
    def test_invalid_input_fails_before_startup(self, runtime, tmp_path, capsys, directory):
        from nemo_gym.cli.eval import e2e_rollout_collection

        config, servers, collector, processor = runtime
        input_path = tmp_path / "missing.jsonl"
        if directory:
            input_path.mkdir()
        config.input_jsonl_fpath = str(input_path)
        with pytest.raises(SystemExit, match="1"):
            e2e_rollout_collection()
        assert "Input file not found or not a file" in capsys.readouterr().out
        processor.run.assert_not_called()
        servers.start.assert_not_called()
        collector.run_from_config.assert_not_called()

    @pytest.mark.parametrize("reuse", [False, True])
    def test_split_preparation_is_preserved(self, runtime, monkeypatch, tmp_path, reuse):
        from nemo_gym.cli import eval as cli_eval

        config, servers, collector, processor = runtime
        config.split = "validation"
        config.reuse_existing_data_preparation = reuse
        agents = [_make_agent_instance_config("agent", [{"name": "data", "type": "validation"}])]
        monkeypatch.setattr(cli_eval.GlobalConfigDictParser, "filter_for_server_instance_configs", lambda *_: agents)
        prepared = tmp_path / "out" / "preprocessed_datasets" / "validation.jsonl"

        def prepare(data_config):
            assert data_config.mode == "train_preparation"
            assert data_config.should_download is True
            assert Path(data_config.output_dirpath) == prepared.parent
            prepared.parent.mkdir(parents=True)
            prepared.write_text("{}\n")

        if reuse:
            prepared.parent.mkdir(parents=True)
            prepared.write_text("{}\n")
        processor.run.side_effect = prepare
        cli_eval.e2e_rollout_collection()
        assert processor.run.call_count == (0 if reuse else 1)
        assert Path(collector.run_from_config.call_args.args[0].input_jsonl_fpath) == prepared
        servers.start.assert_called_once_with(None)
        servers.shutdown.assert_called_once_with()

    def test_explicit_input_reaches_custom_driver(self, runtime, monkeypatch, tmp_path):
        from nemo_gym.cli.eval import e2e_rollout_collection

        config, servers, collector, processor = runtime
        input_path = tmp_path / "subset.jsonl"
        input_path.write_text("{}\n")
        config.input_jsonl_fpath = str(input_path)
        config.rollout_collection_driver = f"{__name__}:test_driver"
        driver = AsyncMock()
        monkeypatch.setattr(__name__ + ".test_driver", driver, raising=False)
        e2e_rollout_collection()
        driver.assert_awaited_once()
        collected, resolved = driver.call_args.args
        assert Path(collected.input_jsonl_fpath) == input_path
        assert resolved["input_jsonl_fpath"] == str(input_path)
        processor.run.assert_not_called()
        collector.run_from_config.assert_not_called()
        servers.shutdown.assert_called_once_with()
