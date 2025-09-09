# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open

from pytest import MonkeyPatch, raises

import nemo_gym.global_config
import nemo_gym.train_data_utils
from nemo_gym.config_types import ResponsesAPIAgentServerInstanceConfig
from nemo_gym.global_config import DictConfig, GlobalConfigDictParser
from nemo_gym.train_data_utils import (
    AvgMinMax,
    DatasetMetrics,
    DatasetValidatorState,
    TrainDataProcessor,
    TrainDataProcessorConfig,
)


def load_multineedle_test_global_config_dict() -> DictConfig:
    return GlobalConfigDictParser().parse_no_environment(
        initial_global_config_dict=DictConfig(
            {
                "config_paths": [
                    "resources_servers/multineedle/configs/multineedle.yaml",
                    "responses_api_models/openai_model/configs/openai_model.yaml",
                ],
                # For policy_model
                "policy_base_url": "",
                "policy_api_key": "",
                "policy_model_name": "",
            }
        ),
    )


class TestLoadAndValidateServerInstanceConfigs:
    def test_load_and_validate_server_instance_configs_sanity(self, monkeypatch: MonkeyPatch) -> None:
        # Fix the port returned
        find_open_port_mock = MagicMock()
        find_open_port_mock.return_value = 12345
        monkeypatch.setattr(nemo_gym.global_config, "find_open_port", find_open_port_mock)

        config = TrainDataProcessorConfig(
            output_dirpath="",
            mode="example_validation",
            should_download=False,
        )
        processor = TrainDataProcessor()
        actual_agent_configs_with_data = processor.load_and_validate_server_instance_configs(
            config=config,
            global_config_dict=load_multineedle_test_global_config_dict(),
        )

        expected_agent_configs_with_data_dict = [
            {
                "name": "multineedle_simple_agent",
                "responses_api_agents": {
                    "simple_agent": {
                        "host": "127.0.0.1",
                        "port": 12345,
                        "entrypoint": "app.py",
                        "datasets": [
                            {
                                "name": "example",
                                "type": "example",
                                "jsonl_fpath": "resources_servers/multineedle/data/example.jsonl",
                                "gitlab_identifier": None,
                                "license": None,
                            }
                        ],
                        "resources_server": {
                            "type": "resources_servers",
                            "name": "multineedle_resources_server",
                        },
                        "model_server": {
                            "type": "responses_api_models",
                            "name": "policy_model",
                        },
                    }
                },
            }
        ]
        actual_agent_configs_with_data_dict = [
            c.model_dump(mode="json", warnings="none") for c in actual_agent_configs_with_data
        ]
        assert expected_agent_configs_with_data_dict == actual_agent_configs_with_data_dict


class TestLoadDatasets:
    def test_load_datasets_sanity(self) -> None:
        config = TrainDataProcessorConfig(
            output_dirpath="",
            mode="example_validation",
            should_download=False,
        )
        processor = TrainDataProcessor()

        server_type_config_dict = {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "example",
                            "type": "example",
                            "jsonl_fpath": "resources_servers/multineedle/data/example.jsonl",
                            "gitlab_identifier": None,
                            "license": None,
                        }
                    ],
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "multineedle_resources_server",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
        processor.load_datasets(
            config=config,
            server_instance_configs=[
                ResponsesAPIAgentServerInstanceConfig(
                    name="multineedle_simple_agent",
                    server_type_config_dict=DictConfig(server_type_config_dict),
                    responses_api_agents=server_type_config_dict["responses_api_agents"],
                ),
            ],
        )

    def test_load_datasets_missing_example_dataset_raises_AssertionError(self) -> None:
        config = TrainDataProcessorConfig(
            output_dirpath="",
            mode="example_validation",
            should_download=False,
        )
        processor = TrainDataProcessor()

        server_type_config_dict = {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "example",
                            "type": "example",
                            "jsonl_fpath": "resources_servers/multineedle/data/example_missing.jsonl",
                            "gitlab_identifier": None,
                            "license": None,
                        }
                    ],
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "multineedle_resources_server",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
        with raises(
            AssertionError,
            match="You must provide the above missing example jsonl files!",
        ):
            processor.load_datasets(
                config=config,
                server_instance_configs=[
                    ResponsesAPIAgentServerInstanceConfig(
                        name="multineedle_simple_agent",
                        server_type_config_dict=DictConfig(server_type_config_dict),
                        responses_api_agents=server_type_config_dict["responses_api_agents"],
                    ),
                ],
            )

    def test_load_datasets_missing_train_dataset_shouldnt_download_raises_AssertionError(
        self,
    ) -> None:
        config = TrainDataProcessorConfig(
            output_dirpath="",
            mode="train_preparation",
            should_download=False,
        )
        processor = TrainDataProcessor()

        server_type_config_dict = {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "train",
                            "type": "train",
                            "jsonl_fpath": "resources_servers/multineedle/data/train.jsonl",
                            "gitlab_identifier": {
                                "dataset_name": "multineedle",
                                "version": "0.0.1",
                                "artifact_fpath": "train.jsonl",
                            },
                            "license": "Apache 2.0",
                        }
                    ],
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "multineedle_resources_server",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
        with raises(
            AssertionError,
            match="Missing local datasets. You must provide local datasets since download is disabled.",
        ):
            processor.load_datasets(
                config=config,
                server_instance_configs=[
                    ResponsesAPIAgentServerInstanceConfig(
                        name="multineedle_simple_agent",
                        server_type_config_dict=DictConfig(server_type_config_dict),
                        responses_api_agents=server_type_config_dict["responses_api_agents"],
                    ),
                ],
            )


class TestValidateSamplesAndAggregateMetrics:
    def test_validate_samples_and_aggregate_metrics_sanity(self, monkeypatch: MonkeyPatch) -> None:
        mock_write_file = mock_open()
        write_filenames = []

        original_open = open

        def custom_open(filename, mode="r"):
            if mode == "w":
                write_filenames.append(filename)
                return mock_write_file()

            return original_open(filename, mode)

        monkeypatch.setattr("builtins.open", custom_open)

        processor = TrainDataProcessor()

        server_type_config_dict = {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "example",
                            "type": "example",
                            "jsonl_fpath": "resources_servers/multineedle/data/example.jsonl",
                            "gitlab_identifier": None,
                            "license": None,
                        }
                    ],
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "multineedle_resources_server",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
        actual_dataset_type_to_aggregate_metrics = processor.validate_samples_and_aggregate_metrics(
            server_instance_configs=[
                ResponsesAPIAgentServerInstanceConfig(
                    name="multineedle_simple_agent",
                    server_type_config_dict=DictConfig(server_type_config_dict),
                    responses_api_agents=server_type_config_dict["responses_api_agents"],
                ),
            ],
        )

        expected_dataset_type_to_aggregate_metrics = {
            "example": DatasetMetrics(
                is_aggregated=False,
                number_of_examples=5,
                number_of_tools=AvgMinMax(is_aggregated=False, total=5, average=10.0, min=2.0, max=2.0),
                json_dumped_number_of_words=AvgMinMax(
                    is_aggregated=False, total=5, average=7520.0, min=1499.0, max=1509.0
                ),
                number_of_turns=AvgMinMax(is_aggregated=False, total=5, average=5.0, min=1.0, max=1.0),
                temperature=AvgMinMax(
                    is_aggregated=False,
                    total=0,
                    average=0,
                    min=float("inf"),
                    max=float("-inf"),
                ),
            )
        }
        assert expected_dataset_type_to_aggregate_metrics == actual_dataset_type_to_aggregate_metrics

        assert write_filenames == [Path("resources_servers/multineedle/data/example_metrics.json")]

    def test_validate_samples_and_aggregate_metrics_conflict_raises_ValueError(self, monkeypatch: MonkeyPatch) -> None:
        mock_write_file = mock_open()
        write_filenames = []

        original_open = open

        def custom_open(filename, mode="r"):
            if mode == "w":
                write_filenames.append(filename)
                return mock_write_file()

            if filename == "resources_servers/multineedle/data/example.jsonl":
                return original_open(filename, mode)
            elif filename == Path("resources_servers/multineedle/data/example_metrics.json"):
                with original_open(filename, mode) as f:
                    read_data = json.loads(f.read())

                read_data["some extra field to cause a conflict"] = "lmao"
                return mock_open(read_data=json.dumps(read_data))()
            else:
                raise NotImplementedError

        monkeypatch.setattr("builtins.open", custom_open)

        processor = TrainDataProcessor()

        server_type_config_dict = {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "example",
                            "type": "example",
                            "jsonl_fpath": "resources_servers/multineedle/data/example.jsonl",
                            "gitlab_identifier": None,
                            "license": None,
                        }
                    ],
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "multineedle_resources_server",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
        with raises(
            ValueError,
            match="Found conflicting aggregate metrics that need to be corrected:",
        ):
            processor.validate_samples_and_aggregate_metrics(
                server_instance_configs=[
                    ResponsesAPIAgentServerInstanceConfig(
                        name="multineedle_simple_agent",
                        server_type_config_dict=DictConfig(server_type_config_dict),
                        responses_api_agents=server_type_config_dict["responses_api_agents"],
                    ),
                ],
            )

        assert write_filenames == [Path("resources_servers/multineedle/data/example_metrics_conflict.json")]

    def test_validate_samples_and_aggregate_metrics_single_sample(self) -> None:
        processor = TrainDataProcessor()

        state = DatasetValidatorState()
        processor._validate_samples_and_aggregate_metrics_single_sample(
            state=state,
            sample_idx=12345,
            sample_dict_str="some non json str",
        )
        assert state.offending_example_idxs == [12345]

        state = DatasetValidatorState()
        processor._validate_samples_and_aggregate_metrics_single_sample(
            state=state,
            sample_idx=12345,
            sample_dict_str='{"some irrelevant key": 2}',
        )
        assert state.offending_example_idxs == [12345]

        state = DatasetValidatorState()
        processor._validate_samples_and_aggregate_metrics_single_sample(
            state=state,
            sample_idx=12345,
            sample_dict_str='{"responses_create_params": {"input": []}, "temperature": 0.1}',
        )
        assert state.offending_example_idxs == []

        expected_metrics = DatasetMetrics(
            is_aggregated=False,
            number_of_examples=1,
            number_of_tools=AvgMinMax(
                is_aggregated=False,
                total=0,
                average=0,
                min=float("inf"),
                max=float("-inf"),
            ),
            json_dumped_number_of_words=AvgMinMax(is_aggregated=False, total=1, average=2.0, min=2.0, max=2.0),
            number_of_turns=AvgMinMax(
                is_aggregated=False,
                total=0,
                average=0,
                min=float("inf"),
                max=float("-inf"),
            ),
            temperature=AvgMinMax(
                is_aggregated=False,
                total=0,
                average=0,
                min=float("inf"),
                max=float("-inf"),
            ),
        )
        assert expected_metrics == state.metrics


class TestCollateSamples:
    def test_collate_samples_sanity(self, monkeypatch: MonkeyPatch) -> None:
        write_filenames_to_mock = dict()

        original_open = open

        def custom_open(filename, mode="r"):
            if mode in ["w", "wb"]:
                write_filenames_to_mock[filename] = mock_open()
                return write_filenames_to_mock[filename]()

            if filename in write_filenames_to_mock:
                write_mock: MagicMock = write_filenames_to_mock[filename]
                written_data = write_mock.return_value.write.call_args_list[0].args[0]
                return mock_open(read_data=written_data)()
            return original_open(filename, mode)

        monkeypatch.setattr("builtins.open", custom_open)

        processor = TrainDataProcessor()

        config = TrainDataProcessorConfig(
            output_dirpath="",
            mode="example_validation",
            should_download=False,
        )
        server_type_config_dict = {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "example",
                            "type": "example",
                            "jsonl_fpath": "resources_servers/multineedle/data/example.jsonl",
                            "gitlab_identifier": None,
                            "license": None,
                        }
                    ],
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "multineedle_resources_server",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
        processor.collate_samples(
            config=config,
            server_instance_configs=[
                ResponsesAPIAgentServerInstanceConfig(
                    name="multineedle_simple_agent",
                    server_type_config_dict=DictConfig(server_type_config_dict),
                    responses_api_agents=server_type_config_dict["responses_api_agents"],
                ),
            ],
            dataset_type_to_aggregate_metrics={
                "example": DatasetMetrics(
                    is_aggregated=False,
                    number_of_examples=5,
                    number_of_tools=AvgMinMax(is_aggregated=False, total=5, average=10.0, min=2.0, max=2.0),
                    json_dumped_number_of_words=AvgMinMax(
                        is_aggregated=False,
                        total=5,
                        average=7520.0,
                        min=1499.0,
                        max=1509.0,
                    ),
                    number_of_turns=AvgMinMax(is_aggregated=False, total=5, average=5.0, min=1.0, max=1.0),
                    temperature=AvgMinMax(
                        is_aggregated=False,
                        total=0,
                        average=0,
                        min=float("inf"),
                        max=float("-inf"),
                    ),
                )
            },
        )

        assert list(write_filenames_to_mock.keys()) == [
            Path("example_metrics.jsonl"),
            Path("resources_servers/multineedle/data/example_prepare.jsonl"),
            Path("example.jsonl"),
        ]

    def test_collate_samples_metrics_conflict_raises_ValueError(self, monkeypatch: MonkeyPatch) -> None:
        write_filenames_to_mock = dict()

        original_open = open

        def custom_open(filename, mode="r"):
            if mode in ["w", "wb"]:
                write_filenames_to_mock[filename] = mock_open()
                return write_filenames_to_mock[filename]()

            if filename in write_filenames_to_mock:
                write_mock: MagicMock = write_filenames_to_mock[filename]
                written_data = write_mock.return_value.write.call_args_list[0].args[0]
                return mock_open(read_data=written_data)()

            if filename == Path("example_metrics.jsonl"):
                read_data = {"some extra field to cause a conflict": "lmao"}
                return mock_open(read_data=json.dumps(read_data))()
            return original_open(filename, mode)

        monkeypatch.setattr("builtins.open", custom_open)

        monkeypatch.setattr(nemo_gym.train_data_utils.Path, "exists", lambda *args, **kwargs: True)

        processor = TrainDataProcessor()

        config = TrainDataProcessorConfig(
            output_dirpath="",
            mode="example_validation",
            should_download=False,
        )
        server_type_config_dict = {
            "responses_api_agents": {
                "simple_agent": {
                    "host": "127.0.0.1",
                    "port": 12345,
                    "entrypoint": "app.py",
                    "datasets": [
                        {
                            "name": "example",
                            "type": "example",
                            "jsonl_fpath": "resources_servers/multineedle/data/example.jsonl",
                            "gitlab_identifier": None,
                            "license": None,
                        }
                    ],
                    "resources_server": {
                        "type": "resources_servers",
                        "name": "multineedle_resources_server",
                    },
                    "model_server": {
                        "type": "responses_api_models",
                        "name": "policy_model",
                    },
                }
            }
        }
        with raises(
            ValueError,
            match="Found conflicting aggregate metrics that need to be corrected:",
        ):
            processor.collate_samples(
                config=config,
                server_instance_configs=[
                    ResponsesAPIAgentServerInstanceConfig(
                        name="multineedle_simple_agent",
                        server_type_config_dict=DictConfig(server_type_config_dict),
                        responses_api_agents=server_type_config_dict["responses_api_agents"],
                    ),
                ],
                dataset_type_to_aggregate_metrics={
                    "example": DatasetMetrics(
                        is_aggregated=False,
                        number_of_examples=5,
                        number_of_tools=AvgMinMax(is_aggregated=False, total=5, average=10.0, min=2.0, max=2.0),
                        json_dumped_number_of_words=AvgMinMax(
                            is_aggregated=False,
                            total=5,
                            average=7520.0,
                            min=1499.0,
                            max=1509.0,
                        ),
                        number_of_turns=AvgMinMax(is_aggregated=False, total=5, average=5.0, min=1.0, max=1.0),
                        temperature=AvgMinMax(
                            is_aggregated=False,
                            total=0,
                            average=0,
                            min=float("inf"),
                            max=float("-inf"),
                        ),
                    )
                },
            )

        assert list(write_filenames_to_mock.keys()) == [
            Path("example_metrics_conflict.json"),
        ]
