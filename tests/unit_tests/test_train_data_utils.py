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

from pydantic import ValidationError
from pytest import MonkeyPatch, raises

import nemo_gym.global_config
import nemo_gym.train_data_utils
from nemo_gym.config_types import DatasetConfig, ResponsesAPIAgentServerInstanceConfig
from nemo_gym.global_config import DictConfig, GlobalConfigDictParser
from nemo_gym.train_data_utils import (
    AvgMinMax,
    DatasetMetrics,
    DatasetValidatorState,
    StringMetrics,
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
                                "num_repeats": 1,
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
                            "num_repeats": 1,
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
                            "num_repeats": 1,
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
                            "jsonl_fpath": "some/nonexiststent/path",
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
                            "num_repeats": 1,
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
                number_of_tools=AvgMinMax(
                    is_aggregated=False,
                    total=5,
                    average=0,
                    min=2.0,
                    max=2.0,
                    median=0,
                    stddev=0,
                ),
                json_dumped_number_of_words=AvgMinMax(
                    is_aggregated=False,
                    total=5,
                    average=0,
                    min=1499.0,
                    max=1509.0,
                    median=0,
                    stddev=0,
                ),
                number_of_turns=AvgMinMax(
                    is_aggregated=False,
                    total=5,
                    average=0,
                    min=1.0,
                    max=1.0,
                    median=0,
                    stddev=0,
                ),
                temperature=AvgMinMax(
                    is_aggregated=False,
                    total=0,
                    average=0,
                    min=float("inf"),
                    max=float("-inf"),
                    median=0,
                    stddev=0,
                ),
                id=AvgMinMax(
                    is_aggregated=True,
                    total=5,
                    average=2.0,
                    min=0.0,
                    max=4.0,
                    median=2.0,
                    stddev=1.58,
                ),
                expected_synonym_values=AvgMinMax(
                    is_aggregated=True,
                    total=10,
                    average=559.0,
                    min=407.0,
                    max=711.0,
                    median=559.0,
                    stddev=160.22,
                ),
                minefield_label_value=AvgMinMax(
                    is_aggregated=True,
                    total=5,
                    average=299.0,
                    min=299.0,
                    max=299.0,
                    median=299.0,
                    stddev=0.0,
                ),
                expected_synonyms=StringMetrics(unique_count=2, total_count=10),
                minefield_label=StringMetrics(unique_count=1, total_count=5),
            )
        }
        assert (
            expected_dataset_type_to_aggregate_metrics.get("example").model_dump()
            == actual_dataset_type_to_aggregate_metrics.get("example").model_dump()
        )

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
                            "num_repeats": 1,
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
                median=0,
                stddev=0,
            ),
            json_dumped_number_of_words=AvgMinMax(
                is_aggregated=False,
                total=1,
                average=0,
                min=2.0,
                max=2.0,
                median=0,
                stddev=0,
            ),
            number_of_turns=AvgMinMax(
                is_aggregated=False,
                total=0,
                average=0,
                min=float("inf"),
                max=float("-inf"),
                median=0,
                stddev=0,
            ),
            temperature=AvgMinMax(
                is_aggregated=False,
                total=0,
                average=0,
                min=float("inf"),
                max=float("-inf"),
                median=0,
                stddev=0,
            ),
        )
        assert expected_metrics.model_dump() == state.metrics.model_dump()

    def test_numeric_close_tolerance_validation(self, monkeypatch: MonkeyPatch) -> None:
        """Test numeric_close with various numeric values to validate tolerance thresholds"""
        processor = TrainDataProcessor()

        test_cases = [
            # Large numbers within tolerance (0.005)
            (100.000, 100.004, True),  # diff = 0.004
            (1.000, 1.005, True),  # at scale boundary
            (10.000, 10.004, True),  # medium scale
            (1000.0, 1000.004, True),  # large scale
            # Large numbers exceeding tolerance
            (100.000, 100.006, False),  # diff = 0.006 > 0.005
            (1000.0, 1000.01, False),  # diff = 0.01 > 0.005
            # Small numbers within tolerance
            (0.500, 0.5004, True),  # diff = 0.0004
            (0.999, 0.9994, True),  # near scale=1
            (0.1, 0.10049, True),  # very small
            # Small numbers exceeding tolerance
            (0.500, 0.5006, False),  # diff = 0.0006 > 0.005
            # Edge cases around scale = 1.0
            (0.999, 1.0039, True),  # crosses boundary, diff = 0.0049
            (0.999, 0.9994, True),  # below boundary, diff = 0.0006
            # Exact equality
            (100.0, 100.0, True),
            (0.5, 0.5, True),
            (1.0, 1.0, True),
        ]

        for prev_value, new_value, should_pass in test_cases:
            mock_write = mock_open()
            prev_metrics = {"test_field": prev_value}

            original_open = open

            def custom_open(filename, mode="r", *args, **kwargs):
                filename_str = str(filename)
                actual_mode = kwargs.get("mode", mode)

                if "test_metrics.json" in filename_str and "r" in actual_mode:
                    return mock_open(read_data=json.dumps(prev_metrics))()
                if "w" in actual_mode or "a" in actual_mode:
                    return mock_write()
                return original_open(filename, actual_mode, *args, **kwargs)

            monkeypatch.setattr("builtins.open", custom_open)
            monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

            new_metrics = {"test_field": new_value}
            result = processor._validate_aggregate_metrics(new_metrics, Path("test_metrics.json"))

            if should_pass:
                assert result is None, f"Expected pass for {prev_value} vs {new_value}"
            else:
                assert result is not None, f"Expected fail for {prev_value} vs {new_value}"
                assert "_conflict.json" in str(result)

            monkeypatch.undo()

    def test_validate_aggregate_metrics_structure(self, monkeypatch: MonkeyPatch) -> None:
        """Test validation of metric structure: extra fields, missing fields, and lists"""
        processor = TrainDataProcessor()

        # Format: (prev_metrics, new_metrics, should_pass)
        test_cases = [
            # Extra fields (allowed)
            (
                {"field1": 10, "field2": 20},
                {"field1": 10, "field2": 20, "field3": 30},  # extra top-level
                True,
            ),
            (
                {"nested": {"a": 1}},
                {"nested": {"a": 1, "b": 2}},  # extra nested
                True,
            ),
            (
                {"field1": 10},
                {"field1": 10, "field2": 20, "field3": {"nested": "value"}},  # multiple extra
                True,
            ),
            # Missing fields (not allowed)
            (
                {"field1": 10, "field2": 20},
                {"field1": 10},  # missing field2
                False,
            ),
            (
                {"nested": {"a": 1, "b": 2}},
                {"nested": {"a": 1}},  # missing nested.b
                False,
            ),
            (
                {"field1": 10, "field2": 20, "field3": 30},
                {"field1": 10},  # missing multiple
                False,
            ),
            # List comparisons (reordering allowed)
            (
                {"items": [1, 2, 3]},
                {"items": [3, 1, 2]},  # reordered
                True,
            ),
            (
                {"items": ["a", "b", "c"]},
                {"items": ["c", "b", "a"]},  # strings reordered
                True,
            ),
            (
                {"items": [1, 1, 2, 3]},
                {"items": [3, 2, 1, 1]},  # duplicates reordered
                True,
            ),
            # Different list elements (fail)
            (
                {"items": [1, 2, 3]},
                {"items": [1, 2, 4]},  # different element
                False,
            ),
            (
                {"items": [1, 2, 3]},
                {"items": [1, 2]},  # different length
                False,
            ),
            (
                {"items": [1, 1, 2]},
                {"items": [1, 2, 2]},  # different duplicates
                False,
            ),
        ]

        for prev_metrics, new_metrics, should_pass in test_cases:
            mock_write = mock_open()
            original_open = open

            def custom_open(filename, mode="r", *args, **kwargs):
                filename_str = str(filename)
                actual_mode = kwargs.get("mode", mode)

                if "test_metrics.json" in filename_str and "r" in actual_mode:
                    return mock_open(read_data=json.dumps(prev_metrics))()
                if "w" in actual_mode or "a" in actual_mode:
                    return mock_write()
                return original_open(filename, actual_mode, *args, **kwargs)

            monkeypatch.setattr("builtins.open", custom_open)
            monkeypatch.setattr("pathlib.Path.exists", lambda self: True)

            result = processor._validate_aggregate_metrics(new_metrics, Path("test_metrics.json"))

            if should_pass:
                assert result is None
            else:
                assert result is not None
                assert "_conflict.json" in str(result)

            monkeypatch.undo()


class TestIterDatasetLines:
    def test_iter_dataset_lines_num_repeats(self, tmp_path):
        """Test _iter_dataset_lines with different num_repeats values"""
        processor = TrainDataProcessor()

        # Create a test dataset file
        test_file = tmp_path / "test_data.jsonl"
        test_data = ['{"id": 1, "content": "line1"}', '{"id": 2, "content": "line2"}']
        test_file.write_text("\n".join(test_data) + "\n")

        # Test default behavior (num_repeats defaults to 1 when not specified)
        config = DatasetConfig(name="test", type="example", jsonl_fpath=str(test_file))
        lines = list(processor._iter_dataset_lines(config))
        expected = ['{"id": 1, "content": "line1"}\n', '{"id": 2, "content": "line2"}\n']
        assert lines == expected

        # Test num_repeats=1 (explicit)
        config = DatasetConfig(name="test", type="example", jsonl_fpath=str(test_file), num_repeats=1)
        lines = list(processor._iter_dataset_lines(config))
        assert lines == expected

        # Test num_repeats=3
        config = DatasetConfig(name="test", type="example", jsonl_fpath=str(test_file), num_repeats=3)
        lines = list(processor._iter_dataset_lines(config))
        expected_repeated = [
            '{"id": 1, "content": "line1"}\n',
            '{"id": 1, "content": "line1"}\n',
            '{"id": 1, "content": "line1"}\n',
            '{"id": 2, "content": "line2"}\n',
            '{"id": 2, "content": "line2"}\n',
            '{"id": 2, "content": "line2"}\n',
        ]
        assert lines == expected_repeated


class TestDatasetConfigNumRepeats:
    def test_dataset_config_num_repeats_valid_values(self):
        """Test DatasetConfig with valid num_repeats values"""

        # Test default (when not specified, defaults to 1)
        config = DatasetConfig(name="test", type="example", jsonl_fpath="test.jsonl")
        assert config.num_repeats == 1

        # Test valid positive integers
        for repeats in [1, 2, 5, 10, 100]:
            config = DatasetConfig(name="test", type="example", jsonl_fpath="test.jsonl", num_repeats=repeats)
            assert config.num_repeats == repeats

    def test_dataset_config_num_repeats_invalid_values(self):
        """Test DatasetConfig with invalid num_repeats values"""

        # Test zero
        with raises(ValidationError, match="Input should be greater than or equal to 1"):
            DatasetConfig(name="test", type="example", jsonl_fpath="test.jsonl", num_repeats=0)

        # Test negative values
        with raises(ValidationError, match="Input should be greater than or equal to 1"):
            DatasetConfig(name="test", type="example", jsonl_fpath="test.jsonl", num_repeats=-1)


class TestNumRepeatsMetricsAggregation:
    def test_validate_samples_with_num_repeats(self, tmp_path):
        """Test that metrics and sample enumeration work correctly with num_repeats"""
        processor = TrainDataProcessor()

        # Test with valid samples and num_repeats=2
        test_file = tmp_path / "test_data.jsonl"
        test_data = [
            '{"responses_create_params": {"input": [{"role": "user", "content": "test1"}]}, "temperature": 0.5}',
            '{"invalid": "sample"}',  # Invalid sample for testing enumeration
        ]
        test_file.write_text("\n".join(test_data) + "\n")

        config = DatasetConfig(name="test", type="example", jsonl_fpath=str(test_file), num_repeats=2)
        state = processor._validate_samples_and_aggregate_metrics_single_dataset(config)

        # Should process 4 samples total (2 lines * 2 repeats), but only valid ones count in metrics
        # The metrics count only tracks successful processing (only 2 successful out of 4 total)
        assert state.metrics.number_of_examples == 2
        # Indices 2,3 should be offending (both repeats of the invalid sample)
        assert state.offending_example_idxs == [2, 3]


class TestNumRepeatsDataPreparation:
    def test_prepare_samples_with_num_repeats(self, tmp_path, monkeypatch: MonkeyPatch):
        """Test that data preparation correctly repeats samples based on num_repeats"""
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
        test_file = tmp_path / "test_data.jsonl"
        test_data = [
            '{"responses_create_params": {"input": [{"role": "user", "content": "test1"}]}}',
            '{"responses_create_params": {"input": [{"role": "user", "content": "test2"}]}}',
        ]
        test_file.write_text("\n".join(test_data) + "\n")

        config = TrainDataProcessorConfig(output_dirpath="", mode="example_validation", should_download=False)
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
                            "jsonl_fpath": str(test_file),
                            "num_repeats": 3,
                            "gitlab_identifier": None,
                            "license": None,
                        }
                    ],
                    "resources_server": {"type": "resources_servers", "name": "test_resources_server"},
                    "model_server": {"type": "responses_api_models", "name": "policy_model"},
                }
            }
        }
        server_config = ResponsesAPIAgentServerInstanceConfig(
            name="test_agent",
            server_type_config_dict=DictConfig(server_type_config_dict),
            responses_api_agents=server_type_config_dict["responses_api_agents"],
        )

        processor.collate_samples(
            config=config,
            server_instance_configs=[server_config],
            dataset_type_to_aggregate_metrics={
                "example": DatasetMetrics(
                    is_aggregated=False,
                    number_of_examples=6,
                    number_of_tools=AvgMinMax(is_aggregated=False, total=6, average=0.0, min=0.0, max=0.0),
                    json_dumped_number_of_words=AvgMinMax(
                        is_aggregated=False, total=6, average=100.0, min=50.0, max=150.0
                    ),
                    number_of_turns=AvgMinMax(is_aggregated=False, total=6, average=1.0, min=1.0, max=1.0),
                    temperature=AvgMinMax(
                        is_aggregated=False, total=0, average=0, min=float("inf"), max=float("-inf")
                    ),
                )
            },
        )

        # Verify 6 writes occurred (2 lines * 3 repeats each)
        expected_prepare_file = test_file.with_name(f"{test_file.stem}_prepare.jsonl")
        prepare_mock = write_filenames_to_mock[expected_prepare_file]
        assert len(prepare_mock.return_value.write.call_args_list) == 6


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
                            "num_repeats": 1,
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
            Path("example_metrics.json"),
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

            if filename == Path("example_metrics.json"):
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
                            "num_repeats": 1,
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
