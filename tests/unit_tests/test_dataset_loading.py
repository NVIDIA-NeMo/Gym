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
from unittest.mock import MagicMock

import pytest
from omegaconf import DictConfig, OmegaConf

import nemo_gym.global_config
import nemo_gym.train_data_utils
from nemo_gym.config_types import DatasetConfig, ServerInstanceConfig
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig
from nemo_gym.train_data_utils import TrainDataProcessor, TrainDataProcessorConfig


HF_TOKEN = "test_hf_token"
CONFIG_OVERRIDES = {
    Path("resources_servers/math_with_judge/configs/math_stack_overflow.yaml"): {
        "math_with_judge": {"resources_servers": {"math_with_judge": {"judge_model_server": {"name": "policy_model"}}}}
    },
    Path("responses_api_agents/mini_swe_agent/configs/mini_swe_agent.yaml"): {
        "mini_swe_simple_agent": {
            "responses_api_agents": {"mini_swe_agent": {"cache_dir_template": "/tmp/nemo-gym-mini-swe-cache"}}
        }
    },
}
HF_DATASET_CONFIG_PATHS = tuple(
    sorted(
        path
        for root in (Path("resources_servers"), Path("responses_api_agents"))
        for path in root.glob("*/configs/*.yaml")
        if "huggingface_identifier" in path.read_text()
    )
)


def test_huggingface_dataset_configs_are_discovered() -> None:
    assert HF_DATASET_CONFIG_PATHS


def _load_agent_configs(config_path: Path) -> list[ServerInstanceConfig]:
    initial_config = OmegaConf.merge(
        GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        {
            "config_paths": [str(config_path)],
            "error_on_almost_servers": False,
        },
        CONFIG_OVERRIDES.get(config_path, {}),
    )

    global_config_dict = GlobalConfigDictParser().parse_no_environment(initial_config)

    processor = TrainDataProcessor()
    return processor.load_and_validate_server_instance_configs(
        config=TrainDataProcessorConfig(output_dirpath="", mode="train_preparation", should_download=True),
        global_config_dict=global_config_dict,
    )


def _replace_with_huggingface_only_datasets(
    agent_configs: list[ServerInstanceConfig],
    tmp_path: Path,
) -> tuple[list[tuple[str, str, str | None, str]], list[ServerInstanceConfig]]:
    expected_downloads = []
    agent_configs_with_hf_datasets = []

    for agent_config in agent_configs:
        hf_datasets = []
        for dataset_index, dataset in enumerate(agent_config.datasets or []):
            if (
                not isinstance(dataset, DatasetConfig)
                or dataset.type not in {"train", "validation"}
                or not dataset.huggingface_identifier
            ):
                continue

            dataset.jsonl_fpath = str(tmp_path / agent_config.name / f"{dataset_index}_{dataset.name}.jsonl")
            hf_datasets.append(dataset)
            expected_downloads.append(
                (
                    dataset.huggingface_identifier.repo_id,
                    dataset.type,
                    dataset.huggingface_identifier.artifact_fpath,
                    dataset.jsonl_fpath,
                )
            )

        if hf_datasets:
            agent_config.get_inner_run_server_config().datasets = hf_datasets
            agent_configs_with_hf_datasets.append(agent_config)

    return expected_downloads, agent_configs_with_hf_datasets


@pytest.mark.parametrize("config_path", HF_DATASET_CONFIG_PATHS, ids=lambda path: str(path))
def test_huggingface_dataset_configs_route_to_hf_download(
    config_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(nemo_gym.global_config, "_find_open_port_using_range", MagicMock(return_value=12345))
    monkeypatch.setattr(
        nemo_gym.train_data_utils,
        "get_global_config_dict",
        lambda: DictConfig({"hf_token": HF_TOKEN}),
    )
    download_hf_dataset_as_jsonl = MagicMock()
    download_jsonl_dataset = MagicMock()
    monkeypatch.setattr(nemo_gym.train_data_utils, "download_hf_dataset_as_jsonl", download_hf_dataset_as_jsonl)
    monkeypatch.setattr(nemo_gym.train_data_utils, "download_jsonl_dataset", download_jsonl_dataset)

    expected_downloads, agent_configs = _replace_with_huggingface_only_datasets(
        agent_configs=_load_agent_configs(config_path),
        tmp_path=tmp_path,
    )
    if not expected_downloads:
        pytest.skip(f"{config_path} has no train/validation Hugging Face datasets")

    TrainDataProcessor().load_datasets(
        config=TrainDataProcessorConfig(
            output_dirpath=str(tmp_path),
            mode="train_preparation",
            should_download=True,
            data_source="huggingface",
        ),
        server_instance_configs=agent_configs,
    )

    download_jsonl_dataset.assert_not_called()
    assert download_hf_dataset_as_jsonl.call_count == len(expected_downloads)

    for call, (repo_id, dataset_type, artifact_fpath, output_fpath) in zip(
        download_hf_dataset_as_jsonl.call_args_list, expected_downloads
    ):
        download_config = call.args[0]
        assert download_config.repo_id == repo_id
        assert download_config.artifact_fpath == artifact_fpath
        assert download_config.output_fpath == output_fpath
        assert download_config.output_dirpath is None
        assert download_config.hf_token == HF_TOKEN
        assert download_config.split == (None if artifact_fpath else dataset_type)
