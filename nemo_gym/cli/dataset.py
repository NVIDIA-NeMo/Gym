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

from nemo_gym.config_types import (
    DeleteJsonlDatasetGitlabConfig,
    DownloadJsonlDatasetGitlabConfig,
    DownloadJsonlDatasetHuggingFaceConfig,
    UploadJsonlDatasetGitlabConfig,
    UploadJsonlDatasetHuggingFaceConfig,
    UploadJsonlDatasetHuggingFaceMaybeDeleteConfig,
)
from nemo_gym.dataset_orchestrator import (
    delete_jsonl_dataset_from_gitlab,
    upload_jsonl_dataset_to_hf_maybe_delete,
)
from nemo_gym.gitlab_utils import download_jsonl_dataset, upload_jsonl_dataset
from nemo_gym.global_config import GlobalConfigDictParserConfig, get_global_config_dict
from nemo_gym.hf_utils import download_hf_dataset_as_jsonl
from nemo_gym.prompt import MaterializePromptsConfig, materialize_prompts
from nemo_gym.train_data_utils import TrainDataProcessor


def upload_jsonl_dataset_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetGitlabConfig.model_validate(global_config)
    upload_jsonl_dataset(config)


def download_jsonl_dataset_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = DownloadJsonlDatasetGitlabConfig.model_validate(global_config)
    download_jsonl_dataset(config)


def upload_jsonl_dataset_to_hf_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetHuggingFaceMaybeDeleteConfig.model_validate(global_config)
    upload_jsonl_dataset_to_hf_maybe_delete(config, delete_from_gitlab=config.delete_from_gitlab)


def download_jsonl_dataset_from_hf_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = DownloadJsonlDatasetHuggingFaceConfig.model_validate(global_config)

    if config.artifact_fpath:
        print(f"Downloading file '{config.artifact_fpath}' from '{config.repo_id}'...")
    else:
        print(f"Downloading '{config.split or 'all'}' split(s) from '{config.repo_id}'...")

    download_hf_dataset_as_jsonl(config)


def delete_jsonl_dataset_from_gitlab_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = DeleteJsonlDatasetGitlabConfig.model_validate(global_config)
    delete_jsonl_dataset_from_gitlab(config.dataset_name)


def upload_jsonl_dataset_to_hf_and_delete_gitlab_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetHuggingFaceConfig.model_validate(global_config)
    upload_jsonl_dataset_to_hf_maybe_delete(config, delete_from_gitlab=True)


def materialize_prompts_cli() -> None:  # pragma: no cover
    """CLI entry point for gym dataset render."""
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )
    config = MaterializePromptsConfig.model_validate(global_config_dict)
    materialize_prompts(config.input_jsonl_fpath, config.prompt_config, config.output_jsonl_fpath)


def prepare_data():  # pragma: no cover
    global_config_dict = get_global_config_dict(
        global_config_dict_parser_config=GlobalConfigDictParserConfig(
            initial_global_config_dict=GlobalConfigDictParserConfig.NO_MODEL_GLOBAL_CONFIG_DICT,
        )
    )

    data_processor = TrainDataProcessor()
    data_processor.run(global_config_dict)
