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
from typing import Union

from nemo_gym.config_types import (
    DownloadJsonlDatasetHuggingFaceConfig,
    UploadJsonlDatasetHuggingFaceConfig,
    UploadJsonlDatasetHuggingFaceMaybeDeleteConfig,
)
from nemo_gym.gitlab_utils import delete_model_from_gitlab, is_model_in_gitlab
from nemo_gym.hf_utils import download_jsonl_dataset as download_jsonl_dataset_from_hf
from nemo_gym.hf_utils import upload_jsonl_dataset
from nemo_gym.server_utils import get_global_config_dict


def upload_jsonl_dataset_to_hf(
    config: Union[UploadJsonlDatasetHuggingFaceMaybeDeleteConfig, UploadJsonlDatasetHuggingFaceConfig],
    delete_from_gitlab: bool = False,
) -> None:  # pragma: no cover
    gitlab_model_name = config.dataset_name

    upload_jsonl_dataset(config)

    if delete_from_gitlab:
        model_exists = is_model_in_gitlab(gitlab_model_name)

        # gitlab model_name must match hf dataset name
        if model_exists:
            print(f"Found model '{gitlab_model_name}' in the registry. Deleting.")
            delete_model_from_gitlab(gitlab_model_name)
            print(f"Deleted '{gitlab_model_name}' from gitlab.")


def upload_jsonl_dataset_to_hf_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetHuggingFaceMaybeDeleteConfig.model_validate(global_config)
    maybe_delete_from_gitlab = config.delete_from_gitlab
    upload_jsonl_dataset_to_hf(config, delete_from_gitlab=maybe_delete_from_gitlab)


def upload_jsonl_dataset_to_hf_and_delete_gitlab_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetHuggingFaceConfig.model_validate(global_config)
    upload_jsonl_dataset_to_hf(config, delete_from_gitlab=True)


def download_jsonl_dataset_from_hf_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = DownloadJsonlDatasetHuggingFaceConfig.model_validate(global_config)
    download_jsonl_dataset_from_hf(config)
