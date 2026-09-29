# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from typing import Union

from nemo_gym.config_types import (
    UploadJsonlDatasetHuggingFaceConfig,
    UploadJsonlDatasetHuggingFaceMaybeDeleteConfig,
)
from nemo_gym.gitlab_utils import delete_model_from_gitlab, is_model_in_gitlab
from nemo_gym.hf_utils import upload_jsonl_dataset as upload_jsonl_dataset_to_hf


def delete_jsonl_dataset_from_gitlab(gitlab_model_name: str) -> None:  # pragma: no cover
    model_exists = is_model_in_gitlab(gitlab_model_name)

    # gitlab model_name must match hf dataset name
    if model_exists:
        confirm_delete = (
            input(
                f"[Nemo-Gym] - Found model '{gitlab_model_name}' in the registry. "
                f"Are you sure you want to delete it from Gitlab? [y/N]: "
            )
            .strip()
            .lower()
        )

        if confirm_delete in ("y", "yes"):
            delete_model_from_gitlab(gitlab_model_name)
            print(f"[Nemo-Gym] - Deleted '{gitlab_model_name}' from Gitlab.")
        else:
            print(f"[Nemo-Gym] - Skipped deletion of '{gitlab_model_name}'.")


def upload_jsonl_dataset_to_hf_maybe_delete(
    config: Union[UploadJsonlDatasetHuggingFaceMaybeDeleteConfig, UploadJsonlDatasetHuggingFaceConfig],
    delete_from_gitlab: bool = False,
) -> None:  # pragma: no cover
    gitlab_model_name = config.dataset_name

    upload_jsonl_dataset_to_hf(config)

    if delete_from_gitlab:
        delete_jsonl_dataset_from_gitlab(gitlab_model_name)


# Backward-compatibility shims (CLI refactor): these CLI entry points moved to `nemo_gym.cli.dataset`.
# Re-exported lazily to avoid a circular import; accessing them emits a DeprecationWarning.
from nemo_gym.cli._compat import moved_attr_getter  # noqa: E402


__getattr__ = moved_attr_getter(
    __name__,
    {
        "upload_jsonl_dataset_to_hf_cli": "nemo_gym.cli.dataset",
        "upload_jsonl_dataset_to_hf_and_delete_gitlab_cli": "nemo_gym.cli.dataset",
        "download_jsonl_dataset_from_hf_cli": "nemo_gym.cli.dataset",
        "delete_jsonl_dataset_from_gitlab_cli": "nemo_gym.cli.dataset",
    },
)
