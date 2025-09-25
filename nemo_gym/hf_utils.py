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
from os import environ
from pathlib import Path

import yaml
from huggingface_hub import HfApi, add_collection_item, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from scripts.update_resource_servers import get_dataset_domain

from nemo_gym.config_types import DownloadJsonlDatasetHuggingFaceConfig, UploadJsonlDatasetHuggingFaceConfig
from nemo_gym.server_utils import get_global_config_dict


def create_huggingface_client(token: str) -> HfApi:  # pragma: no cover
    environ["HF_TOKEN"] = token
    client = HfApi()
    return client


def check_jsonl_format(file_path: str) -> bool:  # pragma: no cover
    """Check for the presence of the expected keys in the dataset"""
    required_keys = {"responses_create_params", "reward_profiles", "expected_answer"}
    missing_keys_info = []

    try:
        with open(file_path, "r") as f:
            for line_number, line in enumerate(f, start=1):
                json_obj = json.loads(line)
                missing_keys = required_keys - json_obj.keys()
                if missing_keys:
                    missing_keys_info.append((line_number, missing_keys))

        if missing_keys_info:
            for line_number, missing_keys in missing_keys_info:
                print(f"Line {line_number} is missing keys: {missing_keys}")
            return False

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error reading or prasing the JSON file: {e}")
        return False

    return True


def upload_jsonl_dataset_to_hf(
    config: UploadJsonlDatasetHuggingFaceConfig,
) -> None:  # pragma: no cover
    client = create_huggingface_client(config.hf_token)

    with open(config.resource_config_path, "r") as f:
        data = yaml.safe_load(f)

    domain = get_dataset_domain(data)

    # TODO: prefix should be Nvidia
    prefix = config.hf_username
    suffix = "68d4abe7a735ee7ae216993e"
    # prefix = "Nvidia"
    # suffix = "68d1e0902765fbacc937bb4f"
    collection = "Nemo-Gym"
    domain = domain.title() if domain else None
    resource_server = config.resource_config_path.split("/")[1]
    dataset_name = config.dataset_name
    repo_id = f"{prefix}/{collection}-{domain}-{resource_server}-{dataset_name}"
    collection_slug = f"{prefix}/{collection}-{suffix}"

    if not check_jsonl_format(config.input_jsonl_fpath):
        print("JSONL file format check failed.")
        return

    try:
        client.create_repo(repo_id=repo_id, token=config.hf_token, repo_type="dataset", private=True)
    except HfHubHTTPError as e:
        print(f"Error creating repo: {e}")

    try:
        client.upload_file(
            path_or_fileobj=config.input_jsonl_fpath,
            path_in_repo=Path(config.input_jsonl_fpath).name,
            repo_id=repo_id,
            token=config.hf_token,
            repo_type="dataset",
        )
    except HfHubHTTPError as e:
        print(f"Error uploading file: {e}")

    try:
        add_collection_item(collection_slug=collection_slug, item_id=repo_id, item_type="dataset")
    except HfHubHTTPError as e:
        print(f"Error adding to collection: {e}")


def upload_jsonl_dataset_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetHuggingFaceConfig.model_validate(global_config)
    upload_jsonl_dataset_to_hf(config)


def download_jsonl_dataset(
    config: DownloadJsonlDatasetHuggingFaceConfig,
) -> None:  # pragma: no cover
    try:
        downloaded_path = hf_hub_download(
            repo_id=config.repo_id,
            repo_type="dataset",
            filename=config.artifact_fpath,
            token=config.hf_token,
        )
        Path(config.output_fpath).write_bytes(Path(downloaded_path).read_bytes())
    except HfHubHTTPError as e:
        print(f"Error downloading file: {e}")


def download_jsonl_dataset_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = DownloadJsonlDatasetHuggingFaceConfig.model_validate(global_config)
    download_jsonl_dataset(config)
