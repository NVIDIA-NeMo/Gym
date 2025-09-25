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
from os import environ
from pathlib import Path

import yaml
from huggingface_hub import HfApi, add_collection_item, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError

from nemo_gym.config_types import DownloadJsonlDatasetHfConfig, UploadJsonlDatasetHuggingFaceConfig
from nemo_gym.server_utils import get_global_config_dict


def create_huggingface_client(token: str) -> HfApi:  # pragma: no cover
    environ["HF_TOKEN"] = token
    client = HfApi()
    return client


def upload_jsonl_dataset_to_hf(
    config: UploadJsonlDatasetHuggingFaceConfig,
) -> None:  # pragma: no cover
    client = create_huggingface_client(config.hf_token)
    domain = None

    with open(config.resource_config_path, "r") as f:
        data = yaml.safe_load(f)

    # TODO: dry up
    def visit_domain(data, level=1):
        nonlocal domain
        if level == 4:
            domain = data.get("domain")
            return
        else:
            for k, v in data.items():
                if level == 2 and k != "resources_servers":
                    continue
                visit_domain(v, level + 1)

    visit_domain(data)

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

    try:
        client.create_repo(repo_id=repo_id, token=config.hf_token, repo_type="dataset", private=True)
    except HfHubHTTPError as e:
        print("ERROR: ", e)

    try:
        client.upload_file(
            path_or_fileobj=config.input_jsonl_fpath,
            path_in_repo=Path(config.input_jsonl_fpath).name,
            repo_id=repo_id,
            token=config.hf_token,
            repo_type="dataset",
        )
    except HfHubHTTPError as e:
        print("ERROR: ", e)

    # Add dataset to collection
    try:
        add_collection_item(collection_slug=collection_slug, item_id=repo_id, item_type="dataset")
    except HfHubHTTPError as e:
        print("ERROR: ", e)


def upload_jsonl_dataset_cli() -> None:  # pragma: no cover
    # TODO: do simple format check
    global_config = get_global_config_dict()
    config = UploadJsonlDatasetHuggingFaceConfig.model_validate(global_config)
    upload_jsonl_dataset_to_hf(config)


def download_jsonl_dataset(
    config: DownloadJsonlDatasetHfConfig,
) -> None:  # pragma: no cover
    try:
        downloaded_path = hf_hub_download(
            repo_id=config.repo_id,
            repo_type="dataset",
            filename=config.filename_in_repo,  # e.g., "data.jsonl"
            token=config.hf_token,
        )
        Path(config.output_fpath).write_bytes(Path(downloaded_path).read_bytes())
    except HfHubHTTPError as e:
        print("ERROR: ", e)


def download_jsonl_dataset_cli() -> None:  # pragma: no cover
    global_config = get_global_config_dict()
    config = DownloadJsonlDatasetHfConfig.model_validate(global_config)
    download_jsonl_dataset(config)
