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
import shutil
from pathlib import Path

import yaml
from datasets import load_dataset
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError
from scripts.update_resource_servers import visit_resource_server

from nemo_gym.config_types import DownloadJsonlDatasetHuggingFaceConfig, UploadJsonlDatasetHuggingFaceConfig


def create_huggingface_client(token: str) -> HfApi:  # pragma: no cover
    client = HfApi(token=token)
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
            print(f"[Nemo-Gym] - Missing keys across {len(missing_keys_info)} lines: {missing_keys}")
            return False

    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"[Nemo-Gym] - Error reading or parsing the JSON file: {e}")
        return False

    return True


def download_hf_dataset_as_jsonl(
    config: DownloadJsonlDatasetHuggingFaceConfig,
) -> None:  # pragma: no cover
    """
    Download a HF dataset and save as JSONL.
    If `artifact_fpath` is provided, downloads that specific file using `hf_hub_download`.
    Otherwise, uses datasets.load_dataset() to handle structured datasets.
    """
    try:
        # artifact_fpath - download raw jsonl file
        if config.artifact_fpath:
            downloaded_path = hf_hub_download(
                repo_id=config.repo_id,
                filename=config.artifact_fpath,
                repo_type="dataset",
                token=config.hf_token,
            )
            output_file = (
                Path(config.output_fpath or config.output_dirpath) / Path(config.artifact_fpath).name
                if config.output_dirpath
                else Path(config.output_fpath)
            )
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # We copy the downloaded file from the cache to the target path
            # to allow renaming (e.g., artifact_fpath="something.jsonl" -> output_fpath="train.jsonl")
            shutil.copy(downloaded_path, output_file)
            print(f"[Nemo-Gym] - Downloaded {config.artifact_fpath} to: {output_file}")
            return

        # no artifact_fpath - use load_dataset() with split (if provided)
        if config.output_fpath:
            # Exact output path specified
            output_file = Path(config.output_fpath)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            ds = load_dataset(config.repo_id, split=config.split, token=config.hf_token)
            ds.to_json(str(output_file))
            print(f"[Nemo-Gym] - Downloaded {config.split} split to: {output_file}")
        else:
            # Output directory specified
            output_dir = Path(config.output_dirpath)
            output_dir.mkdir(parents=True, exist_ok=True)

            if config.split:
                ds = load_dataset(config.repo_id, split=config.split, token=config.hf_token)
                output_file = output_dir / f"{config.split}.jsonl"
                ds.to_json(str(output_file))
                print(f"[Nemo-Gym] - Downloaded {config.split} split to: {output_file}")
            else:
                # Download all
                ds = load_dataset(config.repo_id, token=config.hf_token)
                for split_name, split_data in ds.items():
                    output_file = output_dir / f"{split_name}.jsonl"
                    split_data.to_json(str(output_file))
                    print(f"[Nemo-Gym] - Downloaded {split_name} split to: {output_file}")

    except Exception as e:
        print(f"[Nemo-Gym] - Error downloading/converting dataset: {e}")
        raise


def upload_jsonl_dataset(
    config: UploadJsonlDatasetHuggingFaceConfig,
) -> None:  # pragma: no cover
    client = create_huggingface_client(config.hf_token)
    with open(config.resource_config_path, "r") as f:
        data = yaml.safe_load(f)

    domain = d.lower() + "-" if (d := visit_resource_server(data).to_dict().get("domain")) else ""
    resource_server = config.resource_config_path.split("/")[1]
    dataset_name = config.dataset_name or resource_server
    prefix = config.hf_dataset_prefix + "-" if config.hf_dataset_prefix else ""
    collection_id = (
        f"{config.hf_organization}/{config.hf_collection_name.lower().replace(' ', '-')}-{config.hf_collection_slug}"
    )

    repo_id = f"{config.hf_organization}/{prefix}{domain}{dataset_name}"

    # Dataset format check - only strict check for training data
    is_training = config.split.lower() == "train"
    if is_training and not check_jsonl_format(config.input_jsonl_fpath):
        print("[Nemo-Gym] - JSONL file format check failed.")
        return

    # Repo creation
    try:
        client.create_repo(repo_id=repo_id, token=config.hf_token, repo_type="dataset", private=True, exist_ok=True)
        print(f"[Nemo-Gym] - Repo '{repo_id}' is ready for use")
    except HfHubHTTPError as e:
        if config.create_pr and "403" in str(e):
            print(f"[Nemo-Gym] - Repo '{repo_id}' exists (no create permission, will create PR)")
        else:
            print(f"[Nemo-Gym] - Error creating repo: {e}")
            raise

    # Collection id + addition
    try:
        collection = client.get_collection(collection_id)
        if any(item.item_id.lower() == repo_id.lower() and item.item_type == "dataset" for item in collection.items):
            print(
                f"[Nemo-Gym] - Dataset '{dataset_name}' under Repo '{repo_id}' is already in Collection '{collection_id}'"
            )
        else:
            client.add_collection_item(collection_slug=collection_id, item_id=repo_id, item_type="dataset")
            print(f"[Nemo-Gym] - Dataset '{repo_id}' added to collection '{collection_id}'")
    except HfHubHTTPError as e:
        print(f"[Nemo-Gym] - Error adding to collection: {e}")
        raise

    # File upload
    try:
        commit_info = client.upload_file(
            path_or_fileobj=config.input_jsonl_fpath,
            path_in_repo=Path(config.input_jsonl_fpath).name,
            repo_id=repo_id,
            token=config.hf_token,
            repo_type="dataset",
            create_pr=config.create_pr,
            revision=config.revision,
            commit_message=config.commit_message,
            commit_description=config.commit_description,
        )
        if config.create_pr:
            print(f"[Nemo-Gym] - Pull Request created: {commit_info.pr_url}")
        else:
            print("[Nemo-Gym] - Dataset upload successful")
    except HfHubHTTPError as e:
        print(f"[Nemo-Gym] - Error uploading file: {e}")
        raise
