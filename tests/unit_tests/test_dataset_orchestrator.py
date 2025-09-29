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
from typing import Type, Union
from unittest.mock import MagicMock

from pytest import MonkeyPatch

from nemo_gym.config_types import UploadJsonlDatasetHuggingFaceConfig, UploadJsonlDatasetHuggingFaceMaybeDeleteConfig
from nemo_gym.dataset_orchestrator import delete_jsonl_dataset_from_gitlab, upload_jsonl_dataset_to_hf_maybe_delete


def make_base_config(
    config_class: Type[
        Union[UploadJsonlDatasetHuggingFaceConfig, UploadJsonlDatasetHuggingFaceMaybeDeleteConfig]
    ] = UploadJsonlDatasetHuggingFaceConfig,
    delete_from_gitlab: Union[bool, None] = None,
) -> Union[UploadJsonlDatasetHuggingFaceConfig, UploadJsonlDatasetHuggingFaceMaybeDeleteConfig]:
    kwargs = dict(
        hf_token="test_token",
        hf_organization="test_org",
        hf_collection_name="test_collection",
        hf_collection_slug="test_slug",
        dataset_name="dataset_name",
        input_jsonl_fpath="test/data",
        resource_config_path="test/config",
    )
    if delete_from_gitlab is not None and "delete_from_gitlab" in config_class.__annotations__:
        kwargs["delete_from_gitlab"] = delete_from_gitlab
    return config_class(**kwargs)


class TestDatasetOrchestrator:
    def test_delete_jsonl_dataset_from_gitlab(self, monkeypatch: MonkeyPatch):
        """Test direct call to delete_jsonl_dataset_from_gitlab"""
        mock_input = MagicMock(return_value="y")
        mock_mlflow_client = MagicMock()

        monkeypatch.setattr("builtins.input", mock_input)
        monkeypatch.setattr("nemo_gym.gitlab_utils.create_mlflow_client", lambda: mock_mlflow_client)

        delete_jsonl_dataset_from_gitlab("test_dataset")

        mock_input.assert_called_once()

        mock_mlflow_client.get_registered_model.assert_called_once_with("test_dataset")
        mock_mlflow_client.delete_registered_model.assert_called_once_with("test_dataset")

        assert True

    def test_upload_jsonl_dataset_to_hf_only(self, monkeypatch: MonkeyPatch):
        """Test uploading to HF without deleting"""
        mock_upload = MagicMock()
        mock_delete = MagicMock()

        monkeypatch.setattr("nemo_gym.dataset_orchestrator.upload_jsonl_dataset_to_hf", mock_upload)
        monkeypatch.setattr("nemo_gym.gitlab_utils.delete_model_from_gitlab", mock_delete)

        config = make_base_config()

        upload_jsonl_dataset_to_hf_maybe_delete(config, delete_from_gitlab=False)

        mock_upload.assert_called_once_with(config)
        mock_delete.assert_not_called()

    def test_upload_jsonl_dataset_to_hf_yes_delete(self, monkeypatch: MonkeyPatch):
        """Test Uploading to HF and deleting from Gitlab"""
        mock_upload = MagicMock()
        mock_mlflow_client = MagicMock()
        mock_input = MagicMock(return_value="y")

        monkeypatch.setattr("builtins.input", mock_input)
        monkeypatch.setattr("nemo_gym.gitlab_utils.create_mlflow_client", lambda: mock_mlflow_client)
        monkeypatch.setattr("nemo_gym.dataset_orchestrator.upload_jsonl_dataset_to_hf", mock_upload)

        config = make_base_config(config_class=UploadJsonlDatasetHuggingFaceMaybeDeleteConfig, delete_from_gitlab=True)

        upload_jsonl_dataset_to_hf_maybe_delete(config, delete_from_gitlab=True)

        mock_mlflow_client.get_registered_model.return_value = True
        mock_mlflow_client.get_registered_model.assert_called_once_with(config.dataset_name)
        mock_mlflow_client.delete_registered_model.assert_called_once_with(config.dataset_name)

        mock_upload.assert_called_once_with(config)
        mock_input.assert_called_once()

    def test_upload_jsonl_dataset_to_hf_no_delete_implicit(self, monkeypatch: MonkeyPatch):
        """Test Uploading to HF with no delete param"""
        mock_upload = MagicMock()
        mock_delete = MagicMock()
        monkeypatch.setattr("nemo_gym.dataset_orchestrator.upload_jsonl_dataset_to_hf", mock_upload)
        monkeypatch.setattr("nemo_gym.gitlab_utils.delete_model_from_gitlab", mock_delete)

        config = make_base_config(config_class=UploadJsonlDatasetHuggingFaceMaybeDeleteConfig)

        upload_jsonl_dataset_to_hf_maybe_delete(config, delete_from_gitlab=config.delete_from_gitlab)

        mock_upload.assert_called_once_with(config)
        mock_delete.assert_not_called()

    def test_upload_jsonl_dataset_to_hf_no_delete_explicit(self, monkeypatch: MonkeyPatch):
        """Test Uploading to HF and deleting from Gitlab with delete param"""
        mock_upload = MagicMock()
        mock_delete = MagicMock()
        monkeypatch.setattr("nemo_gym.dataset_orchestrator.upload_jsonl_dataset_to_hf", mock_upload)
        monkeypatch.setattr("nemo_gym.gitlab_utils.delete_model_from_gitlab", mock_delete)

        config = make_base_config(
            config_class=UploadJsonlDatasetHuggingFaceMaybeDeleteConfig, delete_from_gitlab=False
        )

        upload_jsonl_dataset_to_hf_maybe_delete(config, delete_from_gitlab=config.delete_from_gitlab)

        mock_upload.assert_called_once_with(config)
        mock_delete.assert_not_called()
