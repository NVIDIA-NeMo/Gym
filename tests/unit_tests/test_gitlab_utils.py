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
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import requests
from mlflow.exceptions import MlflowException, RestException

import nemo_gym.gitlab_utils as gitlab_utils
from nemo_gym.config_types import ConfigError, DownloadJsonlDatasetGitlabConfig, MLFlowConfig
from nemo_gym.gitlab_utils import _get_model_version


# TODO: Eventually we want to add more tests to ensure that the Gitlab flow does not break
class TestGitlabUtils:
    def test_sanity(self) -> None:
        MLFlowConfig(mlflow_tracking_uri="", mlflow_tracking_token="")

    def test_registry_credentials_alone_do_not_enable_the_exporter(self) -> None:
        config = MLFlowConfig(mlflow_tracking_uri="https://gitlab.example.test", mlflow_tracking_token="t")

        assert not config.is_available


def _rest_404() -> RestException:
    return RestException({"error_code": "INTERNAL_ERROR", "message": "404 Not Found"})


class TestGetModelVersion:
    def test_returns_the_version_when_it_exists(self) -> None:
        model_version = MagicMock()
        client = MagicMock()
        client.get_model_version.return_value = model_version

        assert _get_model_version(client, "sample_dataset", "0.0.1") is model_version

    @pytest.mark.parametrize(
        "payload",
        [
            {"error_code": "RESOURCE_DOES_NOT_EXIST", "message": "Not found"},
            {"error_code": "404", "message": "Not found"},
            {"error_code": "INTERNAL_ERROR", "message": " 404 Not Found "},
            {"error": "404 Not Found"},
        ],
    )
    def test_404_adds_context_without_claiming_a_single_cause_or_leaking_credentials(
        self, monkeypatch: pytest.MonkeyPatch, payload: dict[str, str]
    ) -> None:
        token = "tracking-token-marker"
        registry = "https://gitlab.example.test/api/v4/projects/1/ml/mlflow?credential=uri-marker"
        monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", token)
        error = RestException(payload)
        client = MagicMock(tracking_uri=registry)
        client.get_model_version.side_effect = error

        with pytest.raises(ConfigError) as excinfo:
            _get_model_version(client, "sample_dataset", "0.0.9")

        message = str(excinfo.value)
        assert "dataset 'sample_dataset' version '0.0.9'" in message
        assert "missing and inaccessible resources" in message
        assert token not in message
        assert "uri-marker" not in message
        assert excinfo.value.__cause__ is error
        client.get_registered_model.assert_not_called()

    @pytest.mark.parametrize(
        "payload",
        [
            {"error_code": "INTERNAL_ERROR", "message": "server unavailable"},
            {"error_code": "PERMISSION_DENIED", "message": "denied"},
            {"error_code": "INTERNAL_ERROR", "message": 404},
        ],
    )
    def test_non_404_rest_failure_is_preserved(self, payload: dict[str, str | int]) -> None:
        error = RestException(payload)
        client = MagicMock()
        client.get_model_version.side_effect = error

        with pytest.raises(RestException) as excinfo:
            _get_model_version(client, "sample_dataset", "1")

        assert excinfo.value is error

    def test_transport_failure_is_preserved(self) -> None:
        timeout = requests.Timeout("timed out")
        error = MlflowException("API request failed with timeout")
        error.__cause__ = timeout
        client = MagicMock()
        client.get_model_version.side_effect = error

        with pytest.raises(MlflowException) as excinfo:
            _get_model_version(client, "sample_dataset", "1")

        assert excinfo.value is error
        assert excinfo.value.__cause__ is timeout


def test_dataset_download_reports_missing_version_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = MagicMock()
    client.get_model_version.side_effect = _rest_404()
    monkeypatch.setattr(gitlab_utils, "create_mlflow_client", lambda: client)
    destination = tmp_path / "dataset.jsonl"
    config = DownloadJsonlDatasetGitlabConfig(
        dataset_name="sample_dataset", version="2", artifact_fpath="data.jsonl", output_fpath=str(destination)
    )
    with pytest.raises(ConfigError, match="sample_dataset.*version '2'") as excinfo:
        gitlab_utils.download_jsonl_dataset(config)
    assert isinstance(excinfo.value.__cause__, RestException)
    assert not destination.exists()


def test_successful_dataset_download_is_unchanged(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_TOKEN", "test-token")
    client = MagicMock(tracking_uri="https://gitlab.example.test/api/v4/projects/1/ml/mlflow")
    client.get_model_version.return_value = SimpleNamespace(run_id="run-1")
    repository = SimpleNamespace(repo=SimpleNamespace(artifact_uri="https://gitlab.example.test/artifacts"))
    get_repository = MagicMock(return_value=repository)
    get_artifact = MagicMock(return_value=SimpleNamespace(content=b'{"value": 1}\n'))
    monkeypatch.setattr(gitlab_utils, "create_mlflow_client", lambda: client)
    monkeypatch.setattr(gitlab_utils, "get_artifact_repository", get_repository)
    monkeypatch.setattr(gitlab_utils.requests, "get", get_artifact)
    destination = tmp_path / "dataset.jsonl"
    config = DownloadJsonlDatasetGitlabConfig(
        dataset_name="sample_dataset", version="2", artifact_fpath="data.jsonl", output_fpath=str(destination)
    )

    gitlab_utils.download_jsonl_dataset(config)

    assert destination.read_text() == '{"value": 1}\n'
    get_repository.assert_called_once_with(
        artifact_uri="runs:/run-1", tracking_uri="https://gitlab.example.test/api/v4/projects/1/ml/mlflow"
    )
    get_artifact.assert_called_once_with(
        "https://gitlab.example.test/artifacts/data.jsonl", headers={"Authorization": "Bearer test-token"}
    )
