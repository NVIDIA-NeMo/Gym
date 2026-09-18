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
from unittest.mock import MagicMock

import pytest
import requests
from mlflow.exceptions import RestException
from requests_mock import Mocker

import nemo_gym.gitlab_utils as gitlab_utils
from nemo_gym.config_types import ConfigError, DownloadJsonlDatasetGitlabConfig, MLFlowConfig
from nemo_gym.gitlab_utils import _get_model_version, _gitlab_project_url, _token_is_project_member


# TODO: Eventually we want to add more tests to ensure that the Gitlab flow does not break
class TestGitlabUtils:
    def test_sanity(self) -> None:
        MLFlowConfig(mlflow_tracking_uri="", mlflow_tracking_token="")

    def test_registry_credentials_alone_do_not_enable_the_exporter(self) -> None:
        config = MLFlowConfig(mlflow_tracking_uri="https://gitlab.example.test", mlflow_tracking_token="t")

        assert not config.is_available


def _rest_404() -> RestException:
    return RestException({"error_code": "INTERNAL_ERROR", "message": "404 Not Found"})


class _FakeClient:
    """Stands in for MlflowClient talking to the GitLab-hosted registry."""

    tracking_uri = "https://gitlab.example.test/api/v4/projects/1/ml/mlflow"

    def __init__(self, versions: dict[tuple[str, str], str], models: list[str]) -> None:
        self._versions = versions
        self._models = models

    def get_model_version(self, name: str, version: str) -> str:
        if (name, version) not in self._versions:
            raise _rest_404()
        return self._versions[(name, version)]

    def get_registered_model(self, name: str) -> str:
        if name not in self._models:
            raise _rest_404()
        return name


class TestGitlabProjectUrl:
    def test_strips_the_mlflow_suffix(self) -> None:
        url = _gitlab_project_url("https://gitlab.example.test/api/v4/projects/1/ml/mlflow/")

        assert url == "https://gitlab.example.test/api/v4/projects/1"

    def test_returns_none_for_a_registry_that_is_not_gitlab(self) -> None:
        assert _gitlab_project_url("https://mlflow.example.test") is None


class TestTokenIsProjectMember:
    def test_reads_membership_off_the_project_permissions(self, requests_mock: Mocker) -> None:
        requests_mock.get(
            "https://gitlab.example.test/api/v4/projects/1",
            json={"permissions": {"project_access": {"access_level": 30}, "group_access": None}},
        )

        assert _token_is_project_member("https://gitlab.example.test/api/v4/projects/1", "t") is True

    def test_a_non_member_has_neither_project_nor_group_access(self, requests_mock: Mocker) -> None:
        requests_mock.get(
            "https://gitlab.example.test/api/v4/projects/1",
            json={"permissions": {"project_access": None, "group_access": None}},
        )

        assert _token_is_project_member("https://gitlab.example.test/api/v4/projects/1", "t") is False

    def test_an_unusable_answer_is_reported_as_unknown(self, requests_mock: Mocker) -> None:
        requests_mock.get("https://gitlab.example.test/api/v4/projects/1", json={"message": "401 Unauthorized"})

        assert _token_is_project_member("https://gitlab.example.test/api/v4/projects/1", "t") is None


class TestGetModelVersion:
    def test_returns_the_version_when_it_exists(self) -> None:
        client = _FakeClient({("sample_dataset", "0.0.1"): "mv"}, ["sample_dataset"])

        assert _get_model_version(client, "sample_dataset", "0.0.1") == "mv"

    def test_a_known_dataset_at_an_unknown_version_names_the_version(self) -> None:
        client = _FakeClient({}, ["sample_dataset"])

        with pytest.raises(ConfigError) as excinfo:
            _get_model_version(client, "sample_dataset", "0.0.9")

        assert "but not at version '0.0.9'" in str(excinfo.value)

    def test_a_non_member_token_is_reported_as_an_access_problem(self, requests_mock: Mocker) -> None:
        # GitLab 404s every registry request for a non-member, so the dataset looks absent
        # even though it is published and READY.
        requests_mock.get(
            "https://gitlab.example.test/api/v4/projects/1",
            json={"permissions": {"project_access": None, "group_access": None}},
        )
        client = _FakeClient({}, [])

        with pytest.raises(ConfigError) as excinfo:
            _get_model_version(client, "sample_dataset", "0.0.1")

        assert "not a member of the project" in str(excinfo.value)

    def test_a_member_token_is_told_the_dataset_is_absent(self, requests_mock: Mocker) -> None:
        requests_mock.get(
            "https://gitlab.example.test/api/v4/projects/1",
            json={"permissions": {"project_access": {"access_level": 30}, "group_access": None}},
        )
        client = _FakeClient({}, [])

        with pytest.raises(ConfigError) as excinfo:
            _get_model_version(client, "missing_dataset", "0.0.1")

        assert "'missing_dataset' was not found" in str(excinfo.value)

    def test_an_unreachable_project_endpoint_leaves_both_causes_open(self, requests_mock: Mocker) -> None:
        requests_mock.get("https://gitlab.example.test/api/v4/projects/1", status_code=500, text="boom")
        client = _FakeClient({}, [])

        with pytest.raises(ConfigError) as excinfo:
            _get_model_version(client, "sample_dataset", "0.0.1")

        assert "Either it was never published" in str(excinfo.value)


@pytest.mark.parametrize(
    "error_code,message", [("INTERNAL_ERROR", "server unavailable"), ("PERMISSION_DENIED", "denied")]
)
def test_non_404_failure_is_preserved(error_code: str, message: str) -> None:
    error = RestException({"error_code": error_code, "message": message})
    client = MagicMock()
    client.get_model_version.side_effect = error
    with pytest.raises(RestException) as excinfo:
        _get_model_version(client, "sample_dataset", "1")
    assert excinfo.value is error
    client.get_registered_model.assert_not_called()


def test_secondary_failure_does_not_claim_dataset_is_missing() -> None:
    client = MagicMock(tracking_uri="https://mlflow.example.test")
    client.get_model_version.side_effect = _rest_404()
    client.get_registered_model.side_effect = RestException({"error_code": "INTERNAL_ERROR", "message": "unavailable"})
    with pytest.raises(ConfigError, match="Could not resolve dataset 'sample_dataset' version '1'"):
        _get_model_version(client, "sample_dataset", "1")


def test_non_gitlab_registry_does_not_probe_membership(requests_mock: Mocker) -> None:
    client = _FakeClient({}, [])
    client.tracking_uri = "https://mlflow.example.test"
    with pytest.raises(ConfigError, match="Either it was never published"):
        _get_model_version(client, "sample_dataset", "1")
    assert not requests_mock.called


@pytest.mark.parametrize("response", [{"permissions": None}, {"permissions": []}, {"permissions": "invalid"}])
def test_malformed_permissions_are_unknown(requests_mock: Mocker, response: dict) -> None:
    requests_mock.get("https://gitlab.example.test/api/v4/projects/1", json=response)
    assert _token_is_project_member("https://gitlab.example.test/api/v4/projects/1", "test-token") is None


def test_membership_probe_does_not_follow_redirects(requests_mock: Mocker) -> None:
    url = "https://gitlab.example.test/api/v4/projects/1"
    requests_mock.get(url, status_code=302, headers={"Location": "https://other.example.test/"})
    assert _token_is_project_member(url, "test-token") is None
    assert len(requests_mock.request_history) == 1


def test_membership_transport_failure_is_unknown(requests_mock: Mocker) -> None:
    url = "https://gitlab.example.test/api/v4/projects/1"
    requests_mock.get(url, exc=requests.Timeout)
    assert _token_is_project_member(url, "test-token") is None


@pytest.mark.parametrize("error_code", ["RESOURCE_DOES_NOT_EXIST", "404"])
def test_native_not_found_codes_are_diagnosed(error_code: str) -> None:
    client = MagicMock(tracking_uri="https://mlflow.example.test")
    client.get_model_version.side_effect = RestException({"error_code": error_code, "message": "Not found"})
    with pytest.raises(ConfigError, match="but not at version '2'"):
        _get_model_version(client, "sample_dataset", "2")


def test_dataset_download_reports_missing_version_before_writing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = _FakeClient({}, ["sample_dataset"])
    monkeypatch.setattr(gitlab_utils, "create_mlflow_client", lambda: client)
    destination = tmp_path / "dataset.jsonl"
    config = DownloadJsonlDatasetGitlabConfig(
        dataset_name="sample_dataset", version="2", artifact_fpath="data.jsonl", output_fpath=str(destination)
    )
    with pytest.raises(ConfigError, match="sample_dataset.*version '2'") as excinfo:
        gitlab_utils.download_jsonl_dataset(config)
    assert isinstance(excinfo.value.__cause__, RestException)
    assert not destination.exists()
