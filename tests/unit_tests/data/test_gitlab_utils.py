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
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import requests
from pytest import MonkeyPatch

from nemo_gym.config.config_types import ConfigError, DownloadJsonlDatasetGitlabConfig, MLFlowConfig
from nemo_gym.data.gitlab_utils import create_mlflow_client, download_jsonl_dataset


# TODO: Eventually we want to add more tests to ensure that the Gitlab flow does not break
class TestGitlabUtils:
    def test_sanity(self) -> None:
        MLFlowConfig(mlflow_tracking_uri="", mlflow_tracking_token="")

    def test_registry_credentials_alone_do_not_enable_the_exporter(self) -> None:
        config = MLFlowConfig(mlflow_tracking_uri="https://gitlab", mlflow_tracking_token="t")

        assert not config.is_available

    def test_create_mlflow_client_gives_an_actionable_error_when_mlflow_is_missing(
        self, monkeypatch: MonkeyPatch
    ) -> None:
        # mlflow is an optional extra (`nemo-gym[mlflow]`); force the lazy import in
        # create_mlflow_client to fail the way it would on a plain Gym install.
        monkeypatch.setitem(sys.modules, "mlflow", None)

        with pytest.raises(ConfigError, match=r"pip install nemo-gym\[mlflow\]"):
            create_mlflow_client()

    @pytest.mark.parametrize("status_code", [200, 401, 404, 500])
    def test_download_creates_output_or_preserves_existing_file_on_http_error(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, status_code: int
    ) -> None:
        # README commands download to a relative `data/...` path, which exists only in the Gym root.
        # Fake the registry lookup and the HTTP fetch; mlflow itself is an optional extra.
        client = SimpleNamespace(
            tracking_uri="https://gitlab.invalid/api/v4/projects/1/ml/mlflow",
            get_model_version=lambda name, version: SimpleNamespace(run_id="run"),
        )
        repository = SimpleNamespace(repo=SimpleNamespace(artifact_uri="https://gitlab.invalid/artifacts/"))
        monkeypatch.setattr("nemo_gym.data.gitlab_utils.create_mlflow_client", lambda: client)
        monkeypatch.setitem(sys.modules, "mlflow", ModuleType("mlflow"))
        monkeypatch.setitem(
            sys.modules, "mlflow.artifacts", SimpleNamespace(get_artifact_repository=lambda **_: repository)
        )
        monkeypatch.setitem(
            sys.modules,
            "mlflow.environment_variables",
            SimpleNamespace(MLFLOW_TRACKING_TOKEN=SimpleNamespace(get=lambda: "token")),
        )
        requested = []

        def fake_get(url: str, headers: dict, timeout: tuple[int, int]) -> requests.Response:
            requested.append((url, timeout))
            response = requests.Response()
            response.status_code = status_code
            response._content = b'{"question": "q"}\n' if status_code == 200 else b'{"message": "error"}'
            return response

        monkeypatch.setattr("nemo_gym.data.gitlab_utils.requests.get", fake_get)
        output = tmp_path / "data" / "nested" / "rows.jsonl"

        config = DownloadJsonlDatasetGitlabConfig(
            dataset_name="dataset",
            version="0.0.1",
            artifact_fpath="references/rows.jsonl",
            output_fpath=str(output),
        )
        if status_code == 200:
            download_jsonl_dataset(config)
            assert output.read_text() == '{"question": "q"}\n'
        else:
            output.parent.mkdir(parents=True)
            output.write_text("previous valid data")
            with pytest.raises(requests.HTTPError):
                download_jsonl_dataset(config)
            assert output.read_text() == "previous valid data"

        assert requested == [("https://gitlab.invalid/artifacts/references/rows.jsonl", (10, 120))]
