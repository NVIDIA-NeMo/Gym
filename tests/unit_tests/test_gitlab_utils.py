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
from pytest import MonkeyPatch

from nemo_gym.config_types import ConfigError, DownloadJsonlDatasetGitlabConfig, MLFlowConfig
from nemo_gym.gitlab_utils import create_mlflow_client, download_jsonl_dataset


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

    def test_download_creates_the_output_directory(self, monkeypatch: MonkeyPatch, tmp_path: Path) -> None:
        # README commands download to a relative `data/...` path, which exists only in the Gym root.
        # Fake the registry lookup and the HTTP fetch; mlflow itself is an optional extra.
        client = SimpleNamespace(
            tracking_uri="https://gitlab.invalid/api/v4/projects/1/ml/mlflow",
            get_model_version=lambda name, version: SimpleNamespace(run_id="run"),
        )
        repository = SimpleNamespace(repo=SimpleNamespace(artifact_uri="https://gitlab.invalid/artifacts/"))
        monkeypatch.setattr("nemo_gym.gitlab_utils.create_mlflow_client", lambda: client)
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

        def fake_get(url: str, headers: dict) -> SimpleNamespace:
            requested.append(url)
            return SimpleNamespace(content=b'{"question": "q"}\n')

        monkeypatch.setattr("nemo_gym.gitlab_utils.requests.get", fake_get)
        output = tmp_path / "data" / "nested" / "rows.jsonl"

        download_jsonl_dataset(
            DownloadJsonlDatasetGitlabConfig(
                dataset_name="dataset", version="0.0.1", artifact_fpath="rows.jsonl", output_fpath=str(output)
            )
        )

        assert requested == ["https://gitlab.invalid/artifacts/rows.jsonl"]
        assert output.read_text() == '{"question": "q"}\n'
