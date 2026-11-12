# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from pydantic import ValidationError

import nemo_gym.server_runtime.ray_support
from nemo_gym import PARENT_DIR
from nemo_gym.client import ServerClient
from nemo_gym.server_runtime import entrypoint_ray_enabled
from nemo_gym.server_runtime.ray_support import _connect_server_to_ray
from responses_api_models.genrm_model.common import GenRMModelConfig
from responses_api_models.genrm_model.external_app import GenRMExternalModel, GenRMExternalModelConfig


SERVER_DIR = Path(__file__).resolve().parents[1]


def _config(**overrides) -> GenRMExternalModelConfig:
    fields = dict(
        host="127.0.0.1",
        port=0,
        entrypoint="external_app.py",
        name="genrm",
        model="genrm-model",
        return_token_id_information=False,
        uses_reasoning_parser=True,
        base_url="http://genrm-pool:8000/v1",
    )
    return GenRMExternalModelConfig(**(fields | overrides))


def _server(**overrides) -> GenRMExternalModel:
    return GenRMExternalModel(
        config=_config(**overrides), server_client=MagicMock(spec=ServerClient, global_config_dict={})
    )


def test_entrypoint_imports_neither_ray_nor_vllm() -> None:
    probe = (
        "import importlib.util, json, runpy, sys\n"
        "installed = sorted(m for m in ('ray', 'vllm') if importlib.util.find_spec(m) is not None)\n"
        f"runpy.run_path({str(SERVER_DIR / 'external_app.py')!r}, run_name='not_main')\n"
        "imported = sorted(m for m in ('ray', 'vllm') if m in sys.modules)\n"
        "print(json.dumps({'installed': installed, 'imported': imported}))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(PARENT_DIR), "PATH": ""},
        check=True,
    )
    modules = json.loads(result.stdout.strip().splitlines()[-1])
    # Without Ray in this venv nothing could import it, and the check below would pass for any entrypoint.
    assert "ray" in modules["installed"]
    assert modules["imported"] == []


def test_server_declares_no_ray_where_the_supervisor_reads_it() -> None:
    assert entrypoint_ray_enabled(SERVER_DIR / "external_app.py") is False
    assert entrypoint_ray_enabled(SERVER_DIR / "app.py") is True


def test_server_does_not_join_ray_even_when_the_run_has_a_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_ray = MagicMock()
    monkeypatch.setattr(nemo_gym.server_runtime.ray_support, "initialize_ray", initialize_ray)

    _connect_server_to_ray(GenRMExternalModel, OmegaConf.create({"ray_head_node_address": "10.0.0.1:6379"}))

    initialize_ray.assert_not_called()


def test_config_requires_base_url() -> None:
    with pytest.raises(ValidationError, match="needs `base_url`"):
        _config(base_url=[])


def test_config_carries_every_genrm_field() -> None:
    # A field added to GenRMModelConfig reaches the external config, so GenRMModelMixin can read it from either.
    assert issubclass(GenRMExternalModelConfig, GenRMModelConfig)
    assert _config(supports_principle_role=False).supports_principle_role is False


def test_config_accepts_the_shipped_local_launch_block() -> None:
    # Production recipes inject base_url into the shipped block; switching entrypoint must not need other edits.
    block = OmegaConf.to_container(
        OmegaConf.load(SERVER_DIR / "configs/genrm_model.yaml").genrm_model.responses_api_models.genrm_model
    )
    block |= {"entrypoint": "external_app.py", "base_url": "http://genrm-pool:8000/v1"}

    config = GenRMExternalModelConfig(host="127.0.0.1", port=0, name="genrm", **block)

    assert config.base_url == ["http://genrm-pool:8000/v1"]
    assert config.supports_principle_role is True
    assert config.vllm_serve_kwargs["tensor_parallel_size"] == 8


@pytest.mark.parametrize("supports_principle_role", [True, False])
def test_server_appends_genrm_roles_from_metadata(supports_principle_role: bool) -> None:
    server = _server(supports_principle_role=supports_principle_role)
    body = {
        "messages": [{"role": "user", "content": "Which is better?"}],
        "metadata": {"response_1": "A", "response_2": "B", "principle": "Be concise."},
    }

    body = server._preprocess_chat_completion_create_params(MagicMock(), body)

    principle = [{"role": "principle", "content": "Be concise."}] if supports_principle_role else []
    assert body["messages"] == [
        {"role": "user", "content": "Which is better?"},
        *principle,
        {"role": "response_1", "content": "A"},
        {"role": "response_2", "content": "B"},
    ]
    assert "metadata" not in body
    assert body["model"] == "genrm-model"


def test_inner_vllm_config_route_reports_the_external_pool() -> None:
    client = TestClient(_server().setup_webserver())

    response = client.get("/get_inner_vllm_config")

    assert response.status_code == 200
    assert response.json() == {
        "base_url": ["http://genrm-pool:8000/v1"],
        "api_key": "dummy",  # pragma: allowlist secret
        "model": "genrm-model",
    }
