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
import subprocess
import sys
from importlib.util import find_spec
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
from responses_api_models.local_vllm_model.external_app import LocalVLLMExternalModel, LocalVLLMExternalModelConfig
from responses_api_models.vllm_model.app import VLLMModel


SERVER_DIR = Path(__file__).resolve().parents[1]


def _config(**overrides) -> LocalVLLMExternalModelConfig:
    fields = dict(
        host="127.0.0.1",
        port=0,
        entrypoint="external_app.py",
        name="judge",
        model="judge-model",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        base_url="http://judge-pool:8000/v1",
    )
    return LocalVLLMExternalModelConfig(**(fields | overrides))


def test_entrypoint_imports_neither_ray_nor_vllm() -> None:
    # Without Ray and vLLM installed, an entrypoint that imports them behind a try/except would pass this test,
    # so require them.
    # This server's venv installs both, so only the entrypoint's own imports can keep them out.
    assert find_spec("ray") is not None and find_spec("vllm") is not None
    probe = (
        "import sys, runpy\n"
        f"runpy.run_path({str(SERVER_DIR / 'external_app.py')!r}, run_name='not_main')\n"
        "print(sorted(m for m in ('ray', 'vllm') if m in sys.modules))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(PARENT_DIR), "PATH": ""},
        check=True,
    )
    assert result.stdout.strip().splitlines()[-1] == "[]"


def test_server_declares_no_ray_where_the_supervisor_reads_it() -> None:
    assert entrypoint_ray_enabled(SERVER_DIR / "external_app.py") is False
    assert entrypoint_ray_enabled(SERVER_DIR / "app.py") is True


def test_server_does_not_join_ray_even_when_the_run_has_a_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    initialize_ray = MagicMock()
    monkeypatch.setattr(nemo_gym.server_runtime.ray_support, "initialize_ray", initialize_ray)

    _connect_server_to_ray(LocalVLLMExternalModel, OmegaConf.create({"ray_head_node_address": "10.0.0.1:6379"}))

    initialize_ray.assert_not_called()


@pytest.mark.parametrize("base_url", ["", "  ", [], [""], ["http://judge-pool:8000/v1", " "]])
def test_config_requires_a_nonempty_base_url(base_url: str | list[str]) -> None:
    with pytest.raises(ValidationError, match="needs `base_url`"):
        _config(base_url=base_url)


def test_config_rejects_a_missing_base_url() -> None:
    with pytest.raises(ValidationError, match="base_url"):
        LocalVLLMExternalModelConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="external_app.py",
            name="judge",
            model="judge-model",
            return_token_id_information=False,
            uses_reasoning_parser=False,
        )


def test_config_accepts_the_local_launch_block_unchanged() -> None:
    # A recipe switches between app.py and external_app.py without editing the rest of its block.
    config = _config(
        vllm_serve_kwargs={"tensor_parallel_size": 4},
        vllm_serve_env_vars={"VLLM_RAY_DP_PACK_STRATEGY": "strict"},
        ray_worker_py_executable="/opt/venv/bin/python",
    )
    assert config.base_url == ["http://judge-pool:8000/v1"]
    assert config.vllm_serve_kwargs == {"tensor_parallel_size": 4}


def test_inner_vllm_config_route_reports_the_external_endpoint() -> None:
    server = LocalVLLMExternalModel(
        config=_config(), server_client=MagicMock(spec=ServerClient, global_config_dict={})
    )
    client = TestClient(server.setup_webserver())

    response = client.get("/get_inner_vllm_config")

    assert response.status_code == 200
    assert response.json() == {
        "base_url": ["http://judge-pool:8000/v1"],
        "api_key": "dummy",  # pragma: allowlist secret
        "model": "judge-model",
    }
    assert ("GET", "/get_inner_vllm_config") in LocalVLLMExternalModel.non_generating_model_routes
    assert LocalVLLMExternalModel.non_generating_model_routes >= VLLMModel.non_generating_model_routes
