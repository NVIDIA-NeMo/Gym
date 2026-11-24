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
"""A rollout-attributed model call names the Gym server instance that made it (CALLER_SERVER_HEADER)."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from omegaconf import DictConfig
from pytest import MonkeyPatch, mark

import nemo_gym.client.http_client
from nemo_gym.client.server_client import (
    NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME,
    NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME,
    ServerClient,
    rollout_path_prefix,
)
from nemo_gym.config.config_types import BaseServerConfig, ModelServerRef
from nemo_gym.config.global_config import NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME
from nemo_gym.core.episode_context import CALLER_SERVER_HEADER, rollout_context
from nemo_gym.models.model_call_capture import CaptureStore
from resources_servers.conversational_tool_use_simulation.app import (
    ConversationalToolUseSimulationConfig,
    ConversationalToolUseSimulationServer,
)
from tests.unit_tests.models.test_token_capture import _both_enabled, _server


def _client(monkeypatch: MonkeyPatch, *, observability: bool) -> tuple[ServerClient, AsyncMock]:
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="head", port=11000),
        global_config_dict=DictConfig(
            {
                "observability_enabled": observability,
                "judge_model": {"responses_api_models": {"vllm_model": {"host": "model", "port": 1}}},
                "resources": {"resources_servers": {"judge": {"host": "resources", "port": 2}}},
            }
        ),
    )
    request_mock = AsyncMock(return_value="ok")
    monkeypatch.setattr(
        nemo_gym.client.http_client, "get_global_aiohttp_client", lambda: MagicMock(request=request_mock)
    )
    return server_client, request_mock


def _sent_headers(request_mock: AsyncMock) -> dict:
    return request_mock.await_args.kwargs.get("headers") or {}


@mark.parametrize("source", ["rollout_context", "conversational_judge", "training_path", "subprocess_base_url"])
async def test_rollout_model_call_records_its_caller(monkeypatch: MonkeyPatch, tmp_path: Path, source: str) -> None:
    caller = "conversation_environment"
    monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, caller)
    server_client, request_mock = _client(monkeypatch, observability=True)
    model = TestClient(_server(_both_enabled(tmp_path)).setup_webserver())

    def send_to_model(method, url, **kwargs):
        response = model.request(method, url, **kwargs)
        return SimpleNamespace(
            ok=response.is_success, headers=response.headers, read=AsyncMock(return_value=response.content)
        )

    request_mock.side_effect = send_to_model
    prefix = rollout_path_prefix("4-2", token_capture=source in {"training_path", "subprocess_base_url"})
    if source == "conversational_judge":
        config = ConversationalToolUseSimulationConfig(host="localhost", port=0, entrypoint="", name=caller)
        environment = ConversationalToolUseSimulationServer(config=config, server_client=server_client)
        await environment._call_model(
            model_server=ModelServerRef(type="responses_api_models", name="judge_model"),
            params=config.judge_responses_create_params,
            messages=[],
            rollout_id="4-2",
        )
    elif source == "training_path":
        await server_client.post(server_name="judge_model", url_path=f"{prefix}/v1/responses", json={"input": "hi"})
    else:
        if source == "subprocess_base_url":
            monkeypatch.setenv(NEMO_GYM_MODEL_SERVER_NAME_ENV_VAR_NAME, "judge_model")
            monkeypatch.setenv(NEMO_GYM_MODEL_SERVER_BASE_URL_ENV_VAR_NAME, f"http://model:1{prefix}")
        with rollout_context("4-2"):
            await server_client.post(server_name="judge_model", url_path="/v1/responses", json={"input": "hi"})

    assert request_mock.await_args.kwargs["url"] == f"http://model:1{prefix}/v1/responses"
    [record] = CaptureStore(tmp_path).read("4-2")
    assert record["caller_server"] == caller


@mark.parametrize(
    "config_path, observability, server_name, in_rollout",
    [
        (None, True, "judge_model", True),  # no server instance name to send
        ("my_env_resources_server", False, "judge_model", True),  # no rollout attribution
        ("my_env_resources_server", True, "resources", True),  # not a model server
        ("my_env_resources_server", True, "judge_model", False),  # not inside a rollout
    ],
)
async def test_caller_is_not_sent_outside_rollout_attributed_model_calls(
    monkeypatch: MonkeyPatch, config_path: str | None, observability: bool, server_name: str, in_rollout: bool
) -> None:
    if config_path is None:
        monkeypatch.delenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, raising=False)
    else:
        monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, config_path)
    server_client, request_mock = _client(monkeypatch, observability=observability)

    url_path = "/v1/responses" if server_name == "judge_model" else "/verify"
    if in_rollout:
        with rollout_context("4-2"):
            await server_client.post(server_name=server_name, url_path=url_path)
    else:
        await server_client.post(server_name=server_name, url_path=url_path)

    assert CALLER_SERVER_HEADER not in _sent_headers(request_mock)


@mark.parametrize("path", ["/ng-rollout/4-2/v1/models", "/ng-rollout/4-2/extra/v1/responses"])
async def test_prefixed_non_generation_routes_do_not_name_a_caller(monkeypatch: MonkeyPatch, path: str) -> None:
    monkeypatch.setenv(NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME, "conversation_environment")
    server_client, request_mock = _client(monkeypatch, observability=True)
    await server_client.post(server_name="judge_model", url_path=path)
    assert CALLER_SERVER_HEADER not in _sent_headers(request_mock)
