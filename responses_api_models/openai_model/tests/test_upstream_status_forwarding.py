# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A model server answers an upstream rejected credential or rate limit with that status, never a bare 500."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient

from nemo_gym.client import NeMoGymAsyncOpenAI, ServerClient
from nemo_gym.core import NeMoGymResponse
from nemo_gym.server_runtime import BaseVerifyRequest
from nemo_gym.verification import call_judge, judge_failsafe
from responses_api_models.openai_model.app import SimpleModelServer, SimpleModelServerConfig


def _upstream_error(status: int, body: bytes) -> ClientResponseError:
    request_info = SimpleNamespace(url="https://api.example.com/v1/chat/completions", method="POST", headers={})
    error = ClientResponseError(request_info, (), status=status, message="upstream")
    error.response_content = body
    return error


def _model_server(upstream_status: int, body: bytes) -> SimpleModelServer:
    config = SimpleModelServerConfig(
        host="0.0.0.0",
        port=8081,
        openai_base_url="https://api.example.com/v1",
        openai_api_key="test-key",  # pragma: allowlist secret
        openai_model="test-model",
        entrypoint="",
        name="judge_model_server",
    )
    server = SimpleModelServer(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    server._client = MagicMock(spec=NeMoGymAsyncOpenAI)
    error = _upstream_error(upstream_status, body)
    server._client.create_response = AsyncMock(side_effect=error)
    server._client.create_chat_completion = AsyncMock(side_effect=error)
    return server


def _route_reply(server: SimpleModelServer):
    app = server.setup_webserver()
    server.setup_exception_middleware(app)
    return TestClient(app, raise_server_exceptions=False).post("/v1/responses", json={"input": "hi"})


@pytest.mark.parametrize("status", [401, 403, 429])
def test_the_responses_route_forwards_the_upstream_status_and_body(status) -> None:
    reply = _route_reply(_model_server(status, b'{"error": "rejected"}'))

    assert reply.status_code == status
    assert "rejected" in reply.text


@pytest.mark.parametrize("status", [400, 500, 503])
def test_other_upstream_statuses_stay_a_500_of_the_route(status) -> None:
    assert _route_reply(_model_server(status, b"{}")).status_code == 500


class _JudgeRequest(BaseVerifyRequest):
    pass


async def test_a_judge_behind_the_route_that_the_provider_rejects_names_the_401_in_its_row() -> None:
    reply = _route_reply(_model_server(401, b'{"error": "invalid api key"}'))

    request_info = SimpleNamespace(url="http://judge/v1/responses", method="POST", headers={}, real_url="http://judge")
    judge_reply = MagicMock(status=reply.status_code, ok=False, cookies={})
    judge_reply.content.read = AsyncMock(return_value=reply.content)
    judge_reply.raise_for_status = MagicMock(
        side_effect=ClientResponseError(
            request_info, (), status=reply.status_code, message="Unauthorized", headers=reply.headers
        )
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.post = AsyncMock(return_value=judge_reply)

    @judge_failsafe
    async def verify(body: _JudgeRequest):
        await call_judge(
            server_client,
            server_name="judge",
            url_path="/v1/responses",
            json={},
            response_model=NeMoGymResponse,
        )

    body = _JudgeRequest(
        responses_create_params={"input": "q"},
        response={
            "id": "r",
            "created_at": 0.0,
            "model": "m",
            "object": "response",
            "output": [],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        },
    )
    row = (await verify(body=body)).body

    import json

    data = json.loads(row)
    assert data["failure_kind"] == "judge_failed" and data["mask_sample"] is True
    assert "401" in data["failure_reason"] and "invalid api key" in data["failure_reason"]
    assert "500" not in data["failure_reason"]
    # A rejected credential is not retried: one call.
    assert server_client.post.await_count == 1
