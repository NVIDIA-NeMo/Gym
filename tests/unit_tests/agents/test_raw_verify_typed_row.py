# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""An agent that lets its resources server's /verify failure escape /run still gets a typed row from Collection."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponseError
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.agents.base_responses_api_agent import BaseResponsesAPIAgentConfig, Body, SimpleResponsesAPIAgent
from nemo_gym.client.http_client import get_response_json, raise_for_status
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.failure_transport import FailureHeader
from nemo_gym.core.typed_exceptions import NgFailureEvent, failure_report_for
from nemo_gym.rollouts.rollout_collection import _agent_request_failure_row, _failure_outcome
from nemo_gym.server_runtime.base_resources_server import BaseRunRequest
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentConfig


class _RawVerifyAgent(SimpleResponsesAPIAgent):
    """The shape of the agents that call /verify with a bare raise_for_status."""

    async def responses(self, body=Body()):
        raise NotImplementedError

    async def run(self, request: Request, body: BaseRunRequest):
        verified = await self.server_client.post(server_name="resources", url_path="/verify", json={}, cookies={})
        await raise_for_status(verified)
        return await get_response_json(verified)


def _failed_reply(failure: NgFailureEvent) -> MagicMock:
    """A 500 reply from the resources server whose failure report travels in its header, as on the wire."""
    request_info = SimpleNamespace(url="http://server", method="POST", headers={}, real_url="http://server")
    error = ClientResponseError(
        request_info,
        (),
        status=500,
        message="Internal Server Error",
        headers=FailureHeader.encode(failure_report_for(failure)),
    )
    reply = MagicMock(status=500, ok=False, cookies={})
    reply.content.read = AsyncMock(return_value=b'{"detail":"Internal Server Error"}')
    reply.raise_for_status = MagicMock(side_effect=error)
    return reply


def _resources_failure(kind: str, operation: str = "verify") -> NgFailureEvent:
    failure = NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind=kind,
        failure_reason=f"{operation} failed",
        operation=operation,
        stage="verification",
        http_status=500,
    )
    failure.__cause__ = RuntimeError("grader down")
    return failure


def _run_reply(agent: SimpleResponsesAPIAgent):
    app = agent.setup_webserver()
    agent.setup_exception_middleware(app)
    body = {"responses_create_params": {"input": "hi"}}
    return TestClient(app, raise_server_exceptions=False).post("/run", json=body)


def _saved_row(reply):
    """The sidecar row and typed record Collection saves for a failed /run reply."""
    error = ClientResponseError(
        SimpleNamespace(url="http://agent/run", method="POST", headers={}, real_url="http://agent/run"),
        (),
        status=reply.status_code,
        message="Internal Server Error",
        headers=reply.headers,
    )
    row = _agent_request_failure_row(error, reply.status_code, reply.headers)
    outcome = _failure_outcome({"_ng_run_id": "r", "_ng_task_index": 0, "_ng_rollout_index": 0}, row, "agent")
    return row, outcome


@pytest.mark.parametrize("kind", ["verifier_error", "judge_failed"])
def test_a_raw_verify_failure_in_any_agent_gives_a_typed_row(kind) -> None:
    agent = _RawVerifyAgent(
        config=BaseResponsesAPIAgentConfig(host="", port=0, entrypoint="", name="agent"),
        server_client=MagicMock(spec=ServerClient),
    )
    agent.server_client.post = AsyncMock(return_value=_failed_reply(_resources_failure(kind)))

    reply = _run_reply(agent)
    row, outcome = _saved_row(reply)

    assert reply.status_code == 500
    assert row["_ng_failure_kind"] == kind
    assert outcome.failure.failure_kind == outcome.sidecar_failure_class == kind


def test_simple_agents_own_raw_verify_failure_gives_a_typed_row() -> None:
    config = SimpleAgentConfig(
        host="",
        port=0,
        entrypoint="",
        name="simple",
        model_server=ModelServerRef(type="responses_api_models", name="model"),
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
    )
    agent = SimpleAgent(config=config, server_client=MagicMock(spec=ServerClient))
    model_json = {
        "id": "r",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }

    def ok(payload):
        reply = MagicMock(status=200, ok=True, cookies={})
        reply.read = AsyncMock(return_value=json.dumps(payload))
        return reply

    async def post(*, server_name, url_path, **kwargs):
        if url_path == "/verify":
            return _failed_reply(_resources_failure("verifier_error"))
        return ok(model_json if url_path.endswith("/v1/responses") else {})

    agent.server_client.post = AsyncMock(side_effect=post)

    reply = _run_reply(agent)
    row, outcome = _saved_row(reply)

    assert reply.status_code == 500
    assert row["_ng_failure_kind"] == "verifier_error"
    assert outcome.sidecar_failure_class == "verifier_error"
