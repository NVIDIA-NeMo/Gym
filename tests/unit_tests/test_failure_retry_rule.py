# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
"""Where a failure happened: in a service the model does not control (infra), or in what the model produced."""

import json
from types import SimpleNamespace

import pytest
from aiohttp import ClientConnectorError, ClientResponseError
from fastapi import FastAPI
from fastapi.testclient import TestClient
from multidict import CIMultiDict
from omegaconf import OmegaConf

from environment_servers.legacy_agent.app import LegacyAgentEnvironmentServer, LegacyAgentEnvironmentServerConfig
from nemo_gym.config_types import AgentServerRef
from nemo_gym.failure_kinds import (
    FAILURE_CAUSE_HEADER,
    INFRA,
    MODEL,
    failure_cause_of,
    failure_cause_of_text,
)
from nemo_gym.rollout_collection import (
    NG_FAILURE_CAUSE_KEY,
    NG_FAILURE_CLASS_KEY,
    _agent_request_failure_row,
)
from nemo_gym.server_utils import BaseServerConfig, ServerClient, SimpleServer


def _http_error(status: int, body: bytes = b"", headers: dict | None = None) -> ClientResponseError:
    request_info = SimpleNamespace(method="POST", url="http://agent/run", real_url="http://agent/run")
    error = ClientResponseError(
        request_info=request_info, history=(), status=status, message="boom", headers=CIMultiDict(headers or {})
    )
    error.response_content = body
    return error


def _refused() -> ClientConnectorError:
    return ClientConnectorError(
        SimpleNamespace(host="model", port=8000, ssl=False), OSError(111, "Connection refused")
    )


class TestWhereTheFailureHappened:
    @pytest.mark.parametrize("status", [401, 403, 408, 429, 500, 502, 503, 504])
    def test_a_service_status_is_infra(self, status: int) -> None:
        assert failure_cause_of(_http_error(status)) == INFRA

    @pytest.mark.parametrize("status", [400, 404, 413, 422])
    def test_a_status_answering_what_the_model_sent_is_the_models(self, status: int) -> None:
        """A 400 for a context overflow answers the model's own request."""
        assert failure_cause_of(_http_error(status)) == MODEL

    def test_a_connection_error_wrapped_in_an_agent_error_is_infra(self) -> None:
        try:
            try:
                raise _refused()
            except ClientConnectorError as inner:
                raise RuntimeError("agent step failed") from inner
        except RuntimeError as error:
            assert failure_cause_of(error) == INFRA

    def test_an_exception_group_is_searched(self) -> None:
        assert failure_cause_of(ExceptionGroup("task group", [ValueError("x"), _refused()])) == INFRA

    def test_a_parse_error_of_the_models_output_is_the_models(self) -> None:
        try:
            json.loads("{not json")
        except json.JSONDecodeError as error:
            assert failure_cause_of(error) == MODEL

    def test_an_agents_own_time_budget_is_the_models(self) -> None:
        assert failure_cause_of(TimeoutError("max steps or time used up")) == MODEL

    def test_text_names_a_dead_peer(self) -> None:
        assert failure_cause_of_text("ClientConnectorError(ConnectionKey(host='model', port=8000))") == INFRA
        assert failure_cause_of_text("Cannot connect to host model:8000 ssl:default") == INFRA
        assert failure_cause_of_text("503, message='Service Unavailable', url='http://judge/v1'") == INFRA
        assert failure_cause_of_text("litellm.RateLimitError: judge over quota") == INFRA

    def test_text_naming_only_a_model_output_error_is_the_models(self) -> None:
        assert failure_cause_of_text("Tau2MalformedToolCallError: tool call 0 arguments is invalid JSON") == MODEL
        assert failure_cause_of_text("400, message='prompt is longer than the context window'") == MODEL


class TestAgentRunErrorBySource:
    """Killing the model server mid-run gives agent_run_error with status 500."""

    def test_a_500_carrying_a_dead_model_server_is_retried(self) -> None:
        error = _http_error(
            500, b"Hit an exception in agent calling an inner server: ClientConnectorError(model:8000 refused)"
        )
        row = _agent_request_failure_row(error, 500)

        assert row[NG_FAILURE_CLASS_KEY] == "agent_run_error"
        assert row[NG_FAILURE_CAUSE_KEY] == INFRA

    def test_a_500_from_a_model_output_parse_error_is_final(self) -> None:
        error = _http_error(500, b"JSONDecodeError('Expecting value: line 1 column 1 (char 0)')")
        row = _agent_request_failure_row(error, 500)

        assert row[NG_FAILURE_CLASS_KEY] == "agent_run_error"
        assert row[NG_FAILURE_CAUSE_KEY] == MODEL

    def test_the_cause_a_gym_server_named_wins_over_its_body(self) -> None:
        error = _http_error(500, b"anything", {FAILURE_CAUSE_HEADER: INFRA})
        assert _agent_request_failure_row(error, 500)[NG_FAILURE_CAUSE_KEY] == INFRA

    def test_no_reply_at_all_is_retried(self) -> None:
        row = _agent_request_failure_row(_refused(), None)

        assert row[NG_FAILURE_CLASS_KEY] == "agent_request_failed"


def _agent(breaks_with: BaseException) -> TestClient:
    """A Gym agent server, with Gym's own exception middleware, whose /run raises ``breaks_with``."""
    app = FastAPI()

    @app.post("/run")
    async def run() -> dict:
        raise breaks_with

    server = SimpleNamespace(
        config=SimpleNamespace(name="agent"), get_session_middleware_key=lambda: "SimpleAgent___agent"
    )
    SimpleServer.setup_exception_middleware(server, app)
    return TestClient(app, raise_server_exceptions=False)


class _Relayed:
    """What legacy_agent's ServerClient hands back: the agent's reply, status and headers unchanged."""

    def __init__(self, response) -> None:
        self.status = response.status_code
        self.ok = response.is_success
        self.headers = CIMultiDict(response.headers.items())
        self._body = response.content

    async def read(self) -> bytes:
        return self._body


class _RelayClient(ServerClient):
    agent: object

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Relayed:
        return _Relayed(self.agent.post("/run", content=kwargs.get("data") or b"{}"))


def _through_legacy_agent(breaks_with: BaseException):
    """POST /run to legacy_agent, which relays to a real Gym agent server that raises ``breaks_with``."""
    client = _RelayClient.model_construct(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create({}),
        agent=_agent(breaks_with),
    )
    relay = LegacyAgentEnvironmentServer(
        config=LegacyAgentEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8002,
            entrypoint="app.py",
            agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        ),
        server_client=client,
    )
    return TestClient(relay.setup_webserver()).post("/run", json={})


def _collector_row(response) -> dict:
    """The sidecar row rollout collection builds from the relayed reply."""
    request_info = SimpleNamespace(method="POST", url="http://environment/run", real_url="http://environment/run")
    error = ClientResponseError(
        request_info=request_info,
        history=(),
        status=response.status_code,
        message="Internal Server Error",
        headers=CIMultiDict(response.headers.items()),
    )
    error.response_content = response.content
    return _agent_request_failure_row(error, response.status_code)


class TestThroughLegacyAgent:
    """Many benchmarks run their agent behind legacy_agent, so the cause must survive the relay."""

    def test_a_dead_model_server_is_retried(self) -> None:
        response = _through_legacy_agent(_refused())

        assert response.status_code == 500
        assert response.headers[FAILURE_CAUSE_HEADER] == INFRA

    def test_a_judge_429_or_500_is_retried(self) -> None:
        for status in (429, 500):
            response = _through_legacy_agent(_http_error(status, b"judge overloaded"))

            assert response.headers[FAILURE_CAUSE_HEADER] == INFRA

    def test_a_malformed_model_tool_call_is_final(self) -> None:
        try:
            json.loads('{"path": "unterminated')
        except json.JSONDecodeError as error:
            response = _through_legacy_agent(error)

        assert response.headers[FAILURE_CAUSE_HEADER] == MODEL
        row = _collector_row(response)
        assert row[NG_FAILURE_CLASS_KEY] == "agent_run_error"
