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
"""Execution identity must reach every Gym server on every call.

Checkpointing correlates custody records, session state, and token rows by
``(logical_rollout_id, attempt_index)``. These tests pin the propagation
contract: the identity travels as headers on every downstream call regardless
of observability or token capture, servers rebuild the request context from
those headers, and an explicit attempt index survives logical ids that
themselves end in ``-a{n}``.
"""

from urllib.parse import urlsplit

import orjson
import pytest
from fastapi import FastAPI
from omegaconf import OmegaConf
from pydantic import ConfigDict
from starlette.testclient import TestClient

import nemo_gym.server_utils
from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config_types import BaseServerConfig
from nemo_gym.rollout_correlation import (
    ATTEMPT_INDEX_HEADER,
    ROLLOUT_ID_HEADER,
    RolloutContextMiddleware,
    current_attempt_index,
    current_execution_identity,
    current_rollout_id,
    execution_identity_from_run_body,
    rollout_context,
    split_transport_rollout_id,
)
from nemo_gym.server_utils import ServerClient


# --- transport id splitting ---


@pytest.mark.parametrize(
    ("transport_id", "expected"),
    [
        (None, (None, 0)),
        ("4-2", ("4-2", 0)),
        ("4-2-a3", ("4-2", 3)),
        # Only the rightmost suffix is the attempt marker.
        ("7-1-a2-a3", ("7-1-a2", 3)),
        # ``-a`` followed by non-digits is part of the logical id.
        ("run-apple", ("run-apple", 0)),
        ("run-a", ("run-a", 0)),
    ],
)
def test_split_transport_rollout_id(transport_id, expected) -> None:
    assert split_transport_rollout_id(transport_id) == expected


# --- context helpers ---


def test_no_context_yields_no_identity() -> None:
    assert current_rollout_id() is None
    assert current_attempt_index() is None
    assert current_execution_identity() == (None, None)


def test_attempt_index_derived_from_transport_suffix() -> None:
    with rollout_context("4-2-a3"):
        assert current_rollout_id() == "4-2-a3"
        assert current_attempt_index() == 3
        assert current_execution_identity() == ("4-2", 3)


def test_explicit_attempt_index_wins_over_suffix() -> None:
    with rollout_context("run-a1", attempt_index=0, logical_rollout_id="run-a1"):
        assert current_attempt_index() == 0
        assert current_execution_identity() == ("run-a1", 0)


def test_unsuffixed_transport_id_defaults_to_attempt_zero() -> None:
    with rollout_context("4-2"):
        assert current_attempt_index() == 0
        assert current_execution_identity() == ("4-2", 0)


def test_context_resets_both_variables() -> None:
    with rollout_context("4-2", attempt_index=1):
        with rollout_context("9-9-a2"):
            assert current_execution_identity() == ("9-9", 2)
        assert current_execution_identity() == ("4-2", 1)
    assert current_execution_identity() == (None, None)


# --- run-body identity (unconditional, unlike capture-key derivation) ---


def test_execution_identity_from_run_body() -> None:
    assert execution_identity_from_run_body(None) == (None, None)
    assert execution_identity_from_run_body({"_ng_task_index": 4, "_ng_rollout_index": 2}) == ("4-2", 0)
    assert execution_identity_from_run_body({"_ng_task_index": 4, "_ng_rollout_index": 2, "_ng_attempt_index": 3}) == (
        "4-2",
        3,
    )
    assert execution_identity_from_run_body({"_ng_rollout_id": "run-a1", "_ng_attempt_index": 0}) == ("run-a1", 0)


# --- middleware: headers and path prefix rebuild the context ---


def _identity_app() -> TestClient:
    app = FastAPI()

    @app.get("/echo")
    async def echo() -> dict:
        logical, attempt = current_execution_identity()
        return {
            "rollout_id": current_rollout_id(),
            "logical_id": logical,
            "attempt_index": attempt,
        }

    app.add_middleware(RolloutContextMiddleware)
    return TestClient(app)


def test_middleware_without_identity_leaves_context_empty() -> None:
    body = _identity_app().get("/echo").json()
    assert body == {"rollout_id": None, "logical_id": None, "attempt_index": None}


def test_middleware_reads_identity_headers_without_path_prefix() -> None:
    body = (
        _identity_app()
        .get(
            "/echo",
            headers={ROLLOUT_ID_HEADER: "4-2", ATTEMPT_INDEX_HEADER: "1"},
        )
        .json()
    )
    assert body == {"rollout_id": "4-2-a1", "logical_id": "4-2", "attempt_index": 1}


def test_middleware_attempt_header_is_authoritative() -> None:
    body = _identity_app().get("/echo", headers={ROLLOUT_ID_HEADER: "run-a1", ATTEMPT_INDEX_HEADER: "0"}).json()
    assert body == {"rollout_id": "run-a1", "logical_id": "run-a1", "attempt_index": 0}


def test_middleware_path_prefix_still_sets_context() -> None:
    body = _identity_app().get("/ng-rollout/4-2-a3/echo").json()
    assert body == {"rollout_id": "4-2-a3", "logical_id": "4-2", "attempt_index": 3}


def test_middleware_attempt_header_overrides_path_suffix() -> None:
    body = (
        _identity_app()
        .get(
            "/ng-rollout/run-a1/echo",
            headers={ROLLOUT_ID_HEADER: "run-a1", ATTEMPT_INDEX_HEADER: "0"},
        )
        .json()
    )
    assert body == {"rollout_id": "run-a1", "logical_id": "run-a1", "attempt_index": 0}


def test_middleware_rejects_path_and_header_disagreement() -> None:
    response = _identity_app().get(
        "/ng-rollout/4-2-a3/echo",
        headers={ROLLOUT_ID_HEADER: "4-2", ATTEMPT_INDEX_HEADER: "2"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "execution_identity_mismatch"


def test_middleware_rejects_partial_identity_headers() -> None:
    response = _identity_app().get("/echo", headers={ROLLOUT_ID_HEADER: "4-2"})
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "execution_identity_mismatch"


def test_middleware_rejects_malformed_identity_headers() -> None:
    response = _identity_app().get(
        "/echo",
        headers={ROLLOUT_ID_HEADER: ".bad/id", ATTEMPT_INDEX_HEADER: "soon"},
    )
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "execution_identity_mismatch"


# --- ServerClient: headers attach on every downstream call ---


def _server_client() -> ServerClient:
    config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"x": {"host": "resources.test", "port": 80}}},
            "policy": {"responses_api_models": {"model": {"host": "policy.test", "port": 80}}},
        }
    )
    return ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=config,
    )


class _CapturingDispatch:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def __call__(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})

        class _Response:
            status = 200
            ok = True
            cookies: dict = {}

            async def read(self) -> bytes:
                return b"{}"

        return _Response()


@pytest.mark.asyncio
async def test_server_client_attaches_identity_headers_without_observability(monkeypatch) -> None:
    dispatch = _CapturingDispatch()
    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)

    client = _server_client()
    with rollout_context("4-2-a3"):
        await client.post(server_name="resources", url_path="/tool", json={})

    (call,) = dispatch.calls
    # Observability is off: no capture prefix, but identity headers still attach.
    assert urlsplit(call["url"]).path == "/tool"
    assert call["headers"][ROLLOUT_ID_HEADER] == "4-2"
    assert call["headers"][ATTEMPT_INDEX_HEADER] == "3"


@pytest.mark.asyncio
async def test_server_client_sends_explicit_attempt_index(monkeypatch) -> None:
    dispatch = _CapturingDispatch()
    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)

    client = _server_client()
    with rollout_context("run-a1", attempt_index=0, logical_rollout_id="run-a1"):
        await client.post(server_name="policy", url_path="/v1/responses", json={})

    (call,) = dispatch.calls
    assert call["headers"][ROLLOUT_ID_HEADER] == "run-a1"
    assert call["headers"][ATTEMPT_INDEX_HEADER] == "0"


@pytest.mark.asyncio
async def test_server_client_without_context_sends_no_identity_headers(monkeypatch) -> None:
    dispatch = _CapturingDispatch()
    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)

    client = _server_client()
    await client.post(server_name="resources", url_path="/tool", json={})

    (call,) = dispatch.calls
    headers = call.get("headers") or {}
    assert ROLLOUT_ID_HEADER not in headers
    assert ATTEMPT_INDEX_HEADER not in headers


@pytest.mark.asyncio
async def test_server_client_preserves_caller_supplied_headers(monkeypatch) -> None:
    dispatch = _CapturingDispatch()
    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)

    client = _server_client()
    with rollout_context("4-2"):
        with pytest.raises(ValueError, match="disagrees"):
            await client.post(
                server_name="resources",
                url_path="/tool",
                json={},
                headers={"x-custom": "kept", ROLLOUT_ID_HEADER: "caller-pinned"},
            )
    assert dispatch.calls == []


# --- end to end: agent run establishes identity with capture disabled ---


class _ProbeRunRequest(BaseRunRequest):
    model_config = ConfigDict(extra="allow")


class _ProbeAgent(SimpleResponsesAPIAgent):
    async def responses(self, body):
        raise NotImplementedError

    async def run(self, body: _ProbeRunRequest) -> dict:
        downstream = await self.server_client.post(
            server_name="resources",
            url_path="/probe",
            json={},
        )
        return orjson.loads(await downstream.read())


class _Response:
    def __init__(self, response) -> None:
        self.status = response.status_code
        self.ok = response.is_success
        self.cookies = response.cookies
        self._content = response.content

    async def read(self) -> bytes:
        return self._content


@pytest.mark.asyncio
async def test_agent_run_propagates_identity_with_capture_disabled(monkeypatch) -> None:
    config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"x": {"host": "resources.test", "port": 80}}},
            "agent": {"responses_api_agents": {"agent": {"host": "agent.test", "port": 80}}},
        }
    )
    server_client = ServerClient(
        head_server_config=BaseServerConfig(host="head.test", port=80),
        global_config_dict=config,
    )
    agent = _ProbeAgent(
        config=BaseResponsesAPIAgentConfig(
            host="agent.test",
            port=80,
            entrypoint="app.py",
            name="agent",
        ),
        server_client=server_client,
    )

    resources_app = FastAPI()

    @resources_app.post("/probe")
    async def probe() -> dict:
        logical, attempt = current_execution_identity()
        return {"rollout_id": current_rollout_id(), "logical_id": logical, "attempt_index": attempt}

    resources_app.add_middleware(RolloutContextMiddleware)

    clients = {
        "resources.test": TestClient(resources_app),
        "agent.test": TestClient(agent.setup_webserver()),
    }

    async def dispatch(method: str, url: str, **kwargs):
        parsed = urlsplit(url)
        response = clients[parsed.hostname].request(
            method, parsed.path, json=kwargs.get("json"), headers=kwargs.get("headers")
        )
        return _Response(response)

    monkeypatch.setattr(nemo_gym.server_utils, "request", dispatch)

    run = await server_client.post(
        server_name="agent",
        url_path="/run",
        json={
            "_ng_task_index": 4,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 3,
            "responses_create_params": {"input": "solve"},
        },
    )
    body = orjson.loads(await run.read())

    # No observability, no token capture: the resources server still learned
    # the full execution identity purely from the propagated headers.
    assert body == {"rollout_id": "4-2-a3", "logical_id": "4-2", "attempt_index": 3}
