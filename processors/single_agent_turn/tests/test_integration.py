# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end proof that the episode processor owns the runtime.

Mocks stop at the sandbox provider. The environment below is a real
``SimpleResourcesServer`` served by a real FastAPI app, so ``/sandbox_spec``,
``/seed_session`` and ``/verify`` go over HTTP through the real request models. What this
demonstrates is the claim the processor design rests on: the environment *declares* the
runtime, the processor *owns* it, and the harness and the verifier are handed the same box
without either of them provisioning anything.
"""

import json
from typing import Any, Optional
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseRunRequest,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.processors.single_agent_turn import (
    SingleAgentTurnProcessor,
    SingleAgentTurnProcessorConfig,
    SingleAgentTurnRunRequest,
)
from nemo_gym.server_utils import ServerClient


MODEL_RESPONSE = {
    "id": "response",
    "created_at": 1,
    "model": "model",
    "object": "response",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}


class SandboxedEnvironment(SimpleResourcesServer):
    """An environment whose task needs a runtime and whose verifier scores its final state."""

    # What each hook actually received, so the test can assert on the real round trip.
    seen: dict[str, Any] = {}

    async def sandbox_spec(self, body: BaseRunRequest) -> dict[str, Any]:
        # The spec is a property of the task. This environment says nothing about *where* it runs.
        self.seen["sandbox_spec_called"] = True
        return {"image": "python:3.13-slim"}

    async def seed_session(self, body: BaseSeedSessionRequest) -> BaseSeedSessionResponse:
        # Note what is *absent*: no provider lookup, no AsyncSandbox, no per-session dict to
        # hold a box in, and nothing to tear down in verify().
        self.seen["seed_sandbox"] = body.episode_context.sandbox if body.episode_context else None
        return BaseSeedSessionResponse()

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        self.seen["verify_sandbox"] = body.episode_context.sandbox if body.episode_context else None
        return BaseVerifyResponse(**body.model_dump(), reward=1.0)


class PlainEnvironment(SimpleResourcesServer):
    """An environment that declares no runtime, which is the common case."""

    seen: dict[str, Any] = {}

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        self.seen["verify_sandbox"] = body.episode_context.sandbox if body.episode_context else None
        return BaseVerifyResponse(**body.model_dump(), reward=0.5)


class _HarnessStub:
    """Stands in for the agent server: records the params it was handed, returns a response."""

    def __init__(self) -> None:
        self.episode_context: Optional[dict[str, Any]] = None

    def post(self, payload: Any) -> dict[str, Any]:
        params = (
            payload.model_dump(mode="json")
            if isinstance(payload, NeMoGymResponseCreateParamsNonStreaming)
            else payload
        )
        self.episode_context = params.get("episode_context")
        return MODEL_RESPONSE


class _FakeConnectableSandbox:
    """A provider that can hand a live box to another process, which docker and local cannot."""

    started: list[str] = []
    stopped: list[str] = []

    def __init__(self, provider: Any, spec: Any) -> None:
        self.spec = spec

    async def start(self) -> "_FakeConnectableSandbox":
        type(self).started.append(self.spec.image)
        return self

    async def serialize(self) -> dict[str, Any]:
        return {"sandbox_id": "box-1", "provider": "fake", "image": self.spec.image}

    async def stop(self) -> None:
        type(self).stopped.append("box-1")


def _wire(environment: SimpleResourcesServer, harness: _HarnessStub) -> SingleAgentTurnProcessor:
    """Route the processor's server calls at a real environment app and the harness stub."""
    environment_client = TestClient(environment.setup_webserver())

    processor = SingleAgentTurnProcessor(
        config=SingleAgentTurnProcessorConfig(
            host="127.0.0.1",
            port=12345,
            entrypoint="app.py",
            name="policy__processor",
            agent_server=AgentServerRef(type="responses_api_agents", name="policy"),
            resources_server=ResourcesServerRef(type="resources_servers", name="environment"),
            sandbox_provider={"docker": {}},
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    processor.server_client.global_config_dict = {"observability_enabled": False}

    async def _post(*, server_name: str, url_path: str, json: Any = None, cookies: Any = None, **_: Any) -> MagicMock:
        if server_name == "policy":
            return _as_response(harness.post(json), status=200)
        payload = json.model_dump(mode="python") if hasattr(json, "model_dump") else json
        http = environment_client.post(url_path, json=_jsonable(payload), cookies=cookies or {})
        return _as_response(http.json() if http.content else {}, status=http.status_code)

    processor.server_client.post = _post
    return processor


def _jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload, default=str))


def _as_response(payload: Any, *, status: int) -> MagicMock:
    response = MagicMock(status=status, ok=status < 400, cookies={})
    encoded = json.dumps(payload).encode()

    async def _read() -> bytes:
        return encoded

    response.read = _read
    response.content.read = _read
    return response


@pytest.mark.asyncio
async def test_environment_declares_the_runtime_and_the_processor_owns_it() -> None:
    _FakeConnectableSandbox.started = []
    _FakeConnectableSandbox.stopped = []
    environment = SandboxedEnvironment(
        config=BaseResourcesServerConfig(host="127.0.0.1", port=1, entrypoint="app.py", name="environment"),
        server_client=MagicMock(spec=ServerClient),
    )
    environment.seen = {}
    harness = _HarnessStub()
    processor = _wire(environment, harness)

    with patch("nemo_gym.processors.single_agent_turn.AsyncSandbox", _FakeConnectableSandbox):
        result = await processor.run(
            MagicMock(cookies={}),
            SingleAgentTurnRunRequest(responses_create_params={"input": "question"}),
        )

    assert result.reward == 1.0

    # The processor provisioned from the environment's spec, and tore the box down afterwards.
    assert _FakeConnectableSandbox.started == ["python:3.13-slim"]
    assert _FakeConnectableSandbox.stopped == ["box-1"]

    # Both consumers were handed the same box, and neither had to create one.
    descriptor = {"sandbox_id": "box-1", "provider": "fake", "image": "python:3.13-slim"}
    assert environment.seen["seed_sandbox"] == descriptor
    assert environment.seen["verify_sandbox"] == descriptor
    assert harness.episode_context["sandbox"] == descriptor

    # And the harness was told where the environment is, so it needs none in its own config.
    assert harness.episode_context["env"] == {"type": "resources_servers", "name": "environment"}


@pytest.mark.asyncio
async def test_environment_without_a_runtime_starts_no_sandbox() -> None:
    _FakeConnectableSandbox.started = []
    environment = PlainEnvironment(
        config=BaseResourcesServerConfig(host="127.0.0.1", port=1, entrypoint="app.py", name="environment"),
        server_client=MagicMock(spec=ServerClient),
    )
    environment.seen = {}
    harness = _HarnessStub()
    processor = _wire(environment, harness)

    with patch("nemo_gym.processors.single_agent_turn.AsyncSandbox", _FakeConnectableSandbox):
        result = await processor.run(
            MagicMock(cookies={}),
            SingleAgentTurnRunRequest(responses_create_params={"input": "question"}),
        )

    assert result.reward == 0.5
    # The default `/sandbox_spec` answers 204, so nothing is provisioned for the 103 of 118
    # benchmark instances that need no runtime.
    assert _FakeConnectableSandbox.started == []
    assert environment.seen["verify_sandbox"] is None
    assert harness.episode_context["sandbox"] is None
