# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP wrapping must preserve a seed handler's validation and request injection."""

import inspect
from typing import Annotated, Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from pydantic import BeforeValidator, Field

from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseSeedSessionResponse,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedSessionRequest,
    ResourcesSeedSessionResponse,
    SimpleResourcesServer,
)
from nemo_gym.server_runtime.mcp_auto_exposure import TOKEN_HEADER, _wrap_seed_session, maybe_auto_expose


def _cookie_row(value: object) -> object:
    if isinstance(value, dict) and ("resources_session_id" in value or "episode_id" in value):
        raise ValueError("An episode seed must match the typed request")
    return value


class ValidatedSeedServer(SimpleResourcesServer):
    seeded: list[str] = Field(default_factory=list)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        raise AssertionError("Verification is outside this seed-route test")

    async def seed_session(
        self,
        request: Annotated[Request, "caller"],
        body: ResourcesSeedSessionRequest | Annotated[BaseSeedSessionRequest, BeforeValidator(_cookie_row)],
    ) -> ResourcesSeedSessionResponse | BaseSeedSessionResponse:
        if isinstance(body, ResourcesSeedSessionRequest):
            request.session[SESSION_ID_KEY] = body.resources_session_id
            self.seeded.append(body.resources_session_id)
            return ResourcesSeedSessionResponse(resources_session_id=body.resources_session_id)
        self.seeded.append(request.session[SESSION_ID_KEY])
        return BaseSeedSessionResponse()


@pytest.mark.parametrize(
    "body",
    [
        {"resources_session_id": "session"},
        {"resources_session_id": "session", "episode_id": {"rollout_id": "r"}},
        {"episode_id": {"rollout_id": "r"}, "task_id": {"taskset": "t", "task_id": "1"}},
    ],
)
def test_mcp_seed_preserves_cookie_arm_validator(body: dict) -> None:
    server = ValidatedSeedServer(
        config=BaseResourcesServerConfig(
            name="validated", host="127.0.0.1", port=0, entrypoint="app.py", expose_tools_over_mcp=True
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        rejected = client.post("/seed_session", json=body)
        assert rejected.status_code == 422
        assert server.seeded == []
        assert TOKEN_HEADER not in rejected.text

        # Annotated[Request, ...] must still receive the existing Request, not add a second parameter.
        cookie = client.post("/seed_session", json={"task_id": "a row column", "value": 42})
        assert cookie.status_code == 200, cookie.text
        assert TOKEN_HEADER in cookie.json()["mcp"]["headers"]
        typed = client.post(
            "/seed_session",
            json={
                "resources_session_id": "session",
                "episode_id": {"rollout_id": "r"},
                "task_id": {"taskset": "t", "task_id": "1"},
                "task_data": {},
            },
        )
        assert typed.status_code == 200, typed.text
        assert typed.json()["resources_session_id"] == "session"
        assert TOKEN_HEADER in typed.json()["resources_tools"]["headers"]
        assert len(server.seeded) == 2
        assert server.seeded[-1] == "session"


def _reject_negative(value: object) -> object:
    if isinstance(value, dict) and value.get("value", 0) < 0:
        raise ValueError("value must not be negative")
    return value


class NonNegativeSeed(BaseSeedSessionRequest):
    value: int = 0


class DirectlyAnnotatedBodyServer(ValidatedSeedServer):
    async def seed_session(
        self, request: Request, body: Annotated[NonNegativeSeed, BeforeValidator(_reject_negative)]
    ) -> BaseSeedSessionResponse:
        self.seeded.append(request.session[SESSION_ID_KEY])
        return BaseSeedSessionResponse()


def test_mcp_seed_keeps_a_validator_annotated_directly_on_the_body() -> None:
    server = DirectlyAnnotatedBodyServer(
        config=BaseResourcesServerConfig(
            name="direct", host="127.0.0.1", port=0, entrypoint="app.py", expose_tools_over_mcp=True
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    app = server.setup_webserver()
    maybe_auto_expose(server, app)
    with TestClient(app) as client:
        assert client.post("/seed_session", json={"value": -1}).status_code == 422
        assert server.seeded == []
        ok = client.post("/seed_session", json={"value": 1})
        assert ok.status_code == 200, ok.text
        assert TOKEN_HEADER in ok.json()["mcp"]["headers"]


class CustomRequest(Request):
    pass


@pytest.mark.parametrize("annotated", [False, True])
def test_mcp_seed_reuses_request_subclass_parameter(annotated: bool) -> None:
    app = FastAPI()
    seen = []

    async def seed(body: NonNegativeSeed, request: CustomRequest) -> dict:
        seen.append(request)
        return {"value": body.value, "path": request.url.path}

    if annotated:
        seed.__annotations__["request"] = Annotated[CustomRequest, "caller"]
    app.post("/seed_session")(seed)
    _wrap_seed_session(app, lambda request, body: {"value": body["value"]})
    with TestClient(app) as client:
        result = client.post("/seed_session", json={"value": 3})
    assert result.status_code == 200, result.text
    assert result.json() == {"value": 3, "path": "/seed_session", "mcp": {"value": 3}}
    # FastAPI injects the live Request even when the annotation names a subclass.
    assert len(seen) == 1 and isinstance(seen[0], Request)


def test_mcp_seed_honors_factory_signature_over_annotations() -> None:
    app = FastAPI()
    seen = []

    async def seed(body: Any, request: Request) -> dict:
        seen.append(body)
        return {"value": body.value}

    signature = inspect.signature(seed)
    seed.__signature__ = signature.replace(
        parameters=[
            param.replace(annotation=Annotated[NonNegativeSeed, BeforeValidator(_reject_negative)])
            if name == "body"
            else param
            for name, param in signature.parameters.items()
        ]
    )
    app.post("/seed_session")(seed)
    with TestClient(app) as client:
        assert client.post("/seed_session", json={"value": -1}).status_code == 422
        assert client.post("/seed_session", json={"value": 3}).json() == {"value": 3}
    _wrap_seed_session(app, lambda request, body: {"value": body["value"]})
    with TestClient(app) as client:
        assert client.post("/seed_session", json={"value": -1}).status_code == 422
        result = client.post("/seed_session", json={"value": 3})
    assert result.status_code == 200, result.text
    assert result.json() == {"value": 3, "mcp": {"value": 3}}
    assert len(seen) == 2 and all(isinstance(body, NonNegativeSeed) for body in seen)
