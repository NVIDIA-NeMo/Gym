# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MCP wrapping must preserve a seed handler's validation and request injection."""

from typing import Annotated
from unittest.mock import MagicMock

import pytest
from fastapi import Request
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
from nemo_gym.server_runtime.mcp_auto_exposure import TOKEN_HEADER, maybe_auto_expose


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
