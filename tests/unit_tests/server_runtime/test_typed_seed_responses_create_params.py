# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A typed seed carries the task's responses_create_params; only a seed model that declares them receives them."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import Request
from pydantic import ConfigDict, JsonValue

from nemo_gym.client.server_client import SESSION_ID_KEY, ServerClient
from nemo_gym.core.episode_types import EpisodeId, TaskId
from nemo_gym.server_runtime.base_resources_server import (
    BaseResourcesServerConfig,
    BaseSeedSessionRequest,
    BaseVerifyRequest,
    BaseVerifyResponse,
    ResourcesSeedRejectedError,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)


PARAMS = {"input": [{"role": "user", "content": "hello"}], "tools": []}


class TaskOnlySeed(BaseSeedSessionRequest):
    """A seed model as every server had one: task fields only, nothing else allowed."""

    model_config = ConfigDict(extra="forbid")

    value: int


class PromptSeed(BaseSeedSessionRequest):
    value: int
    responses_create_params: dict[str, JsonValue] | None = None


class RecordingServer(SimpleResourcesServer):
    seeds: dict[str, BaseSeedSessionRequest] = {}

    def has_session_state(self, session_id: str) -> bool:
        return session_id in self.seeds

    async def initialize_session_state(self, session_id: str, body: BaseSeedSessionRequest) -> None:
        self.seeds[session_id] = body

    async def release_session_state(self, session_id: str) -> None:
        self.seeds.pop(session_id, None)

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        return BaseVerifyResponse(**body.model_dump(), reward=0)


class TaskOnlyServer(RecordingServer):
    seed_request_model = TaskOnlySeed


class PromptServer(RecordingServer):
    seed_request_model = PromptSeed


def _server(cls: type[RecordingServer]) -> RecordingServer:
    return cls(
        config=BaseResourcesServerConfig(host="", port=0, entrypoint="", name="resources"),
        server_client=MagicMock(spec=ServerClient),
        seeds={},
    )


def _seed(**kw: Any) -> ResourcesSeedSessionRequest:
    return ResourcesSeedSessionRequest(
        resources_session_id="session",
        episode_id=EpisodeId(rollout_id="rollout"),
        task_id=TaskId(taskset="tasks", task_id="one"),
        task_data={"value": 42},
        **kw,
    )


def _request() -> Request:
    return Request({"type": "http", "session": {SESSION_ID_KEY: "caller-cookie"}})


async def test_a_seed_model_that_declares_them_receives_the_tasks_params() -> None:
    server = _server(PromptServer)
    await server.seed_session(_seed(responses_create_params=PARAMS), request=_request())
    assert server.seeds["session"] == PromptSeed(value=42, responses_create_params=PARAMS)


async def test_a_seed_without_them_leaves_the_field_unset() -> None:
    server = _server(PromptServer)
    await server.seed_session(_seed(), request=_request())
    assert server.seeds["session"].responses_create_params is None


async def test_other_seed_models_validate_task_data_as_before() -> None:
    # A model that forbids unknown fields would refuse responses_create_params: it never sees them.
    server = _server(TaskOnlyServer)
    await server.seed_session(_seed(responses_create_params=PARAMS), request=_request())
    assert server.seeds["session"] == TaskOnlySeed(value=42)
    with pytest.raises(ResourcesSeedRejectedError):
        await _server(TaskOnlyServer).seed_session(
            _seed(responses_create_params=PARAMS).model_copy(update={"task_data": {}}), request=_request()
        )


def test_a_seed_without_them_keeps_the_wire_shape() -> None:
    # Tool servers' seeds carry no params: the key is left out, so a server whose seed request forbids unknown fields
    # (an older Gym) still accepts them, and a seed without the key validates.
    body = _seed().model_dump(mode="json")
    assert "responses_create_params" not in body
    assert ResourcesSeedSessionRequest.model_validate(body).responses_create_params is None
    assert _seed(responses_create_params=PARAMS).model_dump(mode="json")["responses_create_params"] == PARAMS
