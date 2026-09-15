# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, ValidationError

from nemo_gym.base_episode_processor import (
    BaseEpisodeProcessor,
    BaseEpisodeProcessorConfig,
    EpisodeContext,
)
from nemo_gym.base_resources_server import BaseSeedSessionResponse, BaseVerifyResponse, MCPServerMetadata
from nemo_gym.base_responses_api_agent import SimpleResponsesAPIAgent
from nemo_gym.episode import (
    AgentCloseSessionRequest,
    AgentSeedSessionRequest,
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    DirectResourcesToolAccess,
    EpisodeFailure,
    EpisodeId,
    MaterializedTask,
    ResourcesToolAccess,
    TaskId,
)
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


class _TestTaskInput(BaseModel):
    value: str


class _TestRequest(BaseEpisodeRequest[_TestTaskInput]):
    pass


class _TestResponse(BaseEpisodeResponse[str]):
    pass


class _TestProcessor(BaseEpisodeProcessor[_TestRequest, _TestResponse]):
    request_model = _TestRequest
    response_model = _TestResponse

    async def process(self, request: _TestRequest, context: EpisodeContext) -> _TestResponse:
        return _TestResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=request.task.task_input.value,
        )


def _request() -> _TestRequest:
    return _TestRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=1),
        task=MaterializedTask(
            task_id=TaskId(taskset="test", task_id="task"),
            task_input=_TestTaskInput(value="result"),
        ),
    )


def _processor(**config_overrides: Any) -> _TestProcessor:
    config = BaseEpisodeProcessorConfig(
        name="processor",
        host="127.0.0.1",
        port=1234,
        entrypoint="app.py",
        default_episode_timeout_seconds=1,
        cleanup_timeout_seconds=0.01,
        **config_overrides,
    )
    return _TestProcessor(config=config, server_client=MagicMock(spec=ServerClient))


def test_episode_response_requires_exactly_one_outcome() -> None:
    request = _request()
    with pytest.raises(ValidationError, match="exactly one"):
        _TestResponse(episode_id=request.episode_id, task_id=request.task.task_id)
    with pytest.raises(ValidationError, match="exactly one"):
        _TestResponse(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result="result",
            failure=EpisodeFailure(message="failure", terminal=True),
        )


def test_resources_access_can_expose_http_and_mcp() -> None:
    access = ResourcesToolAccess(
        direct_http=DirectResourcesToolAccess(
            base_url="http://resources:8000",
            cookies={"session": "value"},
        ),
        mcp=MCPServerMetadata(
            server_name="resources",
            headers={"Authorization": "Bearer scoped"},
        ),
    )

    assert access.direct_http is not None
    assert access.mcp is not None


def test_empty_seed_response_preserves_legacy_wire_shape() -> None:
    assert BaseSeedSessionResponse().model_dump() == {}
    with pytest.raises(ValidationError):
        DirectResourcesToolAccess(base_url="http://resources:8000", unknown=True)
    with pytest.raises(ValidationError, match="at least one"):
        ResourcesToolAccess()


def test_verify_response_preserves_benchmark_fields() -> None:
    response = BaseVerifyResponse.model_validate(
        {
            "responses_create_params": {"input": "hi"},
            "response": NeMoGymResponse.model_construct(id="response", output=[]),
            "reward": 0.5,
            "benchmark_diagnostic": "kept",
        }
    )
    assert response.model_dump()["benchmark_diagnostic"] == "kept"


def test_processor_binds_concrete_schema_and_rejects_invalid_input() -> None:
    app = _processor().setup_webserver()
    schema = app.openapi()["paths"]["/run"]["post"]["requestBody"]["content"]["application/json"]["schema"]
    assert schema["$ref"].endswith("/_TestRequest")
    assert (
        TestClient(app).post("/run", json={"task": {"task_input": {"value": "missing identity"}}}).status_code == 422
    )


def test_processor_runs_typed_request() -> None:
    response = asyncio.run(_processor().run(_request()))
    assert response.result == "result"


def test_cleanup_is_lifo() -> None:
    calls: list[str] = []
    context = EpisodeContext(
        request=_request(),
        server_client=MagicMock(spec=ServerClient),
        cleanup_timeout_seconds=1,
    )

    async def first() -> None:
        calls.append("first")

    async def second() -> None:
        calls.append("second")

    context.register_cleanup("first", first)
    context.register_cleanup("second", second)
    asyncio.run(context.aclose())
    assert calls == ["second", "first"]


def test_cleanup_is_bounded() -> None:
    calls: list[str] = []
    context = EpisodeContext(
        request=_request(),
        server_client=MagicMock(spec=ServerClient),
        cleanup_timeout_seconds=0.01,
    )

    async def first() -> None:
        calls.append("first")

    async def hung() -> None:
        calls.append("hung")
        await asyncio.sleep(60)

    context.register_cleanup("first", first)
    context.register_cleanup("hung", hung)
    asyncio.run(context.aclose())
    assert calls == ["hung"]


def test_explicit_cleanup_is_bounded_and_final_unwind_retries() -> None:
    attempts = 0
    context = EpisodeContext(
        request=_request(),
        server_client=MagicMock(spec=ServerClient),
        cleanup_timeout_seconds=0.01,
    )

    async def cleanup() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            await asyncio.sleep(60)

    handle = context.register_cleanup("cleanup", cleanup)

    async def run() -> None:
        with pytest.raises(TimeoutError):
            await handle.close()
        await context.aclose()

    asyncio.run(run())
    assert attempts == 2


def test_caller_cancellation_waits_for_cleanup() -> None:
    cleanup_finished = asyncio.Event()

    class _CancelledProcessor(_TestProcessor):
        async def process(self, request: _TestRequest, context: EpisodeContext) -> _TestResponse:
            async def cleanup() -> None:
                await asyncio.sleep(0.01)
                cleanup_finished.set()

            context.register_cleanup("cleanup", cleanup)
            await asyncio.sleep(60)
            raise AssertionError

    async def run() -> None:
        config = _processor().config.model_copy(update={"cleanup_timeout_seconds": 1})
        processor = _CancelledProcessor(config=config, server_client=MagicMock(spec=ServerClient))
        task = asyncio.create_task(processor.run(_request()))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(run())
    assert cleanup_finished.is_set()


def test_capture_key_qualifies_retries() -> None:
    assert EpisodeId(rollout_id="r").capture_key == "r"
    assert EpisodeId(rollout_id="r", attempt=2).capture_key == "r-a2"
    with pytest.raises(ValidationError, match="reserved attempt suffix"):
        EpisodeId(rollout_id="r-a2")


def test_base_agent_exposes_unimplemented_session_routes() -> None:
    class _Sessions(SimpleResponsesAPIAgent):
        async def responses(self, body):
            raise NotImplementedError

        async def run(self, body):
            raise NotImplementedError

    sessions = _Sessions.model_construct(config=MagicMock(num_workers=1), server_client=MagicMock())
    request = MagicMock()
    with pytest.raises(NotImplementedError, match="does not implement"):
        asyncio.run(
            sessions.seed_agent_session(
                request,
                AgentSeedSessionRequest(
                    episode_id=EpisodeId(rollout_id="rollout", attempt=2),
                    task_id=TaskId(taskset="test", task_id="task"),
                ),
            )
        )
    with pytest.raises(NotImplementedError, match="does not implement"):
        asyncio.run(
            sessions.close_agent_session(
                request,
                AgentCloseSessionRequest(agent_session_id="agent-session"),
            )
        )

    paths = sessions.setup_webserver().openapi()["paths"]
    assert "/v1/agent_sessions" in paths
    assert "/v1/agent_sessions/close" in paths
