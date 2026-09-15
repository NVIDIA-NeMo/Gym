# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import orjson
from omegaconf import OmegaConf
from pydantic import ConfigDict

from episode_processors.single_agent.app import (
    SingleAgentEpisodeProcessor,
    SingleAgentEpisodeProcessorConfig,
    _is_retryable_dependency_error,
)
from episode_processors.single_agent_legacy.app import SingleAgentLegacyAdapter
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode import EpisodeId, MaterializedTask, SingleAgentEpisodeRequest, SingleAgentTaskInput, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient


class _Cookie:
    value = "cookie-value"


class _Response:
    ok = True
    cookies = {"session": _Cookie()}

    def __init__(self, body: dict) -> None:
        self.body = orjson.dumps(body)

    async def read(self) -> bytes:
        return self.body


def _agent_response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[],
        tool_choice="auto",
        parallel_tool_calls=True,
        tools=[],
    )


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict]]
    responses: list[_Response]

    async def post(self, server_name: str, url_path: str, **kwargs) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        return self.responses.pop(0)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _processor(*, token_capture: bool = False) -> tuple[SingleAgentEpisodeProcessor, _Client]:
    global_config = OmegaConf.create(
        {
            "resources": {"resources_servers": {"test": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
            "agent": {
                "responses_api_agents": {
                    "test": {
                        "host": "agent",
                        "port": 8001,
                        "entrypoint": "app.py",
                        "token_id_capture": token_capture,
                    }
                }
            },
            "token_id_capture": {"enabled": token_capture},
        }
    )
    response = _agent_response()
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=global_config,
        calls=[],
        responses=[
            _Response({"resources_session_id": "resources-session"}),
            _Response({"agent_session_id": "agent-session"}),
            _Response(response.model_dump(mode="json")),
            _Response(
                {
                    "agent_session_id": "agent-session",
                    "resources_cookies": {"session": "updated-cookie"},
                }
            ),
            _Response(
                {
                    "responses_create_params": {"input": "task"},
                    "response": response.model_dump(mode="json"),
                    "reward": 1.0,
                    "benchmark_field": "preserved",
                }
            ),
            _Response({"resources_session_id": "resources-session"}),
        ],
    )
    config = SingleAgentEpisodeProcessorConfig(
        name="processor",
        host="processor",
        port=8002,
        entrypoint="app.py",
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        agent_server=AgentServerRef(type="responses_api_agents", name="agent"),
        default_episode_timeout_seconds=10,
        cleanup_timeout_seconds=10,
    )
    return SingleAgentEpisodeProcessor(config=config, server_client=client), client


def _request() -> SingleAgentEpisodeRequest:
    return SingleAgentEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(task_source="source", task_id="task"),
            task_input=SingleAgentTaskInput(
                responses_create_params={"input": "task"},
                task_data={"instance_id": "task"},
            ),
        ),
    )


async def test_single_agent_protocol_and_direct_tool_access() -> None:
    processor, client = _processor()
    result = await processor.run(_request())

    assert result.result is not None
    assert result.result.verification.reward == 1.0
    assert result.result.verification.model_dump()["benchmark_field"] == "preserved"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    create_body = client.calls[1][2]["json"]
    assert create_body.task_id == TaskId(task_source="source", task_id="task")
    assert create_body.resources_access.base_url == "http://resources:8000"
    assert create_body.resources_access.cookies == {"session": "cookie-value"}
    assert client.calls[2][2]["cookies"] == {"session": "cookie-value"}
    assert client.calls[3][2]["cookies"] == {"session": "cookie-value"}
    assert client.calls[4][2]["cookies"] == {"session": "updated-cookie"}
    assert client.calls[5][2]["cookies"] == {"session": "updated-cookie"}


async def test_token_capture_keeps_prefixed_twin_route() -> None:
    processor, client = _processor(token_capture=True)
    await processor.run(_request())
    assert client.calls[2][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"


async def test_legacy_compatibility_is_a_separate_processor_deployment() -> None:
    processor, client = _processor()
    adapter = SingleAgentLegacyAdapter(config=processor.config, server_client=client)
    result = await adapter.run_legacy(
        {
            "_ng_task_index": 3,
            "_ng_rollout_index": 2,
            "_ng_attempt_index": 1,
            "instance_id": "task",
            "benchmark_field": "input",
            "responses_create_params": {"input": "task"},
        }
    )

    assert result["reward"] == 1.0
    assert result["benchmark_field"] == "preserved"
    assert result["agent_ref"] == {"name": "agent"}


def test_dependency_failure_messages_are_bounded() -> None:
    processor, _ = _processor()
    error = processor._failure(stage="agent", message="x" * 3000, terminal=False)
    assert len(error.failure.message) == 2000


def test_retry_requires_a_transient_dependency_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid response")) is False
