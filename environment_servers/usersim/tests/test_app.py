# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import orjson
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _create_usersim_generator,
    _is_retryable_dependency_error,
)
from nemo_gym.base_environment_server import BaseEnvironmentServer
from nemo_gym.config_types import AgentServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from resources_servers.usersim.episode_contracts import UserSimEpisodeRequest, UserSimTaskInput


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Response:
    ok = True

    def __init__(self, body: dict[str, Any], *, cookie: str | None = None) -> None:
        self.body = orjson.dumps(body)
        self.cookies = {"session": _Cookie(cookie)} if cookie is not None else {}

    async def read(self) -> bytes:
        return self.body


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict[str, Any]]]
    responses: list[_Response]

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _Response:
        self.calls.append((server_name, url_path, kwargs))
        return self.responses.pop(0)

    def _resolve_base_url(self, server_name: str) -> str:
        return f"http://{server_name}:8000"


def _model_response(response_id: str, text: str) -> dict[str, Any]:
    return {
        "id": response_id,
        "created_at": 1,
        "model": "model",
        "object": "response",
        "output": [
            {
                "id": f"{response_id}-message",
                "content": [{"annotations": [], "text": text, "type": "output_text"}],
                "role": "assistant",
                "status": "completed",
                "type": "message",
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }


def _environment_server(*, token_capture: bool = False) -> tuple[UserSimEnvironmentServer, _Client]:
    servers = {
        "resources": {"resources_servers": {"usersim": {"host": "resources", "port": 8000, "entrypoint": "app.py"}}},
        "user": {
            "responses_api_agents": {
                "user": {
                    "host": "user",
                    "port": 8001,
                    "entrypoint": "app.py",
                    "token_id_capture": token_capture,
                }
            }
        },
        "assistant": {
            "responses_api_agents": {
                "assistant": {
                    "host": "assistant",
                    "port": 8002,
                    "entrypoint": "app.py",
                    "token_id_capture": token_capture,
                }
            }
        },
        "judge": {
            "responses_api_agents": {
                "judge": {
                    "host": "judge",
                    "port": 8003,
                    "entrypoint": "app.py",
                    "token_id_capture": token_capture,
                }
            }
        },
        "summary": {
            "responses_api_agents": {
                "summary": {
                    "host": "summary",
                    "port": 8004,
                    "entrypoint": "app.py",
                    "token_id_capture": token_capture,
                }
            }
        },
        "token_id_capture": {"enabled": token_capture},
    }
    client = _Client(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(servers),
        calls=[],
        responses=[],
    )
    config = UserSimEnvironmentServerConfig(
        name="usersim-environment",
        host="environment",
        port=8005,
        entrypoint="app.py",
        cleanup_timeout_seconds=10,
        resources_server=ResourcesServerRef(type="resources_servers", name="resources"),
        user_agent=AgentServerRef(type="responses_api_agents", name="user"),
        assistant_agent=AgentServerRef(type="responses_api_agents", name="assistant"),
        judge_agent=AgentServerRef(type="responses_api_agents", name="judge"),
        summary_agent=AgentServerRef(type="responses_api_agents", name="summary"),
        resources_tool_transports=["direct_http"],
        max_turns=2,
    )
    return UserSimEnvironmentServer(config=config, server_client=client), client


def _request() -> UserSimEpisodeRequest:
    return UserSimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="usersim:example", task_id="task"),
            task_input=UserSimTaskInput(
                sampling={"locale": "en_US", "seed": 42},
                responses_create_params={
                    "user": {"input": [], "temperature": 0.8},
                    "assistant": {"input": [], "temperature": 0.2},
                    "judge": {"input": []},
                    "summary": {"input": []},
                },
            ),
        ),
    )


def _queue_success_responses(client: _Client) -> None:
    client.responses.extend(
        [
            _Response(
                {
                    "resources_session_id": "resources-session",
                    "scenario": {
                        "persona": {"first_name": "Morgan"},
                        "probe_type": "general_open_ended",
                        "theme": {"type": "recommendation", "description": "Plan dinner."},
                        "goal": "Plan dinner.",
                        "locale": "en_US",
                    },
                    "usersim_context": {
                        "locale": "en_US",
                        "seed": 42,
                        "personas_dataset_version": "0.0.2",
                        "personas_panel_sha256": "a" * 64,
                        "usersim_revision": "b" * 40,
                    },
                },
                cookie="resources-cookie",
            ),
            _Response({"agent_session_id": "user-session"}, cookie="user-cookie"),
            _Response({"agent_session_id": "assistant-session"}, cookie="assistant-cookie"),
            _Response({"agent_session_id": "judge-session"}, cookie="judge-cookie"),
            _Response({"agent_session_id": "summary-session"}, cookie="summary-cookie"),
            _Response(_model_response("user-response", "I need dinner advice.")),
            _Response(_model_response("assistant-response", "Try a lentil curry.")),
            _Response(_model_response("judge-response", "<rating>pass</rating>")),
            _Response(_model_response("summary-response", "The assistant recommended lentil curry.")),
            _Response(
                {
                    "agent_session_id": "summary-session",
                    "resources_cookies": {},
                }
            ),
            _Response(
                {
                    "agent_session_id": "judge-session",
                    "resources_cookies": {},
                }
            ),
            _Response(
                {
                    "agent_session_id": "assistant-session",
                    "resources_cookies": {"session": "resources-cookie"},
                }
            ),
            _Response(
                {
                    "agent_session_id": "user-session",
                    "resources_cookies": {},
                }
            ),
            _Response(
                {
                    "reward": 1.0,
                    "reward_components": {
                        "participants_completed": 1.0,
                        "shared_state_exercised": 0.0,
                        "terminated": 0.0,
                    },
                    "scenario_completed": True,
                    "verifier_data": {},
                }
            ),
            _Response({"resources_session_id": "resources-session"}),
        ]
    )


def test_usersim_generator_uses_instance_api() -> None:
    config = object()
    models = {"user_model": object()}

    class Generator:
        @property
        def config(self) -> object:
            return self._config

        def generate(self, data: dict[str, Any]) -> dict[str, Any]:
            return {
                "config": self.config,
                "model": self.get_model("user_model"),
                "data": data,
            }

    generator = _create_usersim_generator(Generator, config, models)

    assert isinstance(generator, Generator)
    assert generator.generate({"probe_type": "general_open_ended"}) == {
        "config": config,
        "model": models["user_model"],
        "data": {"probe_type": "general_open_ended"},
    }


async def test_usersim_environment_server_runs_native_episode(monkeypatch) -> None:
    environment_server, client = _environment_server()
    _queue_success_responses(client)

    async def fake_run(bridge, _scenario):
        await bridge.invoke("user_model", [{"role": "user", "content": "write user"}], max_tokens=None, tools=None)
        await bridge.invoke(
            "assistant_model",
            [{"role": "user", "content": "I need dinner advice."}],
            max_tokens=128,
            tools=None,
        )
        await bridge.invoke(
            "judge_model",
            [{"role": "user", "content": "judge"}],
            max_tokens=None,
            tools=None,
        )
        await bridge.invoke(
            "summary_model",
            [{"role": "user", "content": "summarize"}],
            max_tokens=None,
            tools=None,
        )
        return {
            "conversation_messages": [
                {"role": "user", "content": "I need dinner advice."},
                {"role": "assistant", "content": "Try a lentil curry."},
            ],
            "conversation_status": True,
            "simulation_outcome": {"status": "ok", "early_stop": True},
            "simulation_traces": [],
        }

    monkeypatch.setattr(environment_server, "_run_usersim", fake_run)
    response = await environment_server.run_request(_request())

    assert isinstance(environment_server, BaseEnvironmentServer)
    assert response.failure is None
    assert response.result is not None
    assert response.result.verification.reward == 1.0
    assert [invocation.role for invocation in response.result.invocations] == ["user", "assistant", "judge", "summary"]
    assert response.result.invocations[1].request.max_output_tokens == 128
    assert response.result.invocations[1].termination_reason == "usersim_early_stop"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions",
        "/v1/agent_sessions",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/ng-rollout/rollout-a2/v1/responses",
        "/ng-rollout/rollout-a2/v1/responses",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert client.calls[1][2]["json"]["tool_accesses"] == []
    [tool_access] = client.calls[2][2]["json"]["tool_accesses"]
    assert tool_access["name"] == "resources.direct_http"
    assert tool_access["cookies"] == {"session": "resources-cookie"}
    assert client.calls[5][2]["cookies"] == {"session": "user-cookie"}
    assert client.calls[6][2]["cookies"] == {"session": "assistant-cookie"}
    assert client.calls[7][2]["cookies"] == {"session": "judge-cookie"}
    assert client.calls[8][2]["cookies"] == {"session": "summary-cookie"}
    verify_body = client.calls[13][2]["json"]
    assert verify_body.task_id == TaskId(taskset="usersim:example", task_id="task")
    assert [invocation.role for invocation in verify_body.verification_input.invocations] == [
        "user",
        "assistant",
        "judge",
        "summary",
    ]


async def test_token_capture_uses_environment_episode_identity(monkeypatch) -> None:
    environment_server, client = _environment_server(token_capture=True)
    _queue_success_responses(client)

    async def fake_run(bridge, _scenario):
        await bridge.invoke("user_model", [{"role": "user", "content": "write user"}], max_tokens=None, tools=None)
        await bridge.invoke(
            "assistant_model",
            [{"role": "user", "content": "hello"}],
            max_tokens=None,
            tools=None,
        )
        await bridge.invoke(
            "judge_model",
            [{"role": "user", "content": "judge"}],
            max_tokens=None,
            tools=None,
        )
        await bridge.invoke(
            "summary_model",
            [{"role": "user", "content": "summarize"}],
            max_tokens=None,
            tools=None,
        )
        return {
            "conversation_messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "answer"},
            ],
            "conversation_status": True,
            "simulation_outcome": {},
        }

    monkeypatch.setattr(environment_server, "_run_usersim", fake_run)
    await environment_server.run_request(_request())

    for call_index in range(5, 9):
        assert client.calls[call_index][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"


def test_dependency_retry_requires_transient_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid contract")) is False
