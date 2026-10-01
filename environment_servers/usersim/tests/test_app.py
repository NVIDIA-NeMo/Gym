# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _apply_activation_parameters,
    _ConversationBridge,
    _is_retryable_dependency_error,
    _response_output_messages,
    _to_responses_input_items,
)
from nemo_gym.base_environment_server import BaseEnvironmentServer
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from resources_servers.usersim.app import UserSimResourcesServer
from resources_servers.usersim.episode_contracts import (
    ActivationRequest,
    AssistantLoopPolicy,
    UserSimEpisodeRequest,
    UserSimTaskInput,
)


class _Cookie:
    def __init__(self, value: str) -> None:
        self.value = value


class _Response:
    ok = True

    def __init__(
        self,
        body: dict[str, Any],
        *,
        cookie: str | None = None,
        cookies: dict[str, str] | None = None,
    ) -> None:
        self.body = orjson.dumps(body)
        values = dict(cookies or {})
        if cookie is not None:
            values["session"] = cookie
        self.cookies = {name: _Cookie(value) for name, value in values.items()}

    async def read(self) -> bytes:
        return self.body


class _Client(ServerClient):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    calls: list[tuple[str, str, dict[str, Any]]]
    responses: list[_Response]

    async def post(self, server_name: str, url_path: str, **kwargs: Any) -> _Response:
        self.calls.append((server_name, url_path, deepcopy(kwargs)))
        response = self.responses.pop(0)
        if url_path == "/seed_session":
            payload = orjson.loads(response.body)
            payload["resources_session_id"] = kwargs["json"].resources_session_id
            response.body = orjson.dumps(payload)
        elif url_path in {"/v1/agent_sessions", "/v1/agent_sessions/close"}:
            payload = orjson.loads(response.body)
            body = kwargs["json"]
            payload["agent_session_id"] = body["agent_session_id"] if isinstance(body, dict) else body.agent_session_id
            response.body = orjson.dumps(payload)
        elif url_path == "/close_session":
            payload = orjson.loads(response.body)
            payload["resources_session_id"] = kwargs["json"].resources_session_id
            response.body = orjson.dumps(payload)
        return response

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
        "usage": {
            "input_tokens": 10,
            "input_tokens_details": {"cached_tokens": 2},
            "output_tokens": 5,
            "output_tokens_details": {"reasoning_tokens": 1},
            "total_tokens": 15,
        },
    }


def _tool_model_response() -> dict[str, Any]:
    response = _model_response("assistant-response", "Done.")
    response["output"] = [
        {
            "id": "reasoning-1",
            "summary": [{"text": "Inspect the tool result.", "type": "summary_text"}],
            "type": "reasoning",
        },
        {
            "id": "fc-1",
            "call_id": "call-1",
            "name": "safe_action",
            "arguments": '{"value":"x"}',
            "type": "function_call",
            "status": "completed",
        },
        {
            "call_id": "call-1",
            "output": '{"ok":true}',
            "type": "function_call_output",
        },
        *response["output"],
    ]
    return response


class _RuntimeDescriptor(BaseModel):
    assistant_tools: list[dict[str, Any]]
    loop_policy: AssistantLoopPolicy


def _runtime_descriptor() -> _RuntimeDescriptor:
    return _RuntimeDescriptor.model_validate(
        {
            "assistant_tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "safe_action",
                        "description": "Perform an action.",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
            "allowed_tool_names": ["safe_action"],
            "initial_user_message": "Perform the action.",
            "loop_policy": {
                "mode": "multi",
                "max_model_calls": 3,
                "max_tool_calls": None,
                "final_synthesis": False,
            },
        }
    )


def test_tool_transcript_converts_to_responses_input_items() -> None:
    assistant_items = _to_responses_input_items(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {"name": "safe_action", "arguments": '{"value":"x"}'},
                }
            ],
        }
    )
    tool_items = _to_responses_input_items({"role": "tool", "content": '{"ok":true}', "tool_call_id": "call-1"})

    assert assistant_items == [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "safe_action",
            "arguments": '{"value":"x"}',
        }
    ]
    assert tool_items == [
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"ok":true}',
        }
    ]


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
                    "model_server": {"type": "responses_api_models", "name": "policy"},
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
                    "model_server": {"type": "responses_api_models", "name": "policy"},
                }
            }
        },
        "policy": {
            "responses_api_models": {
                "vllm_model": {
                    "host": "policy",
                    "port": 8004,
                    "entrypoint": "app.py",
                    "model": "nvidia/example-assistant-model",
                }
            }
        },
        "support": {
            "responses_api_models": {
                "vllm_model": {
                    "host": "support",
                    "port": 8003,
                    "entrypoint": "app.py",
                    "model": "nvidia/example-support-model",
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
        judge_model=ModelServerRef(type="responses_api_models", name="support"),
        summary_model=ModelServerRef(type="responses_api_models", name="support"),
        resources_tool_transports=["direct_http"],
    )
    return UserSimEnvironmentServer(config=config, server_client=client), client


def test_shipped_usersim_servers_disable_ray() -> None:
    assert UserSimResourcesServer.ray_enabled is False
    assert UserSimEnvironmentServer.ray_enabled is False


def _request() -> UserSimEpisodeRequest:
    return UserSimEpisodeRequest(
        episode_id=EpisodeId(rollout_id="rollout", attempt=2),
        task=MaterializedTask(
            task_id=TaskId(taskset="usersim:example", task_id="task"),
            task_input=UserSimTaskInput(
                resolved_row={
                    "persona": {"first_name": "Morgan"},
                    "probe_type": "general_open_ended",
                    "probe_family": "general_open_ended",
                    "probe_variant": "default",
                    "theme": {"type": "recommendation", "description": "Plan dinner."},
                    "locale": "en_US",
                    "trajectory_id": "native-trajectory",
                    "usersim_provenance": {
                        "code_sha": "b3381ae021baac2a6fb314b5f08a55845243017d",  # pragma: allowlist secret
                    },
                    "usersim_config": {"random_seed": 42},
                },
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
                    "resolved_row": {
                        "persona": {"first_name": "Morgan"},
                        "probe_type": "general_open_ended",
                        "probe_family": "general_open_ended",
                        "probe_variant": "default",
                        "theme": {"type": "recommendation", "description": "Plan dinner."},
                        "locale": "en_US",
                        "trajectory_id": "native-trajectory",
                        "usersim_provenance": {
                            "code_sha": "b3381ae021baac2a6fb314b5f08a55845243017d",  # pragma: allowlist secret
                        },
                        "usersim_config": {"random_seed": 42},
                    },
                    "assistant_tools": _runtime_descriptor().assistant_tools,
                },
                cookie="resources-cookie",
            ),
            _Response({"agent_session_id": "user-session"}, cookie="user-cookie"),
            _Response(_model_response("user-response", "I need dinner advice.")),
            _Response({"agent_session_id": "assistant-session"}, cookie="assistant-cookie"),
            _Response(
                _model_response("assistant-response", "Try a lentil curry.")
                | {
                    "_ng_completed_turn": {
                        "transcript": [{"role": "assistant", "content": "Try a lentil curry."}],
                        "model_calls": [
                            {
                                "response": {"role": "assistant", "content": "Try a lentil curry."},
                                "usage": {"input_tokens": 10, "output_tokens": 5},
                            }
                        ],
                    }
                },
                cookies={"session": "assistant-cookie-rotated", "resources": "tool-cookie-rotated"},
            ),
            _Response(
                {
                    "agent_session_id": "assistant-session",
                    "resources_cookies": {"resources": "resources-authoritative"},
                }
            ),
            _Response(_model_response("judge-response", "<rating>pass</rating>")),
            _Response(_model_response("summary-response", "The assistant recommended lentil curry.")),
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


class _FakeConversationRuntime:
    def __init__(self) -> None:
        self.index = 0
        self.submitted: list[dict[str, Any]] = []

    async def advance(self, result: dict[str, Any] | None = None) -> SimpleNamespace:
        if result is not None:
            self.submitted.append(result)
        events = [
            {
                "activation_id": "activation-000001",
                "role": "user",
                "model_alias": "user_model",
                "messages": [{"role": "user", "content": "write user"}],
                "parameters": {},
                "tools": [],
                "continues_turn": False,
            },
            {
                "turn_id": "assistant-turn-000001",
                "role": "assistant",
                "model_alias": "assistant_model",
                "messages": [{"role": "user", "content": "I need dinner advice."}],
                "parameters": {"max_tokens": 128, "temperature": 0.3, "top_p": 0.9},
                "tools": [],
                "loop_policy": {
                    "mode": "none",
                    "max_model_calls": 1,
                    "max_tool_calls": 0,
                    "final_synthesis": False,
                    "replay_reasoning": False,
                    "project_document_tool_results": False,
                },
                "tool_context": {
                    "turn_id": "assistant-turn-000001",
                    "first_tool_turn_idx": 0,
                    "max_tool_calls": 0,
                    "state_snapshot": {},
                },
            },
            {
                "activation_id": "activation-000003",
                "role": "judge",
                "model_alias": "judge_model",
                "messages": [{"role": "user", "content": "judge"}],
                "parameters": {},
                "tools": [],
                "continues_turn": False,
            },
            {
                "activation_id": "activation-000004",
                "role": "summary",
                "model_alias": "summary_model",
                "messages": [{"role": "user", "content": "summarize"}],
                "parameters": {},
                "tools": [],
                "continues_turn": False,
            },
        ]
        if self.index < len(events):
            value = events[self.index]
            self.index += 1
            return SimpleNamespace(to_dict=lambda value=value: value)
        return SimpleNamespace(
            result={
                "conversation_messages": [
                    {"role": "user", "content": "I need dinner advice."},
                    {"role": "assistant", "content": "Try a lentil curry."},
                ],
                "conversation_status": True,
                "simulation_outcome": {"status": "ok", "early_stop": True},
                "simulation_traces": [],
            }
        )

    async def close(self) -> None:
        pass


def test_assistant_activation_preserves_complete_tool_transcript() -> None:
    response = NeMoGymResponse.model_validate(_tool_model_response())
    transcript = _response_output_messages(response)

    assert [message["role"] for message in transcript] == ["assistant", "tool", "assistant"]
    assert transcript[0]["tool_calls"][0]["id"] == "call-1"
    assert transcript[0]["reasoning_content"] == "Inspect the tool result."
    assert transcript[1]["tool_call_id"] == "call-1"


def test_strict_response_format_requires_all_nullable_fields_and_forbids_extras() -> None:
    values: dict[str, Any] = {}
    _apply_activation_parameters(
        values,
        {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "safety_agentic_judgment",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "score": {
                                "anyOf": [{"enum": [1, 2, 3, 4, 5], "type": "integer"}, {"type": "null"}],
                                "default": None,
                            },
                            "reasoning": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
                        },
                    },
                },
            }
        },
        assistant_tools=None,
    )

    schema = values["text"]["format"]["schema"]
    assert values["text"]["format"]["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["score", "reasoning"]
    assert "default" not in schema["properties"]["score"]
    assert "default" not in schema["properties"]["reasoning"]


async def test_conversation_bridge_rejects_role_alias_mismatch_before_invocation() -> None:
    environment_server, client = _environment_server()
    request = _request()
    bridge = _ConversationBridge(
        environment_server,
        request,
        request.task.task_input,
        {},
        {},
        [],
    )

    with pytest.raises(ValueError, match="does not match alias"):
        await bridge.invoke(
            ActivationRequest(
                activation_id="activation-role-mismatch",
                role="judge",
                model_alias="assistant_model",
                messages=[{"role": "user", "content": "Respond as the assistant."}],
                parameters={},
            )
        )

    assert client.calls == []


async def test_usersim_environment_server_runs_native_episode(monkeypatch: pytest.MonkeyPatch) -> None:
    environment_server, client = _environment_server()
    runtime = _FakeConversationRuntime()
    monkeypatch.setattr(environment_server, "_create_conversation_runtime", lambda *_args: runtime)
    _queue_success_responses(client)
    response = await environment_server.run_request(_request())

    assert isinstance(environment_server, BaseEnvironmentServer)
    assert response.failure is None
    assert response.result is not None
    assert response.result.verification.reward == 1.0
    assert [invocation.role for invocation in response.result.invocations] == ["user", "assistant", "judge", "summary"]
    assert response.result.invocations[1].request.max_output_tokens == 128
    assert response.result.invocations[1].request.temperature == 0.3
    assert response.result.invocations[1].request.top_p == 0.9
    assert response.result.invocations[1].response.usage.total_tokens == 15
    assert response.result.invocations[1].termination_reason == "usersim_early_stop"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/agent_sessions/close",
        "/v1/responses",
        "/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert client.calls[1][2]["json"]["tool_accesses"] == []
    [tool_access] = client.calls[3][2]["json"]["tool_accesses"]
    assert tool_access["name"] == "resources.direct_http"
    assert tool_access["cookies"] == {"session": "resources-cookie"}
    assert tool_access["batch_path"] is None
    assert tool_access["tool_call_context"]["turn_id"] == "assistant-turn-000001"
    assert client.calls[2][2]["cookies"] == {"session": "user-cookie"}
    assert client.calls[5][2]["cookies"] == {
        "session": "assistant-cookie-rotated",
        "resources": "tool-cookie-rotated",
    }
    assert client.calls[6][0] == "support"
    assert client.calls[7][0] == "support"
    assert "cookies" not in client.calls[6][2]
    assert "cookies" not in client.calls[7][2]
    assert client.calls[9][2]["cookies"] == {"resources": "resources-authoritative"}
    assert client.calls[0][2]["json"].task_data == _request().task.task_input.model_dump(mode="json")
    verify_body = client.calls[9][2]["json"]
    assert verify_body.task_id == TaskId(taskset="usersim:example", task_id="task")
    assert verify_body.verification_input.resolved_row["trajectory_id"] == "native-trajectory"
    assert [invocation.role for invocation in verify_body.verification_input.invocations] == [
        "user",
        "assistant",
        "judge",
        "summary",
    ]


async def test_token_capture_uses_environment_episode_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    environment_server, client = _environment_server(token_capture=True)
    monkeypatch.setattr(environment_server, "_create_conversation_runtime", lambda *_args: _FakeConversationRuntime())
    _queue_success_responses(client)
    await environment_server.run_request(_request())

    for call_index in (2, 4):
        assert client.calls[call_index][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
    for call_index in (6, 7):
        assert client.calls[call_index][1] == "/v1/responses"


async def test_transcript_evidence_mismatch_remains_infrastructure_contract_error() -> None:
    class EpisodeContractError(ValueError):
        pass

    environment_server, _ = _environment_server()

    class _MismatchRuntime:
        async def advance(self, result=None):
            if result is not None:
                raise EpisodeContractError("Completed assistant transcript and Resources evidence disagree")
            return SimpleNamespace(
                to_dict=lambda: {
                    "turn_id": "turn-1",
                    "role": "assistant",
                    "model_alias": "assistant_model",
                    "messages": [{"role": "user", "content": "act"}],
                    "parameters": {},
                    "tools": [],
                    "loop_policy": {
                        "mode": "none",
                        "max_model_calls": 1,
                        "max_tool_calls": 0,
                        "final_synthesis": False,
                    },
                    "tool_context": {
                        "turn_id": "turn-1",
                        "first_tool_turn_idx": 0,
                        "max_tool_calls": 0,
                        "state_snapshot": {},
                    },
                }
            )

        async def close(self):
            pass

    completed = {
        "turn_id": "turn-1",
        "transcript": [{"role": "assistant", "content": "wrong"}],
        "model_calls": [{"response": {"role": "assistant", "content": "right"}}],
        "evidence": {"turn_id": "turn-1", "rounds": [], "final_effect_state": {}},
    }
    bridge = SimpleNamespace(
        invoke_assistant=AsyncMock(return_value=SimpleNamespace(model_dump=lambda **_kwargs: completed))
    )

    with pytest.raises(EpisodeContractError, match="transcript and Resources evidence disagree"):
        await environment_server._run_usersim(_MismatchRuntime(), bridge)


@pytest.mark.parametrize("error", [RuntimeError("start failed"), asyncio.CancelledError()])
async def test_initial_runtime_advance_always_closes_runtime(error: BaseException) -> None:
    environment_server, _ = _environment_server()
    runtime = SimpleNamespace(
        advance=AsyncMock(side_effect=error),
        close=AsyncMock(),
    )

    with pytest.raises(type(error)):
        await environment_server._run_usersim(runtime, SimpleNamespace())

    runtime.close.assert_awaited_once()


def test_dependency_retry_requires_transient_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid contract")) is False
