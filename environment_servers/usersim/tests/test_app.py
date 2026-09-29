# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import orjson
import pytest
from omegaconf import OmegaConf
from pydantic import ConfigDict

from environment_servers.usersim.app import (
    UserSimEnvironmentServer,
    UserSimEnvironmentServerConfig,
    _configured_model_name,
    _create_usersim_generator,
    _is_retryable_dependency_error,
    _to_responses_input_items,
)
from nemo_gym.base_environment_server import BaseEnvironmentServer
from nemo_gym.config_types import AgentServerRef, ModelServerRef, ResourcesServerRef
from nemo_gym.episode_types import EpisodeId, MaterializedTask, TaskId
from nemo_gym.server_utils import BaseServerConfig, ServerClient
from resources_servers.usersim.episode_contracts import (
    ResolvedUserSimContext,
    UserSimEpisodeRequest,
    UserSimScenario,
    UserSimSeedResponse,
    UserSimTaskInput,
)


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
    }


def _tool_response(response_id: str) -> dict[str, Any]:
    response = _model_response(response_id, "")
    response["output"] = [
        {
            "arguments": '{"city":"Paris"}',
            "call_id": "call-weather",
            "name": "get_weather",
            "type": "function_call",
        }
    ]
    return response


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
        tool_simulation_model=ModelServerRef(type="responses_api_models", name="support"),
        max_turns=2,
    )
    return UserSimEnvironmentServer(config=config, server_client=client), client


def test_configured_model_name_resolves_agent_and_direct_model_targets() -> None:
    server, _ = _environment_server()

    assert _configured_model_name(server, "user_model") == "nvidia/example-assistant-model"
    assert _configured_model_name(server, "assistant_model") == "nvidia/example-assistant-model"
    assert _configured_model_name(server, "judge_model") == "nvidia/example-support-model"
    assert _configured_model_name(server, "summary_model") == "nvidia/example-support-model"
    assert _configured_model_name(server, "api_response_model") == "nvidia/example-support-model"


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
                    "sandbox_access": {
                        "connection": {
                            "kind": "direct",
                            "provider_config_ref": "sandbox",
                            "descriptor": {"sandbox_id": "sandbox-1"},
                        },
                        "workdir": "/workspace",
                    },
                },
                cookie="resources-cookie",
            ),
            _Response({"agent_session_id": "user-session"}, cookie="user-cookie"),
            _Response({"agent_session_id": "assistant-session"}, cookie="assistant-cookie"),
            _Response(_model_response("user-response", "I need dinner advice.")),
            _Response(_tool_response("assistant-response")),
            _Response(_model_response("api-response", '{"temperature": 72}')),
            _Response(_model_response("judge-response", "<rating>pass</rating>")),
            _Response(_model_response("summary-response", "The assistant recommended lentil curry.")),
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


def test_usersim_tool_exchange_converts_to_responses_items() -> None:
    assert _to_responses_input_items(
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call-weather",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
                }
            ],
        }
    ) == [
        {
            "type": "function_call",
            "call_id": "call-weather",
            "name": "get_weather",
            "arguments": '{"city":"Paris"}',
        }
    ]
    assert _to_responses_input_items(
        SimpleNamespace(role="tool", tool_call_id="call-weather", content='{"temperature":72}')
    ) == [
        {
            "type": "function_call_output",
            "call_id": "call-weather",
            "output": '{"temperature":72}',
        }
    ]


@pytest.mark.parametrize("probe_type", ["tool_calling", "safety_agentic", "financial_services"])
async def test_native_generator_owns_each_tool_probe_loop(monkeypatch, probe_type: str) -> None:
    environment_server, _ = _environment_server()
    monkeypatch.setattr("environment_servers.usersim.app._configured_model_name", lambda *_args: "test-model")
    examples_path = Path(__file__).parents[3] / "resources_servers/usersim/data/example.jsonl"
    rows = [json.loads(line) for line in examples_path.read_text().splitlines()]
    task = next(row["task_input"] for row in rows if row["task_input"]["sampling"]["probe_type"] == probe_type)
    seed = UserSimSeedResponse(
        resources_session_id="resources-session",
        scenario=UserSimScenario(
            persona={
                "first_name": "Sarah",
                "last_name": "Johnson",
                "age": 42,
                "city": "Austin",
                "state": "TX",
                "state_abbrev": "TX",
                "education_level": "Bachelor's degree",
                "occupation": "Public school teacher",
            },
            probe_type=probe_type,
            theme={"type": "weather", "description": "Get weather information."},
            locale="en_US",
            probe_data=task.get("probe_data", {}),
        ),
        usersim_context=ResolvedUserSimContext(
            locale="en_US",
            seed=task["sampling"]["seed"],
            personas_dataset_version="0.0.2",
            personas_panel_sha256="a" * 64,
            usersim_revision="b" * 40,
        ),
    )

    class Bridge:
        def __init__(self) -> None:
            self.environment_server = environment_server
            self.calls: list[tuple[str, list[dict[str, Any]] | None]] = []

        async def invoke(self, alias, _messages, *, max_tokens, tools):
            del max_tokens
            self.calls.append((alias, tools))
            tool_calls = None
            content = ""
            if alias == "assistant_model":
                if tools:
                    function = tools[0].get("function", tools[0])
                    properties = function.get("parameters", {}).get("properties", {})
                    arguments = {
                        name: _example_tool_argument(name, schema)
                        for name, schema in properties.items()
                        if name in function.get("parameters", {}).get("required", [])
                    }
                    tool_calls = [
                        SimpleNamespace(
                            id=f"call-{len(self.calls)}",
                            name=function["name"],
                            arguments_json=json.dumps(arguments),
                        )
                    ]
                else:
                    content = "Sure, I can help with that safely."
            elif alias == "user_model":
                content = "Thanks, that is helpful."
            elif alias == "judge_model":
                content = "<explanation>looks fine</explanation>\n<rating>success</rating>"
            elif alias == "summary_model":
                content = "no"
            elif alias == "api_response_model":
                content = '{"status":"ok","result":"success"}'
            return SimpleNamespace(
                message=SimpleNamespace(content=content, reasoning_content=None, tool_calls=tool_calls),
                usage=None,
            )

    bridge = Bridge()
    result = await environment_server._run_usersim(bridge, seed)

    assert result["conversation_status"] is True
    assert any(alias == "assistant_model" and tools for alias, tools in bridge.calls)
    if probe_type == "tool_calling":
        assert any(alias == "api_response_model" for alias, _ in bridge.calls)
    outcome = result["simulation_outcome"]
    if isinstance(outcome, str):
        outcome = json.loads(outcome)
    assert outcome["status"] != "failed"


def _example_tool_argument(name: str, schema: dict[str, Any]) -> Any:
    if name == "date_of_birth":
        return "1984-01-01"
    if name == "full_name":
        return "Sarah Johnson"
    if name in {"city", "location"}:
        return "Austin, TX"
    return {
        "array": [],
        "boolean": False,
        "integer": 1,
        "number": 1.0,
        "object": {},
        "string": "test",
    }.get(schema.get("type"), "test")


async def test_usersim_environment_server_runs_native_episode(monkeypatch) -> None:
    environment_server, client = _environment_server()
    _queue_success_responses(client)

    async def fake_run(bridge, _scenario):
        await bridge.invoke("user_model", [{"role": "user", "content": "write user"}], max_tokens=None, tools=None)
        assistant = await bridge.invoke(
            "assistant_model",
            [{"role": "user", "content": "I need dinner advice."}],
            max_tokens=128,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Look up weather.",
                        "parameters": {
                            "type": "object",
                            "properties": {"city": {"type": "string"}},
                            "required": ["city"],
                        },
                    },
                }
            ],
        )
        assert assistant.message.tool_calls[0].name == "get_weather"
        assert assistant.message.tool_calls[0].arguments_json == '{"city":"Paris"}'
        await bridge.invoke(
            "api_response_model",
            [{"role": "user", "content": "simulate get_weather"}],
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
        bridge.invocations[0] = bridge.invocations[0].model_copy(update={"termination_reason": "stale_reason"})
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
    assert [invocation.role for invocation in response.result.invocations] == [
        "user",
        "assistant",
        "tool_simulation",
        "judge",
        "summary",
    ]
    assert response.result.invocations[1].request.max_output_tokens == 128
    assert response.result.invocations[1].termination_reason == "usersim_early_stop"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions",
        "/ng-rollout/rollout-a2/v1/responses",
        "/ng-rollout/rollout-a2/v1/responses",
        "/v1/responses",
        "/v1/responses",
        "/v1/responses",
        "/v1/agent_sessions/close",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    assert client.calls[1][2]["json"]["tool_accesses"] == []
    assert client.calls[1][2]["json"]["sandbox_access"] is None
    assert client.calls[2][2]["json"]["tool_accesses"] == []
    assert client.calls[2][2]["json"]["sandbox_access"]["workdir"] == "/workspace"
    assert client.calls[3][2]["cookies"] == {"session": "user-cookie"}
    assert client.calls[4][2]["cookies"] == {"session": "assistant-cookie"}
    assert client.calls[4][2]["json"].tools[0]["name"] == "get_weather"
    assert client.calls[5][0] == "support"
    assert client.calls[6][0] == "support"
    assert client.calls[7][0] == "support"
    assert "cookies" not in client.calls[5][2]
    assert "cookies" not in client.calls[6][2]
    verify_body = client.calls[10][2]["json"]
    assert verify_body.task_id == TaskId(taskset="usersim:example", task_id="task")
    assert [invocation.role for invocation in verify_body.verification_input.invocations] == [
        "user",
        "assistant",
        "tool_simulation",
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
            "api_response_model",
            [{"role": "user", "content": "simulate"}],
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

    assert client.calls[3][1] == "/ng-rollout/rollout-a2/v1/responses"
    assert client.calls[4][1] == "/ng-rollout/rollout-a2/training-token-capture/v1/responses"
    for call_index in (5, 6, 7):
        assert client.calls[call_index][1] == "/v1/responses"


async def test_seeded_resources_are_closed_when_seed_response_is_invalid() -> None:
    environment_server, client = _environment_server()
    client.responses.extend(
        [
            _Response({}, cookie="resources-cookie"),
            _Response({"resources_session_id": "resources-session"}),
        ]
    )

    response = await environment_server.run_request(_request())

    assert response.failure is not None
    assert response.failure.stage == "seed"
    assert [path for _, path, _ in client.calls] == ["/seed_session", "/close_session"]


async def test_seeded_agent_is_closed_when_seed_response_is_invalid() -> None:
    environment_server, client = _environment_server()
    _queue_success_responses(client)
    resources_seed = client.responses[0]
    client.responses = [
        resources_seed,
        _Response({"unexpected": True}, cookie="user-cookie"),
        _Response({"agent_session_id": "user-session", "resources_cookies": {}}),
        _Response({"resources_session_id": "resources-session"}),
    ]

    response = await environment_server.run_request(_request())

    assert response.failure is not None
    assert response.failure.stage == "agent"
    assert [path for _, path, _ in client.calls] == [
        "/seed_session",
        "/v1/agent_sessions",
        "/v1/agent_sessions/close",
        "/close_session",
    ]


async def test_assistant_activation_failure_is_verified_as_valid_zero_reward(monkeypatch) -> None:
    environment_server, client = _environment_server()
    _queue_success_responses(client)
    client.responses[4] = _Response({"invalid": "assistant response"})
    client.responses = [
        *client.responses[:5],
        client.responses[7],
        client.responses[8],
        _Response(
            {
                "reward": 0.0,
                "reward_components": {"assistant_quality": 0.0},
                "scenario_completed": False,
                "verifier_data": {},
            }
        ),
        client.responses[10],
    ]

    async def fake_run(bridge, _scenario):
        await bridge.invoke("user_model", [{"role": "user", "content": "write user"}], max_tokens=None, tools=None)
        await bridge.invoke(
            "assistant_model",
            [{"role": "user", "content": "I need dinner advice."}],
            max_tokens=None,
            tools=None,
        )
        raise AssertionError("assistant activation should have failed")

    monkeypatch.setattr(environment_server, "_run_usersim", fake_run)
    response = await environment_server.run_request(_request())

    assert response.failure is None
    assert response.result is not None
    assert response.result.verification.reward == 0.0
    assert response.result.usersim_result.simulation_outcome["failure_attribution"] == "assistant_model"
    assert response.result.invocations[-1].termination_reason == "assistant_model_activation_failed"
    assert [path for _, path, _ in client.calls][-2:] == ["/verify", "/close_session"]


def test_dependency_retry_requires_transient_error() -> None:
    assert _is_retryable_dependency_error(TimeoutError()) is True
    assert _is_retryable_dependency_error(ValueError("invalid contract")) is False
