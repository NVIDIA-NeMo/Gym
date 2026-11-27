# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.client import ServerClient
from responses_api_agents.direct_vision_agent import app as module


class FilesSandbox:
    def __init__(self):
        self.files = {
            "/task/input.json": json.dumps(
                {"system": "scientist", "user": "question", "images": ["a.png", "b.png"]}
            ).encode(),
            "/task/a.png": b"first",
            "/task/b.png": b"second",
            "/tests/reference": b"REFERENCE_SENTINEL",
        }
        self.downloads = []
        self.disconnected = False
        self.upload_failures = 0

    async def download(self, source, destination):
        self.downloads.append(source)
        Path(destination).write_bytes(self.files[source])

    async def upload(self, source, destination):
        if self.upload_failures and destination == "/task/answer.txt":
            self.upload_failures -= 1
            raise OSError("temporary transfer failure")
        self.files[destination] = Path(source).read_bytes()

    async def disconnect(self):
        self.disconnected = True


def seed():
    return {
        "agent_session_id": "a1",
        "episode_id": {"rollout_id": "r1", "attempt": 0},
        "task_id": {"taskset": "images", "task_id": "test"},
        "sandbox_access": {
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": {"id": "borrowed"}},
            "workdir": "/task",
        },
        "agent_timeout_sec": 600,
    }


def make_agent(monkeypatch, *, budget=32768, tool_call=False):
    sandbox = FilesSandbox()
    provider = MagicMock()
    provider.aclose = AsyncMock()
    monkeypatch.setattr(module, "get_global_config_dict", lambda: {"sandbox": {"docker": {}}})
    monkeypatch.setattr(module, "create_provider", lambda config: provider)
    connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(module.AsyncSandbox, "connect", connect)
    message = {"role": "assistant", "content": "ANSWER: A", "reasoning_content": "reason"}
    if tool_call:
        message["tool_calls"] = [{"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}]
    result = {
        "id": "chat-1",
        "created": 1,
        "object": "chat.completion",
        "model": "vision-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    response = SimpleNamespace(ok=True, status=200, read=AsyncMock(return_value=json.dumps(result).encode()))
    server = MagicMock(spec=ServerClient)
    server.post = AsyncMock(return_value=response)
    config = module.DirectVisionAgentConfig(
        name="direct_vision_agent",
        host="127.0.0.1",
        port=18001,
        entrypoint="app.py",
        model_server={"type": "responses_api_models", "name": "policy_model"},
        max_output_tokens=budget,
        chat_template_kwargs={"enable_thinking": True},
    )
    agent = module.DirectVisionAgent(config=config, server_client=server)
    return agent, sandbox, connect, server


@pytest.mark.parametrize("budget", [32768, 131072, 262144])
def test_one_vision_call_preserves_images_reasoning_and_usage(monkeypatch, budget):
    agent, sandbox, connect, server = make_agent(monkeypatch, budget=budget)
    with TestClient(agent.setup_webserver()) as client:
        reply = client.post("/v1/agent_sessions", json=seed())
        assert reply.status_code == 200, reply.text
        reply = client.post("/v1/responses", json={"input": "task instructions"})
        assert reply.status_code == 200, reply.text
        data = reply.json()
        assert data["usage"]["input_tokens"] == 10 and data["usage"]["output_tokens"] == 5
        assert [item["role"] for item in data["output"] if "role" in item] == ["system", "user", "assistant"]
        assert any(item["type"] == "reasoning" for item in data["output"])
        # Exercise the same strict exporter the native Harbor verifier runs.
        from nemo_gym.trajectory import transcript_to_atif

        trajectory = transcript_to_atif(
            [{"type": "message", "role": "user", "content": "task instructions"}],
            data["output"],
            session_id="r1",
            agent_name="direct_vision_agent",
            agent_version="1",
            model_name=data["model"],
            usage=data["usage"],
        )
        assert "policy-input.json" in trajectory.model_dump_json()
        assert client.post("/v1/responses", json={"input": "task instructions"}).status_code == 200
        assert server.post.await_count == 1
        request = server.post.call_args.kwargs["json"]
        assert request["tool_choice"] == "none" and "tools" not in request
        assert request["max_completion_tokens"] == budget
        assert request["chat_template_kwargs"] == {"enable_thinking": True}
        assert request["messages"][1]["content"][1]["image_url"]["url"] == "data:image/png;base64,Zmlyc3Q="
        assert request["messages"][1]["content"][2]["image_url"]["url"] == "data:image/png;base64,c2Vjb25k"
        assert "REFERENCE_SENTINEL" not in json.dumps(request)
        assert sandbox.files["/task/answer.txt"] == b"ANSWER: A"
        assert json.loads(sandbox.files["/logs/artifacts/policy-input.json"]) == request
        assert json.loads(sandbox.files["/logs/artifacts/policy-response.json"])["usage"]["completion_tokens"] == 5
        assert sandbox.downloads == ["/task/input.json", "/task/a.png", "/task/b.png"]
        close = client.post(
            "/v1/agent_sessions/close", json={"agent_session_id": "a1", "episode_id": seed()["episode_id"]}
        )
        assert close.status_code == 200 and sandbox.disconnected
        assert "input_media_in_artifact" in [gap["code"] for gap in close.json()["agent_observations"]["gaps"]]
        assert client.post("/v1/responses", json={"input": "task instructions"}).status_code == 409
    assert connect.await_count == 1


def test_rejects_unseeded_inference_and_missing_sandbox(monkeypatch):
    agent, _, connect, server = make_agent(monkeypatch)
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/responses", json={"input": "q"}).status_code == 409
        body = seed()
        body.pop("sandbox_access")
        assert client.post("/v1/agent_sessions", json=body).status_code == 422
    connect.assert_not_awaited()
    server.post.assert_not_awaited()


def test_tool_reply_is_rejected_without_writing_an_answer(monkeypatch):
    agent, sandbox, _, _ = make_agent(monkeypatch, tool_call=True)
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed()).status_code == 200
        assert client.post("/v1/responses", json={"input": "q"}).status_code >= 400
    assert "/task/answer.txt" not in sandbox.files


@pytest.mark.parametrize("images", [["../secret.png"], ["/tests/secret.png"], ["a.png", "a.png"], [], ["data.json"]])
def test_rejects_unsafe_or_invalid_manifest_before_inference(monkeypatch, images):
    agent, sandbox, _, server = make_agent(monkeypatch)
    sandbox.files["/task/input.json"] = json.dumps({"user": "q", "images": images}).encode()
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed()).status_code == 200
        assert client.post("/v1/responses", json={"input": "q"}).status_code >= 400
    server.post.assert_not_awaited()


def test_failed_artifact_transfer_does_not_repeat_model_call(monkeypatch):
    agent, sandbox, _, server = make_agent(monkeypatch)
    sandbox.upload_failures = 1
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed()).status_code == 200
        assert client.post("/v1/responses", json={"input": "q"}).status_code >= 400
        reply = client.post("/v1/responses", json={"input": "q"})
        assert reply.status_code == 200, reply.text
        assert sandbox.files["/task/answer.txt"] == b"ANSWER: A"
        assert server.post.await_count == 1


@pytest.mark.parametrize("required", [True, False])
def test_seed_rejects_tool_grants_before_connecting(monkeypatch, required):
    agent, _, connect, server = make_agent(monkeypatch)
    body = seed()
    body["tool_accesses"] = [
        {
            "kind": "mcp",
            "name": "search",
            "required": required,
            "connection": {"transport": "streamable_http", "url": "http://tools.example/mcp"},
        }
    ]
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 422
    connect.assert_not_awaited()
    server.post.assert_not_awaited()


def test_request_tools_are_rejected_before_model_call(monkeypatch):
    agent, _, _, server = make_agent(monkeypatch)
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed()).status_code == 200
        reply = client.post(
            "/v1/responses",
            json={"input": "q", "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}]},
        )
        assert reply.status_code == 422, reply.text
    server.post.assert_not_awaited()


@pytest.mark.parametrize("path", ["relative/input.json", "/task/../tests/input.json", "/"])
def test_config_rejects_ambiguous_paths(path):
    with pytest.raises(ValueError, match="absolute file paths"):
        module.DirectVisionAgentConfig(
            name="direct_vision_agent",
            host="127.0.0.1",
            port=18001,
            model_server={"type": "responses_api_models", "name": "policy_model"},
            input_path=path,
        )


def test_failed_connect_closes_provider_without_running_model(monkeypatch):
    agent, _, connect, server = make_agent(monkeypatch)
    connect.side_effect = RuntimeError("sandbox unavailable")
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed()).status_code >= 400
    provider = connect.call_args.kwargs["provider"]
    provider.aclose.assert_awaited_once()
    server.post.assert_not_awaited()


def test_model_timeout_never_writes_a_scored_answer(monkeypatch):
    agent, sandbox, _, server = make_agent(monkeypatch)
    server.post.side_effect = TimeoutError("policy request timeout")
    with TestClient(agent.setup_webserver(), raise_server_exceptions=False) as client:
        assert client.post("/v1/agent_sessions", json=seed()).status_code == 200
        assert client.post("/v1/responses", json={"input": "q"}).status_code >= 400
    assert "/task/answer.txt" not in sandbox.files


def test_entrypoint_starts_the_agent_when_executed_as_a_script(monkeypatch):
    import runpy

    started = []

    def start(cls):
        started.append(cls.__name__)

    monkeypatch.setattr(module.SimpleResponsesAPIAgent, "run_webserver", classmethod(start))
    runpy.run_path(module.__file__, run_name="__main__")
    assert started == ["DirectVisionAgent"]
