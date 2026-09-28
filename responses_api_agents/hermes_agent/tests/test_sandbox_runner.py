# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import threading
import time
from pathlib import Path

import httpx
import openai
import pytest
from run_agent import AIAgent

from responses_api_agents.hermes_agent.sandbox_runner import (
    FileModelRelay,
    _route_model_clients_through,
    _run,
    _write_atomic,
)


def _completion(message: dict) -> dict:
    return {
        "id": "chatcmpl-test",
        "choices": [{"finish_reason": "stop", "index": 0, "message": {"role": "assistant", **message}}],
        "created": 0,
        "model": "model",
        "object": "chat.completion",
    }


class _AgentServer:
    """Answer the runner's model requests in index order, as the controlling agent server does."""

    def __init__(self, exchange_dir: Path, answers: list[dict]) -> None:
        self.exchange_dir = exchange_dir
        self.answers = answers
        self.requests: list[dict] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def __enter__(self) -> "_AgentServer":
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def _serve(self) -> None:
        while not self._stop.is_set():
            request_path = self.exchange_dir / f"model-request-{len(self.requests)}.json"
            if not request_path.exists():
                time.sleep(0.01)
                continue
            self.requests.append(json.loads(request_path.read_text()))
            answer = self.answers[min(len(self.requests), len(self.answers)) - 1]
            _write_atomic(self.exchange_dir / f"model-response-{len(self.requests) - 1}.json", answer)


@pytest.fixture
def restore_process_globals(monkeypatch: pytest.MonkeyPatch) -> None:
    """The runner patches process-wide state; restore it so other tests see the originals."""
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", httpx.HTTPTransport.handle_request)
    monkeypatch.setattr(
        httpx.AsyncHTTPTransport, "handle_async_request", httpx.AsyncHTTPTransport.handle_async_request
    )
    monkeypatch.setattr(AIAgent, "__init__", AIAgent.__init__)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "HERMES_HOME", "TERMINAL_ENV", "TERMINAL_TIMEOUT"):
        monkeypatch.delenv(name, raising=False)


def test_file_model_relay_exchanges_one_request(tmp_path) -> None:
    with _AgentServer(tmp_path, [{"response": _completion({"content": "done"})}]) as server:
        payload = FileModelRelay(tmp_path).exchange({"model": "policy_model"})

    assert server.requests == [{"model": "policy_model"}]
    assert payload["response"]["id"] == "chatcmpl-test"


def test_model_server_errors_keep_their_status(tmp_path, restore_process_globals) -> None:
    _route_model_clients_through(FileModelRelay(tmp_path))
    client = openai.OpenAI(max_retries=0)

    with _AgentServer(tmp_path, [{"error": "context length exceeded", "status": 400}]):
        with pytest.raises(openai.BadRequestError, match="context length exceeded"):
            client.chat.completions.create(model="policy_model", messages=[{"role": "user", "content": "hi"}])


def test_iteration_limit_summary_goes_through_the_relay(tmp_path, restore_process_globals) -> None:
    tool_call = {
        "content": None,
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "terminal", "arguments": json.dumps({"command": "echo hi"})},
            }
        ],
    }
    answers = [
        {"response": _completion(tool_call)},
        {"response": _completion({"content": "summary of the work"})},
    ]
    payload = {
        "agent_session_id": "session",
        "chat_template_kwargs_enabled": False,
        "config_yaml": "model: policy_model\nprovider: auto\n",
        "disabled_toolsets": None,
        "enabled_toolsets": ["terminal"],
        "history": [],
        "max_tokens": 128,
        "max_turns": 1,
        "model": "policy_model",
        "system_message": None,
        "temperature": 0.0,
        "terminal_timeout": 30,
        "user_message": "fix bug",
    }

    with _AgentServer(tmp_path, answers) as server:
        output = _run(payload, tmp_path)

    assert len(server.requests) == 2
    assert all(not request.get("stream") for request in server.requests)
    assert output["result"]["final_response"] == "summary of the work"


def test_clients_hermes_builds_itself_use_the_relay(tmp_path, restore_process_globals) -> None:
    _route_model_clients_through(FileModelRelay(tmp_path))
    answers = [{"response": _completion({"content": "sync"})}, {"response": _completion({"content": "async"})}]

    with _AgentServer(tmp_path, answers) as server:
        # Auxiliary clients find the endpoint through the environment, as Hermes's custom runtime does.
        sync = openai.OpenAI().chat.completions.create(model="m", messages=[{"role": "user", "content": "a"}])
        asynchronous = asyncio.run(
            openai.AsyncOpenAI().chat.completions.create(model="m", messages=[{"role": "user", "content": "b"}])
        )

    assert [sync.choices[0].message.content, asynchronous.choices[0].message.content] == ["sync", "async"]
    assert len(server.requests) == 2
    # Delegated children are AIAgents Hermes constructs itself; the relay cannot stream to them.
    child = AIAgent(
        base_url="http://nemo-gym-model-relay.invalid/v1", api_key="model-relay", model="m", quiet_mode=True
    )
    assert child.use_streaming is False
