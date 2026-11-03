# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest
from aiohttp import ClientConnectionError, ClientResponseError
from fastapi import Request, Response

from nemo_gym.config import ModelServerRef
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.long_conversation_simulation import app as app_module
from responses_api_agents.long_conversation_simulation.app import (
    EmptyOutputError,
    LongConversationSimulationAgent,
    LongConversationSimulationAgentConfig,
    is_transient,
)
from responses_api_agents.long_conversation_simulation.models import LongConversationRunRequest


PACKAGE_DIR = Path(__file__).resolve().parents[1]
TEMPLATES = {
    "assistant_system_prompt_fpath": "SYS <topic> <theme> <previous_plans_summary> <previous_batches>",
    "check_include_question_prompt_fpath": "ASKED? <assistant_response>",
    "check_need_followup_prompt_fpath": "FOLLOW? <topic> <theme> <formatted_history> <assistant_response>",
    "user_answer_question_prompt_fpath": "ANSWER <ai_last_message>",
    "user_followup_question_prompt_fpath": "FOLLOWUP <ai_last_message>",
}


def completion(text: str | None, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "id": "c",
        "choices": [
            {
                "finish_reason": "stop",
                "index": 0,
                "logprobs": None,
                "message": {"content": text, "refusal": None, "role": "assistant"},
            }
        ],
        "created": 0,
        "model": "test-model",
        "object": "chat.completion",
    }
    if usage is not None:
        payload["usage"] = usage
    return payload


def usage(prompt: int, output: int) -> dict[str, Any]:
    return {"prompt_tokens": prompt, "completion_tokens": output, "total_tokens": prompt + output}


class FakeHTTPResponse:
    ok = True
    status = 200

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.cookies: dict[str, str] = {"session": "s"}

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class ScriptedClient:
    """Answers by server: the helper says "no" to every check, unless scripted otherwise."""

    def __init__(self, assistant: list[Any], helper: list[Any] | None = None) -> None:
        self.assistant = list(assistant)
        self.helper = list(helper or [])
        self.calls: list[dict[str, Any]] = []
        self.global_config_dict = {"observability_enabled": False}

    async def post(self, **kwargs: Any) -> FakeHTTPResponse:
        self.calls.append(kwargs)
        queue = self.assistant if kwargs["server_name"] == "assistant_model" else self.helper
        item = queue.pop(0) if queue else completion("no")
        if isinstance(item, BaseException):
            raise item
        return FakeHTTPResponse(item)


@pytest.fixture
def prompt_files(tmp_path: Path) -> dict[str, str]:
    paths = {}
    for field, text in TEMPLATES.items():
        path = tmp_path / f"{field}.txt"
        path.write_text(text, encoding="utf-8")
        paths[field] = str(path)
    return paths


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    async def instant(_seconds: float) -> None:
        return None

    monkeypatch.setattr(app_module.asyncio, "sleep", instant)


def make_agent(
    client: ScriptedClient, prompt_files: dict[str, str], **overrides: Any
) -> LongConversationSimulationAgent:
    config = LongConversationSimulationAgentConfig(
        host="0.0.0.0",
        port=8000,
        entrypoint="app.py",
        name="long_conversation_simulation",
        assistant_model_server=ModelServerRef(type="responses_api_models", name="assistant_model"),
        user_model_server=ModelServerRef(type="responses_api_models", name="user_model"),
        helper_model_server=ModelServerRef(type="responses_api_models", name="helper_model"),
        tiktoken_encoding=None,
        batch_shortfall_allowance=0,
        **prompt_files,
        **overrides,
    )
    return LongConversationSimulationAgent.model_construct(config=config, server_client=client)


def request(path: str) -> Request:
    return Request(
        {
            "type": "http",
            "method": "POST",
            "path": path,
            "headers": [],
            "query_string": b"",
            "server": ("test", 80),
            "client": ("test", 1),
            "scheme": "http",
        }
    )


def body(*batches: list[str]) -> LongConversationRunRequest:
    return LongConversationRunRequest(
        responses_create_params={"input": []},
        topic="Topic",
        theme="Theme",
        batches=[{"messages": messages, "time_anchor": "March-03-2025", "current_plan": "P"} for messages in batches],
        source_id="row-7",
    )


def test_example_row_validates() -> None:
    row = json.loads((PACKAGE_DIR / "data" / "example.jsonl").read_text().splitlines()[0])
    assert LongConversationRunRequest.model_validate(row).batches[0].messages


@pytest.mark.asyncio
async def test_run_returns_transcript_reward_and_assistant_usage(prompt_files: dict[str, str]) -> None:
    client = ScriptedClient(
        assistant=[completion("answer one", usage(10, 3)), completion("answer two", usage(20, 4))],
        helper=[completion("Yes"), completion("no")],  # first check: assistant asked; then no follow-up
    )
    client_user = completion("user reply")
    client.helper.insert(1, client_user)  # the user model shares the non-assistant queue in this fake
    result = await make_agent(client, prompt_files).run(request("/run"), body(["q ->-> 1,1"]))

    assert result.reward == 1.0
    assert [m["content"] for m in result.chat[0]] == ["q ->-> 1,1", "answer one", "user reply", "answer two"]
    assert result.chat[0][2]["question_type"] == "answer_ai_question"
    assert result.response.output[0].content[0].text == "answer two"
    assert result.response.usage.input_tokens == 30 and result.response.usage.output_tokens == 7
    assert result.simulation.batches_completed == 1 and result.simulation.calls["assistant"] == 2
    assert result.source_id == "row-7"  # task fields survive into the rollout record
    assert {call["url_path"] for call in client.calls} == {"/v1/chat/completions"}
    assert [call["server_name"] for call in client.calls[:3]] == ["assistant_model", "helper_model", "user_model"]


@pytest.mark.asyncio
async def test_transient_errors_and_empty_output_are_retried(prompt_files: dict[str, str]) -> None:
    error = ClientResponseError(request_info=None, history=(), status=503)
    client = ScriptedClient(assistant=[error, completion(""), completion("finally")])
    result = await make_agent(client, prompt_files).run(request("/run"), body(["q ->-> 1,1"]))
    assert result.chat[0][1]["content"] == "finally"
    assert result.simulation.failed_calls == {"assistant": 2}


@pytest.mark.asyncio
async def test_partial_conversation_reward_and_stop(prompt_files: dict[str, str]) -> None:
    bad = ClientResponseError(request_info=None, history=(), status=400)
    client = ScriptedClient(assistant=[completion("fine"), bad])
    result = await make_agent(client, prompt_files).run(request("/run"), body(["a ->-> 1,1"], ["b ->-> 2,1"]))
    assert result.reward == 0.5
    assert result.simulation.stopped_at_batch == 1
    assert result.simulation.skipped_questions[0]["error"].startswith("ClientResponseError")


@pytest.mark.asyncio
async def test_run_fails_when_no_batch_completes(prompt_files: dict[str, str]) -> None:
    client = ScriptedClient(assistant=[ClientConnectionError("gone")] * 4)
    with pytest.raises(RuntimeError, match="no batch completed"):
        await make_agent(client, prompt_files).run(request("/run"), body(["a ->-> 1,1"]))


@pytest.mark.asyncio
async def test_call_timeout_counts_as_transient(prompt_files: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0
    real_wait_for = asyncio.wait_for

    async def flaky_wait_for(awaitable, timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            awaitable.close()
            raise asyncio.TimeoutError()
        return await real_wait_for(awaitable, timeout)

    monkeypatch.setattr(app_module.asyncio, "wait_for", flaky_wait_for)
    client = ScriptedClient(assistant=[completion("after timeout")])
    result = await make_agent(client, prompt_files).run(request("/run"), body(["a ->-> 1,1"]))
    assert result.chat[0][1]["content"] == "after timeout"


def test_is_transient() -> None:
    assert is_transient(asyncio.TimeoutError())
    assert is_transient(EmptyOutputError("x"))
    assert is_transient(ClientResponseError(request_info=None, history=(), status=429))
    assert not is_transient(ClientResponseError(request_info=None, history=(), status=400))
    assert not is_transient(ValueError("x"))


@pytest.mark.asyncio
async def test_public_responses_endpoint_forwards_to_the_assistant_model(prompt_files: dict[str, str]) -> None:
    payload = {
        "id": "r",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
    }
    client = ScriptedClient(assistant=[payload])
    response = Response()
    result = await make_agent(client, prompt_files).responses(
        request("/v1/responses"),
        response,
        NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hi"}]),
    )
    assert result.id == "r"
    assert client.calls[0]["server_name"] == "assistant_model" and client.calls[0]["url_path"] == "/v1/responses"


def test_config_file_declares_three_model_servers() -> None:
    text = (PACKAGE_DIR / "configs" / "long_conversation_simulation.yaml").read_text()
    for name in ("conversation_assistant_model", "conversation_user_model", "conversation_helper_model"):
        assert f"name: {name}" in text and f"{name}:\n  _copy: policy_model" in text
