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
import asyncio
import json
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import DirectHTTPToolAccess
from responses_api_agents.arc_agi_3_pivot_agent import app as app_module
from responses_api_agents.arc_agi_3_pivot_agent.app import (
    ArcAgi3PivotAgent,
    ArcAgi3PivotAgentConfig,
    PivotSessionState,
    from_gym_reply,
    to_gym_message,
)


RESOURCES = "http://resources.test"


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status: int = 200):
        self.payload, self.status, self.cookies = payload, status, {}
        self.ok = status < 400
        self.content = self

    async def read(self):
        return json.dumps(self.payload).encode()

    def raise_for_status(self):
        if self.status >= 400:
            raise RuntimeError(f"HTTP {self.status}")


class Servers:
    """The pivot resources server (through the session's direct HTTP grant) and the policy model server: a window
    that ends on the second step."""

    def __init__(
        self, restore_error=None, model_fails=False, token_ids=False, harness_requests=False, step_fails=False
    ):
        self.calls: list[tuple] = []
        self.restore_error, self.model_fails, self.token_ids = restore_error, model_fails, token_ids
        self.step_fails = step_fails
        # /step returns the harness's next request (its history keeps the reply's reasoning as reasoning_content)
        self.harness_requests = harness_requests
        self.sent: list[dict[str, Any]] = []
        self.steps = 0
        self.model_pending = asyncio.Event()
        self.model_blocks = False

    async def resources(self, method: str, url: str, json: Any = None, cookies=None, **_):
        assert method == "POST" and url.startswith(RESOURCES) and cookies == {"session": "resources-1"}
        path = url.removeprefix(RESOURCES)
        self.calls.append(("resources", path, json))
        if path == "/window":
            if self.restore_error:
                return FakeResponse({"messages": [], "restore_error": self.restore_error})
            return FakeResponse(
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {
                            "role": "assistant",
                            "content": "",
                            "reasoning_content": "old thought",
                            "tool_calls": [
                                {
                                    "id": "r1",
                                    "type": "function",
                                    "function": {"name": "python", "arguments": "{}", "index": 0},
                                }
                            ],
                        },
                        {"role": "user", "content": [{"type": "text", "text": "board"}]},
                    ],
                    "tools": [{"type": "function", "function": {"name": "python"}}],
                    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True},
                }
            )
        if path == "/step":
            if self.step_fails:
                return FakeResponse({"detail": "worker gone"}, status=500)
            self.steps += 1
            done = self.steps >= 2
            messages = [{"role": "tool", "tool_call_id": json["message"]["tool_calls"][0]["id"], "content": "ok"}]
            if not done:
                messages.append({"role": "user", "content": "opener"})
            body = {"messages": messages, "done": done, "call": {}}
            if self.harness_requests and not done:
                if not self.sent:
                    self.sent = [{"role": "system", "content": "sys"}, {"role": "user", "content": "board"}]
                self.sent = self.sent + [dict(json["message"], harness_note=True)] + messages
                body["request_messages"] = self.sent
            return FakeResponse(body)
        raise AssertionError(path)

    async def post(self, server_name: str, url_path: str, json: Any = None, cookies=None, **_):
        assert server_name == "policy_model" and url_path.endswith("/v1/chat/completions")
        self.calls.append(("model", url_path, json))
        if self.model_blocks:
            self.model_pending.set()
            await asyncio.Event().wait()
        if self.model_fails:
            return FakeResponse({"error": "boom"}, status=500)
        n = sum(1 for c in self.calls if c[0] == "model")
        message = {
            "role": "assistant",
            "content": f"<think>plan {n}</think>",
            "tool_calls": [
                {"id": f"c{n}", "type": "function", "function": {"name": "python", "arguments": '{"code": "x"}'}}
            ],
        }
        if self.token_ids:  # call 2's prompt keeps call 1's prompt + reply except its last generated token
            message["prompt_token_ids"] = [1, 2, 3] if n == 1 else [1, 2, 3, 7, 9, 9]
            message["generation_token_ids"] = [7, 8]
        return FakeResponse(
            {
                "id": f"chatcmpl-{n}",
                "choices": [{"message": message, "finish_reason": "tool_calls"}],
                "usage": {"completion_tokens": 11, "prompt_tokens": 900},
            }
        )

    def paths(self) -> list[str]:
        return [c[1].removeprefix("/v1/chat") if c[0] == "model" else c[1] for c in self.calls]


@pytest.fixture
def servers(monkeypatch) -> Servers:
    servers = Servers()
    monkeypatch.setattr(app_module, "http_request", servers.resources)
    return servers


def make_agent(servers: Servers) -> ArcAgi3PivotAgent:
    config = ArcAgi3PivotAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="arc_agi_3_pivot_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="pivot"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
    )
    agent = ArcAgi3PivotAgent(config=config, server_client=MagicMock(spec=ServerClient))
    agent.server_client.post = servers.post
    return agent


def seed_body(session_id: str = "agent-1", grants: bool = True) -> AgentSeedSessionRequest:
    grant = DirectHTTPToolAccess(
        name="arc_agi_3_pivot_resources_server.direct_http",
        required=True,
        base_url=RESOURCES,
        cookies={"session": "resources-1"},
    )
    return AgentSeedSessionRequest(
        agent_session_id=session_id,
        episode_id=EpisodeId(rollout_id="rollout-1", attempt=0),
        task_id=TaskId(taskset="arc_agi_3_pivot", task_id="p1"),
        tool_accesses=[grant] if grants else [],
    )


def play(servers: Servers) -> dict[str, Any]:
    """Seed an agent session, run its activation, close it (as the environment server does); the activation's
    response."""
    client = TestClient(make_agent(servers).setup_webserver())
    body = seed_body()
    assert client.post("/v1/agent_sessions", json=body.model_dump(mode="json")).status_code == 200
    response = client.post("/v1/responses", json={"input": [{"role": "user", "content": "x"}]})
    assert response.status_code == 200, response.text
    close = {"agent_session_id": body.agent_session_id, "episode_id": body.episode_id.model_dump(mode="json")}
    assert client.post("/v1/agent_sessions/close", json=close).status_code == 200
    return response.json()


def test_window_loop_until_done(servers):
    result = play(servers)
    assert servers.paths() == ["/window", "/completions", "/step", "/completions", "/step"]

    first = servers.calls[1][2]
    # History reasoning goes as a <think> prefix; top_k and the template kwargs through metadata.
    assert first["messages"][1]["content"] == "<think>old thought</think>"
    # Recorded engine extras (a streaming index) are dropped; the Gym model server refuses them.
    assert first["messages"][1]["tool_calls"] == [
        {"id": "r1", "type": "function", "function": {"name": "python", "arguments": "{}"}}
    ]
    assert json.loads(first["metadata"]["extra_body"]) == {"top_k": 20}
    assert json.loads(first["metadata"]["chat_template_kwargs"]) == {
        "enable_thinking": True,
        "preserve_thinking": True,
    }
    assert first["temperature"] == 1.0 and first["max_tokens"] == 12288 and first["tool_choice"] == "auto"

    step = servers.calls[2][2]
    assert step["message"]["reasoning_content"] == "plan 1" and step["message"]["content"] == ""
    assert step["gen_tokens"] == 11 and step["prompt_tokens"] == 900 and step["finish_reason"] == "tool_calls"

    # The second call extends the first: reply, tool result, opener appended in order.
    second = servers.calls[3][2]["messages"]
    assert [m["role"] for m in second[3:]] == ["assistant", "tool", "user"]
    assert second[3]["content"] == "<think>plan 1</think>"

    assert result["metadata"] is None
    # the last served model response id, so token capture attributes the window to its last call
    assert result["id"] == "chatcmpl-2"
    # the output is every call's served items, as token capture records them (call by call, in order)
    output = result["output"]
    assert [i["type"] for i in output] == ["message", "function_call", "message", "function_call"]
    assert [i["call_id"] for i in output if i["type"] == "function_call"] == ["c1", "c2"]
    assert output[0]["content"][0]["text"] == "<think>plan 1</think>"


def test_next_call_sends_the_harness_request(servers):
    servers.harness_requests = True
    play(servers)
    second = servers.calls[3][2]["messages"]
    # the harness's request as it sent it (its own rendering of the reply), in the Gym chat format
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "user"]
    assert second[2]["harness_note"] is True and second[2]["content"] == "<think>plan 1</think>"


def test_cached_tokens_are_measured_on_token_ids(servers):
    servers.token_ids = True
    play(servers)
    steps = [c[2] for c in servers.calls if c[1] == "/step"]
    assert [s["cached_tokens"] for s in steps] == [None, 4]  # first call: the server's rule; then LCP with [1,2,3,7,8]
    assert "prompt_token_ids" not in steps[0]["message"]


def test_without_token_ids_the_server_derives_cached_tokens(servers):
    play(servers)
    assert all(c[2]["cached_tokens"] is None for c in servers.calls if c[1] == "/step")


def test_restore_error_makes_no_model_call(servers):
    servers.restore_error = "RestoreError: x"
    result = play(servers)
    assert servers.paths() == ["/window"] and result["output"] == []


def test_model_failure_is_reported_not_stepped(servers):
    servers.model_fails = True
    result = play(servers)
    assert servers.paths() == ["/window", "/completions"]
    assert "HTTP 500" in result["metadata"]["agent_error"]


def test_a_seed_without_a_direct_http_grant_is_refused(servers):
    client = TestClient(make_agent(servers).setup_webserver())
    response = client.post("/v1/agent_sessions", json=seed_body(grants=False).model_dump(mode="json"))
    assert response.status_code == 422 and "direct HTTP grant" in response.text


async def test_closing_the_session_cancels_a_pending_activation(servers):
    """The environment server closes the agent session when the episode ends early (its activation was cancelled or
    timed out): the window loop stops with no further model call or step."""
    servers.model_blocks = True
    agent = make_agent(servers)
    state = await agent._seed_agent_session_state(seed_body())
    assert isinstance(state, PivotSessionState)
    state.execution = asyncio.create_task(agent._play(state, "/v1/chat/completions"))
    await asyncio.wait_for(servers.model_pending.wait(), 5)
    closed = await agent._close_agent_session_state(state)
    assert closed.agent_session_id == "agent-1" and state.execution.cancelled()
    assert servers.paths() == ["/window", "/completions"]


async def test_a_failed_step_fails_the_activation(servers):
    """A /step failure ends the activation with an error; the environment server then closes both sessions, which
    releases the window."""
    servers.step_fails = True
    agent = make_agent(servers)
    state = await agent._seed_agent_session_state(seed_body())
    with pytest.raises(RuntimeError, match="HTTP 500"):
        await agent._play(state, "/v1/chat/completions")
    assert servers.paths() == ["/window", "/completions", "/step"]


@pytest.mark.parametrize(
    "message,expected",
    [
        ({"content": "<think>r</think>answer"}, ("r", "answer")),
        ({"content": "answer", "reasoning_content": "r"}, ("r", "answer")),
        ({"content": "answer"}, (None, "answer")),
    ],
)
def test_reasoning_split(message, expected):
    out = from_gym_reply(message)
    assert (out.get("reasoning_content"), out["content"]) == expected
    if expected[0]:
        assert to_gym_message(out)["content"] == f"<think>{expected[0]}</think>{expected[1]}"


@pytest.mark.parametrize(
    "message,content",
    [
        ({"reasoning": "plan"}, "<think>plan</think>answer"),  # the alias alone
        ({"reasoning_content": "plan", "reasoning": "other"}, "<think>plan</think>answer"),
        ({"reasoning_content": "", "reasoning": "plan"}, "<think>plan</think>answer"),  # empty: the alias
        ({}, "answer"),
    ],
)
def test_history_reasoning_aliases(message, content):
    out = to_gym_message({"role": "assistant", "content": "answer", **message})
    assert out["content"] == content and "reasoning" not in out and "reasoning_content" not in out
