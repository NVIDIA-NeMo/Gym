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
import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from responses_api_agents.arc_agi_3_pivot_agent.app import (
    ArcAgi3PivotAgent,
    ArcAgi3PivotAgentConfig,
    ArcAgi3PivotRunRequest,
    from_gym_reply,
    to_gym_message,
)


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
    """The pivot resources server and the policy model server: a window that ends on the second step."""

    def __init__(self, restore_error=None, model_fails=False, token_ids=False, harness_requests=False):
        self.calls: list[tuple] = []
        self.restore_error, self.model_fails, self.token_ids = restore_error, model_fails, token_ids
        # /step returns the harness's next request (its history keeps the reply's reasoning as reasoning_content)
        self.harness_requests = harness_requests
        self.sent: list[dict[str, Any]] = []
        self.steps = 0

    async def post(self, server_name: str, url_path: str, json: Any = None, cookies=None, **_):
        self.calls.append((server_name, url_path, json))
        if url_path == "/seed_session":
            if self.restore_error:
                return FakeResponse({"restore_error": self.restore_error})
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
        if url_path.endswith("/v1/chat/completions"):
            if self.model_fails:
                return FakeResponse({"error": "boom"}, status=500)
            n = sum(1 for c in self.calls if c[1].endswith("/chat/completions"))
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
        if url_path == "/step":
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
        if url_path == "/verify":
            return FakeResponse(json | {"reward": 2.0})
        raise AssertionError(url_path)


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


def run_body() -> ArcAgi3PivotRunRequest:
    return ArcAgi3PivotRunRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "x"}]},
            "verifier_metadata": {"pivot_id": "p1", "pivot": {"pivot_json": "{}", "harness": "30d2cc8b"}},
        }
    )


def request() -> MagicMock:
    r = MagicMock()
    r.cookies = {}
    return r


async def test_window_loop_until_done():
    servers = Servers()
    result = await make_agent(servers).run(request(), run_body())
    paths = [c[1] for c in servers.calls]
    assert paths == ["/seed_session", "/v1/chat/completions", "/step", "/v1/chat/completions", "/step", "/verify"]

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

    verify = servers.calls[-1][2]
    assert verify["agent_error"] is None
    # the last served model response id, so token capture attributes the window to its last call
    assert verify["response"]["id"] == "chatcmpl-2"
    # the output is every call's served items, as token capture records them (call by call, in order)
    output = verify["response"]["output"]
    assert [i["type"] for i in output] == ["message", "function_call", "message", "function_call"]
    assert [i["call_id"] for i in output if i["type"] == "function_call"] == ["c1", "c2"]
    assert output[0]["content"][0]["text"] == "<think>plan 1</think>"
    assert result.reward == 2.0


async def test_next_call_sends_the_harness_request():
    servers = Servers(harness_requests=True)
    await make_agent(servers).run(request(), run_body())
    second = servers.calls[3][2]["messages"]
    # the harness's request as it sent it (its own rendering of the reply), in the Gym chat format
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool", "user"]
    assert second[2]["harness_note"] is True and second[2]["content"] == "<think>plan 1</think>"


async def test_cached_tokens_are_measured_on_token_ids():
    servers = Servers(token_ids=True)
    await make_agent(servers).run(request(), run_body())
    steps = [c[2] for c in servers.calls if c[1] == "/step"]
    assert [s["cached_tokens"] for s in steps] == [None, 4]  # first call: the server's rule; then LCP with [1,2,3,7,8]
    assert "prompt_token_ids" not in steps[0]["message"]
    # Without token ids the server derives it.
    servers = Servers()
    await make_agent(servers).run(request(), run_body())
    assert all(c[2]["cached_tokens"] is None for c in servers.calls if c[1] == "/step")


async def test_restore_error_goes_straight_to_verify():
    servers = Servers(restore_error="RestoreError: x")
    await make_agent(servers).run(request(), run_body())
    assert [c[1] for c in servers.calls] == ["/seed_session", "/verify"]


async def test_model_failure_is_reported_not_stepped():
    servers = Servers(model_fails=True)
    await make_agent(servers).run(request(), run_body())
    assert [c[1] for c in servers.calls] == ["/seed_session", "/v1/chat/completions", "/verify"]
    assert "HTTP 500" in servers.calls[-1][2]["agent_error"]


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
