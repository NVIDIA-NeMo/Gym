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
"""Tests for the sandbox-bound capture interceptor, store, and trajectory assembly."""

import asyncio

from nemo_gym.adapters.capture_store import CaptureStore, assemble_trajectory, has_token_ids
from nemo_gym.adapters.pipeline import AdapterPipeline
from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import AdapterRequest, AdapterResponse, InterceptorContext


def test_capture_registered():
    assert "capture" in InterceptorRegistry.available()


def test_store_roundtrip_is_session_keyed(tmp_path):
    store = CaptureStore(tmp_path)
    store.record("rollout-a", {"request_id": "1", "response": {"choices": []}})
    store.record("rollout-a", {"request_id": "2", "response": {"choices": []}})
    store.record("rollout-b", {"request_id": "3", "response": {"choices": []}})

    assert len(store.read("rollout-a")) == 2
    assert len(store.read("rollout-b")) == 1
    assert store.read("missing") == []
    # one file per session, so a reaped box cannot lose another box's turns
    assert store.path_for("rollout-a") != store.path_for("rollout-b")
    assert store.path_for("rollout-a").exists()


def _fake_upstream(response_body):
    async def _upstream(req: AdapterRequest) -> AdapterResponse:
        # echo what the agent would have sent so the test can assert injection
        req.ctx.extra["_seen_request_body"] = dict(req.body)
        return AdapterResponse(status_code=200, headers={}, body=response_body, latency_ms=4.2, ctx=req.ctx)

    return _upstream


def test_capture_records_exchange_and_injects_flag(tmp_path):
    interceptor = InterceptorRegistry.create(
        "capture",
        {
            "store_dir": str(tmp_path),
            "session_id": "rollout-1",
            "inject_extra_body": {"return_token_id_information": True},
        },
    )
    pipeline = AdapterPipeline([interceptor])

    response_body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi", "generation_token_ids": [5, 6, 7]},
                "prompt_token_ids": [1, 2, 3],
            }
        ]
    }
    req = AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={},
        body={"model": "policy", "messages": [{"role": "user", "content": "hello"}]},
        ctx=InterceptorContext(),
    )

    resp = asyncio.run(pipeline.process(req, upstream_call=_fake_upstream(response_body)))
    assert resp.status_code == 200
    # the token-id flag was injected into the upstream request
    assert resp.ctx.extra["_seen_request_body"]["return_token_id_information"] is True

    exchanges = CaptureStore(tmp_path).read("rollout-1")
    assert len(exchanges) == 1
    recorded = exchanges[0]
    assert recorded["session_id"] == "rollout-1"
    assert recorded["status"] == 200
    assert recorded["response"]["choices"][0]["message"]["generation_token_ids"] == [5, 6, 7]
    assert recorded["request"]["return_token_id_information"] is True


def test_assemble_trajectory_interleaves_tools_and_keeps_token_ids():
    exchanges = [
        {
            "request": {"messages": [{"role": "user", "content": "fix the bug"}]},
            "response": {
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "let me look",
                            "generation_token_ids": [10, 11],
                            "tool_calls": [
                                {"id": "call_1", "function": {"name": "bash", "arguments": "{\"cmd\":\"ls\"}"}}
                            ],
                        },
                        "prompt_token_ids": [1, 2],
                    }
                ]
            },
        },
        {
            "request": {
                "messages": [
                    {"role": "user", "content": "fix the bug"},
                    {"role": "assistant", "content": "let me look"},
                    {"role": "tool", "tool_call_id": "call_1", "content": "file.py"},
                ]
            },
            "response": {
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done", "generation_token_ids": [20, 21, 22]},
                        "prompt_token_ids": [3, 4, 5],
                    }
                ]
            },
        },
    ]

    items = assemble_trajectory(exchanges)
    kinds = [item.type for item in items]
    assert kinds == ["message", "function_call", "function_call_output", "message"]

    first_assistant = items[0]
    assert first_assistant.generation_token_ids == [10, 11]
    assert first_assistant.prompt_token_ids == [1, 2]
    assert items[1].name == "bash"
    assert items[2].call_id == "call_1"
    assert items[2].output == "file.py"
    assert items[3].generation_token_ids == [20, 21, 22]
    assert has_token_ids(items)


def test_assemble_trajectory_tolerates_empty_and_malformed():
    assert assemble_trajectory([]) == []
    assert assemble_trajectory([{"request": {}, "response": {}}]) == []
    assert assemble_trajectory([{"response": {"choices": []}}]) == []


def test_assemble_trajectory_responses_wire_interleaves_tool_outputs():
    exchanges = [
        {
            "request": {"input": "fix it"},
            "response": {
                "output": [
                    {"type": "reasoning", "content": []},
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": "on it"}],
                        "generation_token_ids": [9, 8, 7],
                    },
                    {"type": "function_call", "id": "c1", "call_id": "c1", "name": "shell", "arguments": "{\"cmd\":\"ls\"}"},
                ]
            },
        },
        {
            # the tool result of c1 arrives in the next request's input
            "request": {
                "input": [
                    {"type": "function_call_output", "call_id": "c1", "output": "file.py"},
                ]
            },
            "response": {
                "output": [
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "done"}]},
                ]
            },
        },
    ]
    items = assemble_trajectory(exchanges, wire="responses")
    assert [it.type for it in items] == ["message", "function_call", "function_call_output", "message"]
    assert items[0].generation_token_ids == [9, 8, 7]
    assert items[1].name == "shell"
    assert items[2].call_id == "c1"
    assert items[2].output == "file.py"
