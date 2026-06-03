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
"""Per-interceptor behavior tests — ported from NEL ``tests/test_adapters/test_interceptors.py``
and ``test_interceptors_extended.py``.

Mechanical port from ``nemo_evaluator.adapters.*`` to ``nemo_gym.adapters.*``;
``GracefulError`` re-rooted at ``nemo_gym.adapters.types``.
"""

import pytest

from nemo_gym.adapters.registry import InterceptorRegistry
from nemo_gym.adapters.types import (
    AdapterRequest,
    AdapterResponse,
    GracefulError,
    InterceptorContext,
)


def _req(body=None, **kw):
    return AdapterRequest(
        method="POST",
        path="/v1/chat/completions",
        headers={"content-type": "application/json"},
        body=body or {"model": "test", "messages": [{"role": "user", "content": "hi"}]},
        ctx=InterceptorContext(),
    )


def _resp(body=None, status_code=200):
    return AdapterResponse(
        status_code=status_code,
        headers={},
        body=body or {},
        ctx=InterceptorContext(),
    )


async def test_drop_params():
    ic = InterceptorRegistry.create("drop_params", {"params": ["temperature", "top_p"]})
    req = _req({"model": "test", "messages": [], "temperature": 0.7, "top_p": 0.9, "max_tokens": 100})
    result = await ic.intercept_request(req)
    assert "temperature" not in result.body
    assert "top_p" not in result.body
    assert result.body["max_tokens"] == 100


async def test_modify_tools():
    ic = InterceptorRegistry.create("modify_tools", {"strip_properties": ["x"]})
    req = _req(
        {
            "model": "test",
            "messages": [],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "f",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "x": {"type": "string"},
                                "y": {"type": "integer"},
                            },
                            "required": ["x", "y"],
                        },
                    },
                }
            ],
        }
    )
    result = await ic.intercept_request(req)
    props = result.body["tools"][0]["function"]["parameters"]["properties"]
    assert "x" not in props
    assert "y" in props
    assert "x" not in result.body["tools"][0]["function"]["parameters"]["required"]


async def test_system_message_prepend():
    ic = InterceptorRegistry.create(
        "system_message",
        {
            "system_message": "Be helpful",
            "strategy": "prepend",
        },
    )
    req = _req({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
    result = await ic.intercept_request(req)
    assert result.body["messages"][0] == {"role": "system", "content": "Be helpful"}
    assert result.body["messages"][1] == {"role": "user", "content": "hi"}


async def test_system_message_replace():
    ic = InterceptorRegistry.create(
        "system_message",
        {
            "system_message": "New system",
            "strategy": "replace",
        },
    )
    req = _req(
        {
            "model": "test",
            "messages": [
                {"role": "system", "content": "Old system"},
                {"role": "user", "content": "hi"},
            ],
        }
    )
    result = await ic.intercept_request(req)
    sys_msgs = [m for m in result.body["messages"] if m["role"] == "system"]
    assert len(sys_msgs) == 1
    assert sys_msgs[0]["content"] == "New system"


async def test_payload_modifier_remove():
    ic = InterceptorRegistry.create("payload_modifier", {"params_to_remove": ["stream"]})
    req = _req({"model": "test", "messages": [], "stream": True})
    result = await ic.intercept_request(req)
    assert "stream" not in result.body


async def test_payload_modifier_add():
    ic = InterceptorRegistry.create("payload_modifier", {"params_to_add": {"temperature": 0.5}})
    req = _req({"model": "test", "messages": []})
    result = await ic.intercept_request(req)
    assert result.body["temperature"] == 0.5


async def test_turn_counter_basic():
    ic = InterceptorRegistry.create("turn_counter", {"max_turns": 5})
    for _ in range(5):
        req = _req({"model": "test", "messages": [{"role": "user", "content": "hi"}]})
        await ic.intercept_request(req)

    with pytest.raises(GracefulError, match="Turn budget exhausted"):
        await ic.intercept_request(_req({"model": "test", "messages": [{"role": "user", "content": "hi"}]}))


async def test_turn_counter_session_isolation():
    """Repeats of the same problem get independent turn budgets when
    the proxy injects distinct session_id values."""
    ic = InterceptorRegistry.create("turn_counter", {"max_turns": 3})
    body = {"model": "test", "messages": [{"role": "user", "content": "same prompt"}]}

    for session_id in ("aaa", "bbb"):
        for _ in range(3):
            r = _req(body)
            r.ctx.extra["session_id"] = session_id
            await ic.intercept_request(r)

    r = _req(body)
    r.ctx.extra["session_id"] = "aaa"
    with pytest.raises(GracefulError, match="Turn budget exhausted"):
        await ic.intercept_request(r)

    r = _req(body)
    r.ctx.extra["session_id"] = "ccc"
    await ic.intercept_request(r)
