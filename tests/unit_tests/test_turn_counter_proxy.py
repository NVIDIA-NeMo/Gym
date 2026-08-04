# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tests for the minimal turn-counter proxy."""

from __future__ import annotations

import json

import pytest
from aiohttp import ClientSession, web

from nemo_gym.adapters.turn_counter_proxy import inject_turn_reminder, start_turn_counter_proxy


async def _start_upstream() -> tuple[web.AppRunner, web.TCPSite, str, dict]:
    hits = {"n": 0, "bodies": []}

    async def chat(request: web.Request) -> web.Response:
        hits["n"] += 1
        hits["bodies"].append(await request.json())
        return web.json_response(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }
        )

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    return runner, site, f"http://127.0.0.1:{port}/v1", hits


async def _stop_upstream(runner: web.AppRunner, site: web.TCPSite) -> None:
    await site.stop()
    await runner.cleanup()


def test_inject_threshold_system_message_at_warn_and_urgent():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    inject_turn_reminder(body, n=7, max_turns=10, position="system_message")
    assert len(body["messages"]) == 1  # 70% < 80%: no injection

    inject_turn_reminder(body, n=8, max_turns=10, position="system_message")
    assert body["messages"][-1]["role"] == "system"
    assert "Begin wrapping up" in body["messages"][-1]["content"]
    assert body["messages"][-1]["content"].startswith("[SYSTEM]")

    body = {"messages": [{"role": "user", "content": "hi"}]}
    inject_turn_reminder(body, n=10, max_turns=10, position="system_message")
    assert "URGENT" in body["messages"][-1]["content"]
    assert "final answer NOW" in body["messages"][-1]["content"]


def test_inject_threshold_user_message_appends_without_system_prefix():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    inject_turn_reminder(body, n=9, max_turns=10, position="user_message")
    assert len(body["messages"]) == 1
    assert body["messages"][0]["content"].startswith("hi\n\n")
    assert "Begin wrapping up" in body["messages"][0]["content"]  # 90% → warn, not yet urgent
    assert "[SYSTEM]" not in body["messages"][0]["content"]


@pytest.mark.asyncio
async def test_proxy_allows_up_to_max_turns_then_rejects():
    upstream_runner, upstream_site, upstream_url, hits = await _start_upstream()
    proxy = await start_turn_counter_proxy(
        upstream_base_url=upstream_url,
        api_key="sk-test",
        max_turns=2,
    )
    try:
        async with ClientSession() as client:
            for _ in range(2):
                async with client.post(
                    f"{proxy.base_url}/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                ) as resp:
                    assert resp.status == 200
                    body = await resp.json()
                    assert body["choices"][0]["message"]["content"] == "ok"

            assert proxy.turns_used == 2
            assert hits["n"] == 2

            async with client.post(
                f"{proxy.base_url}/chat/completions",
                json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
            ) as resp:
                assert resp.status == 429
                err = await resp.json()
                assert err["error"]["code"] == "session_budget_exhausted"

            assert proxy.turns_used == 3
            assert hits["n"] == 2  # rejected before upstream
    finally:
        await proxy.stop()
        await _stop_upstream(upstream_runner, upstream_site)


@pytest.mark.asyncio
async def test_proxy_injects_threshold_reminder_into_forwarded_body():
    upstream_runner, upstream_site, upstream_url, hits = await _start_upstream()
    proxy = await start_turn_counter_proxy(
        upstream_base_url=upstream_url,
        api_key="sk-test",
        max_turns=5,
        position="system_message",
    )
    try:
        async with ClientSession() as client:
            # turn 4/5 = 80% → warn reminder
            for _ in range(4):
                async with client.post(
                    f"{proxy.base_url}/chat/completions",
                    json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                ) as resp:
                    assert resp.status == 200

        assert len(hits["bodies"][0]["messages"]) == 1  # turn 1: no reminder
        assert hits["bodies"][3]["messages"][-1]["role"] == "system"
        assert "Begin wrapping up" in hits["bodies"][3]["messages"][-1]["content"]
    finally:
        await proxy.stop()
        await _stop_upstream(upstream_runner, upstream_site)


@pytest.mark.asyncio
async def test_proxy_forwards_authorization_when_missing():
    seen = {"auth": None}

    async def chat(request: web.Request) -> web.Response:
        seen["auth"] = request.headers.get("Authorization")
        await request.read()
        return web.json_response({"ok": True})

    app = web.Application()
    app.router.add_post("/v1/chat/completions", chat)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]  # noqa: SLF001
    upstream_url = f"http://127.0.0.1:{port}/v1"

    proxy = await start_turn_counter_proxy(
        upstream_base_url=upstream_url,
        api_key="sk-injected",
        max_turns=5,
    )
    try:
        async with ClientSession() as client:
            async with client.post(
                f"{proxy.base_url}/chat/completions",
                data=json.dumps({"model": "m"}),
                headers={"Content-Type": "application/json"},
            ) as resp:
                assert resp.status == 200
        assert seen["auth"] == "Bearer sk-injected"
    finally:
        await proxy.stop()
        await site.stop()
        await runner.cleanup()


@pytest.mark.asyncio
async def test_start_rejects_invalid_max_turns():
    with pytest.raises(ValueError, match="max_turns"):
        await start_turn_counter_proxy(
            upstream_base_url="http://127.0.0.1:9/v1",
            api_key="sk",
            max_turns=0,
        )
