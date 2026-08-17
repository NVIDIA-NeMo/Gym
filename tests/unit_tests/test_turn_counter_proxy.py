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

from nemo_gym.adapters.turn_counter_proxy import (
    inject_turn_reminder,
    resolve_reminder_trigger,
    start_turn_counter_proxy,
)


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


@pytest.mark.parametrize(
    ("max_turns", "expected"),
    [
        (1, "per_turn"),
        (4, "per_turn"),  # warn point is turn 4 of 4: too late to act on
        (5, "per_turn"),
        (10, "threshold"),  # warn point is turn 8 of 10: two turns left to wrap up
        (200, "threshold"),
    ],
    ids=["1", "4", "5", "10", "200"],
)
def test_auto_picks_per_turn_reminders_only_for_small_budgets(max_turns, expected):
    assert resolve_reminder_trigger("auto", max_turns) == expected


@pytest.mark.parametrize("trigger", ["threshold", "per_turn"])
def test_explicit_trigger_overrides_the_budget_heuristic(trigger):
    assert resolve_reminder_trigger(trigger, 4) == trigger


def test_invalid_trigger_is_rejected():
    with pytest.raises(ValueError, match="invalid trigger"):
        resolve_reminder_trigger("periodic", 10)


def test_per_turn_reminds_every_turn_and_escalates_on_the_last():
    contents = []
    for n in (1, 2, 3, 4):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        inject_turn_reminder(body, n=n, max_turns=4, position="system_message", trigger="per_turn")
        assert len(body["messages"]) == 2, f"turn {n} got no reminder"
        contents.append(body["messages"][-1]["content"])

    assert "3 turn(s) left" in contents[0]
    assert "1 turn(s) left" in contents[2]
    assert "URGENT" in contents[3] and "final answer NOW" in contents[3]


def test_threshold_trigger_stays_silent_early_even_on_a_small_budget():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    inject_turn_reminder(body, n=1, max_turns=4, position="system_message", trigger="threshold")
    assert len(body["messages"]) == 1


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
async def test_proxy_logs_each_turn_and_the_rejection(caplog):
    """A run must be auditable from the Gym logs: which task, which turn, which cap."""
    upstream_runner, upstream_site, upstream_url, _hits = await _start_upstream()
    with caplog.at_level("INFO", logger="nemo_gym.adapters.turn_counter_proxy"):
        proxy = await start_turn_counter_proxy(
            upstream_base_url=upstream_url,
            api_key="sk-test",
            max_turns=1,
            label="task_42",
        )
        try:
            async with ClientSession() as client:
                for _ in range(2):
                    async with client.post(
                        f"{proxy.base_url}/chat/completions",
                        json={"model": "m", "messages": [{"role": "user", "content": "hi"}]},
                    ) as resp:
                        await resp.read()
        finally:
            await proxy.stop()
            await _stop_upstream(upstream_runner, upstream_site)

    messages = [rec.getMessage() for rec in caplog.records]
    assert any("task_42: enforcing max_turns=1" in m for m in messages)
    assert any("task_42: turn 1/1" in m for m in messages)
    assert any("task_42: REJECTED turn 2" in m for m in messages)
    assert proxy.max_turns == 1 and proxy.label == "task_42"


@pytest.mark.asyncio
async def test_proxy_injects_threshold_reminder_into_forwarded_body():
    upstream_runner, upstream_site, upstream_url, hits = await _start_upstream()
    proxy = await start_turn_counter_proxy(
        upstream_base_url=upstream_url,
        api_key="sk-test",
        max_turns=5,
        position="system_message",
        trigger="threshold",
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
