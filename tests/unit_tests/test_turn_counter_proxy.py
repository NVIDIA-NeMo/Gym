# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

import asyncio
import json

import pytest
from aiohttp import ClientSession, web
from pydantic import ValidationError

from nemo_gym.adapters.turn_counter_proxy import (
    TurnConstraintConfig,
    inject_turn_reminder,
    resolve_reminder_trigger,
    start_turn_counter_proxy,
)


async def _start_upstream(delay: float = 0) -> tuple[web.AppRunner, web.TCPSite, str, dict]:
    hits = {"n": 0, "bodies": []}

    async def chat(request: web.Request) -> web.Response:
        hits["n"] += 1
        hits["bodies"].append(await request.json())
        if delay:
            await asyncio.sleep(delay)
        return web.json_response(
            {
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            }
        )

    app = web.Application(client_max_size=128 * 1024 * 1024)
    app.router.add_post("/v1/chat/completions", chat)
    app.router.add_post("/v1/responses", chat)
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


def test_canonical_proxy_constraint_defaults_to_supported_session_capabilities():
    constraint = TurnConstraintConfig.model_validate({"enforcement": "proxy", "limit": 5})

    assert constraint.scope == "session"
    assert constraint.reminder.trigger == "auto"
    assert constraint.reminder.position == "system_message"


@pytest.mark.parametrize(
    "constraint",
    [
        {"enforcement": "native", "limit": 5},
        {"enforcement": "proxy", "limit": 0},
        {"enforcement": "proxy", "limit": 5, "scope": "subagent"},
        {"enforcement": "proxy", "limit": 5, "exclude_compaction": True},
    ],
)
def test_canonical_proxy_constraint_rejects_unsupported_capabilities(constraint):
    with pytest.raises(ValidationError):
        TurnConstraintConfig.model_validate(constraint)


def test_per_turn_reminds_every_turn_and_escalates_on_the_last():
    contents = []
    for n in (1, 2, 3, 4):
        body = {"messages": [{"role": "user", "content": "hi"}]}
        inject_turn_reminder(body, n=n, max_turns=4, position="system_message", trigger="per_turn")
        assert len(body["messages"]) == 2, f"turn {n} got no reminder"
        assert body["messages"][0]["role"] == "system"
        contents.append(body["messages"][0]["content"])

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
    assert body["messages"][0]["role"] == "system"
    assert "Begin wrapping up" in body["messages"][0]["content"]
    assert body["messages"][0]["content"].startswith("[SYSTEM]")

    body = {"messages": [{"role": "user", "content": "hi"}]}
    inject_turn_reminder(body, n=10, max_turns=10, position="system_message")
    assert "URGENT" in body["messages"][0]["content"]
    assert "final answer NOW" in body["messages"][0]["content"]


def test_system_reminder_merges_with_existing_first_system_message():
    body = {
        "messages": [
            {"role": "system", "content": "Original instructions"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    }
    inject_turn_reminder(body, n=1, max_turns=2, trigger="per_turn")
    assert [message["role"] for message in body["messages"]] == ["system", "user", "assistant"]
    assert body["messages"][0]["content"].startswith("Original instructions\n\n[SYSTEM]")
    assert body["messages"][1]["content"] == "hi"


def test_system_reminder_merges_typed_responses_content():
    body = {
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": "Original instructions"}]},
            {"role": "user", "content": "hi"},
        ]
    }
    inject_turn_reminder(body, n=1, max_turns=2, trigger="per_turn")
    assert [message["role"] for message in body["input"]] == ["system", "user"]
    assert body["input"][0]["content"][-1]["type"] == "input_text"
    assert body["input"][0]["content"][-1]["text"].startswith("[SYSTEM]")


def test_inject_threshold_user_message_appends_without_system_prefix():
    body = {"messages": [{"role": "user", "content": "hi"}]}
    inject_turn_reminder(body, n=9, max_turns=10, position="user_message")
    assert len(body["messages"]) == 1
    assert body["messages"][0]["content"].startswith("hi\n\n")
    assert "Begin wrapping up" in body["messages"][0]["content"]  # 90% → warn, not yet urgent
    assert "[SYSTEM]" not in body["messages"][0]["content"]


@pytest.mark.asyncio
async def test_proxy_does_not_apply_aiohttp_default_deadline_to_policy_requests(monkeypatch):
    from nemo_gym.adapters import turn_counter_proxy as proxy_module

    # The shared transport receives an explicit unlimited generation timeout.
    original_request = proxy_module.upstream_request

    async def check_timeout(*args, **kwargs):
        assert kwargs["timeout"].total is None
        return await original_request(*args, **kwargs)

    monkeypatch.setattr(proxy_module, "upstream_request", check_timeout)
    runner, site, upstream_url, hits = await _start_upstream(delay=0.1)
    proxy = await start_turn_counter_proxy(upstream_base_url=upstream_url, api_key="test", max_turns=1)
    try:
        async with ClientSession() as client:
            async with client.post(f"{proxy.base_url}/chat/completions", json={"messages": []}) as response:
                assert response.status == 200
                assert (await response.json())["choices"][0]["message"]["content"] == "ok"
            async with client.post(f"{proxy.base_url}/chat/completions", json={"messages": []}) as response:
                assert response.status == 400
        assert hits["n"] == 1
        assert proxy.turns_used == 2
    finally:
        await proxy.stop()
        await _stop_upstream(runner, site)


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
                assert resp.status == 400
                err = await resp.json()
                assert err["error"]["code"] == "session_budget_exhausted"

            assert proxy.turns_used == 3
            assert hits["n"] == 2  # rejected before upstream
    finally:
        await proxy.stop()
        await _stop_upstream(upstream_runner, upstream_site)


@pytest.mark.asyncio
async def test_proxy_forwards_multimodal_histories_above_one_megabyte():
    upstream_runner, upstream_site, upstream_url, hits = await _start_upstream()
    proxy = await start_turn_counter_proxy(upstream_base_url=upstream_url, api_key="test", max_turns=1)
    image_url = "data:image/png;base64," + "A" * (2 * 1024 * 1024)
    try:
        async with ClientSession() as client:
            async with client.post(
                f"{proxy.base_url}/chat/completions",
                json={
                    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}]
                },
            ) as response:
                assert response.status == 200
            assert hits["bodies"][0]["messages"][1]["content"][0]["image_url"]["url"] == image_url
        assert proxy.turns_used == 1
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
        assert hits["bodies"][3]["messages"][0]["role"] == "system"
        assert "Begin wrapping up" in hits["bodies"][3]["messages"][0]["content"]
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


@pytest.mark.asyncio
@pytest.mark.parametrize("position", ["system_message", "user_message"])
async def test_responses_proxy_reminds_and_rejects_without_forwarding(position):
    runner, site, upstream, hits = await _start_upstream()
    proxy = await start_turn_counter_proxy(
        upstream_base_url=upstream, api_key="dummy", max_turns=2, position=position, trigger="per_turn"
    )
    try:
        async with ClientSession() as client:
            for attempt in range(1, 4):
                body = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "Solve the task"}]}]}
                async with client.post(f"{proxy.base_url}/responses", json=body) as response:
                    assert response.status == (200 if attempt <= 2 else 400)
                    if attempt == 3:
                        assert (await response.json())["error"]["code"] == "session_budget_exhausted"
        assert hits["n"] == 2
        assert proxy.turns_used == 3
        first, last = (json.dumps(body) for body in hits["bodies"])
        assert "1 turn(s) left" in first
        assert "URGENT" in last
        assert "input_text" in first
    finally:
        await proxy.stop()
        await _stop_upstream(runner, site)


def test_responses_string_input_and_tool_only_input_receive_reminders():
    body = {"input": "Solve the task"}
    inject_turn_reminder(body, n=1, max_turns=2)
    assert body["input"][0]["role"] == "system"
    assert body["input"][1] == {"role": "user", "content": "Solve the task"}
    tool = {"type": "function_call_output", "call_id": "call_1", "output": "done"}
    body = {"input": [tool]}
    inject_turn_reminder(body, n=1, max_turns=2, position="user_message")
    assert body["input"][0] == tool
    assert body["input"][1]["role"] == "user"


@pytest.fixture(autouse=True)
async def shared_transport(monkeypatch):
    from nemo_gym import server_utils

    async with ClientSession() as client:
        monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: client)
        yield


@pytest.mark.parametrize("limit", [True, 1.5, "2"])
def test_limit_must_be_a_positive_integer(limit):
    with pytest.raises(ValidationError):
        TurnConstraintConfig(enforcement="proxy", limit=limit)


@pytest.mark.asyncio
@pytest.mark.parametrize("limit", [1, 2])
async def test_concurrent_admission_and_independent_sessions(limit):
    from nemo_gym.adapters.turn_counter_proxy import TurnConstraintSession

    runner, site, upstream, hits = await _start_upstream(delay=0.02)
    constraint = TurnConstraintConfig(enforcement="proxy", limit=limit)
    session = TurnConstraintSession(
        constraint=constraint, upstream_base_url=upstream, api_key="", harness_version="test"
    )
    other = TurnConstraintSession(
        constraint=constraint, upstream_base_url=upstream, api_key="", harness_version="test"
    )
    try:
        async with session, other, ClientSession() as client:

            async def post(url):
                async with client.post(f"{url}/responses", json={"input": "hello"}) as response:
                    await response.read()
                    return response.status

            statuses = await asyncio.gather(*(post(session.base_url) for _ in range(limit + 5)))
            assert statuses.count(200) == limit
            assert statuses.count(400) == 5
            assert await post(other.base_url) == 200
        realized = session.metadata.realized
        assert realized.observed_count == limit + 5
        assert realized.forwarded_count == limit
        assert realized.rejected_count == 5
        assert realized.exhausted
        assert realized.stop_reason == "session_budget_exhausted"
        assert realized.reminder.trigger == "per_turn"
        assert session.metadata.requested.reminder.trigger == "auto"
        assert realized.runtime_version == "1"
        assert other.metadata.realized.stop_reason == "completed"
        assert not other.metadata.realized.exhausted
        assert hits["n"] == limit + 1
    finally:
        await _stop_upstream(runner, site)


@pytest.mark.asyncio
async def test_unconstrained_session_and_failed_startup_receipt(monkeypatch):
    from nemo_gym.adapters.turn_counter_proxy import TurnConstraintSession

    session = TurnConstraintSession(
        constraint=None, upstream_base_url="http://unused/v1", api_key="", harness_version="t"
    )
    async with session:
        assert session.base_url == "http://unused/v1"
    assert session.metadata is None

    original_start = web.TCPSite.start
    runners = []

    async def fail_start(site):
        await original_start(site)
        runners.append(site._runner)
        raise RuntimeError("startup failed after bind")

    monkeypatch.setattr(web.TCPSite, "start", fail_start)
    session = TurnConstraintSession(
        constraint=TurnConstraintConfig(enforcement="proxy", limit=1),
        upstream_base_url="http://unused/v1",
        api_key="",
        harness_version="t",
    )
    with pytest.raises(RuntimeError, match="startup failed"):
        async with session:
            pytest.fail("must not enter")
    assert not runners[0].sites
    assert session.metadata.realized.stop_reason == "startup_failure"
    assert session.metadata.realized.observed_count == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [RuntimeError("verifier failed"), asyncio.CancelledError()])
async def test_context_closes_listener_and_records_exception(failure):
    from aiohttp import ClientConnectorError

    from nemo_gym.adapters.turn_counter_proxy import TurnConstraintSession

    session = TurnConstraintSession(
        constraint=TurnConstraintConfig(enforcement="proxy", limit=1),
        upstream_base_url="http://unused/v1",
        api_key="",
        harness_version="t",
    )
    with pytest.raises(type(failure)):
        async with session:
            url = session.base_url
            raise failure
    assert session.metadata.realized.stop_reason == (
        "cancelled" if isinstance(failure, asyncio.CancelledError) else "error"
    )
    async with ClientSession() as client:
        with pytest.raises(ClientConnectorError):
            await client.post(f"{url}/responses", json={"input": "hello"})


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["responses", "chat/completions"])
async def test_streaming_capture_prefix_and_upstream_errors(path):
    from nemo_gym.adapters.turn_counter_proxy import TurnConstraintSession

    seen = []
    first_sent = asyncio.Event()
    release_last = asyncio.Event()

    async def upstream(request):
        seen.append((request.path_qs, await request.json()))
        if len(seen) == 1:
            return web.json_response({"error": {"code": "rate_limit"}}, status=429)
        if len(seen) == 2:
            return web.json_response({"error": {"code": "overloaded"}}, status=503)
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        first_sent.set()
        await release_last.wait()
        await response.write(b"data: [DONE]\n\n")
        return response

    app = web.Application()
    app.router.add_post(f"/ng-rollout/abc/v1/{path}", upstream)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    session = TurnConstraintSession(
        constraint=TurnConstraintConfig(enforcement="proxy", limit=3),
        upstream_base_url=f"http://127.0.0.1:{port}/ng-rollout/abc/v1",
        api_key="",
        harness_version="t",
    )
    try:
        async with session, ClientSession() as client:
            assert "/ng-rollout/" not in session.base_url
            url = f"{session.base_url}/{path}?test=1"
            payload = (
                {"input": "hello"} if path == "responses" else {"messages": [{"role": "user", "content": "hello"}]}
            )
            for status in (429, 503):
                async with client.post(url, json=payload) as response:
                    assert response.status == status
            async with client.post(url, json=payload) as response:
                await asyncio.wait_for(first_sent.wait(), timeout=2)
                assert await asyncio.wait_for(response.content.readexactly(13), timeout=2) == b"data: first\n\n"
                release_last.set()
                assert await response.read() == b"data: [DONE]\n\n"
            assert len(seen) == 3
            assert seen[0][0] == f"/ng-rollout/abc/v1/{path}?test=1"
        assert session.metadata.realized.forwarded_count == 3
        assert not session.metadata.realized.exhausted  # natural finish on N
    finally:
        release_last.set()
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("disconnect", [True, False])
async def test_disconnect_and_context_shutdown_release_upstream(monkeypatch, disconnect):
    from nemo_gym.adapters import turn_counter_proxy as module

    responses = []
    stopped = asyncio.Event()

    async def stream(request):
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        await response.write(b"data: first\n\n")
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    app = web.Application()
    app.router.add_post("/v1/responses", stream)
    runner = web.AppRunner(app, handler_cancellation=True)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    original = module.upstream_request

    async def tracked(*args, **kwargs):
        response = await original(*args, **kwargs)
        responses.append(response)
        return response

    monkeypatch.setattr(module, "upstream_request", tracked)
    session = module.TurnConstraintSession(
        constraint=TurnConstraintConfig(enforcement="proxy", limit=1),
        upstream_base_url=f"http://127.0.0.1:{port}/v1",
        api_key="",
        harness_version="t",
    )
    try:
        async with ClientSession() as client:
            async with session:
                response = await client.post(f"{session.base_url}/responses", json={"input": "hello"})
                assert await response.content.readexactly(13) == b"data: first\n\n"
                if disconnect:
                    response.close()
                    await asyncio.wait_for(stopped.wait(), timeout=2)
            assert responses[0].closed
            await asyncio.wait_for(stopped.wait(), timeout=2)
            response.close()
    finally:
        await runner.cleanup()


def test_only_typed_budget_errors_are_terminal():
    from nemo_gym.adapters.turn_counter_proxy import is_turn_budget_exhausted

    error = RuntimeError("session_budget_exhausted")
    assert not is_turn_budget_exhausted(error)
    error.body = {"error": {"code": "rate_limit"}}
    assert not is_turn_budget_exhausted(error)
    error.body = {"error": {"code": "session_budget_exhausted"}}
    assert is_turn_budget_exhausted(error)


@pytest.mark.asyncio
async def test_connection_failure_consumes_attempt_without_internal_retry(monkeypatch):
    from aiohttp import ClientConnectionError

    from nemo_gym import server_utils
    from nemo_gym.adapters.turn_counter_proxy import TurnConstraintSession

    attempts = 0

    class BrokenTransport:
        async def request(self, **kwargs):
            nonlocal attempts
            attempts += 1
            raise ClientConnectionError("connection failed")

    monkeypatch.setattr(server_utils, "get_global_aiohttp_client", lambda: BrokenTransport())
    session = TurnConstraintSession(
        constraint=TurnConstraintConfig(enforcement="proxy", limit=2),
        upstream_base_url="http://unused/v1",
        api_key="",
        harness_version="t",
    )
    async with session, ClientSession() as client:
        for status in (502, 502, 400):
            async with client.post(f"{session.base_url}/responses", json={"input": "hello"}) as response:
                assert response.status == status
                assert (await response.json())["error"]["code"] == (
                    "upstream_failure" if status == 502 else "session_budget_exhausted"
                )
    assert attempts == 2
    assert session.metadata.realized.forwarded_count == 2
    assert session.metadata.realized.rejected_count == 1


@pytest.mark.asyncio
async def test_budget_stop_does_not_mask_other_exception():
    from nemo_gym.adapters.turn_counter_proxy import TurnConstraintSession

    runner, site, upstream, _ = await _start_upstream()
    session = TurnConstraintSession(
        constraint=TurnConstraintConfig(enforcement="proxy", limit=1),
        upstream_base_url=upstream,
        api_key="",
        harness_version="t",
    )
    try:
        with pytest.raises(RuntimeError, match="infrastructure"):
            async with session, ClientSession() as client:
                for expected in (200, 400):
                    async with client.post(f"{session.base_url}/responses", json={"input": "hi"}) as response:
                        assert response.status == expected
                raise RuntimeError("infrastructure failure")
        assert session.metadata.realized.exhausted
        assert session.metadata.realized.stop_reason == "error"
    finally:
        await runner.cleanup()
