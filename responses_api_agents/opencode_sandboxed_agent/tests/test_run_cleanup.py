# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI, HTTPException, Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import NeMoGymResponse
from responses_api_agents.opencode_sandboxed_agent import app
from responses_api_agents.opencode_sandboxed_agent.tests.test_app import TestOpenCodeSandboxedAgent as AppTests


class Response:
    ok = True

    def __init__(self, payload, cookies=None):
        self.payload = payload
        self.cookies = cookies or {}

    async def json(self):
        if isinstance(self.payload, BaseException):
            raise self.payload
        return self.payload

    async def read(self):
        return json.dumps(self.payload).encode()


@pytest.fixture
def case(monkeypatch):
    client = MagicMock(spec=ServerClient)
    server = app.OpenCodeSandboxedAgent(config=AppTests()._create_config(), server_client=client)
    context = SimpleNamespace(
        server=server,
        events=[],
        seed={"sandbox_handle": "resource-sandbox", "workdir": "/task/repo"},
        close_error=None,
        start=server._start_sandbox,
    )
    sandbox = SimpleNamespace(
        stop=AsyncMock(side_effect=lambda: context.events.append("stop")),
        disconnect=AsyncMock(side_effect=lambda: context.events.append("disconnect")),
    )
    context.sandbox = sandbox
    request = Request({"type": "http", "headers": [(b"cookie", b"caller=kept")], "session": {SESSION_ID_KEY: "trial"}})
    context.request = request
    context.body = app.OpenCodeSandboxedAgentRunRequest(responses_create_params={"input": "Repair the task"})

    async def post(*, server_name, url_path, json, cookies):
        context.events.append(url_path)
        if url_path == "/seed_session":
            return Response(context.seed, {"session": "owner-cookie"})
        if url_path == "/verify":
            assert cookies == {"caller": "kept", "session": "owner-cookie", "sandbox_id": "trial"}
            return Response(json | {"reward": 1.0})
        assert url_path == "/close_session"
        assert json == {}
        assert cookies == {"caller": "kept", "session": "owner-cookie"}
        if context.close_error:
            raise context.close_error
        return Response({"resources_session_id": "owner-session"})

    client.post = AsyncMock(side_effect=post)
    monkeypatch.setattr(server, "_start_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(server, "_seed_tool_servers", AsyncMock(return_value={}))

    async def responses(request, body):
        context.events.append("responses")
        server._sandbox_id_to_run_result["trial"] = {
            "opencode_results_fpath": "/artifacts/export.json",
            "opencode_run_stdout": "complete",
            "opencode_run_stderr": "",
            "opencode_finished": True,
            "opencode_export_found": True,
        }
        return NeMoGymResponse(
            id="answer",
            created_at=0,
            model="model",
            object="response",
            output=[],
            tool_choice=body.tool_choice,
            tools=body.tools,
            parallel_tool_calls=body.parallel_tool_calls,
        )

    monkeypatch.setattr(app.OpenCodeSandboxedAgent, "responses", AsyncMock(side_effect=responses))
    return context


@pytest.mark.parametrize(
    "seed",
    [
        ValueError("invalid JSON"),
        [],
        None,
        {"sandbox_descriptor": {}},
        {"sandbox_descriptor": None},
        {"sandbox_handle": None},
        {"sandbox_handle": ""},
        {"sandbox_handle": " "},
        {"sandbox_handle": 0},
        {"sandbox_descriptor": None, "sandbox_handle": ""},
    ],
)
async def test_unreadable_seed_closes_resource_owner_before_any_agent_work(case, seed):
    case.seed = seed
    with pytest.raises(ValueError):
        await case.server.run(case.request, case.body)
    assert case.events == ["/seed_session", "/close_session"]
    case.server._start_sandbox.assert_not_awaited()
    case.server.responses.assert_not_awaited()
    assert not case.server._run_sandbox_states


@pytest.mark.parametrize("stage", ["_seed_tool_servers", "_start_sandbox"])
async def test_setup_failure_closes_owner_without_verification(case, stage):
    getattr(case.server, stage).side_effect = RuntimeError("setup failed")
    with pytest.raises(RuntimeError, match="setup failed"):
        await case.server.run(case.request, case.body)
    assert case.events == ["/seed_session", "/close_session"]
    case.server.responses.assert_not_awaited()
    assert not case.server._run_sandbox_states


async def test_success_keeps_original_owner_cookies_and_disconnects_borrowed_client(case):
    output = await case.server.run(case.request, case.body)
    assert output.reward == 1
    assert case.events == ["/seed_session", "responses", "/verify", "/close_session", "disconnect"]
    case.sandbox.stop.assert_not_awaited()
    assert not case.server._run_sandbox_states
    assert not case.server._sandbox_id_to_sandbox


async def test_failed_owner_close_remains_retriable_after_client_disconnect(case):
    case.close_error = RuntimeError("owner close failed")
    with pytest.raises(RuntimeError, match="owner close failed"):
        await case.server.run(case.request, case.body)
    state = case.server._run_sandbox_states["trial"]
    assert state.resources_cookies == {"caller": "kept", "session": "owner-cookie"}
    assert state.sandbox is None
    case.sandbox.disconnect.assert_awaited_once()
    case.close_error = None
    await case.server._close_run_sandbox("trial")
    assert case.events.count("/close_session") == 2
    case.sandbox.disconnect.assert_awaited_once()
    assert not case.server._run_sandbox_states


async def test_failed_disconnect_retains_handle_and_does_not_repeat_acknowledged_owner_close(case):
    case.sandbox.disconnect.side_effect = RuntimeError("disconnect failed")
    with pytest.raises(RuntimeError, match="disconnect failed"):
        await case.server.run(case.request, case.body)
    assert case.server._sandbox_id_to_sandbox["trial"] is case.sandbox
    assert case.server._run_sandbox_states["trial"].resources_cookies is None
    case.sandbox.disconnect.side_effect = None
    await case.server._close_run_sandbox("trial")
    assert case.events.count("/close_session") == 1
    assert not case.server._sandbox_id_to_sandbox
    assert not case.server._run_sandbox_states


async def test_failed_owned_stop_is_visible_and_shutdown_retries_the_retained_handle(case):
    case.seed = {}
    case.sandbox.stop.side_effect = RuntimeError("stop failed")
    with pytest.raises(RuntimeError, match="stop failed"):
        await case.server.run(case.request, case.body)
    assert case.server._sandbox_id_to_sandbox["trial"] is case.sandbox
    case.sandbox.disconnect.assert_not_awaited()
    assert "/close_session" not in case.events
    case.sandbox.stop.side_effect = None
    async with case.server._drain_sessions_on_shutdown(FastAPI()):
        pass
    assert case.sandbox.stop.await_count == 2
    assert not case.server._run_sandbox_states
    assert not case.server._sandbox_id_to_sandbox


async def test_cancellation_during_tool_discovery_still_closes_owner(case):
    started = asyncio.Event()

    async def discover(*args):
        started.set()
        await asyncio.Event().wait()

    case.server._seed_tool_servers.side_effect = discover
    task = asyncio.create_task(case.server.run(case.request, case.body))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=1)
    assert case.events == ["/seed_session", "/close_session"]
    case.server._start_sandbox.assert_not_awaited()
    assert not case.server._run_sandbox_states


@pytest.mark.parametrize("fail_connect", [False, True])
async def test_serialized_connection_preserves_descriptor_and_releases_provider_on_attach_failure(
    case, monkeypatch, fail_connect
):
    descriptor = {"sandbox_id": "borrowed", "workdir": "/original", "user": "task-user", "ports": [8080]}
    case.seed = {"sandbox_descriptor": descriptor, "workdir": "/task/repo"}
    provider = SimpleNamespace(aclose=AsyncMock(), name="opensandbox")
    constructor = MagicMock(connect=AsyncMock(return_value=case.sandbox))
    if fail_connect:
        constructor.connect.side_effect = RuntimeError("attach failed")
    monkeypatch.setattr(app, "AsyncSandbox", constructor)
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *args: {})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *args: {})
    monkeypatch.setattr(app, "create_provider", lambda *args: provider)
    monkeypatch.setattr(case.server, "_start_sandbox", case.start)
    if fail_connect:
        with pytest.raises(RuntimeError, match="attach failed"):
            await case.server.run(case.request, case.body)
        provider.aclose.assert_awaited_once()
        assert case.events == ["/seed_session", "/close_session"]
    else:
        await case.server.run(case.request, case.body)
        case.sandbox.disconnect.assert_awaited_once()
    constructor.connect.assert_awaited_once_with(descriptor | {"workdir": "/task/repo"}, provider=provider)
    assert descriptor["workdir"] == "/original"
    assert not case.server._run_sandbox_states


@pytest.mark.parametrize("cancel", [False, True])
async def test_owner_cleanup_timeout_retains_cookies_but_releases_borrowed_client(case, cancel):
    case.server.config.session_close_timeout_seconds = 0.01
    post = case.server.server_client.post.side_effect
    closing = asyncio.Event()

    async def delayed_post(**kwargs):
        if kwargs["url_path"] == "/close_session":
            closing.set()
            await asyncio.Event().wait()
        return await post(**kwargs)

    case.server.server_client.post.side_effect = delayed_post
    task = asyncio.create_task(case.server.run(case.request, case.body))
    await asyncio.wait_for(closing.wait(), timeout=1)
    if cancel:
        task.cancel()
        await asyncio.sleep(0)
        task.cancel()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(task, timeout=1)
    case.sandbox.disconnect.assert_awaited_once()
    assert case.server._run_sandbox_states["trial"].resources_cookies is not None


async def test_an_active_run_cannot_be_replaced_while_resource_seed_is_pending(case):
    started, proceed = asyncio.Event(), asyncio.Event()
    post = case.server.server_client.post.side_effect

    async def wait_on_seed(**kwargs):
        if kwargs["url_path"] == "/seed_session":
            started.set()
            await proceed.wait()
        return await post(**kwargs)

    case.server.server_client.post.side_effect = wait_on_seed
    first = asyncio.create_task(case.server.run(case.request, case.body))
    await asyncio.wait_for(started.wait(), timeout=1)
    try:
        with pytest.raises(HTTPException) as caught:
            await case.server.run(case.request, case.body)
        assert caught.value.status_code == 409
        assert case.server._run_sandbox_states["trial"].active
        assert not case.events
    finally:
        proceed.set()
        await asyncio.wait_for(first, timeout=1)
    assert case.events.count("/seed_session") == case.events.count("/close_session") == 1


async def test_concurrent_cleanup_retries_share_the_same_pending_owner_close(case):
    case.close_error = RuntimeError("owner unavailable")
    with pytest.raises(RuntimeError, match="owner unavailable"):
        await case.server.run(case.request, case.body)
    case.close_error = None
    started, proceed = asyncio.Event(), asyncio.Event()
    post = case.server.server_client.post.side_effect

    async def wait_on_close(**kwargs):
        if kwargs["url_path"] == "/close_session":
            started.set()
            await proceed.wait()
        return await post(**kwargs)

    case.server.server_client.post.side_effect = wait_on_close
    first = asyncio.create_task(case.server._close_run_sandbox("trial"))
    await asyncio.wait_for(started.wait(), timeout=1)
    second = asyncio.create_task(case.server._close_run_sandbox("trial"))
    proceed.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=1)
    assert case.events.count("/close_session") == 2  # The failed close plus one shared retry.
    case.sandbox.disconnect.assert_awaited_once()
    assert not case.server._run_sandbox_states


@pytest.mark.parametrize("borrowed", [True, False])
async def test_repeated_task_cancellation_waits_for_owner_close_or_owned_stop(case, borrowed):
    running, closing, proceed = asyncio.Event(), asyncio.Event(), asyncio.Event()
    post = case.server.server_client.post.side_effect

    async def execute(*args):
        running.set()
        await asyncio.Event().wait()

    async def wait_on_close(**kwargs):
        if kwargs["url_path"] == "/close_session":
            closing.set()
            await proceed.wait()
        return await post(**kwargs)

    async def wait_on_stop():
        closing.set()
        await proceed.wait()

    case.server.responses.side_effect = execute
    if borrowed:
        case.server.server_client.post.side_effect = wait_on_close
    else:
        case.seed = {}
        case.sandbox.stop.side_effect = wait_on_stop
    task = asyncio.create_task(case.server.run(case.request, case.body))
    try:
        await asyncio.wait_for(running.wait(), timeout=1)
        task.cancel()
        await asyncio.wait_for(closing.wait(), timeout=1)
        task.cancel()
        await asyncio.sleep(0)
        assert not task.done()
        proceed.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=1)
    finally:
        proceed.set()
        await asyncio.gather(task, return_exceptions=True)
    if borrowed:
        case.sandbox.disconnect.assert_awaited_once()
        case.sandbox.stop.assert_not_awaited()
    else:
        case.sandbox.stop.assert_awaited_once()
        case.sandbox.disconnect.assert_not_awaited()
    assert not case.server._run_sandbox_states
    assert not case.server._sandbox_id_to_sandbox


async def test_cancelled_cleanup_waiter_does_not_release_or_cancel_the_active_close(case):
    case.close_error = RuntimeError("owner unavailable")
    with pytest.raises(RuntimeError, match="owner unavailable"):
        await case.server.run(case.request, case.body)
    state = case.server._run_sandbox_states["trial"]
    case.close_error = None
    started, proceed = asyncio.Event(), asyncio.Event()
    post = case.server.server_client.post.side_effect

    async def wait_on_close(**kwargs):
        if kwargs["url_path"] == "/close_session":
            started.set()
            await proceed.wait()
        return await post(**kwargs)

    case.server.server_client.post.side_effect = wait_on_close
    first = asyncio.create_task(case.server._close_run_sandbox("trial"))
    second = None
    try:
        await asyncio.wait_for(started.wait(), timeout=1)
        second = asyncio.create_task(case.server._close_run_sandbox("trial"))
        await asyncio.sleep(0)
        second.cancel()
        await asyncio.sleep(0)
        second.cancel()
        await asyncio.sleep(0)
        assert not first.done() and not second.done()
        assert state.close_lock.locked()
        proceed.set()
        await asyncio.wait_for(first, timeout=1)
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(second, timeout=1)
    finally:
        proceed.set()
        await asyncio.gather(first, *([second] if second else []), return_exceptions=True)
    assert not state.close_lock.locked()
    assert case.events.count("/close_session") == 2
    assert not case.server._run_sandbox_states


async def test_cleanup_lock_timeout_preserves_the_current_holder_and_pending_owner(case):
    case.close_error = RuntimeError("owner unavailable")
    with pytest.raises(RuntimeError, match="owner unavailable"):
        await case.server.run(case.request, case.body)
    case.close_error = None
    state = case.server._run_sandbox_states["trial"]
    case.server.config.session_close_timeout_seconds = 0.01
    await state.close_lock.acquire()
    try:
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(case.server._close_run_sandbox("trial"), timeout=1)
        assert state.close_lock.locked()
        assert state.resources_cookies == {"caller": "kept", "session": "owner-cookie"}
        assert case.events.count("/close_session") == 1
    finally:
        state.close_lock.release()
    await case.server._close_run_sandbox("trial")
    assert not case.server._run_sandbox_states
