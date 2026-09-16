# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.base_resources_server import BaseRunRequest
from nemo_gym.server_utils import SESSION_ID_KEY
from resources_servers.terminal_bench_4 import agent as lifecycle
from resources_servers.terminal_bench_4.handoff import AgentTermination


@pytest.mark.parametrize("reason", ["completed", "timeout", "nonzero_exit", "infrastructure_error", "cancelled"])
async def test_borrowed_agent_always_verifies_and_releases(monkeypatch, reason):
    body = BaseRunRequest(responses_create_params={"input": []})
    sandbox = MagicMock(release=AsyncMock())
    connect = AsyncMock(return_value=sandbox)
    monkeypatch.setattr(lifecycle.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(lifecycle, "create_provider", lambda _: MagicMock())
    monkeypatch.setattr(lifecycle, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(lifecycle, "raise_for_status", AsyncMock())
    seed = {
        "session_id": "test",
        "sandbox": {"provider": "cpu", "sandbox_id": "remote"},
        "instruction": "Official instruction",
        "agent_timeout_sec": 10,
        "setup_timeout_sec": 5,
    }
    monkeypatch.setattr(
        lifecycle, "get_response_json", AsyncMock(side_effect=[seed, {"agent_timeout_sec": 7}, {"reward": 1}])
    )
    post = AsyncMock(return_value=SimpleNamespace(cookies={"resources-session": "cookie"}))
    agent = SimpleNamespace(
        config=SimpleNamespace(
            name="test",
            resources_server=SimpleNamespace(name="resources"),
            model_server=SimpleNamespace(name="model"),
            sandbox_providers={"cpu": {"local": {}}},
        ),
        server_client=SimpleNamespace(post=post),
        rollout_id_from_run=lambda _: "rollout",
    )

    async def execute(sandbox, seed, budget):
        assert budget == 7
        if reason == "timeout":
            raise TimeoutError()
        if reason == "cancelled":
            raise asyncio.CancelledError()
        if reason == "infrastructure_error":
            raise ConnectionError("model unavailable")
        return lifecycle.empty_response(body.responses_create_params, "model"), AgentTermination(reason=reason), {}

    run = lifecycle.run_borrowed(
        agent,
        SimpleNamespace(cookies={"incoming": "cookie"}, session={SESSION_ID_KEY: "client"}),
        body,
        setup=AsyncMock(),
        execute=execute,
    )
    if reason == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await run
    else:
        assert (await run)["reward"] == 1
    assert body.responses_create_params.input == []
    assert connect.await_args.kwargs["owns_sandbox"] is False
    assert [call.kwargs["url_path"] for call in post.await_args_list] == ["/seed_session", "/start_session", "/verify"]
    verify = post.await_args_list[-1].kwargs
    assert verify["cookies"] == {"incoming": "cookie", "resources-session": "cookie"}
    assert verify["json"]["termination"]["reason"] == reason
    assert verify["json"]["session_id"] == "test"
    sandbox.release.assert_awaited_once()


@pytest.mark.parametrize("failure", ["alias", "connect", "setup", "cancel"])
async def test_failed_attachment_or_setup_cancels_owned_session(tmp_path, monkeypatch, failure):
    monkeypatch.chdir(tmp_path)
    sandbox = MagicMock(release=AsyncMock())
    provider = MagicMock(aclose=AsyncMock())
    connect = AsyncMock(
        side_effect=ConnectionError("attach failed") if failure == "connect" else None, return_value=sandbox
    )
    monkeypatch.setattr(lifecycle.AsyncSandbox, "connect", connect)
    monkeypatch.setattr(lifecycle, "create_provider", lambda _: provider)
    monkeypatch.setattr(lifecycle, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(lifecycle, "raise_for_status", AsyncMock())
    seed = {
        "session_id": "test",
        "sandbox": {"provider": "cpu", "sandbox_id": "box"},
        "instruction": "task",
        "agent_timeout_sec": 1,
        "setup_timeout_sec": 1,
    }
    monkeypatch.setattr(lifecycle, "get_response_json", AsyncMock(side_effect=[seed, {}]))
    post = AsyncMock(return_value=SimpleNamespace(cookies={}))
    agent = SimpleNamespace(
        config=SimpleNamespace(
            name="test",
            resources_server=SimpleNamespace(name="resources"),
            model_server=SimpleNamespace(name="model"),
            sandbox_providers={} if failure == "alias" else {"cpu": {"local": {}}},
        ),
        server_client=SimpleNamespace(post=post),
        rollout_id_from_run=lambda _: None,
    )
    setup = AsyncMock(side_effect=asyncio.CancelledError() if failure == "cancel" else RuntimeError("setup failed"))
    execute = AsyncMock()
    expected = {
        "alias": ValueError,
        "connect": ConnectionError,
        "setup": RuntimeError,
        "cancel": asyncio.CancelledError,
    }[failure]
    with pytest.raises(expected):
        await lifecycle.run_borrowed(
            agent,
            SimpleNamespace(cookies={}, session={SESSION_ID_KEY: "client"}),
            BaseRunRequest(responses_create_params={"input": []}),
            setup=setup,
            execute=execute,
        )
    assert [c.kwargs["url_path"] for c in post.await_args_list] == ["/seed_session", "/cancel_session"]
    execute.assert_not_awaited()
    if failure == "connect":
        provider.aclose.assert_awaited_once()
    if failure in {"setup", "cancel"}:
        sandbox.release.assert_awaited_once()
