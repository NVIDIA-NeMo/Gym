# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The exec-limit preflight through the mini-SWE agent's /run: no harness, no policy call, a terminal failure row."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.rollouts import NG_TERMINAL_KEY, is_terminal_failure
from resources_servers.terminal_bench_4 import lifecycle
from resources_servers.terminal_bench_4.models import SandboxedVerifyRequest, TerminalBench4RunRequest
from resources_servers.terminal_bench_4.task import TaskSettings
from resources_servers.terminal_bench_4.tests.test_terminal_bench_science import (
    DATASET_REF,
    exec_limit_server,
    preparation,
)


module = pytest.importorskip("responses_api_agents.miniswe_sandboxed_agent.app")


async def test_a_rejected_exec_limit_reaches_collection_as_a_terminal_failure_without_a_policy_call(
    tmp_path, monkeypatch
):
    from responses_api_agents.miniswe_sandboxed_agent.models import MiniSWERunRequest

    server = exec_limit_server(tmp_path, monkeypatch, 7200)
    ref = next(task["ref"] for task in preparation.load_manifest()["tasks"] if task["name"] == "leaky-bloch-meep")
    task = SimpleNamespace(
        name="terminal-bench-science/leaky-bloch-meep",
        config=TaskSettings.model_validate(
            {
                "environment": {"docker_image": "a"},
                "verifier": {"timeout_sec": 7201, "environment": {"docker_image": "v"}},
            }
        ),
    )
    monkeypatch.setattr(server._loader, "load", AsyncMock(return_value=task))
    monkeypatch.setattr(lifecycle, "Environment", MagicMock(side_effect=AssertionError("no sandbox may be built")))

    agent = module.MiniSWESandboxedAgent(
        config=module.MiniSWESandboxedConfig(
            host="localhost",
            port=1,
            name="agent",
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "terminal_bench_science"},
            model_server={"type": "responses_api_models", "name": "policy"},
            artifacts_dir=tmp_path / "agent",
            shutdown_timeout_sec=0.02,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    harnesses, connects = [], AsyncMock()
    monkeypatch.setattr(module, "MiniSWEHarness", lambda **kwargs: harnesses.append(kwargs))
    monkeypatch.setattr(module.AsyncSandbox, "connect", connects)
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(side_effect=lambda r: r.value))
    monkeypatch.setattr(module, "get_server_url", lambda name: "http://policy:8000")
    resources_request = SimpleNamespace(session={SESSION_ID_KEY: "agent-owner"})
    calls = []

    # Only the HTTP hop between the two servers is replaced: each call goes to the real resources server.
    async def post(*, server_name, url_path, json, cookies, **kwargs):
        calls.append((server_name, url_path))
        if url_path == "/seed_session":
            value = await server.seed_session(resources_request, TerminalBench4RunRequest.model_validate(json))
        elif url_path == "/verify":
            value = await server.verify(resources_request, SandboxedVerifyRequest.model_validate(json))
        else:
            raise AssertionError(url_path)
        return SimpleNamespace(value=value.model_dump(mode="json"), cookies={"session": "seeded"})

    agent.server_client.post = AsyncMock(side_effect=post)
    body = MiniSWERunRequest(
        responses_create_params={"input": []},
        task_name=task.name,
        task_ref=ref,
        dataset_ref=DATASET_REF,
        rollout_id="rollout-1",
    )
    try:
        result = await agent.run(Request({"type": "http", "session": {SESSION_ID_KEY: "owner"}, "headers": []}), body)
    finally:
        await agent.shutdown()

    row = result.model_dump()
    assert calls == [("terminal_bench_science", "/seed_session"), ("terminal_bench_science", "/verify")]
    assert harnesses == [] and not connects.await_count
    assert row["_ng_failure_class"] == "infrastructure_error"
    assert row["_ng_failure_type"] == "ExecLimitConfigurationError"
    assert row[NG_TERMINAL_KEY] is True
    assert is_terminal_failure(row)
