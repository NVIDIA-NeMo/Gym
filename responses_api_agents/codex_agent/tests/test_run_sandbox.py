# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The flat /run endpoint borrows the task sandbox without changing the resource protocol."""

import asyncio
import json
import tomllib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException, Request

from nemo_gym.agents import AgentSessionSetupError
from nemo_gym.config import SKILLS_REF_KEY_NAME, ResourcesServerRef
from nemo_gym.core import encode_model_sampling
from nemo_gym.server_runtime import NEMO_GYM_MCP_METADATA_KEY
from responses_api_agents.codex_agent import app
from responses_api_agents.codex_agent.tests.test_app import FakeAioHTTPResponse
from responses_api_agents.codex_agent.tests.test_sandbox_sessions import setup as setup


@pytest.fixture
def run_case(setup, monkeypatch):
    agent, sandbox = setup
    agent.config.resources_server = ResourcesServerRef(type="resources_servers", name="tasks")
    agent.config.sandbox_provider = "sandbox"
    agent.config.concurrency = 1
    agent.sem = asyncio.Semaphore(1)
    agent.server_client.global_config_dict["token_id_capture"] = {"enabled": True, "all_agents": True}
    agent.server_client.global_config_dict["tasks"] = {
        "resources_servers": {"swe_rebench": {"host": "resources.example", "port": 9001}}
    }
    agent.server_client._build_server_base_url.side_effect = lambda config: f"http://{config['host']}:{config['port']}"
    monkeypatch.setattr(agent, "_create_response", AsyncMock(side_effect=AssertionError("host execution forbidden")))
    sandbox.expected_workdir = "/task/repository"
    case = SimpleNamespace(
        agent=agent,
        sandbox=sandbox,
        request=Request({"type": "http", "headers": [(b"cookie", b"upstream=kept")]}),
        body=app.CodexAgentRunRequest.model_validate(
            {
                "responses_create_params": {"input": "Repair the task", "instructions": "Keep the public API"},
                "_ng_rollout_id": "task-run-a2",
                "_ng_model_sampling": {"temperature": 1.0, "top_p": 1.0, "max_output_tokens": 1024},
                "instance_id": "org__repo-12",
            }
        ),
        seed={"sandbox_descriptor": {"sandbox_id": "task", "workdir": "/task/repository"}},
        calls=[],
        failure=None,
    )

    async def post(**kwargs):
        endpoint = kwargs["url_path"]
        case.calls.append(endpoint)
        if endpoint == "/seed_session":
            response = FakeAioHTTPResponse(case.seed, cookies={"resources": "seed-cookie"})
            if case.failure == "seed_json":
                response.read = AsyncMock(return_value=b"not json")
            return response
        assert kwargs["cookies"] == {"upstream": "kept", "resources": "seed-cookie"}
        if endpoint == "/verify":
            sandbox.disconnect.assert_awaited_once()
            sandbox.stop.assert_not_awaited()
            if case.failure == "verify":
                raise RuntimeError("verify failed")
            # Verifiers may clear their response cookie; cleanup must retain the original seed cookie.
            return FakeAioHTTPResponse(kwargs["json"] | {"reward": 1}, cookies={"resources": ""})
        assert endpoint == "/close_session" and kwargs["json"] == {}
        if case.failure == "close":
            raise RuntimeError("resource close failed")
        if case.failure == "close_timeout":
            await asyncio.sleep(1)
        return FakeAioHTTPResponse({})

    agent.server_client.post = AsyncMock(side_effect=post)
    return case


@pytest.mark.parametrize("timed_out", [False, True])
async def test_run_uses_task_runtime_preserves_capture_and_closes_before_verify(run_case, timed_out):
    case = run_case
    case.sandbox.result["timed_out"] = timed_out
    result = await asyncio.wait_for(case.agent.run(case.request, case.body), timeout=5)
    assert result.reward == 1
    assert result.agent_timed_out is timed_out
    assert result.finished_naturally is (not timed_out)
    assert result.response.status == ("incomplete" if timed_out else "completed")
    assert result.response.metadata["harness_execution"] == "sandbox"
    assert result.response.metadata["codex_version"] == "0.144.4"
    assert result.ng_agent_observations.source == "codex"
    assert case.calls == ["/seed_session", "/verify", "/close_session"]
    case.agent._create_response.assert_not_awaited()
    case.sandbox.stop.assert_not_awaited()
    assert case.agent._local_setup_task is None
    directory = case.sandbox.directory
    payload = json.loads(case.sandbox.files[f"{directory}/input.json"])
    assert payload["prompt"] == "Repair the task"
    assert payload["cwd"] == "/task/repository"
    assert payload["command"][:2] == [
        "/tmp/nemo-gym-codex-node-22.19.0-0.144.4/node/bin/node",
        "/tmp/nemo-gym-codex-node-22.19.0-0.144.4/codex/node_modules/@openai/codex/bin/codex.js",
    ]
    assert "HOME" not in payload["env"] and "PATH" not in payload["env"]
    assert payload["env"]["CODEX_HOME"] == f"{directory}/home/.codex"
    settings = tomllib.loads(case.sandbox.files[f"{directory}/home/.codex/config.toml"])
    sampling = encode_model_sampling(case.body.model_sampling)
    assert settings["model_providers"]["gym"]["base_url"] == (
        f"http://model.example:9000/ng-rollout/task-run-a2/training-token-capture/ng-sampling/{sampling}/v1"
    )
    assert settings["model_providers"]["gym"]["http_headers"] == {"x-session-id": "task-run-a2"}
    assert settings["developer_instructions"] == "Keep the public API"
    assert not any(path.startswith("/task/") for path in case.sandbox.files)


async def test_run_stages_private_skills_and_resource_mcp_grants(run_case, tmp_path):
    case = run_case
    skill = tmp_path / "skills" / "repair"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: repair\ndescription: Repair a task.\n---\nCheck the tests.")
    (skill / "helper.txt").write_text("supporting material")
    case.body.model_extra[SKILLS_REF_KEY_NAME] = {"path": str(skill.parent)}
    case.seed[NEMO_GYM_MCP_METADATA_KEY] = {
        "server_name": "task_tools",
        "url_path": "/mcp",
        "headers": {"X-NeMo-Gym-Session-Token": "test-session"},
    }
    result = await asyncio.wait_for(case.agent.run(case.request, case.body), timeout=5)
    directory = case.sandbox.directory
    assert case.sandbox.files[f"{directory}/home/.codex/skills/repair/helper.txt"] == "supporting material"
    settings = tomllib.loads(case.sandbox.files[f"{directory}/home/.codex/config.toml"])
    assert settings["mcp_servers"]["task_tools"] == {
        "url": "http://resources.example:9001/mcp",
        "http_headers": {"X-NeMo-Gym-Session-Token": "test-session"},
        "required": True,
    }
    assert "test-session" not in result.model_dump_json()


async def test_failed_sandbox_response_preserves_error_and_is_not_finished_naturally(run_case):
    case = run_case
    records = [json.loads(line) for line in case.sandbox.events.splitlines()]
    records[-1][1] = {"type": "turn.failed", "error": {"message": "unexpected status 503"}}
    case.sandbox.events = "\n".join(json.dumps(record) for record in records)
    case.sandbox.result["return_code"] = 1
    result = await asyncio.wait_for(case.agent.run(case.request, case.body), timeout=5)
    assert result.response.status == "failed"
    assert result.response.error.message == "unexpected status 503"
    assert result.response.output[-1].type == "message"
    assert not result.finished_naturally
    assert case.calls == ["/seed_session", "/verify", "/close_session"]


@pytest.mark.parametrize(
    "seed",
    [
        {"sandbox_handle": "task", "workdir": "/task/repository"},
        {"sandbox_descriptor": {}},
        {"sandbox_descriptor": None},
        {"sandbox_descriptor": {"sandbox_id": "task"}},
        {"sandbox_descriptor": {"sandbox_id": "task", "workdir": "relative"}},
        [],
    ],
)
async def test_invalid_resource_sandbox_fails_without_host_fallback_and_closes(run_case, seed):
    case = run_case
    case.seed = seed
    with pytest.raises((HTTPException, ValueError)):
        await case.agent.run(case.request, case.body)
    assert case.calls == ["/seed_session", "/close_session"]
    case.agent._create_response.assert_not_awaited()
    case.sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("failure", ["seed_json", "connect", "install", "execute", "verify", "close", "close_timeout"])
async def test_run_cleanup_covers_failed_setup_execution_and_verification(run_case, monkeypatch, failure):
    case = run_case
    case.failure = failure
    if failure == "connect":
        monkeypatch.setattr(app.AsyncSandbox, "connect", AsyncMock(side_effect=RuntimeError("connect failed")))
        monkeypatch.setattr(app, "create_provider", lambda _: SimpleNamespace(aclose=AsyncMock()))
    elif failure == "install":
        original_exec = case.sandbox.exec.side_effect

        async def execute(command, **kwargs):
            if "install_codex_runtime.sh" in command:
                raise RuntimeError("install failed")
            return await original_exec(command, **kwargs)

        case.sandbox.exec.side_effect = execute
    elif failure == "execute":
        case.sandbox.launch.side_effect = RuntimeError("execute failed")
    elif failure == "close_timeout":
        case.agent.config.session_close_timeout_seconds = 0.02
    with pytest.raises((RuntimeError, ValueError, HTTPException, TimeoutError)):
        await asyncio.wait_for(case.agent.run(case.request, case.body), timeout=5)
    assert case.calls[-1] == "/close_session"
    assert ("/verify" in case.calls) is (failure in {"verify", "close", "close_timeout"})
    case.agent._create_response.assert_not_awaited()
    case.sandbox.stop.assert_not_awaited()


async def test_run_retries_cleanup_state_returned_by_failed_seed(run_case, monkeypatch):
    case = run_case
    state = SimpleNamespace(close=AsyncMock())
    error = AgentSessionSetupError(state, error=RuntimeError("setup failed before close failed"))
    monkeypatch.setattr(case.agent, "_seed_agent_session_state", AsyncMock(side_effect=error))
    with pytest.raises(RuntimeError, match="setup failed before close failed"):
        await case.agent.run(case.request, case.body)
    state.close.assert_awaited_once_with(case.agent.config.session_close_timeout_seconds)
    assert case.calls == ["/seed_session", "/close_session"]


async def test_cancelled_run_stops_harness_and_closes_resource_owner(run_case):
    case = run_case
    case.sandbox.blocked = True
    task = asyncio.create_task(case.agent.run(case.request, case.body))
    try:
        await asyncio.wait_for(case.sandbox.started.wait(), timeout=2)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=3)
    finally:
        if not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
    assert case.sandbox.signals == ["SIGTERM"]
    case.sandbox.disconnect.assert_awaited_once()
    case.sandbox.stop.assert_not_awaited()
    assert case.calls == ["/seed_session", "/close_session"]
