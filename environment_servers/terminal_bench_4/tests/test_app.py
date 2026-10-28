# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
from http.cookies import SimpleCookie
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, Request

from environment_servers.terminal_bench_4 import app as module
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient


def row():
    return dict(
        task_name="terminal-bench/example",
        task_ref="task-digest",
        dataset_ref="dataset-digest",
        rollout_id="smoke-1",
        responses_create_params={"input": []},
    )


def request():
    return Request({"type": "http", "session": {SESSION_ID_KEY: "owner"}, "headers": []})


@pytest.fixture
def setup(monkeypatch):
    server = module.TerminalBench4EnvironmentServer(
        config=module.TerminalBench4EnvironmentConfig(
            name="env",
            host="localhost",
            port=1,
            entrypoint="app.py",
            resources_server={"type": "resources_servers", "name": "resources"},
            agent_server={"type": "responses_api_agents", "name": "agent"},
            agent_max_timeout_sec=20,
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    state = dict(calls=[], close_confirmed=True, agent_error=None, mcp=[], status="completed", seed_error=None)

    async def post(*, server_name, url_path, json, cookies):
        state["calls"].append((url_path, json, dict(cookies)))
        if url_path == "/seed_session":
            if state["seed_error"]:
                raise state["seed_error"]
            value = dict(
                session_id="resource-session",
                instruction="Solve the real task",
                agent_timeout_sec=100,
                mcp_servers=state["mcp"],
                sandbox_access={
                    "connection": {"provider_config_ref": "cpu", "descriptor": {"sandbox_id": "borrowed"}},
                    "workdir": "/app",
                    "user": "task-user",
                },
            )
        elif url_path == "/v1/agent_sessions":
            value = dict(agent_session_id=json["agent_session_id"])
        elif url_path.endswith("/v1/responses"):
            if state["agent_error"]:
                raise state["agent_error"]
            value = dict(
                id="response",
                created_at=0,
                model="policy",
                object="response",
                output=[],
                parallel_tool_calls=False,
                tool_choice="auto",
                tools=[],
                status=state["status"],
            )
        elif url_path == "/v1/agent_sessions/close":
            value = dict(
                agent_session_id=json["agent_session_id"],
                cleanup_confirmed=state["close_confirmed"],
                agent_observations={"source": "opencode"},
            )
        elif url_path == "/verify":
            value = json | {"reward": 1, "evaluation_completed": json["agent_started"]}
        elif url_path == "/close_session":
            value = {"closed": True}
        else:
            raise AssertionError(url_path)
        return SimpleNamespace(value=value, cookies=SimpleCookie({"session": server_name}))

    server.server_client.post = AsyncMock(side_effect=post)
    monkeypatch.setattr(module, "raise_for_status", AsyncMock())
    monkeypatch.setattr(module, "get_response_json", AsyncMock(side_effect=lambda reply: reply.value))
    return server, state


async def test_borrowed_sandbox_prompt_timeout_and_close_before_verify(setup):
    server, state = setup
    result = await server.run_legacy(request(), row())
    assert result["evaluation_completed"] and result["reward"] == 1
    assert result["ng_agent_observations"]["source"] == "opencode"
    paths = [path for path, _, _ in state["calls"]]
    assert paths == [
        "/seed_session",
        "/v1/agent_sessions",
        "/ng-rollout/smoke-1/v1/responses",
        "/v1/agent_sessions/close",
        "/verify",
        "/close_session",
    ]
    seed = state["calls"][1][1]
    assert seed["sandbox_access"]["connection"]["descriptor"]["sandbox_id"] == "borrowed"
    assert seed["sandbox_access"]["user"] == "task-user"
    activation = state["calls"][2][1]
    assert activation["input"] == "Solve the real task"
    assert activation["metadata"]["agent_timeout_sec"] == "20.0"
    assert state["calls"][4][2] == {"session": "resources"}
    assert result["termination"]["reason"] == "completed"


async def test_uncertain_cleanup_never_verifies(setup):
    server, state = setup
    state["close_confirmed"] = False
    result = await server.run_legacy(request(), row())
    assert "did not confirm termination" in result["_ng_failure_message"]
    assert "/verify" not in [path for path, _, _ in state["calls"]]
    assert state["calls"][-1][0] == "/close_session"


@pytest.mark.parametrize("error", [RuntimeError("broken transport"), TimeoutError("lost activation reply")])
async def test_execution_errors_are_ungraded_and_closed(setup, error):
    server, state = setup
    state["agent_error"] = error
    result = await server.run_legacy(request(), row())
    assert result["termination"]["reason"] == "infrastructure_error"
    assert not result["evaluation_completed"]
    assert state["calls"][-1][0] == "/close_session"


async def test_unsupported_task_tools_do_not_execute(setup):
    server, state = setup
    state["mcp"] = [{"name": "required-tool"}]
    result = await server.run_legacy(request(), row())
    assert not any(path.endswith("/v1/responses") for path, _, _ in state["calls"])
    assert not result["evaluation_completed"]
    assert "MCP/skills" in result["termination"]["detail"]


async def test_native_timeout_retains_gradable_partial_work(setup):
    server, state = setup
    state["status"] = "incomplete"
    result = await server.run_legacy(request(), row())
    assert result["evaluation_completed"]
    assert result["termination"]["reason"] == "timeout"


async def test_lost_seed_reply_still_closes_resources(setup):
    server, state = setup
    state["seed_error"] = TimeoutError("lost reply")
    await server.run_legacy(request(), row())
    assert [path for path, _, _ in state["calls"]] == ["/seed_session", "/close_session"]
    assert state["calls"][0][1] == state["calls"][1][1]


async def test_retried_rollout_executes_once_and_conflicting_request_fails(setup):
    server, state = setup
    first, second = await asyncio.gather(server.run_legacy(request(), row()), server.run_legacy(request(), row()))
    assert first == second
    assert sum(path == "/seed_session" for path, _, _ in state["calls"]) == 1
    with pytest.raises(HTTPException, match="another request"):
        await server.run_legacy(request(), row() | {"task_ref": "different"})


def test_recipe_resolves_without_changing_miniswe(monkeypatch):
    from pathlib import Path

    from omegaconf import OmegaConf

    from nemo_gym.global_config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
        get_first_server_config_dict,
    )

    monkeypatch.chdir(Path(__file__).resolve().parents[3])
    for harness in ("opencode", "miniswe"):
        config = GlobalConfigDictParser().parse(
            GlobalConfigDictParserConfig(
                skip_load_from_cli=True,
                skip_load_from_dotenv=True,
                offline=True,
                initial_global_config_dict=OmegaConf.create(
                    {
                        "config_paths": [
                            f"benchmarks/terminal_bench_4/{harness}.yaml",
                            "responses_api_models/vllm_model/configs/vllm_model.yaml",
                        ],
                        "policy_base_url": "http://unused.invalid/v1",
                        "policy_api_key": "unused",
                        "policy_model_name": "test-model",
                    }
                ),
            )
        )
        resource = get_first_server_config_dict(config, "terminal_bench_4")
        agent = get_first_server_config_dict(config, f"terminal_bench_4_{harness}")
        environment = get_first_server_config_dict(config, f"terminal_bench_4_{harness}_environment_server")
        assert environment.agent_server.name == f"terminal_bench_4_{harness}"
        assert agent.datasets[0].prepare_script == "benchmarks/terminal_bench_4/prepare.py"
        if harness == "opencode":
            assert resource.environment.sandbox_provider == "tb4_sandbox"
            assert agent.opencode_version == "1.17.11"
            assert agent.opencode_config.tools is None
            assert agent.opencode_config.permission.task == "deny"
            assert agent.opencode_config.permission.webfetch == "deny"
            assert agent.opencode_config.permission.websearch == "deny"
        else:
            assert "opensandbox" in resource.environment.sandbox_provider
            assert agent.harness.step_limit == 500
