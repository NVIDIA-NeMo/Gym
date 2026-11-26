# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A task's own MCP servers and skills directory in a borrowed-sandbox OpenCode session."""

import json
from types import SimpleNamespace

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.core import NgFailureEvent
from responses_api_agents.opencode_agent.app import TASK_CONTEXT_FILE, _task_mcp_servers
from responses_api_agents.opencode_agent.runtime import OPENCODE_V2_VERSION
from responses_api_agents.opencode_agent.tests.test_mcp_grants import grant, read_params
from responses_api_agents.opencode_agent.tests.test_sandbox_sessions import seed, setup  # noqa: F401


def declare(sandbox, context, *, return_code=0, error_type=None) -> None:
    """Make the sandbox answer the read of the task context file with ``context`` (text as written, or JSON)."""
    stdout = context if isinstance(context, str) else json.dumps(context)
    run_exec = sandbox.run_exec

    async def with_task_context(command, **kwargs):
        if TASK_CONTEXT_FILE in command:
            return SimpleNamespace(return_code=return_code, stdout=stdout, stderr="", error_type=error_type)
        return await run_exec(command, **kwargs)

    sandbox.exec.side_effect = with_task_context


def connect(sandbox, names) -> None:
    """Stand in for required_mcp.mjs: record every named server as connected before the run exits."""
    wait_exit = sandbox.wait_exit

    async def record_then_exit():
        sandbox.files[f"{sandbox.directory}/mcp-status.json"] = json.dumps(
            {name: {"status": "connected"} for name in names}
        )
        return await wait_exit()

    sandbox.wait_exit = record_then_exit


def launched_config(sandbox) -> dict:
    return json.loads(read_params(sandbox.files[f"{sandbox.directory}/params.sh"])["env"]["OPENCODE_CONFIG_CONTENT"])


def test_task_mcp_servers_and_skills_reach_opencode(setup):  # noqa: F811
    agent, sandbox = setup
    agent.config.opencode_config = {"permission": {"*": "allow", "skill": "deny", "bash": {"*": "allow"}}}
    declare(
        sandbox,
        {
            "mcp_servers": [
                {"name": "playwright", "transport": "sse", "url": "http://playwright-mcp:3080/sse", "args": []},
                {"name": "files", "transport": "stdio", "command": "npx", "args": ["-y", "files-server"]},
            ],
            "skills_dir": "/app/site/.agents/skills",
        },
    )
    connect(sandbox, ["playwright", "files"])
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        assert created.json()["supports_task_tool_context"] is True
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
        assert result.status_code == 200, result.text
    config = launched_config(sandbox)
    assert config["mcp"] == {
        # OpenCode's remote entry tries streamable HTTP and then SSE, so it serves the task's sse server.
        "playwright": {"type": "remote", "url": "http://playwright-mcp:3080/sse", "enabled": True, "oauth": False},
        "files": {"type": "local", "command": ["npx", "-y", "files-server"], "enabled": True},
    }
    # The task's own servers are required: the plugin stops the run when one did not connect.
    assert json.loads(sandbox.files[f"{sandbox.directory}/mcp-grants.json"]) == {
        "required": ["playwright", "files"],
        "optional": [],
    }
    assert config["skills"] == {"paths": ["/app/site/.agents/skills"]}
    # The allow follows the profile's other rules, and the configured profile itself is left as it was.
    assert list(config["permission"])[-1] == "skill"
    assert config["permission"]["skill"] == "allow"
    assert agent.config.opencode_config["permission"]["skill"] == "deny"


def test_a_task_without_declarations_changes_nothing(setup):  # noqa: F811
    agent, sandbox = setup
    agent.config.opencode_config = {"permission": {"*": "allow", "skill": "deny"}}
    with TestClient(agent.setup_webserver()) as client:
        created = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
        assert created.status_code == 200, created.text
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 200
    config = launched_config(sandbox)
    assert "mcp" not in config and "skills" not in config
    assert config["permission"] == {"*": "allow", "skill": "deny"}


def test_a_granted_task_server_keeps_the_grant_and_becomes_required(setup):  # noqa: F811
    agent, sandbox = setup
    declare(sandbox, {"mcp_servers": [{"name": "task_tools", "transport": "sse", "url": "http://sidecar:1/sse"}]})
    connect(sandbox, ["task_tools"])
    body = seed().model_dump(mode="json")
    body["tool_accesses"] = [grant("task_tools", required=False)]
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=body).status_code == 200
        assert client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"}).status_code == 200
    assert launched_config(sandbox)["mcp"]["task_tools"]["url"] == "http://resources.example:8000/task_tools/mcp"
    assert json.loads(sandbox.files[f"{sandbox.directory}/mcp-grants.json"]) == {
        "required": ["task_tools"],
        "optional": [],
    }


def test_a_task_server_that_does_not_connect_fails_the_run(setup):  # noqa: F811
    agent, sandbox = setup
    declare(sandbox, {"mcp_servers": [{"name": "playwright", "transport": "sse", "url": "http://sidecar:1/sse"}]})
    wait_exit = sandbox.wait_exit

    async def fail_then_exit():
        sandbox.files[f"{sandbox.directory}/mcp-status.json"] = json.dumps(
            {"playwright": {"status": "failed", "error": "connection refused"}}
        )
        return await wait_exit()

    sandbox.wait_exit = fail_then_exit
    with TestClient(agent.setup_webserver()) as client:
        assert client.post("/v1/agent_sessions", json=seed().model_dump(mode="json")).status_code == 200
        result = client.post("/ng-rollout/opencode-smoke-a2/v1/responses", json={"input": "task"})
    assert result.status_code == 502
    assert result.json()["detail"] == (
        "Required Gym MCP tools could not be initialized: playwright: failed (connection refused)"
    )


@pytest.mark.parametrize(
    "context, reason",
    [
        ("{not json", "is not JSON"),
        (["playwright"], "is not a JSON object"),
        ({"mcp_servers": {"name": "playwright"}}, "mcp_servers is not a list"),
        ({"mcp_servers": [{"transport": "sse", "url": "http://sidecar:1/sse"}]}, "without a name"),
        ({"mcp_servers": [{"name": "playwright", "transport": "sse"}]}, "needs a url"),
        ({"mcp_servers": [{"name": "files", "transport": "stdio"}]}, "needs a command"),
        (
            {"mcp_servers": [{"name": "a", "url": "http://sidecar:1/sse"}, {"name": "a", "url": "http://b:1/sse"}]},
            "more than once",
        ),
        ({"skills_dir": "skills"}, "skills_dir must be an absolute"),
    ],
)
def test_a_declaration_opencode_cannot_be_given_refuses_the_seed(setup, context, reason):  # noqa: F811
    agent, sandbox = setup
    declare(sandbox, context)
    with TestClient(agent.setup_webserver()) as client:
        response = client.post("/v1/agent_sessions", json=seed().model_dump(mode="json"))
    assert response.status_code == 422, response.text
    assert reason in response.json()["detail"]
    sandbox.launch.assert_not_awaited()


@pytest.mark.parametrize("failure", [{"return_code": 1}, {"error_type": "sandbox"}])
async def test_an_unreadable_task_context_fails_the_seed_without_running(setup, failure):  # noqa: F811
    """Running the task without the tools it declares would score a silent zero, so the seed fails instead."""
    agent, sandbox = setup
    declare(sandbox, "", **failure)
    request = Request({"type": "http", "session": {}})
    with pytest.raises(NgFailureEvent, match=f"could not read {TASK_CONTEXT_FILE}") as failed:
        await agent.seed_agent_session(request, seed())
    assert failed.value.boundary == "harness"
    assert failed.value.operation == "opencode.setup"
    assert failed.value.stage == "seed"
    assert not request.session
    sandbox.launch.assert_not_awaited()
    assert not any(record.state is not None for record in agent._session_records.values())


def test_opencode_2_does_not_declare_task_tool_context(setup):  # noqa: F811
    agent, _ = setup
    assert agent.supports_task_tool_context()
    agent.config.opencode_version = OPENCODE_V2_VERSION
    assert not agent.supports_task_tool_context()


def test_task_servers_take_the_configured_mcp_timeout():
    servers = _task_mcp_servers(
        {
            "mcp_servers": [
                {"name": "remote", "transport": "streamable-http", "url": "http://sidecar:1/mcp"},
                {"name": "local", "transport": "stdio", "command": "server", "args": [1]},
            ]
        },
        900000,
    )
    assert servers == {
        "remote": {
            "type": "remote",
            "url": "http://sidecar:1/mcp",
            "enabled": True,
            "oauth": False,
            "timeout": 900000,
        },
        "local": {"type": "local", "command": ["server", "1"], "enabled": True, "timeout": 900000},
    }
    assert _task_mcp_servers({}, None) == {}
