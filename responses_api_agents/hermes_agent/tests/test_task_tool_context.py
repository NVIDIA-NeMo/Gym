# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A task's own MCP servers and skills directory in a Hermes session."""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from nemo_gym.agents import AgentSeedRejectedError
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_runtime import MCPToolAccess
from responses_api_agents.hermes_agent import app as hermes_app
from responses_api_agents.hermes_agent.task_context import (
    TASK_CONTEXT_FILE,
    read_task_context,
    task_mcp_servers,
    task_skills_dir,
)
from responses_api_agents.hermes_agent.tests.test_sandbox import agent  # noqa: F401


CONTEXT = {
    "mcp_servers": [
        {"name": "playwright", "transport": "sse", "url": "http://playwright-mcp:3080/sse", "args": []},
        {"name": "docs", "transport": "streamable-http", "url": "http://docs:8080/mcp"},
        {"name": "files", "transport": "stdio", "command": "npx", "args": ["-y", 8080]},
    ],
    "skills_dir": "/app/site/.agents/skills",
}
OFF = {"resources": False, "prompts": False}


def grant(name: str, *, required: bool) -> MCPToolAccess:
    return MCPToolAccess.model_validate(
        {"kind": "mcp", "name": name, "required": required, "connection": {"url": f"http://resources:8000/{name}/mcp"}}
    )


def sandbox_answering(stdout: str = "", *, return_code: int = 0, error_type: str | None = None):
    result = SimpleNamespace(return_code=return_code, stdout=stdout, stderr="", error_type=error_type)
    return SimpleNamespace(exec=AsyncMock(return_value=result))


def test_task_servers_become_hermes_mcp_servers():
    assert task_mcp_servers(CONTEXT) == {
        "playwright": {"url": "http://playwright-mcp:3080/sse", "transport": "sse", "tools": OFF},
        "docs": {"url": "http://docs:8080/mcp", "tools": OFF},
        "files": {"command": "npx", "args": ["-y", "8080"], "tools": OFF},
    }
    assert task_mcp_servers({}) == {}
    assert task_skills_dir(CONTEXT) == "/app/site/.agents/skills"
    assert task_skills_dir({"skills_dir": None}) is None


@pytest.mark.parametrize(
    "context, reason",
    [
        ({"mcp_servers": {"name": "playwright"}}, "mcp_servers is not a list"),
        ({"mcp_servers": [{"transport": "sse", "url": "http://sidecar:1/sse"}]}, "without a name"),
        ({"mcp_servers": [{"name": "playwright", "transport": "sse"}]}, "needs a url"),
        ({"mcp_servers": [{"name": "files", "transport": "stdio"}]}, "needs a command"),
        (
            {"mcp_servers": [{"name": "a", "url": "http://a:1/mcp"}, {"name": "a", "url": "http://b:1/mcp"}]},
            "more than once",
        ),
    ],
)
def test_a_declaration_hermes_cannot_be_given_is_refused(context, reason):
    with pytest.raises(AgentSeedRejectedError, match=reason):
        task_mcp_servers(context)


@pytest.mark.parametrize("skills_dir", ["skills", "/app/../etc", 7])
def test_a_skills_directory_must_be_an_absolute_sandbox_path(skills_dir):
    with pytest.raises(AgentSeedRejectedError, match="skills_dir must be an absolute"):
        task_skills_dir({"skills_dir": skills_dir})


async def test_the_task_context_is_read_from_the_borrowed_sandbox():
    sandbox = sandbox_answering(json.dumps(CONTEXT))
    assert await read_task_context(sandbox) == CONTEXT
    assert TASK_CONTEXT_FILE in sandbox.exec.await_args.args[0]
    assert await read_task_context(sandbox_answering("")) == {}
    with pytest.raises(AgentSeedRejectedError, match="is not JSON"):
        await read_task_context(sandbox_answering("{not json"))
    with pytest.raises(AgentSeedRejectedError, match="is not a JSON object"):
        await read_task_context(sandbox_answering('["playwright"]'))


@pytest.mark.parametrize("failure", [{"return_code": 1}, {"error_type": "sandbox"}])
async def test_an_unreadable_task_context_fails_instead_of_passing_for_none(failure):
    """Running the task without the tools it declares would score a silent zero."""
    with pytest.raises(RuntimeError, match="could not read"):
        await read_task_context(sandbox_answering("", **failure))


def test_the_config_carries_the_task_servers_and_skills(agent):  # noqa: F811
    assert agent.supports_task_tool_context()
    config = yaml.safe_load(agent._build_config([grant("docs", required=False)], CONTEXT))
    # A server that a grant already names keeps the grant's connection.
    assert config["mcp_servers"]["docs"]["url"] == "http://resources:8000/docs/mcp"
    assert config["mcp_servers"]["playwright"] == {
        "url": "http://playwright-mcp:3080/sse",
        "transport": "sse",
        "tools": OFF,
    }
    assert config["mcp_servers"]["files"] == {"command": "npx", "args": ["-y", "8080"], "tools": OFF}
    assert config["skills"] == {"external_dirs": ["/app/site/.agents/skills"]}
    plain = yaml.safe_load(agent._build_config())
    assert "mcp_servers" not in plain and "skills" not in plain


def test_the_worker_checks_the_task_servers_and_keeps_their_tools_enabled(agent):  # noqa: F811
    body = NeMoGymResponseCreateParamsNonStreaming(input="task")
    payload = agent._worker_payload(
        body,
        agent_session_id="session",
        rollout_id="episode-a1",
        mcp_accesses=[grant("docs", required=False), grant("search", required=True)],
        task_context=CONTEXT,
    )
    assert payload["mcp_servers"] == ["docs", "search", "playwright", "files"]
    # The task's own servers are required, also the one an optional grant names.
    assert payload["required_mcp_servers"] == ["search", "playwright", "docs", "files"]
    # The fixture restricts the toolsets to the terminal; the task's tools and skills stay reachable.
    assert payload["enabled_toolsets"] == ["terminal", "docs", "search", "playwright", "files", "skills"]
    plain = agent._worker_payload(body, agent_session_id="session", rollout_id="episode-a1", mcp_accesses=[])
    assert plain["mcp_servers"] == [] and plain["required_mcp_servers"] == []
    assert plain["enabled_toolsets"] == ["terminal"]


async def test_a_task_server_named_like_a_hermes_toolset_refuses_the_seed(agent, monkeypatch):  # noqa: F811
    monkeypatch.setattr(hermes_app, "builtin_toolsets", lambda: {"terminal", "skills"})
    body = SimpleNamespace(tool_accesses=[])
    monkeypatch.setattr(agent.__class__, "effective_tool_accesses", lambda self, request: [])
    await agent._check_task_tools(body, CONTEXT)
    with pytest.raises(AgentSeedRejectedError, match="collide with Hermes toolsets: terminal"):
        await agent._check_task_tools(body, {"mcp_servers": [{"name": "terminal", "url": "http://sidecar:1/mcp"}]})
