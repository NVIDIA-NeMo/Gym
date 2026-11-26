# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A task's own MCP servers and skills directory in a sandboxed NOOA session."""

import json
import sys
from typing import Any
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from nooa.agentdoc import doc
from nooa_bench.bench_agent import BenchAgent

from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming, NeMoGymResponseFunctionToolCall
from nemo_gym.sandbox import SandboxExecResult
from responses_api_agents.nooa_agent.app import NOOAAgent
from responses_api_agents.nooa_agent.result import is_transient_infrastructure_error
from responses_api_agents.nooa_agent.runner import NOOARunRequest
from responses_api_agents.nooa_agent.sandbox_entrypoint import SandboxInput
from responses_api_agents.nooa_agent.task_context import TASK_CONTEXT_FILE, task_mcp_servers
from responses_api_agents.nooa_agent.task_tools import TaskMCPServer, agent_class_with_task_tools, open_task_tools
from responses_api_agents.nooa_agent.tests.test_app import request, seed
from responses_api_agents.nooa_agent.tests.test_app import result as run_result
from responses_api_agents.nooa_agent.tests.test_config import ExampleAgent, agent_config, invocation_config
from responses_api_agents.nooa_agent.tests.test_gym_llm import FakeHTTPResponse, model_response
from responses_api_agents.nooa_agent.tests.test_runner import ResourceUsingAgent, policy_runner
from responses_api_agents.nooa_agent.tests.test_sandbox_sessions import access, close, setup  # noqa: F401


# An MCP server that counts the calls one process has answered: a new session starts a new process at zero.
COUNTER_SERVER = """
from mcp.server import MCPServer

server = MCPServer("counter")
calls = 0


@server.tool()
def count() -> str:
    \"\"\"Count the calls this server process has answered.\"\"\"
    global calls
    calls += 1
    return str(calls)


server.run()
"""


def declare(sandbox, context, *, return_code=0, error_type=None) -> None:
    """Make the sandbox answer the read of the task context file with ``context`` (text as written, or JSON)."""
    stdout = context if isinstance(context, str) else json.dumps(context)

    async def with_task_context(command, **kwargs):
        if TASK_CONTEXT_FILE in command:
            return SandboxExecResult(stdout=stdout, stderr="", return_code=return_code, error_type=error_type)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    sandbox.exec = AsyncMock(side_effect=with_task_context)


async def test_task_mcp_servers_and_skills_reach_the_worker(setup):  # noqa: F811
    instance, sandbox, _, _ = setup
    declare(
        sandbox,
        {
            "mcp_servers": [
                {"name": "playwright", "transport": "sse", "url": "http://playwright-mcp:3080/sse", "args": []},
                {"name": "file-tools", "transport": "stdio", "command": "npx", "args": ["-y", 1]},
                {"name": "search", "url": "http://search:1/mcp"},
            ],
            "skills_dir": "/app/site/.agents/skills",
        },
    )
    seeded = await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    assert seeded.supports_task_tool_context is True
    state = instance._require_agent_session("session")
    servers = [
        TaskMCPServer(
            name="playwright", attribute="playwright", transport="sse", url="http://playwright-mcp:3080/sse"
        ),
        TaskMCPServer(name="file-tools", attribute="file_tools", transport="stdio", command="npx", args=["-y", "1"]),
        TaskMCPServer(name="search", attribute="search", transport="streamable-http", url="http://search:1/mcp"),
    ]
    assert state.task_mcp_servers == servers
    assert state.task_skills_dir == "/app/site/.agents/skills"

    state.runner.run = AsyncMock(return_value=run_result())
    await instance.responses(
        request(session_id="session", rollout_id="rollout-a2"), NeMoGymResponseCreateParamsNonStreaming(input="task")
    )
    sent: NOOARunRequest = state.runner.run.await_args.args[0]
    assert sent.task_mcp_servers == servers
    assert sent.task_skills_dir == "/app/site/.agents/skills"
    # The declarations cross into the sandbox in the worker's input file.
    await state.runner._stage_activation(sent)
    staged = SandboxInput.model_validate_json(sandbox.files[f"{state.runner.directory}/input.json"])
    assert staged.request.task_mcp_servers == servers
    assert staged.request.task_skills_dir == "/app/site/.agents/skills"
    await close(instance)


async def test_a_task_without_declarations_sends_none(setup):  # noqa: F811
    instance, _, _, _ = setup
    await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    state = instance._require_agent_session("session")
    assert state.task_mcp_servers == []
    assert state.task_skills_dir is None
    await close(instance)


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
            {
                "mcp_servers": [
                    {"name": "a-b", "url": "http://sidecar:1/sse"},
                    {"name": "a_b", "url": "http://b:1/sse"},
                ]
            },
            "more than once",
        ),
        ({"mcp_servers": [{"name": "9lives", "url": "http://sidecar:1/sse"}]}, "cannot name an agent attribute"),
        ({"mcp_servers": [{"name": "context", "url": "http://sidecar:1/sse"}]}, "collides with ExampleAgent.context"),
        ({"skills_dir": "skills"}, "skills_dir must be an absolute"),
    ],
)
async def test_a_declaration_nooa_cannot_be_given_refuses_the_seed(setup, context, reason):  # noqa: F811
    instance, sandbox, _, _ = setup
    declare(sandbox, context)
    with pytest.raises(HTTPException) as refused:
        await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    assert refused.value.status_code == 422
    assert reason in refused.value.detail
    await close(instance)
    sandbox.disconnect.assert_awaited_once()


@pytest.mark.parametrize("failure", [{"return_code": 1}, {"error_type": "sandbox"}])
async def test_an_unreadable_task_context_fails_the_seed_without_running(setup, failure):  # noqa: F811
    """Running the task without the tools it declares would score a silent zero, so the seed fails instead."""
    instance, sandbox, _, _ = setup
    declare(sandbox, "", **failure)
    with pytest.raises(RuntimeError, match=f"could not read {TASK_CONTEXT_FILE}"):
        await instance.seed_agent_session(request(), seed(sandbox_access=access()))
    with pytest.raises(HTTPException, match="closing"):
        instance._require_agent_session("session")
    await close(instance)


def test_embedded_nooa_does_not_declare_task_tool_context(setup):  # noqa: F811
    instance, _, _, _ = setup
    assert instance.supports_task_tool_context()
    embedded = NOOAAgent(config=agent_config(nooa=invocation_config()), server_client=instance.server_client)
    assert not embedded.supports_task_tool_context()


def test_a_server_name_becomes_an_agent_attribute():
    servers = task_mcp_servers({"mcp_servers": [{"name": "my-tools", "url": "http://sidecar:1/mcp"}]}, ExampleAgent)
    assert [(server.name, server.attribute) for server in servers] == [("my-tools", "my_tools")]
    assert task_mcp_servers({}, ExampleAgent) == []


def counter(tmp_path) -> TaskMCPServer:
    script = tmp_path / "counter_server.py"
    script.write_text(COUNTER_SERVER)
    return TaskMCPServer(
        name="call-counter", attribute="call_counter", transport="stdio", command=sys.executable, args=[str(script)]
    )


async def test_the_model_calls_a_task_server_through_one_held_session(tmp_path):
    runner, _ = policy_runner(ResourceUsingAgent)
    model_requests = []
    arguments = json.dumps({"code": "print(await self.call_counter.count(), await self.call_counter.count())"})

    async def post(*, json: Any, **_: Any) -> FakeHTTPResponse:
        model_requests.append(json)
        first = len(model_requests) == 1
        output = NeMoGymResponseFunctionToolCall(
            id=f"fc-{len(model_requests)}",
            call_id=f"call-{len(model_requests)}",
            name="execute_python" if first else "return_result",
            arguments=arguments if first else '{"result":"done"}',
        )
        return FakeHTTPResponse(model_response(output, response_id=f"response-{len(model_requests)}"))

    runner._server_client.post = AsyncMock(side_effect=post)
    result = await runner.run(
        NOOARunRequest(
            responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
            model_url_path="/v1/responses",
            task_mcp_servers=[counter(tmp_path)],
            task_skills_dir="/app/site/.agents/skills",
        )
    )
    assert result.return_value == "done"
    prompt = model_requests[0].model_dump_json()
    # The server is on the agent as a typed tool object, and the skills directory is named.
    assert "call_counter: CallCounterTool" in prompt
    assert "Task skills are in /app/site/.agents/skills. Read the relevant SKILL.md files." in prompt
    executed = next(tool for tool in result.trajectory.tool_calls if tool.tool_call_id == "call-1")
    # Two calls answered by one server process: the session was held, not opened for each call.
    assert "1 2" in executed.output


async def test_the_benchmark_agent_documents_the_tools_of_a_task_server(tmp_path):
    tools, _ = await open_task_tools([counter(tmp_path)])
    # The expression BenchAgent renders itself with in its prompt.
    rendered = doc(agent_class_with_task_tools(BenchAgent, tools), concise=True)
    assert "call_counter: CallCounterTool" in rendered
    assert "async def count(self) -> Any:" in rendered
    assert "Count the calls this server process has answered." in rendered
    assert agent_class_with_task_tools(BenchAgent, {}) is BenchAgent


async def test_a_task_server_that_does_not_connect_fails_the_run_as_an_outage():
    runner, calls = policy_runner(ResourceUsingAgent)
    absent = TaskMCPServer(
        name="absent",
        attribute="absent",
        transport="stdio",
        command=sys.executable,
        args=["-c", "raise SystemExit(1)"],
    )
    with pytest.raises(ConnectionError, match="The task's MCP server 'absent' did not connect") as failed:
        await runner.run(
            NOOARunRequest(
                responses_create_params=NeMoGymResponseCreateParamsNonStreaming(input="question"),
                model_url_path="/v1/responses",
                task_mcp_servers=[absent],
            )
        )
    assert is_transient_infrastructure_error(failed.value)
    assert not calls
