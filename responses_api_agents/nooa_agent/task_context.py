# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Harbor task's own MCP servers and skills directory, read by the NOOA controller at seed."""

import json
import keyword
from collections.abc import Mapping
from pathlib import PurePosixPath
from shlex import quote
from typing import Any

from pydantic import JsonValue

from nemo_gym.agents import AgentSeedRejectedError
from nemo_gym.sandbox import AsyncSandbox
from responses_api_agents.nooa_agent.task_tools import TaskMCPServer


# Where a Harbor environment server leaves the task's own MCP servers and skills directory in the agent's sandbox.
TASK_CONTEXT_FILE = "/tmp/.nemo-gym/task.json"


async def read_task_context(sandbox: AsyncSandbox) -> dict[str, JsonValue]:
    """The task's own MCP servers and skills directory, from the file a Harbor environment server writes.

    A read that ran and found nothing returns ``{}``. A provider failure raises, because running the task without
    the tools it declares would score a silent zero. A file that is not a JSON object is the task's own error, so
    the seed is refused rather than retried.
    """
    path = quote(TASK_CONTEXT_FILE)
    # A file that exists but cannot be read fails the read instead of passing for an absent one.
    result = await sandbox.exec(f"if [ -e {path} ]; then cat {path}; fi", cwd="/", timeout_s=30)
    if result.error_type is not None or result.return_code != 0:
        raise RuntimeError(
            f"The sandbox could not read {TASK_CONTEXT_FILE}: {result.error_type or result.return_code}"
        )
    text = (result.stdout or "").strip()
    if not text:
        return {}
    try:
        context = json.loads(text)
    except json.JSONDecodeError as error:
        raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE} is not JSON") from error
    if not isinstance(context, dict):
        raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE} is not a JSON object")
    return context


def _declares(agent_class: type, attribute: str) -> bool:
    return hasattr(agent_class, attribute) or any(
        attribute in vars(base).get("__annotations__", {}) for base in agent_class.__mro__
    )


def task_mcp_servers(task_context: Mapping[str, Any], agent_class: type) -> list[TaskMCPServer]:
    """The MCP servers the task itself declares, each with the attribute its tools take on ``agent_class``.

    A server with a URL is reached over streamable HTTP, or over SSE when the task says so; a ``stdio`` server is
    started by the worker in the sandbox. A declaration that is neither, or a name the agent cannot carry as an
    attribute, is the task's own error, so the seed is refused rather than retried.
    """
    declared = task_context.get("mcp_servers") or []
    if not isinstance(declared, list):
        raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE}: mcp_servers is not a list")
    servers: dict[str, TaskMCPServer] = {}
    for server in declared:
        name = server.get("name") if isinstance(server, Mapping) else None
        if not isinstance(name, str) or not name:
            raise AgentSeedRejectedError("The task declares an MCP server without a name")
        attribute = name.replace("-", "_")
        if attribute in servers:
            raise AgentSeedRejectedError(f"The task declares the MCP server {name!r} more than once")
        if not attribute.isidentifier() or attribute.startswith("_") or keyword.iskeyword(attribute):
            raise AgentSeedRejectedError(f"NOOA cannot name an agent attribute after the task's MCP server {name!r}")
        if _declares(agent_class, attribute):
            raise AgentSeedRejectedError(
                f"The task's MCP server {name!r} collides with {agent_class.__name__}.{attribute}"
            )
        url, command, args = server.get("url"), server.get("command"), server.get("args") or []
        if server.get("transport") == "stdio":
            if not isinstance(command, str) or not command or not isinstance(args, list):
                raise AgentSeedRejectedError(
                    f"The task's stdio MCP server {name!r} needs a command and a list of args"
                )
            servers[attribute] = TaskMCPServer(
                name=name, attribute=attribute, transport="stdio", command=command, args=[str(arg) for arg in args]
            )
        elif isinstance(url, str) and url:
            transport = "sse" if server.get("transport") == "sse" else "streamable-http"
            servers[attribute] = TaskMCPServer(name=name, attribute=attribute, transport=transport, url=url)
        else:
            raise AgentSeedRejectedError(f"The task's MCP server {name!r} needs a url")
    return list(servers.values())


def task_skills_dir(task_context: Mapping[str, Any]) -> str | None:
    """The task's skills directory, an absolute sandbox path, or ``None`` when the task declares none."""
    skills_dir = task_context.get("skills_dir")
    if skills_dir is None or skills_dir == "":
        return None
    if (
        not isinstance(skills_dir, str)
        or "\x00" in skills_dir
        or not PurePosixPath(skills_dir).is_absolute()
        or ".." in PurePosixPath(skills_dir).parts
    ):
        raise AgentSeedRejectedError(
            f"The task's skills_dir must be an absolute, normalized sandbox path: {skills_dir!r}"
        )
    return skills_dir
