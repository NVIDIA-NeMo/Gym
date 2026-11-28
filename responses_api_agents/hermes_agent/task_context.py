# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A Harbor task's own MCP servers and skills directory, as Hermes configuration."""

import json
from collections.abc import Mapping
from pathlib import PurePosixPath
from shlex import quote
from typing import Any

from pydantic import JsonValue

from nemo_gym.agents import AgentSeedRejectedError
from nemo_gym.sandbox import AsyncSandbox


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


def task_mcp_servers(task_context: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Hermes ``mcp_servers`` entries for the MCP servers the task itself declares, by name.

    A server with a URL is reached over streamable HTTP, or over SSE when the task says so; a ``stdio`` server is
    started by Hermes in the sandbox. A declaration that is neither is the task's own error, so the seed is refused
    rather than retried.
    """
    declared = task_context.get("mcp_servers") or []
    if not isinstance(declared, list):
        raise AgentSeedRejectedError(f"{TASK_CONTEXT_FILE}: mcp_servers is not a list")
    servers: dict[str, dict[str, Any]] = {}
    for server in declared:
        name = server.get("name") if isinstance(server, Mapping) else None
        if not isinstance(name, str) or not name:
            raise AgentSeedRejectedError("The task declares an MCP server without a name")
        if name in servers:
            raise AgentSeedRejectedError(f"The task declares the MCP server {name!r} more than once")
        url, command, args = server.get("url"), server.get("command"), server.get("args") or []
        transport = server.get("transport")
        if transport == "stdio":
            if not isinstance(command, str) or not command or not isinstance(args, list):
                raise AgentSeedRejectedError(
                    f"The task's stdio MCP server {name!r} needs a command and a list of args"
                )
            entry: dict[str, Any] = {"command": command, "args": [str(arg) for arg in args]}
        elif isinstance(url, str) and url:
            entry = {"url": url}
            if transport == "sse":
                entry["transport"] = "sse"
        else:
            raise AgentSeedRejectedError(f"The task's MCP server {name!r} needs a url")
        # As for a grant: the server is there for its tools, so Hermes' resource and prompt helper tools stay off.
        entry["tools"] = {"resources": False, "prompts": False}
        servers[name] = entry
    return servers


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
