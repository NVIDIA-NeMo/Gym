# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate owner-provided Harbor task tools into OpenCode's native configuration."""

from pathlib import PurePosixPath
from shlex import quote
from typing import Literal

from pydantic import BaseModel, Field, JsonValue, model_validator

from nemo_gym.sandbox import AsyncSandbox


TASK_CONTEXT_FILE = "/tmp/.nemo-gym/task.json"


class TaskMCPServer(BaseModel):
    name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9_-]+$")
    transport: Literal["stdio", "sse", "streamable-http", "http"] = "sse"
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_connection(self) -> "TaskMCPServer":
        if not (self.command if self.transport == "stdio" else self.url):
            raise ValueError("Task MCP server requires a command or URL")
        return self


class TaskToolContext(BaseModel):
    mcp_servers: list[TaskMCPServer] = Field(default_factory=list)
    skills_dir: str | None = None

    @model_validator(mode="after")
    def validate_declarations(self) -> "TaskToolContext":
        if len({server.name for server in self.mcp_servers}) != len(self.mcp_servers):
            raise ValueError("Task MCP server names must be unique")
        if self.skills_dir and not PurePosixPath(self.skills_dir).is_absolute():
            raise ValueError("Task skills directory must be absolute")
        return self

    def opencode_config(self) -> dict[str, JsonValue]:
        """Use native MCP transport negotiation and explicit skill discovery paths."""
        config: dict[str, JsonValue] = {}
        if self.mcp_servers:
            config["mcp"] = {
                server.name: (
                    {"type": "local", "command": [server.command, *server.args], "enabled": True}
                    if server.transport == "stdio"
                    else {"type": "remote", "url": server.url, "enabled": True, "oauth": False}
                )
                for server in self.mcp_servers
            }
        if self.skills_dir:
            config["skills"] = {"paths": [self.skills_dir]}
        return config


async def read_task_context(sandbox: AsyncSandbox) -> TaskToolContext:
    """Read the optional owner declaration; distinguish missing files from failed reads."""
    result = await sandbox.exec(
        f"if [ -e {TASK_CONTEXT_FILE} ]; then cat {TASK_CONTEXT_FILE}; else printf '{{}}'; fi",
        timeout_s=30,
    )
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f"Unable to read task tool context: {result.error_type or result.stderr}")
    context = TaskToolContext.model_validate_json(result.stdout or "{}")
    if context.skills_dir:
        result = await sandbox.exec(f"test -d {quote(context.skills_dir)}", timeout_s=30)
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(f"Task skills directory is unavailable: {context.skills_dir}")
    return context
