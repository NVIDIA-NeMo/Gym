# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness-independent task context resolved during resource provisioning."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SandboxMCPServer(BaseModel):
    """An MCP service reachable from inside the task sandbox."""

    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    transport: Literal["stdio", "sse", "streamable-http"]
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_transport(self):
        if self.transport == "stdio":
            if not self.command or self.url is not None:
                raise ValueError("stdio MCP requires command and no URL")
        elif not self.url or self.command is not None or self.args:
            raise ValueError("HTTP MCP requires URL and no command or args")
        return self


class AgentTaskContext(BaseModel):
    """Task requirements supplied by resources and honored by a compatible harness.

    Instruction fills an empty task input when resources resolves it from a pinned
    package. MCP connections are sandbox-local, not connections from the server.
    """

    model_config = ConfigDict(extra="forbid")
    instruction: str | None = None
    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    skills_dir: str | None = None
    mcp_servers: list[SandboxMCPServer] = Field(default_factory=list)
