# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Harness-independent task context resolved during resource provisioning."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SandboxMCPServer(BaseModel):
    """An MCP service reachable from inside the task sandbox."""

    model_config = ConfigDict(extra="forbid")
    name: str
    transport: Literal["stdio", "sse", "streamable-http"]
    url: str | None = None
    command: str | None = None
    args: list[str] = Field(default_factory=list)


class AgentTaskContext(BaseModel):
    """Task requirements supplied by resources and honored by a compatible harness.

    Instruction replaces the task input when resources resolves it from a pinned
    package. MCP connections are sandbox-local, not connections from the server.
    """

    model_config = ConfigDict(extra="forbid")
    instruction: str | None = None
    timeout_sec: float | None = Field(default=None, gt=0)
    user: str | int | None = None
    skills_dir: str | None = None
    mcp_servers: list[SandboxMCPServer] = Field(default_factory=list)
