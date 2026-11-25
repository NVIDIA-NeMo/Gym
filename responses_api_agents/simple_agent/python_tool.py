# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Simple Agent's Python tools: the model's code runs in a Python interpreter beside the agent loop, in its sandbox.

A Python tool is a harness tool. The agent server's config defines it (``python_tools``) and a task opts into it by
name in its ``agent_context`` (``python_tools``), as a Harbor task declares the tools it needs. Only a sandboxed loop
runs one: each tool gets one worker process (``python_tool_worker.py``) in the sandbox the environment lends, started
on the tool's first call and kept for the loop's activation, so names a call defines stay defined for later calls of
that activation. ``python_tool_client.py`` starts and calls the workers.

This module holds only the settings, which the agent server's config imports; the worker client loads only in the
sandbox.
"""

import builtins
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints


# A requirement pinned to one version, such as numpy==2.5.5 or pandas[performance]==3.1.1.
PinnedRequirement = Annotated[
    str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*(\[[A-Za-z0-9._,-]+\])?==[A-Za-z0-9.+!_-]+$")
]
PythonName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
ModuleName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*(\.[A-Za-z_][A-Za-z0-9_]*)*$")]
BuiltinName = Literal[tuple(sorted(vars(builtins)))]


class PythonToolConfig(BaseModel):
    """How one Python tool runs. Every task that opts into the tool by name gets these settings."""

    model_config = ConfigDict(extra="forbid")

    timeout_seconds: int = Field(
        default=10,
        ge=1,
        description="How long one call's code may run before it fails with 'code timed-out'.",
    )
    requirements: list[PinnedRequirement] = Field(
        default_factory=list,
        description=(
            "Packages for the tool's own Python runtime, which this server builds and uploads into the sandbox at "
            "seed, each pinned with ==, for example numpy==2.5.5. Their own dependencies resolve when a host first "
            "builds the runtime. Empty, the tool runs on the agent loop's runtime."
        ),
    )
    preload: dict[PythonName, ModuleName] = Field(
        default_factory=dict,
        description="Global names bound to modules before the first call, for example {np: numpy}.",
    )
    builtins: list[BuiltinName] | None = Field(
        default=None,
        description="The builtins the code may use. Unset, all of them.",
    )
    memory_mib: int | None = Field(
        default=None,
        ge=64,
        description=(
            "The most memory, in MiB, the tool's worker may allocate (its RLIMIT_DATA). Code that asks for more gets "
            "a MemoryError in the tool's output, instead of the sandbox running out of memory and stopping the agent "
            "loop with it. Code that holds CAP_SYS_RESOURCE, such as root in a sandbox that keeps it, can raise the "
            "limit, so the sandbox's memory limit is the boundary: keep this below it, since the sandbox also holds "
            "the loop. Unset, only the sandbox's limit applies."
        ),
    )


class PythonToolSpec(BaseModel):
    """What the sandboxed loop needs to start a tool's worker: its config and where its files are in the sandbox."""

    model_config = ConfigDict(extra="forbid")

    config: PythonToolConfig
    python: str
    worker: str
    workdir: str | None = None
