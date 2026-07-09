# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The external-harness adapter contract.

The agent orchestrator is harness-agnostic: it creates a sandbox, installs the
harness, prepares it, launches it, harvests the outcome, verifies, and builds a
trajectory. Everything a specific CLI needs that differs from the others lives
behind this small contract:

  - HarnessSpec        the declarative, config-facing core; adapters subclass it
                       to add their own typed, config-overridable fields.
  - HarnessAdapter     the per-harness code: how to install it (runtime), wire
                       its endpoint and any config files into the sandbox
                       (prepare), and launch it (launch).

Only these three methods should ever need to know which CLI is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class HarnessSpec(BaseModel):
    """Fields every external harness shares. Adapters subclass this to add
    harness-specific fields; the whole thing is populated from YAML.

    This is the one type on the contract that crosses the config boundary, so it
    stays a pydantic model: it validates YAML input and forbids unknown keys."""

    model_config = ConfigDict(extra="forbid")

    name: str
    pinned_version: Optional[str] = None  # pin the CLI version for reproducibility
    timeout_s: float = 1200.0
    setup_timeout_s: float = 600.0
    extra_env: dict[str, str] = Field(default_factory=dict)
    # Request fingerprints that mark a model call as non-task (title/summary
    # calls), excluded from training by the trajectory builder.
    aux_fingerprints: list[str] = Field(default_factory=list)


@dataclass
class RuntimeRequirement:
    """What must be installed into a fresh sandbox before the harness launches.
    Commands run in the sandbox; skip them by returning an empty list (e.g. when
    the image already bakes the harness in)."""

    setup_cmds: list[list[str]] = field(default_factory=list)


@dataclass
class RolloutEndpoints:
    """Where this rollout's model calls go. ``model_base_url`` is already
    rollout-scoped (it carries the /ng-rollout/<id> prefix) and is the root of
    the gate; each adapter appends whatever dialect suffix its client uses."""

    model_base_url: str
    api_key: str = "dummy_key"
    model_name: str = ""


@dataclass
class SeedResult:
    """What the environment returned for this rollout. ``files`` seed the world.
    ``mcp_servers`` carries any tools the environment lends over MCP: one entry
    per server name -> {"type", "url", "headers"}, where ``url`` is already
    reachable from inside the sandbox and ``headers`` holds the per-rollout
    signed session token. Empty when the environment lends no tools."""

    files: dict[str, str] = field(default_factory=dict)
    mcp_servers: dict[str, dict] = field(default_factory=dict)
    # What the env wants pulled out of the sandbox as the graded outcome. Both
    # optional — empty means the env grades the assembled response instead of a
    # sandbox artifact.
    #   harvest_files: files the harness writes, read back verbatim
    #     (e.g. ["answer.txt"]).
    #   harvest_commands: commands run in the sandbox whose output is captured
    #     (e.g. ["git -C /testbed diff"] to extract a patch for SWE-bench).
    # This is the whole job-shape extension point: the harness runs whatever the
    # env declares and forwards it to verify; it never learns the job shape.
    harvest_files: list[str] = field(default_factory=list)
    harvest_commands: list[str] = field(default_factory=list)


@runtime_checkable
class HarnessAdapter(Protocol):
    """One per harness, registered by name. Owns everything CLI-specific."""

    name: str
    spec_model: type[HarnessSpec]

    def runtime(self, spec: HarnessSpec) -> RuntimeRequirement: ...

    async def prepare(
        self, sandbox: Any, spec: HarnessSpec, seed: SeedResult, endpoints: RolloutEndpoints, workdir: str
    ) -> None:
        """Write the harness's own config files into the sandbox (e.g. an
        endpoint config, or an MCP config for env-lent tools). No-op for
        harnesses that take their endpoint via env and are lent no tools."""

    def launch(
        self,
        spec: HarnessSpec,
        seed: SeedResult,
        endpoints: RolloutEndpoints,
        task_prompt: str,
        workdir: str,
    ) -> tuple[list[str], dict[str, str]]:
        """Return (argv, env) to run the harness once, non-interactively."""
