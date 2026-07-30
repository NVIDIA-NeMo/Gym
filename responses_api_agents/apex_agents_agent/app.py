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
"""Apex Agents agent server.

Eval-first **wrap** of Archipelago's ``react_toolbelt`` agent, run inside a Gym
provider-neutral sandbox (``nemo_gym.sandbox``, default Apptainer — the cvdp /
PR #2076 pattern). ``/run`` opens an ``AsyncSandbox`` from the Archipelago
environment image, uploads the world/task assets, and ``exec``s the guest
``sandbox_entrypoint.py`` which (inside the box, over localhost) boots the env,
populates it, configures ``/apps``, runs the Archipelago agent against ``/mcp/``
with the policy served by Gym's Model Server, and snapshots the result. The host
downloads the snapshot + trajectory and hands them to the Apex resources server
for grading.

Status: ``build_spec`` (below) is implemented and unit-tested. The ``run`` body
(sandbox exec + Archipelago agent CLI) is wired in Task 3.2 once Phase 0 has
confirmed the env image, the sandbox exec path, and the Archipelago agent
trajectory schema against the pinned upstream ref.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from nemo_gym.base_responses_api_agent import BaseResponsesAPIAgentConfig
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.sandbox import SandboxSpec


# The Archipelago environment image (env gateway + 11 MCP app servers, and the
# agents runner). Built once from the upstream ``environment`` image in Phase 0;
# may be a bare docker ref, an explicit ``.sif`` path, or a ``docker://`` uri.
DEFAULT_ENV_IMAGE = "archipelago-environment:latest"

# Guest entrypoint injected into the sandbox and run as ``python sandbox_entrypoint.py``.
_ENTRYPOINT_SOURCE_PATH = Path(__file__).with_name("sandbox_entrypoint.py")


class ApexAgentsAgentConfig(BaseResponsesAPIAgentConfig):
    """Config for the Apex Agents agent (sandbox-wrapped Archipelago agent)."""

    resources_server: ResourcesServerRef
    model_server: ModelServerRef

    # The Archipelago react_toolbelt config passed through to the agent runner.
    orchestrator_model: str = ""
    agent_config: Dict[str, Any] = {
        "agent_config_id": "react_toolbelt_agent",
        "agent_config_values": {"max_steps": 50, "timeout": 3600},
    }
    agent_timeout_s: int = 3600
    concurrency: int = 8

    # Sandbox wiring (provider-neutral). ``sandbox_provider`` is a single-key
    # provider config, e.g. ``{"apptainer": {}}`` (default), ``{"docker": {}}``,
    # or ``{"opensandbox": {}}`` — the backend is config, not code. ``sandbox_spec``
    # carries extra spec fields (``provider_options``, ``ttl_s``, ...).
    image: str = DEFAULT_ENV_IMAGE
    sandbox_provider: Dict[str, Any] = {"apptainer": {}}
    sandbox_spec: Dict[str, Any] = {}


def _read_entrypoint_source() -> str:
    """Return the guest entrypoint script (``sandbox_entrypoint.py``) verbatim."""
    return _ENTRYPOINT_SOURCE_PATH.read_text()


def build_spec(
    task_info: dict,
    model_url: str,
    *,
    image: str = DEFAULT_ENV_IMAGE,
    orchestrator_model: str = "",
    provider_options: Optional[dict] = None,
) -> SandboxSpec:
    """Build the per-task ``SandboxSpec`` for one Apex rollout.

    The policy is reached from inside the sandbox at ``NV_MODEL_URL`` (Gym's
    Model Server), mirroring the cvdp agent. Task identity travels as env vars so
    the guest entrypoint can locate the uploaded world/task assets and label the
    trajectory. Large inputs (world zip, initial messages) are uploaded
    separately via ``AsyncSandbox.upload`` rather than through the spec.
    """
    return SandboxSpec(
        image=image,
        workdir="/app",
        env={
            "NV_MODEL_URL": model_url,
            "APEX_TASK_ID": task_info["task_id"],
            "APEX_WORLD_ID": task_info["world_id"],
            "APEX_ORCHESTRATOR_MODEL": orchestrator_model,
        },
        provider_options=provider_options or {},
    )
