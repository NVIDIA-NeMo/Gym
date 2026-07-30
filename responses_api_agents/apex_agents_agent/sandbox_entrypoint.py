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
"""Guest entrypoint — runs INSIDE the Apex sandbox.

Ports the flow of Archipelago's ``examples/hugging_face_task/main.py`` into a
single in-box script so the host only has to ``upload`` inputs, ``exec`` this,
and ``download`` outputs (no host->sandbox port needed).

Contract (host <-> guest):
  Inputs, uploaded by the host before exec:
    /inputs/world.zip              seeded world snapshot (filesystem + .apps_data)
    /inputs/task_files/**          optional per-task input files
    /inputs/initial_messages.json  [system, user] messages (user = task prompt)
    /inputs/mcp_config.json        the 11-app MCP server config
  Env vars (set via SandboxSpec.env by build_spec):
    NV_MODEL_URL                   OpenAI-compatible policy endpoint (Gym Model Server)
    APEX_ORCHESTRATOR_MODEL        LiteLLM model string for the Archipelago agent
    APEX_TASK_ID / APEX_WORLD_ID   task identity (labels only)
  Outputs, written here for the host to download:
    /output/initial_snapshot.zip   copy of the seeded world (grader "before")
    /output/final_snapshot.zip     env state after the agent ran (grader "after")
    /output/trajectory.json        Archipelago AgentTrajectoryOutput (carries `status`)

NOTE (Phase 0 dependency): the concrete steps below — the env service launch
command, the exact ``/data/populate`` / ``/apps`` / ``/data/snapshot`` calls, and
the Archipelago ``agents/runner`` CLI invocation + its trajectory schema — must
be filled in against the pinned upstream ref during the Phase 0 sandbox spike
(Task 0.2/0.3), which is the first thing that runs on an Apptainer-capable box.
This skeleton fixes the host<->guest contract so the host side (Task 3.2) can be
built and unit-tested against it in the meantime.
"""

from __future__ import annotations

import os
from pathlib import Path


INPUTS = Path("/inputs")
OUTPUT = Path("/output")
ENV_BASE_URL = "http://localhost:8080"


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    _ = os.environ.get("NV_MODEL_URL", "")

    # 1. start the env service (background), wait for ENV_BASE_URL/health
    # 2. POST /data/populate  <- /inputs/world.zip (filesystem + .apps_data), then task_files overlay
    # 3. copy the seeded world to /output/initial_snapshot.zip
    # 4. POST /apps  <- /inputs/mcp_config.json
    # 5. run Archipelago agents/runner against ENV_BASE_URL/mcp/ with
    #    --initial-messages /inputs/initial_messages.json and
    #    --orchestrator-model $APEX_ORCHESTRATOR_MODEL (policy at $NV_MODEL_URL),
    #    writing the trajectory to /output/trajectory.json
    # 6. POST /data/snapshot -> tar.gz -> /output/final_snapshot.zip
    raise NotImplementedError(
        "sandbox_entrypoint: fill in against the pinned Archipelago ref during the "
        "Phase 0 sandbox spike (Task 0.2). The host<->guest contract above is fixed."
    )


if __name__ == "__main__":  # pragma: no cover - guest entrypoint
    raise SystemExit(main())
