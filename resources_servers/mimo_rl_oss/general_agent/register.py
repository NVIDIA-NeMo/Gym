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
import os
from pathlib import Path

from huggingface_hub import snapshot_download
from mimoagent.environments.datasets import DATASET_REGISTRY

from resources_servers.mimo_rl_oss.general_agent.environment import GeneralAgentEnvironment


REPO = "XiaomiMiMo/MiMo-V2.6-RL-oss"
# Written into the box at setup so mimoagent can expose the task's MCP tools to native loops.
MCP_CONFIG_PATH = "/work/_setup/mcp_servers.json"


class SingleBoxGeneralAgentEnvironment(GeneralAgentEnvironment):
    """MiMo's general_agent environment with main and sidecar in one box.

    The task scripts need mcp 1.x in ``/opt/openai-agents-venv``, which MiMo's internal images ship. The released
    images only have mcp 2.x in the system python3, so setup builds that venv.
    """

    # GA_JUDGE_API selects MiMo's chat-completions judge path, which self-hosted vLLM judges need.
    VERIFY_ENV_FORWARD = (*GeneralAgentEnvironment.VERIFY_ENV_FORWARD, "GA_JUDGE_API")

    def _setup_dataset_specific(self) -> None:
        venv = os.path.dirname(os.path.dirname(self.VENV_PYTHON))
        res = self.execute(
            f"[ -x {self.VENV_PYTHON} ] || {{ python3 -m venv {venv} && {venv}/bin/pip install -q 'mcp>=1.9,<2'; }}",
            cwd="/",
            timeout=600,
        )
        if res.get("returncode") != 0:
            raise RuntimeError(f"{self.instance_id}: building the MCP venv failed: {res.get('output', '')[-800:]}")
        super()._setup_dataset_specific()


DATASET_REGISTRY.setdefault("general_agent", SingleBoxGeneralAgentEnvironment)


def resolve_task_dir(instance: dict) -> dict:
    """Download the row's env tree (envs/<task>, ~16 MB) on demand and point the instance at it."""
    rel = instance.get("env_task_dir") or ""
    if instance.get("dataset_type") != "general_agent" or os.path.isabs(rel):
        return instance
    root = snapshot_download(REPO, repo_type="dataset", allow_patterns=[f"general/{rel}/*"])
    return {**instance, "env_task_dir": str(Path(root) / "general" / rel)}
