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

"""Adapter for the Pi CLI (@earendil-works/pi-coding-agent).

Unlike Claude Code, Pi takes a custom endpoint from a config file
(~/.pi/agent/models.json), not an env var, so prepare() writes that file
pointing a provider at the rollout-scoped gate. Pi speaks the OpenAI
chat-completions dialect, which the gate captures like any other.
"""

from __future__ import annotations

import json
from typing import Any

from responses_api_agents.external_harness.sandbox_utils import sandbox_write_text

from .base import HarnessSpec, RolloutEndpoints, RuntimeRequirement, SeedResult


class PiSpec(HarnessSpec):
    name: str = "pi"
    pinned_version: str | None = "0.80.2"
    provider_name: str = "gym_gate"  # the provider entry written into models.json
    home: str = "/root"  # Pi reads $HOME/.pi/agent/models.json


class PiAdapter:
    name = "pi"
    spec_model = PiSpec

    def runtime(self, spec: HarnessSpec) -> RuntimeRequirement:
        pkg = "@earendil-works/pi-coding-agent"
        if spec.pinned_version:
            pkg = f"{pkg}@{spec.pinned_version}"
        return RuntimeRequirement(setup_cmds=[["npm", "install", "-g", pkg]])

    async def prepare(
        self, sandbox: Any, spec: HarnessSpec, seed: SeedResult, endpoints: RolloutEndpoints, workdir: str
    ) -> None:
        assert isinstance(spec, PiSpec)
        # Pi's openai-completions provider posts to <baseUrl>/chat/completions,
        # so baseUrl is the rollout gate root + /v1.
        models = {
            "providers": {
                spec.provider_name: {
                    "baseUrl": endpoints.model_base_url.rstrip("/") + "/v1",
                    "api": "openai-completions",
                    "apiKey": endpoints.api_key or "dummy_key",
                    "models": [{"id": endpoints.model_name, "reasoning": False}],
                }
            }
        }
        await sandbox_write_text(sandbox, f"{spec.home}/.pi/agent/models.json", json.dumps(models, indent=2))

    def launch(
        self,
        spec: HarnessSpec,
        seed: SeedResult,
        endpoints: RolloutEndpoints,
        task_prompt: str,
        workdir: str,
    ) -> tuple[list[str], dict[str, str]]:
        assert isinstance(spec, PiSpec)
        instruction = f"{task_prompt}\n\nThe project files are in your working directory: {workdir}"
        cmd = [
            "pi",
            "--print",
            "--mode",
            "json",
            "--no-session",
            "--provider",
            spec.provider_name,
            "--model",
            endpoints.model_name,
            instruction,
        ]
        env = {
            "HOME": spec.home,
            "PI_SKIP_VERSION_CHECK": "1",
            "PI_TELEMETRY": "0",
            **spec.extra_env,
        }
        return cmd, env


ADAPTER = PiAdapter()
