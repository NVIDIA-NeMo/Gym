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
"""Small compatibility and guidance extensions to Harbor's existing installed agents.

Harbor owns installation, execution, skills, timeouts, and trace conversion.
These subclasses supply missing guidance settings and correct pinned Hermes installation paths/probes.
"""

import hashlib
import json
import tempfile
from pathlib import Path

import yaml
from harbor.agents.installed.codex import Codex, CodexOptions
from harbor.agents.installed.hermes import Hermes
from harbor.environments.base import BaseEnvironment, ExecResult


class GuidanceOptions(CodexOptions):
    model_catalog_path: str
    model_catalog_sha256: str


class CodexGuidance(Codex):
    options_model = GuidanceOptions

    @staticmethod
    def name() -> str:
        return "skill-trigger-codex-guidance"

    async def setup(self, environment: BaseEnvironment) -> None:
        await super().setup(environment)
        raw = Path(self.options.model_catalog_path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.options.model_catalog_sha256:
            raise ValueError("Codex model catalog hash mismatch")
        catalog = json.loads(raw)
        model = self.model_name.split("/", 1)[-1]
        selected = [entry for entry in catalog["models"] if entry.get("slug") == model]
        if len(selected) != 1:
            raise ValueError("Expected exactly one matching model in the Codex catalog")
        selected[0]["include_skills_usage_instructions"] = True
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "models.json"
            path.write_text(json.dumps(catalog))
            await environment.upload_file(path, "/tmp/skill-trigger-models.json")


class HermesCompat(Hermes):
    """Keep Harbor execution with a consistent Hermes install/runtime home."""

    def get_version_command(self) -> str:
        return 'export PATH="$HOME/.local/bin:$PATH"; HERMES_HOME=/tmp/hermes hermes --version'

    async def exec_as_agent(
        self,
        environment: BaseEnvironment,
        command: str,
        env: dict[str, str] | None = None,
        cwd: str | None = None,
        timeout_sec: int | None = None,
    ) -> ExecResult:
        # Harbor 0.25.0's install command uses a nonexistent `version` subcommand.
        if command.endswith("&& hermes version"):
            command = command.removesuffix("hermes version") + "hermes --version"
            # Install where Harbor runs Hermes, rather than the installer's ~/.hermes default.
            env = {**(env or {}), "HERMES_HOME": "/tmp/hermes"}
            command = command.replace(
                "/hermes-agent/main/scripts/install.sh",
                "/hermes-agent/d923a5eb487a17c0a68801bba9ef0f56c81040a8/scripts/install.sh",
            )
        return await super().exec_as_agent(environment, command=command, env=env, cwd=cwd, timeout_sec=timeout_sec)


class HermesGuidance(HermesCompat):
    @staticmethod
    def name() -> str:
        return "skill-trigger-hermes-guidance"

    @staticmethod
    def _build_config_yaml(model: str, max_turns: int | None = None) -> str:
        # Harbor 0.25.0 constructs this config internally; it has no override for this setting.
        config = yaml.safe_load(Hermes._build_config_yaml(model, max_turns))
        config.setdefault("agent", {})["tool_use_enforcement"] = True
        return yaml.safe_dump(config)
