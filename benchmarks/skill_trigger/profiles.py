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

Harbor owns installation, execution, skills, timeouts, and per-session trace conversion.
These subclasses select Codex session logs, supply guidance settings, and correct Hermes installation paths/probes.
"""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import yaml
from harbor.agents.agent_auth_resolution import ResolvedModelConnection
from harbor.agents.installed.codex import Codex, CodexOptions
from harbor.agents.installed.hermes import Hermes
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.models.trajectories import Trajectory

from benchmarks.skill_trigger.scoring import EvidenceError


class GuidanceOptions(CodexOptions):
    model_catalog_path: str
    model_catalog_sha256: str


class CodexCompat(Codex):
    """Convert the CLI's main session and retain its delegated-session evidence.

    Harbor 0.25 otherwise selects the lexicographically last rollout, which can
    belong to a child agent. Installation and model execution remain unchanged.
    """

    def convert_trajectory(self, logs_dir: Path) -> Trajectory:
        def events(path: Path) -> list[dict]:
            values = []
            for line in path.read_text().splitlines():
                try:
                    value = json.loads(line)
                except ValueError:
                    continue
                if isinstance(value, dict):
                    values.append(value)
            return values

        roots = {
            event["thread_id"]
            for event in events(logs_dir / "codex.txt")
            if event.get("type") == "thread.started" and event.get("thread_id")
        }
        if len(roots) != 1:
            raise EvidenceError("Expected exactly one Codex CLI main-session ID")
        root = roots.pop()
        sessions = {}
        children = {}
        for path in sorted((logs_dir / "sessions").rglob("*.jsonl")):
            metadata = next((e.get("payload") for e in events(path) if e.get("type") == "session_meta"), None)
            if not isinstance(metadata, dict) or not metadata.get("id"):
                raise EvidenceError("Codex rollout lacks session identity")
            session_id = metadata["id"]
            if session_id in sessions:
                if sessions[session_id][0].read_bytes() != path.read_bytes():
                    raise EvidenceError("Conflicting Codex session logs")
                continue
            source = metadata.get("source")
            spawn = (source.get("subagent") or {}).get("thread_spawn") or {} if isinstance(source, dict) else {}
            parent = spawn.get("parent_thread_id")
            sessions[session_id] = (path, metadata, parent)
            if parent:
                children.setdefault(parent, []).append(session_id)
        if root not in sessions or sessions[root][2] or sessions[root][1].get("forked_from_id"):
            raise EvidenceError("Codex CLI main session is missing, delegated, or forked")
        visited = set()

        def convert(session_id: str) -> Trajectory:
            if session_id in visited:
                raise EvidenceError("Cyclic Codex delegation")
            visited.add(session_id)
            path = sessions[session_id][0]
            with tempfile.TemporaryDirectory() as folder:
                shutil.copyfile(path, Path(folder) / path.name)
                trajectory = self._convert_events_to_trajectory(Path(folder))
            if trajectory is None or trajectory.session_id != session_id:
                raise EvidenceError("Codex conversion selected the wrong session")
            descendants = [convert(child) for child in sorted(children.get(session_id, []))]
            data = trajectory.model_dump(mode="json")
            data["trajectory_id"] = "codex-" + session_id
            data["subagent_trajectories"] = [child.model_dump(mode="json") for child in descendants] or None
            for step in data["steps"]:
                for result in (step.get("observation") or {}).get("results") or []:
                    content = json.dumps(result.get("content"))
                    refs = [
                        {"trajectory_id": child.trajectory_id, "session_id": child.session_id}
                        for child in descendants
                        if child.session_id in content
                    ]
                    if refs:
                        result["subagent_trajectory_ref"] = refs
            # A successful spawn can be recorded even when its child log was not collected.
            spawned = set()
            calls = {}
            for event in events(path):
                item = event.get("payload") or {}
                if event.get("type") != "response_item":
                    continue
                if item.get("type") == "function_call":
                    calls[item.get("call_id")] = (item.get("name") or "").rsplit(".", 1)[-1]
                elif item.get("type") == "function_call_output" and calls.get(item.get("call_id")) == "spawn_agent":
                    try:
                        output = json.loads(item.get("output", ""))
                    except (ValueError, TypeError):
                        continue
                    if isinstance(output, dict) and output.get("agent_id"):
                        spawned.add(output["agent_id"])
            if spawned - set(children.get(session_id, [])):
                data["agent"]["extra"] = {**(data["agent"].get("extra") or {}), "uncollected_subagents": True}
            return Trajectory.model_validate(data)

        return convert(root)


class CodexGuidance(CodexCompat):
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
    def _build_config_yaml(
        model: str, max_turns: int | None = None, *, connection: ResolvedModelConnection | None = None
    ) -> str:
        # Harbor 0.25.0 constructs this config internally; it has no override for this setting.
        config = yaml.safe_load(Hermes._build_config_yaml(model, max_turns, connection=connection))
        config.setdefault("agent", {})["tool_use_enforcement"] = True
        return yaml.safe_dump(config)
