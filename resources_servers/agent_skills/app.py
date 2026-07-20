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

"""Hidden-check resources server for sandboxed agent-skill coding tasks."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from nemo_gym import PARENT_DIR
from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.reward_profile import compute_pass_majority_metrics, highest_k_metrics
from nemo_gym.sandbox import resolve_provider_config, resolve_provider_metadata, sandbox_spec_from_mapping
from resources_servers.agent_skills.verifier import SandboxPatchVerifier


class AgentSkillCheckSuiteConfig(BaseModel):
    """Server-side definition of a fixture and its hidden check command."""

    sandbox_spec: dict[str, Any]
    check_command: str
    workspace: str = "/workspace/nemo-gym"
    hidden_files: dict[str, str] = Field(default_factory=dict)
    hidden_file_paths: dict[str, str] = Field(default_factory=dict)
    check_cwd: str = "/tmp/nemo_gym_hidden_checks"
    timeout: int = 1800
    user: str | int | None = "root"
    check_user: str | int | None = "nobody"
    max_patch_bytes: int = 10 * 1024 * 1024
    max_log_chars: int = 20_000


class AgentSkillsResourcesServerConfig(BaseResourcesServerConfig):
    sandbox_provider: str | dict[str, Any]
    check_suites: dict[str, AgentSkillCheckSuiteConfig] = Field(default_factory=dict)
    concurrency: int = 8
    cleanup_timeout: int = 30


class AgentSkillsVerifyRequest(BaseVerifyRequest):
    model_config = ConfigDict(extra="allow")

    verifier_metadata: dict[str, Any] = Field(default_factory=dict)
    workspace_patch: str
    workspace_base_revision: str


class AgentSkillsVerifyResponse(BaseVerifyResponse):
    model_config = ConfigDict(extra="allow")

    task_id: str | None = None
    check_suite_id: str | None = None
    status: str
    correctness: float
    completeness: float = 0.0
    convention_compliance: float = 0.0
    verifier_base_revision: str | None = None
    verifier_elapsed_seconds: float | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class AgentSkillsResourcesServer(SimpleResourcesServer):
    config: AgentSkillsResourcesServerConfig

    def model_post_init(self, context: Any) -> None:
        self._semaphore = asyncio.Semaphore(self.config.concurrency)

    @staticmethod
    def _score_fn(result: dict[str, Any]) -> dict[str, float]:
        return {
            "task_success": float(result.get("reward", 0) > 0),
            "correctness": float(result.get("correctness", 0)),
            "completeness": float(result.get("completeness", 0)),
            "convention_compliance": float(result.get("convention_compliance", 0)),
        }

    def compute_metrics(self, tasks: list[list[dict[str, Any]]]) -> dict[str, Any]:
        return compute_pass_majority_metrics(tasks, score_fn=self._score_fn)[0]

    def get_key_metrics(self, agent_metrics: dict[str, Any]) -> dict[str, Any]:
        key: dict[str, Any] = {}
        for name in ("mean/input_tokens", "mean/output_tokens", "mean/verifier_elapsed_seconds"):
            if name in agent_metrics:
                key[name] = agent_metrics[name]
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@1[avg-of-{k}]",
                score_names=["task_success", "correctness", "completeness", "convention_compliance"],
            )
        )
        key.update(
            highest_k_metrics(
                agent_metrics,
                "pass@{k}",
                score_names=["task_success", "correctness", "completeness", "convention_compliance"],
            )
        )
        return key

    async def verify(self, body: AgentSkillsVerifyRequest) -> AgentSkillsVerifyResponse:
        metadata = body.verifier_metadata
        task_id = metadata.get("task_id")
        suite_id = metadata.get("check_suite_id")
        response_data = body.model_dump(
            exclude={
                "workspace_patch",
                "task_id",
                "check_suite_id",
                "status",
                "correctness",
                "completeness",
                "convention_compliance",
                "verifier_base_revision",
                "verifier_elapsed_seconds",
                "details",
                "reward",
            }
        )

        if not isinstance(suite_id, str) or suite_id not in self.config.check_suites:
            return AgentSkillsVerifyResponse(
                **response_data,
                reward=0.0,
                task_id=task_id,
                check_suite_id=suite_id,
                status="unknown_check_suite",
                correctness=0.0,
                details={"reason": f"Unknown check_suite_id: {suite_id!r}"},
            )
        if not body.workspace_patch:
            return AgentSkillsVerifyResponse(
                **response_data,
                reward=0.0,
                task_id=task_id,
                check_suite_id=suite_id,
                status="empty_patch",
                correctness=0.0,
                details={"reason": "Agent returned no workspace patch"},
            )

        suite = self.config.check_suites[suite_id]
        try:
            global_config = self.server_client.global_config_dict
            provider = resolve_provider_config(self.config.sandbox_provider, global_config)
            provider_metadata = resolve_provider_metadata(self.config.sandbox_provider, global_config)
            spec = sandbox_spec_from_mapping(
                suite.sandbox_spec,
                default_workdir=suite.workspace,
                default_metadata=provider_metadata,
                metadata={
                    "nemo_gym_resources_server": "agent_skills",
                    "check_suite_id": suite_id,
                    "task_id": str(task_id or "unknown")[:63],
                },
            )
            verifier = SandboxPatchVerifier(
                provider=provider,
                spec=spec,
                workspace=spec.workdir or suite.workspace,
                check_command=suite.check_command,
                timeout_s=suite.timeout,
                hidden_files=suite.hidden_files,
                hidden_file_paths={
                    relative_path: self._resolve_hidden_file_path(source_path)
                    for relative_path, source_path in suite.hidden_file_paths.items()
                },
                check_cwd=suite.check_cwd,
                user=suite.user,
                check_user=suite.check_user,
                max_patch_bytes=suite.max_patch_bytes,
                max_log_chars=suite.max_log_chars,
                cleanup_timeout_s=self.config.cleanup_timeout,
            )
            async with self._semaphore:
                result = await verifier.verify(
                    patch=body.workspace_patch,
                    expected_base_revision=body.workspace_base_revision,
                )
        except Exception as exc:
            return AgentSkillsVerifyResponse(
                **response_data,
                reward=0.0,
                task_id=task_id,
                check_suite_id=suite_id,
                status="verifier_error",
                correctness=0.0,
                details={"reason": str(exc)},
            )

        component_scores = self._parse_component_scores(result.stdout)
        correctness = component_scores.get("correctness", float(result.passed))
        task_success = float(result.passed and component_scores.get("task_success", 1.0) > 0)
        return AgentSkillsVerifyResponse(
            **response_data,
            reward=task_success,
            task_id=task_id,
            check_suite_id=suite_id,
            status=result.status,
            correctness=correctness,
            completeness=component_scores.get("completeness", 0.0),
            convention_compliance=component_scores.get("convention_compliance", 0.0),
            verifier_base_revision=result.verifier_base_revision,
            verifier_elapsed_seconds=result.elapsed_seconds,
            details={
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.return_code,
                "error_type": result.error_type,
            },
        )

    @staticmethod
    def _resolve_hidden_file_path(source_path: str) -> Path:
        path = Path(source_path)
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        return cwd_path if cwd_path.exists() else PARENT_DIR / path

    @staticmethod
    def _parse_component_scores(stdout: str) -> dict[str, float]:
        for line in reversed(stdout.splitlines()):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("status") != "summary" or not isinstance(payload.get("scores"), dict):
                continue
            return {
                str(name): float(value)
                for name, value in payload["scores"].items()
                if isinstance(value, (int, float)) and not isinstance(value, bool)
            }
        return {}


if __name__ == "__main__":
    AgentSkillsResourcesServer.run_webserver()
