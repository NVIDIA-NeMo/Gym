# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Translate LAB task pointers and model routing for the general Harbor bridge."""

from __future__ import annotations

import string
from pathlib import Path

from harbor.models.job.config import JobConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import TrialResult
from pydantic import model_validator

from nemo_gym.config import ModelServerRef, get_first_server_config_dict, get_global_config_dict
from responses_api_agents.harbor_agent_general.app import (
    HarborAgent,
    HarborAgentConfig,
    HarborRunRequest,
    HarborVerifyResponse,
)


class LegalAgentBenchHarborConfig(HarborAgentConfig):
    """Harbor's native job configuration plus Gym policy routing."""

    model_server: ModelServerRef

    @model_validator(mode="after")
    def require_unmounted_agent_logs(self) -> LegalAgentBenchHarborConfig:
        if self.job.environment.kwargs.get("auto_mounting"):
            raise ValueError("LAB host logs must not be writable from the agent sandbox; disable auto_mounting")
        return self


class LegalAgentBenchHarborRequest(HarborRunRequest):
    """Accept existing LAB pointers without changing their public identity."""

    instance_id: str

    @model_validator(mode="before")
    @classmethod
    def task_from_instance_id(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        instance_id = value.get("instance_id")
        if not isinstance(instance_id, str) or not instance_id.startswith("legal_agent_bench::"):
            raise ValueError("instance_id must be 'legal_agent_bench::<task_name>'")
        task_name = instance_id.removeprefix("legal_agent_bench::")
        allowed_characters = string.ascii_letters + string.digits + "_-"
        if not task_name or any(character not in allowed_characters for character in task_name):
            raise ValueError("LAB task name must contain only letters, numbers, underscores and hyphens")
        if value.get("task_name", task_name) != task_name:
            raise ValueError("task_name does not match instance_id")
        return {**value, "task_name": task_name}


class LegalAgentBenchHarborBridge(HarborAgent):
    """Retain LAB pointers while Harbor owns scheduling, grading and ATIF conversion."""

    # Gym reads this entrypoint's declarations before importing its base class.
    ray_enabled = True
    config: LegalAgentBenchHarborConfig

    def _configure_job_for_run(self, job_config: JobConfig, body: HarborRunRequest) -> None:
        model = get_first_server_config_dict(get_global_config_dict(), self.config.model_server.name)
        identity = body.model_dump(by_alias=True)
        identity["_ng_rollout_id"] = body.capture_rollout_id
        base_url = self.base_url_for_run(f"http://{model['host']}:{model['port']}", identity)
        kwargs = job_config.agents[0].kwargs
        kwargs["api_base"] = base_url + "/v1"
        kwargs["responses_create_params"] = body.responses_create_params.model_dump(exclude_unset=True)

    def success_response(self, body: HarborRunRequest, trial_dir: Path) -> HarborVerifyResponse:
        trial = TrialResult.model_validate_json(TrialPaths(trial_dir).result_path.read_text())
        rewards = (trial.verifier_result.rewards if trial.verifier_result is not None else None) or {}
        if rewards.get("verifier_error") or rewards.get("judge_error_count"):
            raise RuntimeError("LAB verification did not complete; inspect the retained per-judge reports")
        agent_metadata = (trial.agent_result.metadata if trial.agent_result is not None else None) or {}
        if agent_metadata.get("model_error"):
            raise RuntimeError(f"LAB model request failed: {agent_metadata['model_error']}")
        response = super().success_response(body, trial_dir)
        if agent_metadata.get("finished_cleanly") is False:
            response.response = response.response.model_copy(
                update={
                    "status": "incomplete",
                    "metadata": {
                        **(response.response.metadata or {}),
                        "lab_finish_reason": agent_metadata.get("finish_reason") or "incomplete",
                    },
                }
            )
        return response.model_copy(update={"lab_metrics": rewards})

    async def run(self, body: LegalAgentBenchHarborRequest) -> HarborVerifyResponse:
        """Execute a LAB pointer through the shared Harbor lifecycle."""
        return await super().run(body)


if __name__ == "__main__":
    LegalAgentBenchHarborBridge.run_webserver()
