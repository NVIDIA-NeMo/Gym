# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resources Server for Legal Agent Bench: LAB Harbor tasks served and graded through harbor_tasks."""

from __future__ import annotations

import json
import shlex
from typing import Literal

from fastapi import FastAPI
from pydantic import ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

from resources_servers.harbor_tasks.app import (
    HarborTaskSession,
    HarborTasksResourcesServer,
    HarborTasksResourcesServerConfig,
)
from resources_servers.legal_agent_bench.prepare import (
    DEFAULT_RUNTIME_IMAGE,
    DEFAULT_RUNTIME_TASKS_DIR,
    DEFAULT_SKILLS_DIR,
    DEFAULT_TASKS_DIR,
    ensure_assets,
    hydrate_runtime_tasks,
)


RewardMode = Literal["full_task", "criteria_pass_rate"]
JUDGE_CONFIG_TO_ENV = {
    "judge_base_url": "LAB_JUDGE_BASE_URL",
    "judge_api_key": "LAB_JUDGE_API_KEY",  # pragma: allowlist secret
    "judge_model_name": "LAB_JUDGE_MODEL",
    "judge_temperature": "LAB_JUDGE_TEMPERATURE",
    "judge_request_timeout_seconds": "LAB_JUDGE_REQUEST_TIMEOUT_SECONDS",
    "judge_max_retries": "LAB_JUDGE_MAX_RETRIES",
    "judge_structured_output": "LAB_JUDGE_STRUCTURED_OUTPUT",
    "judge_parse_repair_attempts": "LAB_JUDGE_PARSE_REPAIR_ATTEMPTS",
    "judge_repair_max_tokens": "LAB_JUDGE_REPAIR_MAX_TOKENS",
    "judge_max_tokens": "LAB_JUDGE_MAX_TOKENS",
    "judge_parallelism": "LAB_JUDGE_PARALLELISM",
}


VDR_DIR = "/workspace/vdr"


class LegalAgentBenchResourcesServerConfig(HarborTasksResourcesServerConfig):
    model_config = ConfigDict(extra="allow")

    harbor_tasks_cache_dir: str = str(DEFAULT_TASKS_DIR)
    harbor_tasks_dir: str = str(DEFAULT_RUNTIME_TASKS_DIR)
    harness_skills_dir: str = str(DEFAULT_SKILLS_DIR)
    auto_prepare_assets: bool = True
    runtime_image: str = Field(
        default=DEFAULT_RUNTIME_IMAGE,
        description="Prebuilt image from any task's environment/Dockerfile; every LAB task shares it.",
    )
    reward_mode: RewardMode = "full_task"
    judge_base_url: str | None = Field(default=None, description="OpenAI-compatible base URL for the LAB judge.")
    judge_api_key: str | None = Field(default=None, description="API key for the LAB judge endpoint.")
    judge_model_name: str | None = Field(default=None, description="Model identifier sent to the LAB judge endpoint.")
    judge_temperature: float | None = Field(
        default=None,
        ge=0,
        description="Optional sampling temperature for LAB judge requests.",
    )
    judge_request_timeout_seconds: PositiveFloat = Field(
        default=90,
        description="Timeout for one LAB judge request.",
    )
    judge_max_retries: PositiveInt = Field(default=1, description="Maximum attempts for each LAB judge request.")
    judge_structured_output: bool = Field(
        default=True,
        description="Request structured judge output before falling back to plain text.",
    )
    judge_parse_repair_attempts: NonNegativeInt = Field(
        default=1,
        description="Maximum attempts to repair an unparseable judge response.",
    )
    judge_repair_max_tokens: PositiveInt = Field(
        default=4096,
        description="Maximum output tokens for judge response repair.",
    )
    judge_max_tokens: PositiveInt = Field(default=4096, description="Maximum output tokens for LAB judge requests.")
    judge_parallelism: PositiveInt = Field(default=6, description="Maximum concurrent LAB judge requests per task.")


class LegalAgentBenchResourcesServer(HarborTasksResourcesServer):
    """Prepare immutable source assets and a credential-isolated runtime tree, then serve it as Harbor tasks."""

    ray_enabled = False

    config: LegalAgentBenchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        assets = ensure_assets(
            tasks_dir=self.config.harbor_tasks_cache_dir,
            skills_dir=self.config.harness_skills_dir,
            allow_download=self.config.auto_prepare_assets,
        )
        verifier_env = _build_verifier_env(self.config)
        hydrate_runtime_tasks(
            assets["tasks"],
            self.config.harbor_tasks_dir,
            verifier_env=verifier_env,
            reward_mode=self.config.reward_mode,
            docker_image=self.config.runtime_image,
            cache_is_validated=True,
        )
        return super().setup_webserver()

    async def prepare_sandbox(self, session: HarborTaskSession) -> None:
        """Stage the task's documents; the agent reads them from the sandbox, never from the task directory."""
        task_dir = session.task.paths.task_dir
        docs_dir = task_dir / json.loads((task_dir / "task.json").read_text(encoding="utf-8")).get(
            "docs_dir", "documents"
        )
        environment = session.environment
        await environment.exec(f"rm -rf {VDR_DIR} && mkdir -p /workspace", user="root")
        await environment.upload_dir(docs_dir, VDR_DIR)
        result = await environment.exec(f"chmod -R a+rwX {shlex.quote(VDR_DIR)}", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Cannot stage Legal Agent Bench documents: {result.stderr or result.stdout}")


def _build_verifier_env(config: LegalAgentBenchResourcesServerConfig) -> dict[str, str]:
    env: dict[str, str] = {}
    for config_key, env_key in JUDGE_CONFIG_TO_ENV.items():
        value = getattr(config, config_key)
        if value in (None, "", "****"):
            continue
        if config_key == "judge_model_name" and not str(value).startswith("openai-compatible/"):
            value = f"openai-compatible/{value}"
        env[env_key] = str(value).lower() if isinstance(value, bool) else str(value)
    return env


if __name__ == "__main__":
    LegalAgentBenchResourcesServer.run_webserver()
