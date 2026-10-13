# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource server lifecycle for Legal Agent Bench."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import ConfigDict, Field, NonNegativeInt, PositiveFloat, PositiveInt

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from resources_servers.legal_agent_bench.prepare import (
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


class LegalAgentBenchResourcesServerConfig(BaseResourcesServerConfig):
    model_config = ConfigDict(extra="allow")

    harbor_tasks_cache_dir: str = str(DEFAULT_TASKS_DIR)
    harbor_tasks_dir: str = str(DEFAULT_RUNTIME_TASKS_DIR)
    harness_skills_dir: str = str(DEFAULT_SKILLS_DIR)
    auto_prepare_assets: bool = True
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


class LegalAgentBenchResourcesServer(SimpleResourcesServer):
    """Prepare immutable source assets and a credential-isolated runtime tree."""

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
            cache_is_validated=True,
        )
        return super().setup_webserver()

    async def verify(self, body: BaseVerifyRequest) -> BaseVerifyResponse:
        # Harbor executes the task-local verifier and the agent bridge returns its reward.
        return BaseVerifyResponse(**body.model_dump(), reward=0.0)


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
