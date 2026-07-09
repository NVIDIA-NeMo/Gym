# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resource server lifecycle for Legal Agent Bench."""

from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import ConfigDict, PositiveInt

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
JUDGE_ENV_KEYS = {
    "judge_base_url": "LAB_JUDGE_BASE_URL",
    "judge_api_key": "LAB_JUDGE_API_KEY",
    "judge_model_name": "LAB_JUDGE_MODEL",
}


class LegalAgentBenchResourcesServerConfig(BaseResourcesServerConfig):
    model_config = ConfigDict(extra="allow")

    harbor_tasks_cache_dir: str = str(DEFAULT_TASKS_DIR)
    harbor_tasks_dir: str = str(DEFAULT_RUNTIME_TASKS_DIR)
    harness_skills_dir: str = str(DEFAULT_SKILLS_DIR)
    auto_prepare_assets: bool = True
    reward_mode: RewardMode = "full_task"
    judge_parallelism: PositiveInt = 6


class LegalAgentBenchResourcesServer(SimpleResourcesServer):
    """Prepare immutable source assets and a credential-isolated runtime tree."""

    config: LegalAgentBenchResourcesServerConfig

    def setup_webserver(self) -> FastAPI:
        assets = ensure_assets(
            tasks_dir=self.config.harbor_tasks_cache_dir,
            skills_dir=self.config.harness_skills_dir,
            allow_download=self.config.auto_prepare_assets,
        )
        verifier_env = _load_judge_env_from_global_config()
        verifier_env["LAB_JUDGE_PARALLELISM"] = str(self.config.judge_parallelism)
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


def _load_judge_env_from_global_config() -> dict[str, str]:
    from nemo_gym.global_config import get_global_config_dict

    config = get_global_config_dict()

    env: dict[str, str] = {}
    for config_key, env_key in JUDGE_ENV_KEYS.items():
        value = config.get(config_key)
        if value and value != "****":
            value = str(value)
            if config_key == "judge_model_name" and not value.startswith("openai-compatible/"):
                value = f"openai-compatible/{value}"
            env[env_key] = value
    return env


if __name__ == "__main__":
    LegalAgentBenchResourcesServer.run_webserver()
