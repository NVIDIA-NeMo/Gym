# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""LAB-specific preparation and grading policy for the Harbor Environment Server."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any, Literal

from fastapi import HTTPException
from pydantic import (
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    PrivateAttr,
)

from environment_servers.harbor.app import (
    HarborEnvironmentServer,
    HarborEnvironmentServerConfig,
    HarborEpisodeRequest,
    HarborTasksetConfig,
)
from environment_servers.harbor.session import HarborSession
from nemo_gym.config import get_global_config_dict
from nemo_gym.core import NeMoGymResponse
from nemo_gym.data.tasks.harbor.task import HarborTask, HarborTurn
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.transfer import upload_dir
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from nemo_gym.utils.gym_commit import installed_gym_commit


RewardMode = Literal["full_task", "criteria_pass_rate"]
JUDGE_CONFIG_TO_ENV = {
    "judge_model_name": "LAB_JUDGE_MODEL",
    "judge_base_url": "LAB_JUDGE_BASE_URL",
    "judge_api_key": "LAB_JUDGE_API_KEY",  # pragma: allowlist secret
    "judge_temperature": "LAB_JUDGE_TEMPERATURE",
    "judge_request_timeout_seconds": "LAB_JUDGE_REQUEST_TIMEOUT_SECONDS",
    "judge_max_retries": "LAB_JUDGE_MAX_RETRIES",
    "judge_structured_output": "LAB_JUDGE_STRUCTURED_OUTPUT",
    "judge_parse_repair_attempts": "LAB_JUDGE_PARSE_REPAIR_ATTEMPTS",
    "judge_repair_max_tokens": "LAB_JUDGE_REPAIR_MAX_TOKENS",
    "judge_max_tokens": "LAB_JUDGE_MAX_TOKENS",
    "judge_parallelism": "LAB_JUDGE_PARALLELISM",
}


class LegalAgentBenchEnvironmentConfig(HarborEnvironmentServerConfig):
    model_config = ConfigDict(extra="forbid")

    prepared_tasksets_file: str = "resources_servers/legal_agent_bench/data/environment/tasksets.json"
    reward_mode: RewardMode = "full_task"
    judge_model_name: str | None = None
    judge_base_url: str | None = Field(default=None, description="OpenAI-compatible base URL for the LAB judge.")
    judge_api_key: str | None = Field(default=None, description="API key for the LAB judge endpoint.")
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


class LegalAgentBenchEnvironmentServer(HarborEnvironmentServer):
    """Harbor owns execution; LAB adds verifier settings and judge-failure routing."""

    config: LegalAgentBenchEnvironmentConfig
    _gym_commit: str | None = PrivateAttr(default=None)

    def model_post_init(self, context: object, /) -> None:
        if (
            not self.config.tasksets
            or not self.config.tasksets.get("legal_agent_bench", HarborTasksetConfig(folder=".", tasks={})).tasks
        ):
            mapping = Path(self.config.prepared_tasksets_file)
            if not mapping.is_file():
                raise ValueError(
                    "Prepare LAB tasks with `gym eval prepare --config benchmarks/legal_agent_bench/config.yaml` first"
                )
            self.config.tasksets = {
                name: HarborTasksetConfig.model_validate(value)
                for name, value in json.loads(mapping.read_text()).items()
            }
        self._gym_commit = installed_gym_commit()
        values = {name: "" for name in JUDGE_CONFIG_TO_ENV.values()}
        values.update(_build_verifier_env(self.config))
        values["LEGAL_AGENT_BENCH_REWARD_MODE"] = self.config.reward_mode
        self.config.secrets = {**self.config.secrets, **values}
        super().model_post_init(context)

    async def _upload_task_skills(self, sandbox: AsyncSandbox, task: HarborTask, workdir: str) -> None:
        """Stage LAB's public inputs before Harbor starts the agent session."""
        await super()._upload_task_skills(sandbox, task, workdir)
        # Task folders contain private verifier files too; only this prepared input tree reaches the agent.
        for name in ("vdr", "skills"):
            await upload_dir(sandbox, task.path / "environment" / name, f"{workdir}/{name}")

    async def _verified_outcome(
        self,
        session: HarborSession,
        request: HarborEpisodeRequest,
        agent_response: NeMoGymResponse,
        turn: tuple[HarborTurn, Any] | None = None,
    ) -> dict[str, Any]:
        outcome = await super()._verified_outcome(session, request, agent_response, turn=turn)
        if outcome.get("mask_sample"):
            # Use the failure boundary for sidecar routing and retries, not just mask_sample.
            raise HTTPException(503, f"{outcome.get('failure_reason')}; reports: {outcome.get('verifier_logs_dir')}")
        return outcome

    async def _run_verifier(
        self,
        session: HarborSession,
        trajectory: bytes,
        *,
        response: NeMoGymResponse | None = None,
    ) -> dict[str, Any]:
        if response is not None:
            entry = get_global_config_dict()[self.config.agent_server.name]["responses_api_agents"]
            kind = next(iter(entry))
            settings = entry[kind]
            pins = {
                key: settings[key] for key in ("codex_version", "claude_code_version", "max_turns") if key in settings
            }
            if kind == "hermes_agent":
                project = Path(__file__).resolve().parents[2] / "responses_api_agents/hermes_agent/pyproject.toml"
                dependencies = tomllib.loads(project.read_text())["project"]["dependencies"]
                pins["runtime_pin"] = next(item for item in dependencies if item.startswith("hermes-agent@"))
            metadata = dict(response.metadata or {})
            metadata["lab_harness_provenance"] = json.dumps(
                {"agent_id": kind, "harness_pin": pins, "gym_commit": self._gym_commit}
            )
            response = response.model_copy(update={"metadata": metadata})
        return await super()._run_verifier(session, trajectory, response=response)

    def _score(
        self,
        task: HarborTask,
        settings: Any,
        result: SandboxExecResult,
        logs_dir: Path,
        started: float,
        *,
        timed_out: bool,
        rewards: dict[str, float] | None,
        problem: str | None,
    ) -> dict[str, Any]:
        result = super()._score(
            task, settings, result, logs_dir, started, timed_out=timed_out, rewards=rewards, problem=problem
        )
        missing_aggregate = rewards is None and not timed_out or rewards is not None and "reward" not in rewards
        failed_grading = bool(rewards and (rewards.get("verifier_error") or rewards.get("judge_error_count")))
        if missing_aggregate or failed_grading:
            # Missing judge results are operational failures; measured verifier timeouts retain Harbor's policy.
            result.update(self._masked("lab_judge_failed", "LAB grading did not complete; inspect verifier reports"))
        return result


def _build_verifier_env(config: LegalAgentBenchEnvironmentConfig) -> dict[str, str]:
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
    LegalAgentBenchEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = LegalAgentBenchEnvironmentServer.run_webserver()
