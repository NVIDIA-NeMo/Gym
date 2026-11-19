# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate a RoboDojo case with the installed EmberBridge and Nanoverse engine."""

import asyncio
import logging
import tempfile
from pathlib import Path
from statistics import mean
from typing import Literal

from fastapi import Body, HTTPException
from pydantic import Field, PrivateAttr

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.core.failure_kinds import ENVIRONMENT_SERVER_FAILED
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.robodojo_agent.cases import bind_case
from responses_api_agents.robodojo_agent.episode import Episode, EpisodeError


LOGGER = logging.getLogger(__name__)


class RoboDojoAgentConfig(BaseResponsesAPIAgentConfig):
    bridge_config: Path | None = None
    panel_manifest: Path | None = None
    case_validity: Path | None = None
    bridge_python: Path | None = None
    simulation_python: Path | None = None
    task_policy: Path | None = None
    output_root: Path
    evidence_export_root: Path | None = None
    model_name: str
    model_server: ModelServerRef
    horizon_profile: Literal["native_x1", "native_x2"] = "native_x2"
    episode_timeout_seconds: float = Field(default=10800, gt=0, allow_inf_nan=False)
    cleanup_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)


class RoboDojoRequest(BaseRunRequest):
    case_id: str = Field(min_length=1)


class RoboDojoResponse(BaseVerifyResponse):
    case_id: str
    assigned_case_id: str
    layout_sha256: str
    simulator: Literal["isaaclab3_newton"] = "isaaclab3_newton"
    outcome_class: str | None = None
    native_score: float | None = None
    extended_score: float | None = None
    native_success: bool | None = None
    extended_success: bool | None = None
    evidence_dir: str
    video_path: str | None = None


class RoboDojoAgent(SimpleResponsesAPIAgent):
    config: RoboDojoAgentConfig
    _gpu: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)

    def multiple_workers_unsupported_reason(self) -> str:
        return "one RoboDojo agent instance owns one local GPU"

    async def responses(self, body: NeMoGymResponseCreateParamsNonStreaming = Body()) -> NeMoGymResponse:
        raise NotImplementedError("EmberBridge owns the RoboDojo policy loop")

    async def run(self, body: RoboDojoRequest = Body()) -> RoboDojoResponse:
        """Bind the Gym task, launch the source engine and return its measured grade."""
        config = self.config
        if (
            config.bridge_config is None
            or config.panel_manifest is None
            or config.bridge_python is None
            or config.simulation_python is None
        ):
            raise HTTPException(422, "Configure the installed RoboDojo runtime, frozen panel and Gym model server")
        model_url = self.resolve_model_base_url(config.model_server.name, self.rollout_id_from_run(body))
        try:
            bound, case = bind_case(
                template_path=config.bridge_config,
                panel_path=config.panel_manifest,
                validity_path=config.case_validity,
                case_id=body.case_id,
                horizon=config.horizon_profile,
                model_url=model_url,
                model_name=config.model_name,
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise HTTPException(422, str(error)) from error

        config.output_root.mkdir(parents=True, exist_ok=True)
        directory = Path(tempfile.mkdtemp(prefix="episode-", dir=config.output_root.resolve()))
        runner = Episode(
            directory,
            config.bridge_python,
            config.simulation_python,
            config.task_policy,
            config.episode_timeout_seconds,
            config.cleanup_timeout_seconds,
        )
        response = NeMoGymResponse(
            id=directory.name,
            created_at=0,
            model=config.model_name,
            object="response",
            output=[],
            parallel_tool_calls=False,
            tools=[],
            tool_choice="none",
            metadata={"episode_engine": "ember_bridge"},
        )
        common = {
            **body.model_dump(),
            "response": response,
            "assigned_case_id": case.case_id,
            "layout_sha256": case.layout.sha256,
            "evidence_dir": str(directory),
        }
        try:
            async with self._gpu:
                outcome, scores = await runner.run(bound)
            if scores.task_step_limit != case.native_step_limit or scores.control_budget != bound["step_limit"]:
                raise EpisodeError("Native horizon does not match the frozen case")
            if config.evidence_export_root is not None:
                directory = await asyncio.to_thread(runner.export_evidence, config.evidence_export_root)
                common["evidence_dir"] = str(directory)
        except (EpisodeError, OSError, TimeoutError, ValueError, KeyError) as error:
            LOGGER.warning(
                "RoboDojo case %s could not be scored; evidence in %s", body.case_id, directory, exc_info=True
            )
            return RoboDojoResponse(
                **common,
                reward=0,
                mask_sample=True,
                failure_kind=ENVIRONMENT_SERVER_FAILED,
                failure_reason=f"{type(error).__name__}: RoboDojo native measurement failed; see evidence_dir",
            )
        video = directory / "native" / "evaluation" / "sensors.mp4"
        return RoboDojoResponse(
            **common,
            reward=float(scores.within_budget),
            outcome_class=outcome,
            native_score=scores.score_within_native,
            extended_score=scores.score_within_budget,
            native_success=scores.within_native,
            extended_success=scores.within_budget,
            video_path=str(video) if video.is_file() else None,
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        result = await super().aggregate_metrics(body)
        scored = [
            row for row in body.verify_responses if not row.get("mask_sample") and row.get("native_score") is not None
        ]
        metrics = {
            "num/scored_cases": len(scored),
            "num/unscored_cases": len(body.verify_responses) - len(scored),
        }
        if scored:
            metrics.update(
                {
                    f"mean/{key}": mean(row[key] for row in scored)
                    for key in ("native_score", "extended_score", "native_success", "extended_success")
                }
            )
        result.agent_metrics.update(metrics)
        result.key_metrics.update(metrics)
        return result


if __name__ == "__main__":
    RoboDojoAgent.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = RoboDojoAgent.run_webserver()
