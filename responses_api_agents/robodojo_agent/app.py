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
from pydantic import ConfigDict, Field, PrivateAttr

from nemo_gym.agents import BaseResponsesAPIAgentConfig, SimpleResponsesAPIAgent
from nemo_gym.config import AggregateMetrics, AggregateMetricsRequest, ModelServerRef
from nemo_gym.core import NeMoGymResponse, NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.core.failure_kinds import (
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    AGENT_TIMEOUT,
    ENVIRONMENT_PROTOCOL_VIOLATION,
    ENVIRONMENT_SERVER_FAILED,
    HARNESS_TERMINATION_UNCONFIRMED,
    MODEL_CONTEXT_LIMIT,
)
from nemo_gym.server_runtime import BaseRunRequest, BaseVerifyResponse, observe_verifier
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.robodojo_agent.cases import ROBODOJO_SIMULATORS, RoboDojoSimulator, bind_case
from responses_api_agents.robodojo_agent.episode import CleanupUnconfirmed, Episode, EpisodeError


LOGGER = logging.getLogger(__name__)
POLICY_FAILURE_KINDS = {
    "policy_failure.decision_budget": AGENT_LIMIT,
    "policy_failure.context": MODEL_CONTEXT_LIMIT,
    "policy_failure.no_progress": AGENT_INVALID_OUTPUT,
}


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
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(min_length=1)


class RoboDojoResponse(BaseVerifyResponse):
    case_id: str
    assigned_case_id: str
    layout_sha256: str
    simulator: RoboDojoSimulator = "isaaclab3_newton"
    outcome_class: str | None = None
    native_score: float | None = None
    extended_score: float | None = None
    native_success: bool | None = None
    extended_success: bool | None = None
    evidence_dir: str
    video_path: str | None = None
    trajectory_path: str | None = None


class RoboDojoAgent(SimpleResponsesAPIAgent):
    config: RoboDojoAgentConfig
    _gpu: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _cleanup_unconfirmed: bool = PrivateAttr(default=False)

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
            "responses_create_params": body.responses_create_params,
            "case_id": body.case_id,
            "response": response,
            "assigned_case_id": case.case_id,
            "layout_sha256": case.layout.sha256,
            "simulator": ROBODOJO_SIMULATORS[bound["simulator_backend"]],
            "evidence_dir": str(directory),
        }
        outcome = None
        failure_kind = None
        failure_reason = None
        async with self._gpu:
            try:
                if self._cleanup_unconfirmed:
                    raise CleanupUnconfirmed("An earlier episode did not release the GPU")
                outcome, scores = await runner.run(bound)
            except CleanupUnconfirmed as error:
                self._cleanup_unconfirmed = True
                failure_kind, failure_reason = HARNESS_TERMINATION_UNCONFIRMED, str(error)
            except asyncio.CancelledError:
                self._cleanup_unconfirmed = True
                raise
            except TimeoutError:
                failure_kind, failure_reason = AGENT_TIMEOUT, "RoboDojo supervisor exceeded its episode deadline"
            except (EpisodeError, OSError, ValueError, KeyError) as error:
                failure_kind = ENVIRONMENT_SERVER_FAILED
                failure_reason = f"{type(error).__name__}: RoboDojo native measurement failed"
                LOGGER.warning("RoboDojo case %s could not be scored", body.case_id, exc_info=True)
            else:
                if scores.task_step_limit != case.native_step_limit or scores.control_budget != bound["step_limit"]:
                    failure_kind = ENVIRONMENT_PROTOCOL_VIOLATION
                    failure_reason = "Native horizon does not match the frozen case"
                elif outcome == "policy_failure.deadline":
                    failure_kind = AGENT_TIMEOUT
                    failure_reason = "The episode deadline expired without verified model-side attribution"
        if config.evidence_export_root is not None:
            try:
                directory = await asyncio.to_thread(runner.export_evidence, config.evidence_export_root)
                common["evidence_dir"] = str(directory)
            except OSError:
                LOGGER.warning("Could not export RoboDojo evidence from %s", directory, exc_info=True)
        video = directory / "native" / "evaluation" / "sensors.mp4"
        trajectory = directory / "native" / "actor" / "transcript.jsonl"
        common.update(
            outcome_class=outcome,
            video_path=str(video) if video.is_file() else None,
            trajectory_path=str(trajectory) if trajectory.is_file() else None,
        )
        if failure_kind is not None:
            return RoboDojoResponse(
                **common,
                reward=0,
                mask_sample=True,
                failure_kind=failure_kind,
                failure_reason=failure_reason,
            )

        async def verified_grade() -> RoboDojoResponse:
            return RoboDojoResponse(
                **common,
                reward=float(scores.within_budget),
                native_score=scores.score_within_native,
                extended_score=scores.score_within_budget,
                native_success=scores.within_native,
                extended_success=scores.within_budget,
                failure_kind=POLICY_FAILURE_KINDS.get(outcome),
                failure_reason=outcome if outcome in POLICY_FAILURE_KINDS else None,
            )

        return await observe_verifier(verified_grade, enabled=True, verifier_name=config.name)()

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
