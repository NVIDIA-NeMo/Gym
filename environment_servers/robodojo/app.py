# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run RoboDojo episodes through EmberBridge and return native Nanoverse grades."""

import asyncio
import logging
import secrets
import tempfile
from contextlib import asynccontextmanager, nullcontext
from pathlib import Path
from statistics import mean
from typing import Literal, Self

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, JsonValue, PrivateAttr, model_validator

from environment_servers.robodojo.cases import (
    ROBODOJO_SIMULATORS,
    RoboDojoSimulator,
    bind_case,
    parse_pool_cases,
)
from environment_servers.robodojo.episode import CleanupUnconfirmed, Episode, EpisodeError
from environment_servers.robodojo.native_pool import NativePool
from environment_servers.robodojo.opencode_relay import mount_opencode_relay
from nemo_gym.client import apply_rollout_prefix
from nemo_gym.config import (
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    ModelServerRef,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    AGENT_TIMEOUT,
    ENVIRONMENT_SERVER_FAILED,
    HARNESS_TERMINATION_UNCONFIRMED,
    MODEL_CONTEXT_LIMIT,
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)
from nemo_gym.core.failure_kinds import ENVIRONMENT_PROTOCOL_VIOLATION
from nemo_gym.server_runtime import (
    BaseEnvironmentServer,
    BaseEnvironmentServerConfig,
    BaseVerifyResponse,
    CleanupContext,
    observe_verifier,
)
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint


LOGGER = logging.getLogger(__name__)
POLICY_FAILURE_KINDS = {
    "policy_failure.decision_budget": AGENT_LIMIT,
    "policy_failure.deadline": AGENT_TIMEOUT,
    "policy_failure.context": MODEL_CONTEXT_LIMIT,
    "policy_failure.no_progress": AGENT_INVALID_OUTPUT,
}


class RoboDojoTaskInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    case_id: str = Field(min_length=1)
    responses_create_params: NeMoGymResponseCreateParamsNonStreaming = Field(
        default_factory=lambda: NeMoGymResponseCreateParamsNonStreaming(input=[])
    )


class RoboDojoEnvironmentConfig(BaseEnvironmentServerConfig):
    agent_server: AgentServerRef
    model_server: ModelServerRef
    model_name: str
    bridge_config: Path | None = None
    panel_manifest: Path | None = None
    case_validity: Path | None = None
    bridge_python: Path | None = None
    simulation_python: Path | None = None
    task_policy: Path | None = None
    output_root: Path
    evidence_export_root: Path | None = None
    horizon_profile: Literal["native_x1", "native_x2"] = "native_x2"
    episode_timeout_seconds: float = Field(default=10800, gt=0, allow_inf_nan=False)
    cleanup_timeout_seconds: float = Field(default=180, gt=0, allow_inf_nan=False)
    pool_cases: str | None = None
    pool_launcher: Path | None = None
    pool_startup_seconds: float = Field(default=1200, gt=0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_pool(self) -> Self:
        if (self.pool_cases is None) != (self.pool_launcher is None):
            raise ValueError("Configure both RoboDojo pool cases and the pool launcher")
        if self.pool_cases is not None:
            parse_pool_cases(self.pool_cases)
            if self.case_validity is None:
                raise ValueError("RoboDojo pool cases require frozen validity data")
        return self


class RoboDojoVerifyResponse(BaseVerifyResponse):
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


class RoboDojoEnvironmentServer(
    BaseEnvironmentServer[BaseEpisodeRequest[RoboDojoTaskInput], BaseEpisodeResponse[dict[str, JsonValue]]]
):
    config: RoboDojoEnvironmentConfig
    request_model = BaseEpisodeRequest[RoboDojoTaskInput]
    response_model = BaseEpisodeResponse[dict[str, JsonValue]]
    _gpu: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _pool_lock: asyncio.Lock = PrivateAttr(default_factory=asyncio.Lock)
    _pool: NativePool | None = PrivateAttr(default=None)
    _active_pool_cases: set[str] = PrivateAttr(default_factory=set)
    _cleanup_unconfirmed: bool = PrivateAttr(default=False)
    _relays: dict[str, str] = PrivateAttr(default_factory=dict)
    _relay_mounted: bool = PrivateAttr(default=False)

    def setup_webserver(self) -> FastAPI:
        app = super().setup_webserver()
        mount_opencode_relay(app, self._relays, server_name=self.config.model_server.name)
        self._relay_mounted = True
        parent = app.router.lifespan_context

        @asynccontextmanager
        async def lifespan(application: FastAPI):
            try:
                async with parent(application) as state:
                    yield state
            finally:
                if self._pool is not None:
                    await self._pool.close(self.config.cleanup_timeout_seconds)

        app.router.lifespan_context = lifespan
        return app

    def multiple_workers_unsupported_reason(self) -> str:
        return "one RoboDojo environment server instance owns one local GPU"

    def _resolve_model_base_url(self, rollout_id: str | None) -> str:
        server_config = get_first_server_config_dict(
            self.server_client.global_config_dict, self.config.model_server.name
        )
        base_url = self.server_client._build_server_base_url(server_config)
        prefixed = apply_rollout_prefix(base_url, rollout_id)
        return f"{prefixed}/v1"

    async def _pool_slot(self, case_id: str, bound: dict, binding: dict, runner: Episode) -> dict:
        config = self.config
        cases = parse_pool_cases(config.pool_cases)
        if case_id not in cases:
            raise ValueError("This case is outside the configured RoboDojo pool")
        async with self._pool_lock:
            if self._cleanup_unconfirmed:
                raise CleanupUnconfirmed("An earlier episode did not release the GPU")
            if self._pool is None:
                cohort = {key: bind_case(case_id=key, **binding)[0] for key in cases}
                pool = NativePool()
                try:
                    await pool.start(
                        config.bridge_python,
                        config.pool_launcher,
                        cohort,
                        runner.environment,
                        (config.evidence_export_root or config.output_root) / "native-pool.log",
                        config.pool_startup_seconds,
                    )
                except BaseException:
                    try:
                        await pool.close(config.cleanup_timeout_seconds)
                    except BaseException:
                        self._cleanup_unconfirmed = True
                        raise
                    raise
                self._pool = pool
            if case_id not in self._pool.seats:
                raise ValueError("This case is missing from the started RoboDojo pool")
            if case_id in self._active_pool_cases:
                raise ValueError("This RoboDojo pool case is already running")
            self._active_pool_cases.add(case_id)
            seat = self._pool.seats[case_id]
        runner.pool_evidence = Path(seat["native_pool_evidence"]).resolve()
        return {**seat, "model": bound["model"]}

    async def run(
        self, request: BaseEpisodeRequest[RoboDojoTaskInput], cleanup: CleanupContext
    ) -> BaseEpisodeResponse[dict[str, JsonValue]]:
        """Bind the Gym task, launch the EmberBridge engine, and return its measured grade."""
        del cleanup
        config = self.config
        task_input = request.task.task_input
        if (
            config.bridge_config is None
            or config.panel_manifest is None
            or config.bridge_python is None
            or config.simulation_python is None
        ):
            raise HTTPException(422, "Configure the installed RoboDojo runtime, frozen panel and Gym model server")
        model_url = self._resolve_model_base_url(request.episode_id.capture_key)
        binding = {
            "template_path": config.bridge_config,
            "panel_path": config.panel_manifest,
            "validity_path": config.case_validity,
            "horizon": config.horizon_profile,
            "model_url": model_url,
            "model_name": config.model_name,
        }
        try:
            bound, case = bind_case(case_id=task_input.case_id, **binding)
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
            "responses_create_params": task_input.responses_create_params,
            "case_id": task_input.case_id,
            "response": response,
            "assigned_case_id": case.case_id,
            "layout_sha256": case.layout.sha256,
            "simulator": ROBODOJO_SIMULATORS[bound["simulator_backend"]],
            "evidence_dir": str(directory),
        }
        outcome = None
        failure_kind = None
        failure_reason = None
        relay_id: str | None = None
        claimed_pool_case: str | None = None
        if self._relay_mounted:
            relay_id = secrets.token_urlsafe(18)
            self._relays[relay_id] = model_url
            relay_host = "127.0.0.1" if config.host in ("0.0.0.0", "::", "") else config.host
            relay_url = f"http://{relay_host}:{config.port}/_opencode_relay/{relay_id}/v1"
            binding["model_url"] = relay_url
            bound["model"]["base_url"] = relay_url
        try:
            async with self._gpu if config.pool_cases is None else nullcontext():
                try:
                    if self._cleanup_unconfirmed:
                        raise CleanupUnconfirmed("An earlier episode did not release the GPU")
                    if config.pool_cases is not None:
                        bound = await self._pool_slot(task_input.case_id, bound, binding, runner)
                        claimed_pool_case = task_input.case_id
                    outcome, scores = await runner.run(bound)
                except CleanupUnconfirmed as error:
                    self._cleanup_unconfirmed = True
                    failure_kind, failure_reason = HARNESS_TERMINATION_UNCONFIRMED, str(error)
                except asyncio.CancelledError:
                    self._cleanup_unconfirmed = True
                    raise
                except TimeoutError:
                    failure_kind = ENVIRONMENT_SERVER_FAILED
                    failure_reason = "RoboDojo supervisor exceeded its episode deadline"
                except (EpisodeError, OSError, RuntimeError, ValueError, KeyError) as error:
                    failure_kind = ENVIRONMENT_SERVER_FAILED
                    failure_reason = f"{type(error).__name__}: RoboDojo native measurement failed"
                    LOGGER.warning("RoboDojo case %s could not be scored", task_input.case_id, exc_info=True)
                else:
                    if (
                        scores.task_step_limit != case.native_step_limit
                        or scores.control_budget != bound["step_limit"]
                    ):
                        failure_kind = ENVIRONMENT_PROTOCOL_VIOLATION
                        failure_reason = "Native horizon does not match the frozen case"
        finally:
            if claimed_pool_case is not None:
                self._active_pool_cases.discard(claimed_pool_case)
            if relay_id is not None:
                self._relays.pop(relay_id, None)
        if config.evidence_export_root is not None:
            try:
                directory = await asyncio.to_thread(runner.export_evidence, config.evidence_export_root)
                common["evidence_dir"] = str(directory)
            except (OSError, ValueError, KeyError):
                LOGGER.warning("Could not export RoboDojo evidence from %s", directory, exc_info=True)
        video = directory / "native" / "evaluation" / "sensors.mp4"
        trajectory = directory / "native" / "actor" / "transcript.jsonl"
        common.update(
            outcome_class=outcome,
            video_path=str(video) if video.is_file() else None,
            trajectory_path=str(trajectory) if trajectory.is_file() else None,
        )
        if failure_kind is not None:
            verify_response = RoboDojoVerifyResponse(
                **common,
                reward=0,
                mask_sample=True,
                failure_kind=failure_kind,
                failure_reason=failure_reason,
            )
            return self.response_model(
                episode_id=request.episode_id,
                task_id=request.task.task_id,
                result=verify_response.model_dump(mode="json"),
            )

        async def verified_grade() -> RoboDojoVerifyResponse:
            return RoboDojoVerifyResponse(
                **common,
                reward=float(scores.within_budget),
                native_score=scores.score_within_native,
                extended_score=scores.score_within_budget,
                native_success=scores.within_native,
                extended_success=scores.within_budget,
                failure_kind=POLICY_FAILURE_KINDS.get(outcome),
                failure_reason=outcome if outcome in POLICY_FAILURE_KINDS else None,
            )

        graded = await observe_verifier(verified_grade, enabled=True, verifier_name=config.name)()
        return self.response_model(
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            result=graded.model_dump(mode="json"),
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest = Body()) -> AggregateMetrics:
        scored = [
            row for row in body.verify_responses if not row.get("mask_sample") and row.get("native_score") is not None
        ]
        metrics: dict[str, float | int] = {
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
        return AggregateMetrics(agent_metrics=metrics, key_metrics=metrics)


if __name__ == "__main__":
    RoboDojoEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = RoboDojoEnvironmentServer.run_webserver()
