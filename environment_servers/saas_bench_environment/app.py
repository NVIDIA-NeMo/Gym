# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SaaS episode lifecycle and native verification through an agent session."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from pydantic import Field, PrivateAttr, ValidationError

from nemo_gym.agents import (
    AgentCloseSessionRequest,
    AgentCloseSessionResponse,
    AgentSeedSessionRequest,
    AgentSeedSessionResponse,
)
from nemo_gym.client import get_response_json, raise_for_status
from nemo_gym.config import (
    ATTEMPT_INDEX_KEY_NAME,
    TOKEN_CAPTURE_BLOCK,
    TOKEN_CAPTURE_PATH_SEGMENT,
    AgentServerRef,
    AggregateMetrics,
    AggregateMetricsRequest,
    get_first_server_config_dict,
)
from nemo_gym.core import (
    AGENT_COMMAND_NOT_ALLOWED,
    AGENT_INVALID_OUTPUT,
    AGENT_LIMIT,
    AGENT_RUN_ERROR,
    AGENT_TIMEOUT,
    ENVIRONMENT_SERVER_FAILED,
    JUDGE_FAILED,
    JUDGE_UNPARSEABLE,
    MODEL_CONTEXT_LIMIT,
    VERIFIER_ERROR,
    BaseEpisodeRequest,
    BaseEpisodeResponse,
    EpisodeId,
    FailureClass,
    FailureReportRow,
    MaterializedTask,
    NeMoGymResponse,
    OutcomeAction,
    StageOutcomePolicyConfig,
    StagePolicyConfig,
    TaskId,
    choose_stage_outcome,
    maybe_rollout_id_from_run_body,
    model_sampling_from_run_body,
    model_sampling_path_segment,
    skips_token_capture,
)
from nemo_gym.server_runtime import AgentContext, BaseEnvironmentServer, BaseEnvironmentServerConfig, CleanupContext
from nemo_gym.utils import is_nemo_gym_fastapi_entrypoint
from responses_api_agents.saas_bench_agent.response import actor_response
from responses_api_agents.saas_bench_agent.runtime import DEFAULT_BUNDLE_ROOT, check_bundle_code
from responses_api_agents.saas_bench_agent.schemas import SaasBenchRunRequest, SaasBenchVerifyResponse


_MODEL_ENDINGS = {
    "malformed_tool_call": AGENT_INVALID_OUTPUT,
    "model_protocol_error": AGENT_INVALID_OUTPUT,
    "output_token_limit": AGENT_LIMIT,
    "context_limit": MODEL_CONTEXT_LIMIT,
    "timeout": AGENT_TIMEOUT,
    "agent_error": AGENT_INVALID_OUTPUT,
    "permission_abort": AGENT_COMMAND_NOT_ALLOWED,
}


def default_outcome_policy() -> StageOutcomePolicyConfig:
    """Grade model endings from app state; retry evaluation failures."""
    return StageOutcomePolicyConfig(
        by_stage={
            "verification": StagePolicyConfig(default_action=OutcomeAction.FAIL),
            "agent": StagePolicyConfig(
                by_failure_class={
                    FailureClass.MODEL: OutcomeAction.VERIFY,
                    FailureClass.EVALUATION: OutcomeAction.RETRY,
                },
                default_action=OutcomeAction.RETRY,
            ),
        }
    )


class SaasBenchEnvironmentConfig(BaseEnvironmentServerConfig):
    source_root: str = str(DEFAULT_BUNDLE_ROOT)
    agent_server: AgentServerRef
    harness: Literal["opencode", "browser_use"] = "opencode"
    model_name: str = "policy"
    outcome_policy: StageOutcomePolicyConfig = Field(default_factory=default_outcome_policy)
    default_episode_timeout_seconds: float | None = 7500
    cleanup_timeout_seconds: float = 210


class SaasBenchEpisodeRequest(BaseEpisodeRequest[SaasBenchRunRequest]):
    pass


class SaasBenchEpisodeResponse(BaseEpisodeResponse[SaasBenchVerifyResponse]):
    pass


class SaasBenchEnvironmentServer(BaseEnvironmentServer[SaasBenchEpisodeRequest, SaasBenchEpisodeResponse]):
    """Own episode ordering, typed failures, native grading, metrics and bounded session cleanup."""

    config: SaasBenchEnvironmentConfig
    _prepared: bool = PrivateAttr(default=False)
    applies_model_sampling = True
    request_model = SaasBenchEpisodeRequest
    response_model = SaasBenchEpisodeResponse

    def _prepare_runtime(self):
        if not self._prepared:
            source = Path(self.config.source_root)
            if not source.is_absolute():
                source = Path(__file__).resolve().parents[2] / source
            source = source.resolve(strict=True)
            check_bundle_code(source)
            sys.path.insert(0, str(source))
            self._prepared = True

    def setup_webserver(self) -> FastAPI:
        app = FastAPI()
        self.setup_environment_checkpoint(app)
        app.post("/run")(self.run_flat)
        app.post("/aggregate_metrics")(self.aggregate_metrics)
        return app

    async def run_flat(self, row: dict) -> dict:
        typed = "episode_id" in row and "task" in row
        try:
            if typed:
                request = SaasBenchEpisodeRequest.model_validate(row)
            else:
                body = SaasBenchRunRequest.model_validate(row)
                identity = maybe_rollout_id_from_run_body(body)
                if not identity:
                    raise HTTPException(400, "SaaS Bench requires _ng_rollout_id; enable Gym observability")
                request = SaasBenchEpisodeRequest(
                    episode_id=EpisodeId.from_capture_key(identity),
                    model_sampling=model_sampling_from_run_body(row),
                    skip_token_capture=skips_token_capture(row),
                    task=MaterializedTask(
                        task_id=TaskId(
                            taskset=self.config.agent_server.name, task_id=body.verifier_metadata["task_id"]
                        ),
                        task_input=body,
                    ),
                )
        except ValidationError as error:
            raise RequestValidationError(error.errors(include_url=False)) from None
        response = await self.run_request(request)
        if typed:
            return response.model_dump(mode="json")
        if response.result is not None:
            result = response.result.model_dump(mode="json", by_alias=True)
            result["agent_ref"] = {"name": self.config.agent_server.name}
            if response.failure_report is not None:
                FailureReportRow.merged_into(result, response.failure_report)
            return result
        failure = response.failure
        result = {
            "_ng_failure_class": failure.failure_kind or ENVIRONMENT_SERVER_FAILED,
            "_ng_failure_kind": failure.failure_kind,
            "_ng_failure_terminal": failure.terminal,
            "_ng_failure_message": failure.failure_reason,
            "_ng_failure_stage": failure.stage,
            "agent_ref": {"name": self.config.agent_server.name},
        }
        if response.failure_report is not None:
            FailureReportRow.merged_into(result, response.failure_report)
        return result

    async def run(self, request: SaasBenchEpisodeRequest, cleanup: CleanupContext) -> SaasBenchEpisodeResponse:
        self._prepare_runtime()
        session_id = "saas-" + uuid4().hex
        cookies = {}

        async def close():
            reply = await self.server_client.post(
                self.config.agent_server.name,
                "/v1/agent_sessions/close",
                json=AgentCloseSessionRequest(agent_session_id=session_id, episode_id=request.episode_id),
                cookies=cookies,
                _control=True,
            )
            await raise_for_status(reply)
            receipt = AgentCloseSessionResponse.model_validate(await get_response_json(reply))
            if receipt.agent_session_id != session_id or receipt.cleanup_confirmed is not True:
                raise RuntimeError("SaaS session cleanup was not confirmed")

        cleanup.register_cleanup("saas-agent", close)
        body = request.task.task_input.model_copy(deep=True)
        body.capture_rollout_id = request.episode_id.rollout_id
        setattr(body, ATTEMPT_INDEX_KEY_NAME, request.episode_id.attempt)
        if request.task.task_id.task_id != body.verifier_metadata["task_id"]:
            raise ValueError("SaaS task identity mismatch")
        if body.verifier_metadata.get("harness", self.config.harness) != self.config.harness:
            raise ValueError("SaaS environment harness does not match the prepared task")
        seed = AgentSeedSessionRequest(
            agent_session_id=session_id,
            episode_id=request.episode_id,
            task_id=request.task.task_id,
            agent_context=AgentContext(
                values={
                    "episode": {
                        **body.model_dump(mode="json", by_alias=True),
                        "_ng_rollout_id": request.episode_id.rollout_id,
                        ATTEMPT_INDEX_KEY_NAME: request.episode_id.attempt,
                    }
                }
            ),
        )
        reply = await self.server_client.post(
            self.config.agent_server.name, "/v1/agent_sessions", json=seed, cookies=cookies
        )
        await raise_for_status(reply)
        seeded = AgentSeedSessionResponse.model_validate(await get_response_json(reply))
        if seeded.agent_session_id != session_id:
            raise RuntimeError("SaaS agent returned a different session identity")
        cookies.update({key: value.value for key, value in reply.cookies.items()})
        reply = await self.server_client.post(
            self.config.agent_server.name,
            self._agent_responses_path(request),
            json=body.responses_create_params,
            cookies=cookies,
        )
        await raise_for_status(reply)
        payload = await get_response_json(reply)
        native = payload.pop("saas_bench_native_result")
        response = NeMoGymResponse.model_validate(payload)
        verified = self._response(body, native, response)
        return SaasBenchEpisodeResponse(episode_id=request.episode_id, task_id=request.task.task_id, result=verified)

    def _agent_responses_path(self, request: SaasBenchEpisodeRequest) -> str:
        block = self.server_client.global_config_dict.get(TOKEN_CAPTURE_BLOCK) or {}
        config = get_first_server_config_dict(self.server_client.global_config_dict, self.config.agent_server.name)
        capture = (
            not request.skip_token_capture
            and bool(block.get("enabled"))
            and (block.get("all_agents") or config.get("token_id_capture"))
        )
        segment = f"/{TOKEN_CAPTURE_PATH_SEGMENT}" if capture else ""
        sampling = model_sampling_path_segment(self.model_sampling_segment(request, trained=True))
        return f"/ng-rollout/{request.episode_id.capture_key}{segment}{sampling}/v1/responses"

    def _response(
        self, body: SaasBenchRunRequest, result: dict, response: NeMoGymResponse | None = None
    ) -> SaasBenchVerifyResponse:
        from saas_bench.opencode.core import measured_score
        from saas_bench.opencode.observability import native_trajectory
        from saas_bench.opencode.protocol import has_model_turn

        native = result.get("native_trajectory", result.get("trajectory", []))
        model_turn = result.get("model_turn_observed") is True or has_model_turn(native)
        masked = result.get("mask_sample", True)
        kind = None
        decision = None
        # Judge/readback faults take precedence over the actor's ending: there
        # is no usable evaluation verdict, even when the actor also failed.
        diagnostics = result.get("verifier_diagnostics") or {}
        judge_faults = (result.get("verify") or {}).get("judge_failures") or diagnostics.get("judge_failures") or []
        if judge_faults:
            kind = (
                JUDGE_UNPARSEABLE
                if all(f["failure_kind"] == JUDGE_UNPARSEABLE for f in judge_faults)
                else JUDGE_FAILED
            )
        elif result.get("failure_kind") == "saas_bench:verifier" or diagnostics.get("unavailable"):
            kind = VERIFIER_ERROR
        elif (
            result.get("failure_kind") == "saas_bench:infrastructure"
            or result.get("failure_origin") == "evaluation"
            or self.config.harness == "browser_use"
            and result.get("failure_kind") == "saas_bench:model_ending"
            and result.get("failure_origin") != "model"
        ):
            kind = AGENT_RUN_ERROR
        else:
            termination = (result.get("execution_health") or {}).get("termination")
            kind = (
                (AGENT_TIMEOUT if model_turn else AGENT_RUN_ERROR)
                if result.get("timed_out") or termination == "timeout"
                else _MODEL_ENDINGS.get(termination)
            )
            if kind is None and result.get("failure_kind"):
                kind = AGENT_RUN_ERROR
        if kind is not None:
            stage = "verification" if kind in {VERIFIER_ERROR, JUDGE_FAILED, JUDGE_UNPARSEABLE} else "agent"
            decision = choose_stage_outcome(
                failure_kind=kind, http_status=None, stage=stage, config=self.config.outcome_policy
            )
            masked = decision.action not in {OutcomeAction.COUNT_AS_ZERO, OutcomeAction.VERIFY}
        reward = (
            0.0
            if masked or decision and decision.action == OutcomeAction.COUNT_AS_ZERO
            else measured_score(result["verify"])
        )
        response = response or actor_response(
            body, result, harness=self.config.harness, model_name=self.config.model_name
        )
        return SaasBenchVerifyResponse.model_validate(
            {
                **body.model_dump(by_alias=True),
                "response": response,
                "reward": reward,
                "mask_sample": masked,
                "failure_kind": kind,
                "failure_outcome_action": decision.action if decision else None,
                "failure_outcome_attribution": decision.attribution if decision else None,
                "native_failure_kind": result.get("failure_kind"),
                "native_metrics": {} if masked else {"weighted_score": reward, "execution": 1.0},
                "native_diagnostics": {}
                if masked
                else {
                    "task_id": body.verifier_metadata["task_id"],
                    "source_sha256": body.verifier_metadata["source_sha256"],
                    "rollout_id": maybe_rollout_id_from_run_body(body),
                    "verifier": result["verify"],
                    "graded_response_id": response.id,
                    "graded_response_sha256": hashlib.sha256(response.model_dump_json().encode()).hexdigest(),
                },
                "failure_reason": result.get("error") or result.get("execution_error"),
                "execution_error": result.get("execution_error"),
                "task_id": body.verifier_metadata["task_id"],
                "category_id": body.verifier_metadata["category_id"],
                "artifacts": result["artifacts"],
                "verify_status": result.get("verify", {}).get("status", "ERROR"),
                "trajectory": native,
                "native_trajectory": native,
                "harness": self.config.harness,
                "execution_health": result.get("execution_health", {}),
                "harness_version": result.get("harness_version"),
                "verifier_diagnostics": result.get("verifier_diagnostics", {}),
                "ng_trajectory": native_trajectory(
                    native,
                    body.verifier_metadata["task_id"],
                    maybe_rollout_id_from_run_body(body) or "unknown",
                    result.get("execution_health", {}),
                )
                if self.config.harness == "opencode"
                else None,
            }
        )

    async def aggregate_metrics(self, body: AggregateMetricsRequest) -> AggregateMetrics:
        rows = body.verify_responses
        valid = [r for r in rows if not r.get("mask_sample", False)]
        metrics = {
            "saas_bench/num_scored": len(valid),
            "saas_bench/num_masked": len(rows) - len(valid),
            "saas_bench/num_verifier_errors": sum(
                r.get("failure_kind") in {"saas_bench:verifier", VERIFIER_ERROR} for r in rows
            ),
        }
        health = [r["execution_health"] for r in rows if r.get("execution_health")]
        metrics["saas_bench/num_execution_health_unobserved"] = len(rows) - len(health)
        if health:
            metrics["saas_bench/tool_calls"] = sum(h["tool_calls"] for h in health)
            metrics["saas_bench/tool_errors"] = sum(h["tool_errors"] for h in health)
            metrics["saas_bench/execution_healthy_rate"] = sum(h["execution_healthy"] for h in health) / len(health)
            metrics["saas_bench/malformed_terminations"] = sum(
                h["termination"] in {"malformed_tool_call", "model_protocol_error"} for h in health
            )
            metrics["saas_bench/incomplete_tasks"] = sum(h.get("task_success") is False for h in health)
        if valid:
            metrics["saas_bench/mean_score"] = sum(r["reward"] for r in valid) / len(valid)
            metrics["saas_bench/pass_rate"] = sum(r["verify_status"] == "PASS" for r in valid) / len(valid)
            for domain in sorted({r["category_id"] for r in valid}):
                subset = [r for r in valid if r["category_id"] == domain]
                metrics[f"saas_bench/{domain}/mean_score"] = sum(r["reward"] for r in subset) / len(subset)
        return AggregateMetrics(
            agent_metrics=metrics,
            key_metrics={k: v for k, v in metrics.items() if k in {"saas_bench/mean_score", "saas_bench/pass_rate"}},
        )


if __name__ == "__main__":
    SaasBenchEnvironmentServer.run_webserver()
elif is_nemo_gym_fastapi_entrypoint(__file__):
    app = SaasBenchEnvironmentServer.run_webserver()
