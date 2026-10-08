# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate materialized UserSim rows and verify Environment-owned results."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Mapping, Sequence
from time import monotonic
from types import SimpleNamespace
from typing import Any

from fastapi import HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    ResourcesCloseSessionRequest,
    ResourcesCloseSessionResponse,
    ResourcesSeedSessionRequest,
    SimpleResourcesServer,
)
from nemo_gym.config_types import ModelServerRef
from nemo_gym.episode_types import EpisodeId, TaskId
from nemo_gym.failure_kinds import JUDGE_FAILED, VERIFIER_ERROR, is_namespaced, is_registered
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from resources_servers.nemo_user_sim.episode_contracts import (
    UserSimSeedResponse,
    UserSimSimulationResult,
    UserSimTaskInput,
    UserSimVerification,
    UserSimVerifyRequest,
)


PROBE_SCORERS = {
    "financial_services": "financial_services",
    "health_decision_support_disclosure": "health_disclosure_concealment",
    "health_general_disclosure": "health_disclosure_concealment",
    "health_therapy_disclosure": "health_disclosure_concealment",
    "health_triage_disclosure": "health_disclosure_concealment",
    "identity_disclosure": "identity_disclosure",
    "safety_agentic": "safety_agentic",
    "safety_chat_pressure": "safety_chat_pressure",
    "sov_ai_dynamic": "sov_ai_dynamic",
    "sov_ai_facts": "sov_ai_facts",
    "sov_ai_multilingual_parity": "sov_ai_multilingual_parity",
    "tool_calling": "tool_use",
}
ASSISTANT_QUALITY_AXES = ("helpfulness", "accuracy", "coherence")
logger = logging.getLogger(__name__)


class UserSimResourcesServerConfig(BaseResourcesServerConfig):
    """Configure materialized-row validation and trajectory verification."""

    usersim_revision: str = Field(
        "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee",  # pragma: allowlist secret
        pattern=r"^[0-9a-f]{40}$",
    )
    nemotron_personas_version: str = "0.0.2"
    probe_scorer_model: ModelServerRef
    judge_max_output_tokens: int = Field(16_384, gt=0)
    judge_max_output_tokens_non_ascii: int = Field(32_768, gt=0)
    model_call_timeout_seconds: float = Field(300.0, gt=0)
    evaluation_timeout_seconds: float = Field(1200.0, gt=0)
    session_ttl_seconds: float = Field(1980.0, gt=0)


class SeededUserSimEpisode(BaseModel):
    """Immutable Resources snapshot for one Environment-owned episode."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    episode_id: EpisodeId
    task_id: TaskId
    resolved_row: dict[str, Any]
    resolved_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    expires_at: float


class _ResourcesModelFacade:
    """Async UserSim evaluator facade backed by a Gym Model Server."""

    def __init__(self, server: "UserSimResourcesServer", model: ModelServerRef) -> None:
        self.server = server
        self.model = model
        config = get_first_server_config_dict(server.server_client.global_config_dict, model.name)
        self.model_name = str(config.get("model") or model.name)

    async def acompletion(self, messages: Sequence[Any], **kwargs: Any) -> SimpleNamespace:
        unsupported = set(kwargs) - {
            "max_tokens",
            "max_completion_tokens",
            "reasoning_effort",
            "response_format",
            "temperature",
        }
        if unsupported:
            raise NotImplementedError(f"Unsupported UserSim evaluator options: {sorted(unsupported)}")
        values: dict[str, Any] = {
            "input": [_to_responses_input(message) for message in messages],
        }
        max_tokens = kwargs.get("max_tokens") or kwargs.get("max_completion_tokens")
        if max_tokens is not None:
            values["max_output_tokens"] = max_tokens
        if kwargs.get("temperature") is not None:
            values["temperature"] = kwargs["temperature"]
        if kwargs.get("reasoning_effort") is not None:
            values["reasoning"] = {"effort": kwargs["reasoning_effort"]}
        response_format = kwargs.get("response_format")
        if response_format is not None:
            json_schema = response_format.get("json_schema")
            if response_format.get("type") != "json_schema" or not isinstance(json_schema, Mapping):
                raise NotImplementedError(f"Unsupported response format: {response_format!r}")
            schema_format = {
                "type": "json_schema",
                "name": json_schema["name"],
                "schema": json_schema["schema"],
            }
            strict = json_schema.get("strict")
            if strict is not None:
                schema_format["strict"] = strict
            values["text"] = {"format": schema_format}
        request_params = NeMoGymResponseCreateParamsNonStreaming.model_validate(values)
        try:
            async with asyncio.timeout(self.server.config.model_call_timeout_seconds):
                response = await self.server.server_client.post(
                    server_name=self.model.name,
                    url_path="/v1/responses",
                    json=request_params,
                )
                await raise_for_status(response)
                gym_response = NeMoGymResponse.model_validate(await get_response_json(response))
        except TimeoutError as error:
            raise TimeoutError(
                f"Timed out after {self.server.config.model_call_timeout_seconds}s waiting for {self.model.name}"
            ) from error
        usage = gym_response.usage
        return SimpleNamespace(
            message=SimpleNamespace(content=_response_text(gym_response), reasoning_content=None, tool_calls=None),
            usage=(
                SimpleNamespace(input_tokens=usage.input_tokens, output_tokens=usage.output_tokens)
                if usage is not None
                else None
            ),
        )


def _create_evaluator(config: Any, models: Mapping[str, Any]) -> Any:
    """Adapt UserSim's Data Designer evaluator for one hosted verification."""
    from usersim.engine.evaluator.generator import TrajectoryEvaluatorGenerator

    class _HostedTrajectoryEvaluator(TrajectoryEvaluatorGenerator):
        def __init__(self) -> None:
            self._config = config
            self._models = dict(models)
            self._initialize()

        def get_model(self, alias: str) -> Any:
            return self._models[alias]

        def get_models(self) -> dict[str, Any]:
            return dict(self._models)

    return _HostedTrajectoryEvaluator()


class UserSimResourcesServer(SimpleResourcesServer):
    """Validate task identity and verify the Environment-owned UserSim result."""

    ray_enabled = False
    config: UserSimResourcesServerConfig
    session_id_to_seed: dict[str, SeededUserSimEpisode] = {}
    closed_resources_session_ids: dict[str, float] = {}

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest,
    ) -> UserSimSeedResponse:
        try:
            task = UserSimTaskInput.model_validate(body.task_data)
            _validate_resolved_row(
                task.resolved_row,
                expected_revision=self.config.usersim_revision,
                expected_personas_version=self.config.nemotron_personas_version,
            )
        except (ValidationError, ValueError) as error:
            detail = error.errors() if isinstance(error, ValidationError) else str(error)
            raise HTTPException(status_code=422, detail=detail) from error
        session_id = request.session[SESSION_ID_KEY]
        self._expire_sessions()
        resolved_row = task.resolved_row
        if body.resources_session_id in self.closed_resources_session_ids:
            raise HTTPException(status_code=409, detail="Resources session is already closed")
        existing = self.session_id_to_seed.get(session_id)
        if existing is not None:
            if (
                existing.resources_session_id != body.resources_session_id
                or existing.episode_id != body.episode_id
                or existing.task_id != body.task_id
                or existing.resolved_row_sha256 != _row_digest(resolved_row)
            ):
                raise HTTPException(status_code=409, detail="Resources session is already seeded for another task")
            return UserSimSeedResponse(
                resources_session_id=existing.resources_session_id,
                sandbox_access=None,
            )
        self.session_id_to_seed[session_id] = SeededUserSimEpisode(
            resources_session_id=body.resources_session_id,
            episode_id=body.episode_id,
            task_id=body.task_id,
            resolved_row=resolved_row,
            resolved_row_sha256=_row_digest(resolved_row),
            expires_at=monotonic() + self.config.session_ttl_seconds,
        )
        return UserSimSeedResponse(
            resources_session_id=body.resources_session_id,
            sandbox_access=None,
        )

    async def verify(
        self,
        request: Request,
        body: UserSimVerifyRequest,
    ) -> UserSimVerification:
        seeded = self._seeded_episode(request)
        verification_input = body.verification_input
        if body.episode_id != seeded.episode_id or body.task_id != seeded.task_id:
            raise HTTPException(status_code=409, detail="Verification identity does not match the seeded episode")
        if _row_digest(verification_input.resolved_row) != seeded.resolved_row_sha256:
            raise HTTPException(status_code=409, detail="Verified resolved row does not match the seeded task")

        usersim_result = verification_input.usersim_result
        result_row = usersim_result.model_dump(mode="python")
        trajectory_id = seeded.resolved_row.get("trajectory_id")
        if trajectory_id and result_row.get("trajectory_id") != trajectory_id:
            raise HTTPException(status_code=409, detail="UserSim result trajectory_id does not match the seeded task")

        if usersim_result.simulation_outcome.get("failure_attribution") == "assistant_model":
            participants_completed = {"user", "assistant"} <= _conversation_roles(usersim_result)
            outcome = usersim_result.simulation_outcome
            return UserSimVerification(
                reward=0.0,
                mask_sample=False,
                failure_kind=_usersim_failure_kind(outcome.get("failure_class")),
                failure_reason=outcome.get("failure_detail") or outcome.get("failure_reason"),
                reward_components={
                    "participants_completed": float(participants_completed),
                    "conversation_status": float(usersim_result.conversation_status),
                    "scorer_applied": 0.0,
                    "scorer_pass": 0.0,
                    "trajectory_evaluator_applied": 0.0,
                    "assistant_quality": 0.0,
                },
                scenario_completed=False,
                usersim_result=usersim_result,
                verifier_data={
                    "episode_interaction_protocol": verification_input.episode_interaction_protocol,
                    "resolved_row_sha256": seeded.resolved_row_sha256,
                    "scorer_result": None,
                    "assistant_eval": {
                        "skipped": True,
                        "skipped_reason": "assistant_model activation failed",
                    },
                    "normalized_axis_scores": {},
                    "scenario_completed": False,
                },
            )
        evaluation_exception: Exception | None = None
        try:
            async with asyncio.timeout(self.config.evaluation_timeout_seconds):
                assistant_eval, normalized_scores, assistant_quality = await self._evaluate(
                    seeded.resolved_row,
                    usersim_result,
                )
        except Exception as error:
            logger.exception("UserSim trajectory evaluator failed")
            evaluation_exception = error
            assistant_eval = {
                "axes": {},
                "scorers": {},
                "error": f"{type(error).__name__}: {error}",
            }
            normalized_scores = {}
            assistant_quality = None

        probe_type = str(seeded.resolved_row["probe_type"])
        scorer_name = _scorer_for_row(seeded.resolved_row)
        scorer_result = assistant_eval.get("scorers", {}).get(scorer_name) if scorer_name else None
        if scorer_name is None:
            scorer_result_state = "not_applicable"
        else:
            from usersim.taxonomy.eval_cell import scorer_state

            scorer_result_state = scorer_state(scorer_result)
        scorer_error = (
            scorer_result.get("error")
            if scorer_result_state == "error" and isinstance(scorer_result, Mapping)
            else None
        )
        status_proposal = (
            scorer_result.get("status_proposal")
            if scorer_result_state == "ok" and isinstance(scorer_result, Mapping)
            else None
        )
        scorer_pass: bool | None
        if scorer_result_state == "not_applicable":
            scorer_pass = True
        elif status_proposal is True:
            scorer_pass = True
        elif status_proposal is False:
            scorer_pass = False
        else:
            scorer_pass = None
        outcome = usersim_result.simulation_outcome
        simulation_failed = outcome.get("status") == "failed"
        failure_attribution = outcome.get("failure_attribution")
        evaluator_error = assistant_eval.get("error")
        participants_completed = {"user", "assistant"} <= _conversation_roles(usersim_result)
        scenario_completed = usersim_result.conversation_status and participants_completed and scorer_pass is True
        quality_missing = scenario_completed and assistant_quality is None and not assistant_eval.get("skipped", False)
        scorer_inconclusive = scorer_result_state == "ok" and scorer_pass is None
        mask_sample = (simulation_failed and failure_attribution != "assistant_model") or bool(
            scorer_error or evaluator_error or quality_missing or scorer_inconclusive
        )
        failure_kind = _usersim_failure_kind(outcome.get("failure_class")) if simulation_failed else None
        failure_reason = outcome.get("failure_detail") if simulation_failed else None
        if scorer_error:
            failure_kind = VERIFIER_ERROR
            failure_reason = str(scorer_error)
        elif evaluator_error:
            failure_kind = VERIFIER_ERROR
            failure_reason = str(evaluator_error)
        elif scorer_inconclusive:
            failure_kind = "usersim:scorer_inconclusive"
            failure_reason = f"{scorer_name} returned status_proposal=None"
        elif quality_missing:
            failure_kind = JUDGE_FAILED
            failure_reason = "Trajectory judge returned no assistant quality scores"
        if isinstance(evaluation_exception, TimeoutError):
            failure_reason = (
                f"Trajectory evaluation exceeded the {self.config.evaluation_timeout_seconds}s overall timeout"
            )
        reward = assistant_quality if scenario_completed and assistant_quality is not None else 0.0
        return UserSimVerification(
            reward=reward,
            mask_sample=mask_sample,
            failure_kind=failure_kind,
            failure_reason=failure_reason,
            reward_components={
                "participants_completed": float(participants_completed),
                "conversation_status": float(usersim_result.conversation_status),
                "scorer_applied": float(scorer_name is not None and scorer_result_state == "ok"),
                "scorer_pass": float(scorer_pass is True),
                "trajectory_evaluator_applied": float(not assistant_eval.get("skipped", False)),
                "assistant_quality": assistant_quality or 0.0,
                **{f"quality.{axis}": score for axis, score in normalized_scores.items()},
            },
            scenario_completed=scenario_completed,
            usersim_result=usersim_result,
            verifier_data={
                "episode_interaction_protocol": verification_input.episode_interaction_protocol,
                "resolved_row_sha256": seeded.resolved_row_sha256,
                "probe_type": probe_type,
                "scorer_name": scorer_name,
                "scorer_state": scorer_result_state,
                "scorer_status_proposal": status_proposal,
                "scorer_result": scorer_result,
                "assistant_eval": assistant_eval,
                "normalized_axis_scores": normalized_scores,
                "scenario_completed": scenario_completed,
            },
        )

    async def _evaluate(
        self,
        resolved_row: dict[str, Any],
        usersim_result: UserSimSimulationResult,
    ) -> tuple[dict[str, Any], dict[str, float], float | None]:
        from usersim.engine.evaluator.config import TrajectoryEvaluatorConfig
        from usersim.taxonomy.eval_cell import normalize_axis_score, score_from_eval_cell

        scorer_name = _scorer_for_row(resolved_row)
        config = TrajectoryEvaluatorConfig(
            name="assistant_eval",
            judges=[{"alias": "judge_model"}],
            scorers=[scorer_name] if scorer_name else [],
            skip_if_existing=False,
            max_judge_tokens=self.config.judge_max_output_tokens,
            max_judge_tokens_non_ascii=self.config.judge_max_output_tokens_non_ascii,
        )
        model = _ResourcesModelFacade(self, self.config.probe_scorer_model)
        evaluator = _create_evaluator(config, {"judge_model": model})
        trajectory = {
            **resolved_row,
            **usersim_result.model_dump(mode="python"),
        }
        generated = await evaluator.agenerate(trajectory)
        raw_eval = generated["assistant_eval"]
        evaluation = json.loads(raw_eval) if isinstance(raw_eval, str) else raw_eval
        normalized_scores: dict[str, float] = {}
        for axis in evaluation.get("envelope", {}).get("axes", []):
            score = score_from_eval_cell(evaluation, axis)
            if score is not None:
                normalized_scores[axis] = normalize_axis_score(axis, score)
        quality_scores = [normalized_scores.get(axis) for axis in ASSISTANT_QUALITY_AXES]
        assistant_quality = (
            sum(score for score in quality_scores if score is not None) / len(ASSISTANT_QUALITY_AXES)
            if all(score is not None for score in quality_scores)
            else None
        )
        return evaluation, normalized_scores, assistant_quality

    async def close_resources_session(
        self,
        request: Request,
        body: ResourcesCloseSessionRequest,
    ) -> ResourcesCloseSessionResponse:
        session_id = request.session[SESSION_ID_KEY]
        self._expire_sessions()
        seeded = self.session_id_to_seed.get(session_id)
        if seeded is not None:
            if body.resources_session_id != seeded.resources_session_id or body.episode_id != seeded.episode_id:
                raise HTTPException(status_code=409, detail="Resources session does not match the active episode")
            del self.session_id_to_seed[session_id]
        self.closed_resources_session_ids.setdefault(
            body.resources_session_id,
            monotonic() + self.config.session_ttl_seconds,
        )
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    def _seeded_episode(self, request: Request) -> SeededUserSimEpisode:
        self._expire_sessions()
        session_id = request.session[SESSION_ID_KEY]
        seeded = self.session_id_to_seed.get(session_id)
        if seeded is None:
            raise HTTPException(status_code=404, detail="No active NeMo UserSim task. Call /seed_session first.")
        return seeded

    def _expire_sessions(self) -> None:
        """Discard sessions older than the configured episode and cleanup window."""
        now = monotonic()
        for session_id, seeded in tuple(self.session_id_to_seed.items()):
            if seeded.expires_at <= now:
                del self.session_id_to_seed[session_id]
        for resources_session_id, expires_at in tuple(self.closed_resources_session_ids.items()):
            if expires_at <= now:
                del self.closed_resources_session_ids[resources_session_id]


def _validate_resolved_row(
    row: Mapping[str, Any],
    *,
    expected_revision: str,
    expected_personas_version: str,
) -> None:
    required = ("persona", "probe_type", "conversation_language", "trajectory_id", "usersim_config")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"Resolved UserSim row is missing required fields: {missing}")
    if row.get("probe_variant") == "guarded":
        raise ValueError("Guarded health probes are unsupported by the pinned UserSim revision")
    provenance = _decode_provenance(row.get("usersim_provenance"))
    if provenance.get("code_sha") != expected_revision:
        raise ValueError(
            f"Resolved UserSim row revision {provenance.get('code_sha')!r} does not match {expected_revision!r}"
        )
    personas_version = provenance.get("nemotron_personas_version")
    if personas_version != "synthetic" and personas_version != expected_personas_version:
        raise ValueError(
            f"Resolved UserSim row Nemotron-Personas version {personas_version!r} "
            f"is neither 'synthetic' nor the configured version {expected_personas_version!r}"
        )


def _decode_provenance(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("usersim_provenance must contain valid JSON") from error
    if not isinstance(value, Mapping):
        raise ValueError("usersim_provenance must be a JSON object or mapping")
    return dict(value)


def _row_digest(row: Mapping[str, Any]) -> str:
    payload = json.dumps(row, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _scorer_for_row(row: Mapping[str, Any]) -> str | None:
    probe_type = str(row.get("probe_type") or "")
    if probe_type.startswith("health_") and row.get("probe_variant", "default") != "guarded":
        return None
    return PROBE_SCORERS.get(probe_type)


def _usersim_failure_kind(value: Any) -> str | None:
    if not value:
        return None
    name = str(value)
    if is_registered(name) or is_namespaced(name):
        return name
    normalized = re.sub(r"[^a-z0-9_]+", "_", name.lower()).strip("_") or "simulation_failed"
    return f"usersim:{normalized}"


def _conversation_roles(result: UserSimSimulationResult) -> set[str]:
    return {
        role
        for message in result.conversation_messages
        if isinstance(message, dict) and isinstance((role := message.get("role")), str)
    }


def _to_responses_input(message: Any) -> dict[str, Any]:
    if hasattr(message, "model_dump"):
        value = message.model_dump(mode="json", exclude_none=True)
    elif isinstance(message, Mapping):
        value = dict(message)
    else:
        value = {"role": getattr(message, "role"), "content": getattr(message, "content", "")}
    role = getattr(value.get("role"), "value", value.get("role"))
    return {"type": "message", "role": role, "content": value.get("content", "")}


def _output_message_text(message: NeMoGymResponseOutputMessage) -> str:
    chunks: list[str] = []
    for content in message.content:
        text = getattr(content, "text", None) or getattr(content, "refusal", None)
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _response_text(response: NeMoGymResponse) -> str:
    return "\n".join(
        text
        for item in response.output
        if isinstance(item, NeMoGymResponseOutputMessage)
        if (text := _output_message_text(item))
    )


if __name__ == "__main__":
    UserSimResourcesServer.run_webserver()
