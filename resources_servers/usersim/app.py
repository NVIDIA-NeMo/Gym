# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate prepared UserSim rows and verify Environment-owned native results."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
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
from nemo_gym.global_config import get_first_server_config_dict
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
)
from nemo_gym.server_utils import SESSION_ID_KEY, get_response_json, raise_for_status
from resources_servers.usersim.episode_contracts import (
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
    """Configure prepared-row validation and native trajectory verification."""

    usersim_revision: str = Field(
        "a5f676bf6dc5a73914c8a0860f97c10dd2c214ee",  # pragma: allowlist secret
        pattern=r"^[0-9a-f]{40}$",
    )
    probe_scorer_model: ModelServerRef | None = None
    model_call_timeout_seconds: float = Field(300.0, gt=0)


class SeededUserSimEpisode(BaseModel):
    """Immutable Resources snapshot for one Environment-owned episode."""

    model_config = ConfigDict(extra="forbid")

    resources_session_id: str
    episode_id: EpisodeId
    task_id: TaskId
    resolved_row: dict[str, Any]
    resolved_row_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


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
        response_format = kwargs.get("response_format")
        if response_format is not None:
            json_schema = response_format.get("json_schema")
            if response_format.get("type") != "json_schema" or not isinstance(json_schema, Mapping):
                raise NotImplementedError(f"Unsupported response format: {response_format!r}")
            strict = json_schema.get("strict", True)
            values["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": json_schema["name"],
                    "schema": json_schema["schema"],
                    "strict": strict,
                }
            }
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

    config: UserSimResourcesServerConfig
    session_id_to_seed: dict[str, SeededUserSimEpisode] = {}

    async def seed_session(
        self,
        request: Request,
        body: ResourcesSeedSessionRequest,
    ) -> UserSimSeedResponse:
        try:
            task = UserSimTaskInput.model_validate(body.task_data)
            _validate_resolved_row(task.resolved_row, expected_revision=self.config.usersim_revision)
        except (ValidationError, ValueError) as error:
            detail = error.errors() if isinstance(error, ValidationError) else str(error)
            raise HTTPException(status_code=422, detail=detail) from error
        session_id = request.session[SESSION_ID_KEY]
        if session_id in self.session_id_to_seed:
            raise HTTPException(status_code=409, detail="Resources session is already seeded")
        resolved_row = task.resolved_row
        self.session_id_to_seed[session_id] = SeededUserSimEpisode(
            resources_session_id=body.resources_session_id,
            episode_id=body.episode_id,
            task_id=body.task_id,
            resolved_row=resolved_row,
            resolved_row_sha256=_row_digest(resolved_row),
        )
        return UserSimSeedResponse(
            resources_session_id=body.resources_session_id,
            sandbox_access=None,
            resolved_row=resolved_row,
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

        native_result = verification_input.usersim_result
        result_row = native_result.model_dump(mode="python")
        trajectory_id = seeded.resolved_row.get("trajectory_id")
        if trajectory_id and result_row.get("trajectory_id") != trajectory_id:
            raise HTTPException(status_code=409, detail="Native result trajectory_id does not match the seeded task")

        try:
            assistant_eval, normalized_scores, assistant_quality = await self._evaluate(
                seeded.resolved_row,
                native_result,
            )
        except Exception as error:
            logger.exception("Native UserSim trajectory evaluator failed")
            assistant_eval = {
                "axes": {},
                "scorers": {},
                "error": f"{type(error).__name__}: {error}",
            }
            normalized_scores = {}
            assistant_quality = None

        probe_type = str(seeded.resolved_row["probe_type"])
        scorer_name = _scorer_for_row(seeded.resolved_row)
        native_scores = assistant_eval.get("scorers", {}).get(scorer_name) if scorer_name else None
        scorer_error = native_scores.get("error") if isinstance(native_scores, Mapping) else None
        native_scorer_pass = scorer_name is None or (
            isinstance(native_scores, Mapping) and native_scores.get("status_proposal") is True and not scorer_error
        )
        outcome = native_result.simulation_outcome
        native_failure = outcome.get("status") == "failed"
        failure_attribution = outcome.get("failure_attribution")
        evaluator_error = assistant_eval.get("error")
        mask_sample = (native_failure and failure_attribution != "assistant_model") or bool(
            scorer_error or evaluator_error
        )
        failure_kind = outcome.get("failure_class") if native_failure else None
        failure_reason = outcome.get("failure_detail") if native_failure else None
        if scorer_error:
            failure_kind = "probe_scorer_error"
            failure_reason = str(scorer_error)
        elif evaluator_error:
            failure_kind = "trajectory_evaluator_error"
            failure_reason = str(evaluator_error)

        participants_completed = {"user", "assistant"} <= _conversation_roles(native_result)
        scenario_completed = native_result.conversation_status and participants_completed and native_scorer_pass
        reward = assistant_quality if scenario_completed and assistant_quality is not None else 0.0
        return UserSimVerification(
            reward=reward,
            mask_sample=mask_sample,
            failure_kind=failure_kind,
            failure_reason=failure_reason,
            reward_components={
                "participants_completed": float(participants_completed),
                "native_conversation_status": float(native_result.conversation_status),
                "native_scorer_applied": float(scorer_name is not None),
                "native_scorer_pass": float(native_scorer_pass),
                "trajectory_evaluator_applied": float(not assistant_eval.get("skipped", False)),
                "assistant_quality": assistant_quality or 0.0,
                **{f"quality.{axis}": score for axis, score in normalized_scores.items()},
            },
            scenario_completed=scenario_completed,
            native_usersim_result=native_result,
            verifier_data={
                "invocations": [invocation.model_dump(mode="json") for invocation in verification_input.invocations],
                "episode_interaction_protocol": verification_input.episode_interaction_protocol,
                "resolved_row_sha256": seeded.resolved_row_sha256,
                "probe_type": probe_type,
                "native_scorer_name": scorer_name,
                "native_scores": native_scores,
                "assistant_eval": assistant_eval,
                "normalized_axis_scores": normalized_scores,
                "scenario_completed": scenario_completed,
            },
        )

    async def _evaluate(
        self,
        resolved_row: dict[str, Any],
        native_result: UserSimSimulationResult,
    ) -> tuple[dict[str, Any], dict[str, float], float | None]:
        from usersim.engine.evaluator.config import TrajectoryEvaluatorConfig
        from usersim.taxonomy.eval_cell import normalize_axis_score, score_from_eval_cell

        if self.config.probe_scorer_model is None:
            return (
                {
                    "axes": {},
                    "scorers": {},
                    "skipped": True,
                    "skipped_reason": "missing_probe_scorer_model",
                },
                {},
                None,
            )
        scorer_name = _scorer_for_row(resolved_row)
        config = TrajectoryEvaluatorConfig(
            name="assistant_eval",
            judges=[{"alias": "judge_model"}],
            scorers=[scorer_name] if scorer_name else [],
            skip_if_existing=False,
        )
        model = _ResourcesModelFacade(self, self.config.probe_scorer_model)
        evaluator = _create_evaluator(config, {"judge_model": model})
        trajectory = {
            **resolved_row,
            **native_result.model_dump(mode="python"),
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
        seeded = self._seeded_episode(request)
        if body.resources_session_id != seeded.resources_session_id or body.episode_id != seeded.episode_id:
            raise HTTPException(status_code=409, detail="Resources session does not match the active episode")
        del self.session_id_to_seed[session_id]
        return ResourcesCloseSessionResponse(resources_session_id=body.resources_session_id)

    def _seeded_episode(self, request: Request) -> SeededUserSimEpisode:
        session_id = request.session[SESSION_ID_KEY]
        if session_id not in self.session_id_to_seed:
            raise RuntimeError("No active NeMo UserSim task. Call /seed_session first.")
        return self.session_id_to_seed[session_id]


def _validate_resolved_row(row: Mapping[str, Any], *, expected_revision: str) -> None:
    required = ("persona", "probe_type", "conversation_language", "trajectory_id", "usersim_config")
    missing = [name for name in required if not row.get(name)]
    if missing:
        raise ValueError(f"Resolved UserSim row is missing required fields: {missing}")
    provenance = _decode_provenance(row.get("usersim_provenance"))
    if provenance.get("code_sha") != expected_revision:
        raise ValueError(
            f"Resolved UserSim row revision {provenance.get('code_sha')!r} does not match {expected_revision!r}"
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
