# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project NOOA outcomes into the response and evidence returned to Gym."""

import aiohttp
from openai.types.responses.response_error import ResponseError

from nemo_gym.core import NeMoGymResponse
from nemo_gym.trajectory import AgentObservationBundle
from responses_api_agents.nooa_agent.observability import ensure_verifier_final_message, finalize_observation_gaps
from responses_api_agents.nooa_agent.runner import NOOARunResult
from responses_api_agents.nooa_agent.termination import NOOATerminationReason, cleanup_errors


def set_response_lifecycle(
    response: NeMoGymResponse, reason: NOOATerminationReason | None, error: str | None
) -> NeMoGymResponse:
    """Keep policy termination visible without assigning an episode reward."""
    if reason is None:
        return response.model_copy(update={"status": "completed", "error": None})
    if reason in {
        NOOATerminationReason.POLICY_BUDGET_EXCEEDED,
        NOOATerminationReason.INVALID_POLICY_OUTPUT,
        NOOATerminationReason.CODE_RETRIES_EXHAUSTED,
        NOOATerminationReason.RESULT_VALIDATION_EXHAUSTED,
        NOOATerminationReason.MAX_ITERATIONS_EXCEEDED,
        NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED,
        NOOATerminationReason.LOOP_DETECTED,
        # The worker's execution budget ran out: the rollout ends where it stands and is graded as such,
        # like any other policy termination, so the benchmark scores the miss instead of retrying it.
        NOOATerminationReason.TIMEOUT_EXCEEDED,
    }:
        return response.model_copy(update={"status": "incomplete", "error": None})
    if reason is NOOATerminationReason.CANCELLED:
        return response.model_copy(update={"status": "cancelled", "error": None})
    return response.model_copy(
        update={
            "status": "failed",
            "error": ResponseError(
                # Responses error codes are a closed OpenAI enum. Keep the
                # NOOA-specific reason in the message and rollout metadata.
                code="server_error",
                message=error or f"NOOA execution terminated with {reason}.",
            ),
        }
    )


def finalize_run_result(run_result: NOOARunResult) -> tuple[NeMoGymResponse, AgentObservationBundle]:
    """Preserve the final answer and capture gaps for a completed or partial run."""
    verify_response, verify_gaps = ensure_verifier_final_message(run_result.episode.response, run_result.return_value)
    observations = finalize_observation_gaps(
        run_result.episode.observations,
        extra_gaps=verify_gaps,
        termination_reason=run_result.termination_reason,
        termination_error=run_result.termination_error,
    )
    observations.gaps = list({gap.model_dump_json(): gap for gap in observations.gaps}.values())
    return set_response_lifecycle(
        verify_response, run_result.termination_reason, run_result.termination_error
    ), observations


# litellm is not a Gym dependency, so recognize its transient model errors by class, not by import.
_LITELLM_TRANSIENT = {"Timeout", "ServiceUnavailableError"}


def _is_litellm_transient(error: BaseException) -> bool:
    return any(
        cls.__module__.split(".")[0] == "litellm" and cls.__name__ in _LITELLM_TRANSIENT for cls in type(error).__mro__
    )


def is_transient_infrastructure_error(error: BaseException) -> bool:
    """Recognize retryable transport failures in task and retained cleanup errors."""
    seen: set[int] = set()
    pending = [error]
    while pending:
        error = pending.pop()
        if id(error) in seen:
            continue
        seen.add(id(error))
        pending.extend(cleanup_errors(error))
        if isinstance(
            error, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, TimeoutError, ConnectionError)
        ):
            return True
        if _is_litellm_transient(error):
            return True
        if isinstance(error, aiohttp.ClientResponseError):
            if error.status in {408, 425, 429} or error.status >= 500:
                return True
            continue
        cause = error.__cause__ or error.__context__
        if cause is not None:
            pending.append(cause)
    return False
