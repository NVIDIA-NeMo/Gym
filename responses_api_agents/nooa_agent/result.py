# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Project NOOA outcomes into the response and evidence returned to Gym."""

import aiohttp
from openai.types.responses.response_error import ResponseError

from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.rollout_observability import AgentObservationBundle
from responses_api_agents.nooa_agent.observability import ensure_verifier_final_message, finalize_observation_gaps
from responses_api_agents.nooa_agent.runner import NOOARunResult


def set_response_lifecycle(response: NeMoGymResponse, reason: str | None, error: str | None) -> NeMoGymResponse:
    """Keep policy termination visible without assigning an episode reward."""
    if reason is None:
        return response.model_copy(update={"status": "completed", "error": None})
    status = "incomplete" if reason in {"timeout", "cancelled", "agent_run_timeout", "timeout_exceeded"} else "failed"
    return response.model_copy(
        update={
            "status": status,
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


def is_transient_infrastructure_error(error: BaseException) -> bool:
    """Recognize retryable transport failures through NOOA exception wrapping."""
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        if isinstance(
            error, (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, TimeoutError, ConnectionError)
        ):
            return True
        if isinstance(error, aiohttp.ClientResponseError):
            return error.status in {408, 425, 429} or error.status >= 500
        cause = error.__cause__ or error.__context__
        if cause is None:
            break
        error = cause
    return False
