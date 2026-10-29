# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA termination reasons shared by runtime and response projection."""

from enum import StrEnum
from types import FrameType

from aiohttp import ClientResponseError
from nooa.errors import GenerationError, IterationLimitExceeded, LoopDetectedError
from nooa.strategies.codeact import CodeActSession, CodeActStrategy
from nooa.strategies.predict import PredictStrategy

from responses_api_agents.nooa_agent.gym_llm import (
    InvalidPolicyOutputError,
    OutputTokenLimitExceeded,
    PolicyCallBudgetExceeded,
)


class NOOATerminationReason(StrEnum):
    """Stable wire values describing why a NOOA rollout stopped."""

    POLICY_BUDGET_EXCEEDED = "policy_budget_exceeded"
    MAX_ITERATIONS_EXCEEDED = "max_iterations_exceeded"
    MAX_OUTPUT_TOKENS_EXCEEDED = "max_output_tokens_exceeded"
    LOOP_DETECTED = "loop_detected"
    INVALID_POLICY_OUTPUT = "invalid_policy_output"
    MODEL_REJECTED = "model_rejected"
    TIMEOUT_EXCEEDED = "timeout_exceeded"
    CANCELLED = "cancelled"
    INFRASTRUCTURE_ERROR = "infrastructure_error"


def cleanup_errors(error: BaseException) -> tuple[BaseException, ...]:
    """Return cleanup failures retained on the primary exception, including itself."""
    return tuple(item for item in getattr(error, "nooa_cleanup_errors", ()) if isinstance(item, BaseException))


def _raising_frame(error: BaseException) -> FrameType | None:
    if error.__traceback__ is None:
        return None
    traceback = error.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    return traceback.tb_frame


def _generation_error_frame(error: BaseException) -> FrameType | None:
    # GenerationError is also available to arbitrary agent code. Recognize the
    # pinned strategies by the code that raised it, not its type or message alone.
    return _raising_frame(error) if type(error) is GenerationError else None


def _is_codeact_iteration_exhaustion(error: BaseException) -> bool:
    # The pinned NOOA uses the same error type for iteration and retry exhaustion.
    frame = _generation_error_frame(error)
    if frame is None or frame.f_code is not CodeActStrategy._run_generation.__code__:
        return False
    session = frame.f_locals.get("session")
    return (
        isinstance(session, CodeActSession)
        and session.max_iterations is not None
        and session.iteration >= session.max_iterations
        and session.error_count < session.max_retries
        and str(error) == str(session.build_failure_error())
    )


def _is_codeact_loop_stop(error: BaseException) -> bool:
    # Only the pinned loop guard's model-driven stop is a gradable outcome.
    frame = _raising_frame(error) if type(error) is LoopDetectedError else None
    return frame is not None and frame.f_code in (
        CodeActStrategy._loop_guard_before.__code__,
        CodeActStrategy._loop_guard_after.__code__,
    )


def policy_termination(
    error: BaseException | None, *, model_error: BaseException | None = None
) -> tuple[NOOATerminationReason | None, str | None]:
    """Recognize native stops and model errors without hiding cleanup or outer failures."""
    seen: set[int] = set()
    while isinstance(error, Exception) and id(error) not in seen:
        seen.add(id(error))
        if cleanup_errors(error):
            break
        if error is model_error and isinstance(error, ClientResponseError) and 400 <= error.status <= 599:
            # Report this invocation's terminal model error for the environment's
            # outcome policy; unrelated HTTP and cleanup failures remain separate.
            return NOOATerminationReason.MODEL_REJECTED, str(error)
        if isinstance(error, OutputTokenLimitExceeded):
            return NOOATerminationReason.MAX_OUTPUT_TOKENS_EXCEEDED, str(error)
        if isinstance(error, PolicyCallBudgetExceeded):
            return NOOATerminationReason.POLICY_BUDGET_EXCEEDED, str(error)
        if isinstance(error, InvalidPolicyOutputError):
            return NOOATerminationReason.INVALID_POLICY_OUTPUT, str(error)
        if isinstance(error, IterationLimitExceeded):
            # NOOA names this stop (raised by CodeActSession.build_failure_error, outside the generation frame);
            # the frame inspection below serves older pins.
            return NOOATerminationReason.MAX_ITERATIONS_EXCEEDED, str(error)
        if _is_codeact_iteration_exhaustion(error):
            return NOOATerminationReason.MAX_ITERATIONS_EXCEEDED, str(error)
        if _is_codeact_loop_stop(error):
            return NOOATerminationReason.LOOP_DETECTED, str(error)
        frame = _generation_error_frame(error)
        if frame is None or frame.f_code not in (
            CodeActStrategy._run_generation.__code__,
            PredictStrategy._execute_inner.__code__,
        ):
            break
        error = error.__cause__
    return None, None
