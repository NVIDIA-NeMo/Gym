# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA termination reasons shared by runtime and response projection."""

from enum import StrEnum
from types import FrameType

from nooa.errors import GenerationError
from nooa.strategies.codeact import CodeActStrategy
from nooa.strategies.predict import PredictStrategy

from responses_api_agents.nooa_agent.gym_llm import InvalidPolicyOutputError, PolicyCallBudgetExceeded


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


def _generation_error_frame(error: BaseException) -> FrameType | None:
    # GenerationError is also available to arbitrary agent code. Recognize the
    # pinned strategies by the code that raised it, not its type or message alone.
    if type(error) is not GenerationError or error.__traceback__ is None:
        return None
    traceback = error.__traceback__
    while traceback.tb_next is not None:
        traceback = traceback.tb_next
    return traceback.tb_frame


def is_policy_limit_stop(error: BaseException | None) -> bool:
    """Recognize policy stops through native wrappers without hiding outer failures."""
    seen: set[int] = set()
    while isinstance(error, Exception) and id(error) not in seen:
        seen.add(id(error))
        if getattr(error, "nooa_cleanup_errors", ()) or isinstance(
            getattr(error, "nooa_cleanup_error", None), Exception
        ):
            return False
        if isinstance(error, (PolicyCallBudgetExceeded, InvalidPolicyOutputError)):
            return True
        frame = _generation_error_frame(error)
        if frame is None or frame.f_code not in (
            CodeActStrategy._run_generation.__code__,
            PredictStrategy._execute_inner.__code__,
        ):
            return False
        error = error.__cause__
    return False
