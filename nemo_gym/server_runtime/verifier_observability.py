# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe whole verify executions independently of their captured model calls."""

from __future__ import annotations

import functools
import inspect
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar
from typing import ParamSpec, TypeVar
from uuid import uuid4

from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, Response

from nemo_gym.core.episode import (
    current_rollout_id,
    verifier_invocation_context,
)
from nemo_gym.core.failure_kinds import TRANSPORT_TIMEOUT, VERIFIER_ERROR
from nemo_gym.failure_transport import FailureHeader, observe_failure
from nemo_gym.server_runtime.episode_types import FailureBoundary
from nemo_gym.trajectory.trajectory_record import (
    NG_VERIFIER_INVOCATIONS_KEY,
    GradingOperationObservation,
    ModelCallRef,
    SandboxObservation,
    VerifierInvocation,
    grading_result_binding,
)
from nemo_gym.typed_exceptions import NgFailureEvent, failure_report_for
from nemo_gym.verification.judge import JudgeGrading, _judge_grading


logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


_active_verifier: ContextVar[VerifierInvocation | None] = ContextVar("active_verifier_grading", default=None)


def observe_verifier_sandbox(observation: SandboxObservation | Mapping[str, object]) -> None:
    """Bind owner-observed sandbox facts to the active verification.

    These facts do not certify remote liveness or dependency-free grading.
    """
    invocation = _active_verifier.get()
    if invocation is None:
        return
    try:
        observed = SandboxObservation.model_validate(observation)
    except Exception:
        logger.warning("Malformed sandbox lifecycle observation omitted")
        return
    if invocation.sandbox_observations is None:
        invocation.sandbox_observations = []
    invocation.sandbox_observations.append(observed)


def observe_programmatic_grading(calculate: Callable[[], T]) -> T:
    """Execute a reviewed synchronous scorer once and record its actual result in the active invocation.

    This is an explicit call-site contract for pure calculations, not a purity detector for arbitrary callbacks.
    Judge/SDK/sandbox paths must not use it to label a returned result as programmatic.
    """
    if inspect.iscoroutinefunction(calculate):
        raise TypeError("programmatic grading must be synchronous")
    invocation = _active_verifier.get()
    operation = GradingOperationObservation(operation_id=f"grade-{uuid4().hex}", dependencies=[])
    if invocation is not None:
        if invocation.grading_operations is None:
            invocation.grading_operations = []
        invocation.grading_operations.append(operation)
    try:
        result = calculate()
        if inspect.isawaitable(result):
            raise TypeError("programmatic grading returned an awaitable")
    except BaseException:
        operation.outcome = "failed"
        raise
    try:
        operation.result_binding_sha256 = grading_result_binding(result)
    except Exception:
        # Observation must not turn a successfully calculated grade into a verifier failure.
        operation.result_binding_sha256 = None
    operation.outcome = "completed" if operation.result_binding_sha256 is not None else "unknown"
    return result


async def observe_llm_judge_grading(calculate: Callable[[], Awaitable[T]]) -> T:
    """Execute a reviewed async LLM-judge grading once and record its actual result in the active invocation.

    The judge replies parsed by ``call_judge`` during ``calculate`` become the operation's dependencies. It completes
    only when every judge reply was identified, at least one was, the call site marked every reply with
    ``record_parsed_judge_verdict`` and the result binds; otherwise it stays unknown. Errors propagate unchanged.
    """
    invocation = _active_verifier.get()
    operation = GradingOperationObservation(operation_id=f"grade-{uuid4().hex}", kind="llm_judge", dependencies=[])
    if invocation is not None:
        if invocation.grading_operations is None:
            invocation.grading_operations = []
        invocation.grading_operations.append(operation)
    grading = JudgeGrading()
    token = _judge_grading.set(grading)
    try:
        result = await calculate()
    except BaseException:
        operation.outcome = "failed"
        raise
    finally:
        _judge_grading.reset(token)
        operation.dependencies = list(grading.dependencies)
    try:
        operation.result_binding_sha256 = grading_result_binding(result)
    except Exception:
        operation.result_binding_sha256 = None
    complete = (
        bool(grading.dependencies)
        and len(grading.parsed_dependencies) == len(grading.dependencies)
        and not grading.unidentified_calls
    )
    operation.outcome = "completed" if complete and operation.result_binding_sha256 is not None else "unknown"
    return result


def merge_verifier_call_observation(
    invocations: dict[str, VerifierInvocation],
    exchange: Mapping[str, object],
    *,
    model_call_id: str,
) -> None:
    """Link a captured attempt to its caller-declared verifier invocation, when supplied."""
    invocation_id = exchange.get("verifier_invocation_id")
    if invocation_id is None:
        return
    if not isinstance(invocation_id, str) or not invocation_id.strip():
        raise ValueError("verifier_invocation_id must be a nonblank string")
    reference = ModelCallRef(model_call_id=model_call_id)
    existing = invocations.get(invocation_id)
    if existing is None:
        invocations[invocation_id] = VerifierInvocation(invocation_id=invocation_id, model_calls=[reference])
    else:
        existing.model_calls.append(reference)


def _finish(invocation: VerifierInvocation, result: object) -> None:
    if isinstance(result, Response):
        if result.status_code >= 400:
            invocation.status = "failed"
            invocation.failure_kind = VERIFIER_ERROR
            return
        try:
            data = json.loads(result.body)
        except (ValueError, AttributeError):
            return
    else:
        data = result.model_dump() if isinstance(result, BaseModel) else result
    if not isinstance(data, Mapping):
        return
    failure_kind = data.get("failure_kind")
    invocation.failure_kind = failure_kind if isinstance(failure_kind, str) else None
    if data.get("_ng_failure_class"):
        invocation.status = "failed"
    elif data.get("evaluation_completed") is False:
        invocation.status = "incomplete"
    else:
        invocation.status = "completed"


def observe_verifier(verify_fn: Callable[P, Awaitable[T]], *, enabled: bool) -> Callable[P, Awaitable[T | Response]]:
    """Return verification evidence with its result, leaving persistence to collection.

    HTTP failures keep their status and carry evidence in the error body. Cancellation
    still propagates; a cancelled or lost response cannot deliver in-memory evidence.
    """
    if not enabled:
        return verify_fn

    @functools.wraps(verify_fn)
    async def wrapped(*args: P.args, **kwargs: P.kwargs) -> T | Response:
        if current_rollout_id() is None:
            return await verify_fn(*args, **kwargs)
        invocation = VerifierInvocation(invocation_id=f"verify-{uuid4().hex}")
        with verifier_invocation_context(invocation.invocation_id):
            token = _active_verifier.set(invocation)
            try:
                result = await verify_fn(*args, **kwargs)
            except Exception as error:
                invocation.status = "incomplete" if isinstance(error, TimeoutError) else "failed"
                invocation.failure_kind = TRANSPORT_TIMEOUT if isinstance(error, TimeoutError) else VERIFIER_ERROR
                # The reply below is an HTTP error, not a raised one, so record the failure for this request's
                # failure report; its operation_id joins it to the VerifierInvocation. Recorded by type only.
                failure = NgFailureEvent(
                    boundary=FailureBoundary.RESOURCES_SERVER,
                    failure_kind=invocation.failure_kind,
                    failure_reason=f"Verification failed: {type(error).__name__}",
                    operation="verify",
                    operation_id=invocation.invocation_id,
                    stage="verification",
                    http_status=error.status_code if isinstance(error, HTTPException) else 500,
                    exception_type=type(error).__name__,
                )
                failure.__cause__ = error
                observe_failure(failure)
                if isinstance(error, HTTPException):
                    result = JSONResponse(
                        {"detail": error.detail}, status_code=error.status_code, headers=error.headers
                    )
                else:
                    logger.exception("Verification failed.")
                    result = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
                # The returned reply replaces the raised error, so it carries the error's selected failure and causes.
                report = failure_report_for(failure)
                inner = FailureHeader.decode(result.headers)
                if inner is not None:
                    report = report.merged(inner) if report.failure_id is not None else inner.merged(report)
                result.headers.update(FailureHeader.encode(report))
            else:
                _finish(invocation, result)
            finally:
                _active_verifier.reset(token)

        evidence = [invocation.model_dump(mode="json")]
        if isinstance(result, Response):
            # Preserve the original status, headers, cookies and background cleanup.
            # Non-JSON responses cannot carry structured observations.
            try:
                data = json.loads(result.body)
            except (ValueError, AttributeError):
                return result
            if not isinstance(data, dict):
                return result
            data[NG_VERIFIER_INVOCATIONS_KEY] = evidence
            result.body = JSONResponse(jsonable_encoder(data)).body
            result.headers["content-length"] = str(len(result.body))
            return result
        if isinstance(result, BaseModel):
            return result.model_copy(update={NG_VERIFIER_INVOCATIONS_KEY: [invocation]})
        if isinstance(result, Mapping):
            return {**result, NG_VERIFIER_INVOCATIONS_KEY: evidence}
        return result

    return wrapped
