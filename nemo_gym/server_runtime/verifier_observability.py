# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Observe whole verify executions independently of their captured model calls."""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Awaitable, Callable, Mapping
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
from nemo_gym.trajectory.trajectory_record import NG_VERIFIER_INVOCATIONS_KEY, ModelCallRef, VerifierInvocation


logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")


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
            try:
                result = await verify_fn(*args, **kwargs)
            except Exception as error:
                invocation.status = "incomplete" if isinstance(error, TimeoutError) else "failed"
                invocation.failure_kind = TRANSPORT_TIMEOUT if isinstance(error, TimeoutError) else VERIFIER_ERROR
                if isinstance(error, HTTPException):
                    result = JSONResponse(
                        {"detail": error.detail}, status_code=error.status_code, headers=error.headers
                    )
                else:
                    logger.exception("Verification failed.")
                    result = JSONResponse({"detail": "Internal Server Error"}, status_code=500)
            else:
                _finish(invocation, result)

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
