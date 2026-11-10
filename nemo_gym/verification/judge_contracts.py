# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Judge failure handling and grading context shared by server boundaries."""

import functools
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from nemo_gym.core.failure_kinds import JUDGE_FAILED, PROVIDER_CREDENTIALS_REJECTED, PROVIDER_QUOTA_EXHAUSTED
from nemo_gym.core.failure_records import FailureBoundary, SdkErrorDetails
from nemo_gym.core.failure_transport import FailureReportRow, report_for_current_request
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.trajectory.trajectory_record import ModelCallRef
from nemo_gym.utils.exception_inspection import bounded_text


@dataclass
class JudgeGrading:
    """Judge facts of one reviewed LLM-judge grading execution; set only by ``observe_llm_judge_grading``."""

    dependencies: list[ModelCallRef] = field(default_factory=list)
    unidentified_calls: int = 0
    parsed_dependencies: set[int] = field(default_factory=set)


_judge_grading: ContextVar[JudgeGrading | None] = ContextVar("judge_grading", default=None)


class JudgeError(Exception):
    """A parser-declared judge failure; judge_failsafe routes the row to the failures sidecar."""


def judge_call_failed(
    message: str,
    *,
    operation: str,
    sdk_error: SdkErrorDetails | None = None,
    operation_id: str | None = None,
    attempt_id: str | None = None,
    http_status: int | None = None,
) -> NgFailureEvent:
    """The resources server boundary's ``judge_failed``; what was observed stays on the explicit cause."""
    return NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind=JUDGE_FAILED,
        failure_reason=bounded_text(message, 2000),
        operation=operation,
        operation_id=operation_id,
        attempt_id=attempt_id,
        http_status=http_status,
        stage="verification",
        foreign_error=sdk_error,
        message_args=(message,),
    )


JUDGE_PROVIDER_STOP_KINDS = (PROVIDER_QUOTA_EXHAUSTED, PROVIDER_CREDENTIALS_REJECTED)
"""The kinds of a judge provider that cannot grade until an operator acts: its quota resets or its key is fixed."""


def judge_provider_stopped(message: str, *, failure_kind: str, operation: str) -> NgFailureEvent:
    """The resources server boundary's stop of a judge provider that refuses to grade (quota or credentials).

    ``judge_failsafe`` routes it like a judge failure, so the row keeps its saved answer and recovery grades it again
    from that answer; the kind tells recovery and the run that no attempt can succeed until the operator acts.
    """
    if failure_kind not in JUDGE_PROVIDER_STOP_KINDS:
        raise ValueError(f"not a judge provider stop kind: {failure_kind!r}")
    return NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind=failure_kind,
        failure_reason=bounded_text(message, 2000),
        operation=operation,
        stage="verification",
        message_args=(message,),
    )


def is_judge_provider_stop(error: BaseException) -> bool:
    """Whether ``error`` is a ``judge_provider_stopped`` event."""
    return (
        isinstance(error, NgFailureEvent)
        and error.boundary == FailureBoundary.RESOURCES_SERVER
        and error.failure_kind in JUDGE_PROVIDER_STOP_KINDS
    )


def is_judge_failure(error: BaseException) -> bool:
    """Whether ``error`` is a judge failure: the ``JudgeError`` a parser raises, or a ``judge_failed`` event."""
    return isinstance(error, JudgeError) or (
        isinstance(error, NgFailureEvent)
        and error.boundary == FailureBoundary.RESOURCES_SERVER
        and error.failure_kind == JUDGE_FAILED
    )


JUDGE_FAILURES = (JudgeError, NgFailureEvent)
"""Types to catch around a judge call; narrow with ``is_judge_failure`` where other Gym failures can occur."""


def judge_failsafe(verify_fn: Callable) -> Callable:
    """Wrap verify() so a judge failure returns a sidecar-routed row (reward 0.0, the
    routing keys, the request's ``response`` carried) instead of propagating.
    functools.wraps keeps verify's signature so FastAPI injects the same params
    (``body``, and ``request`` for servers that take it); ``*args, **kwargs`` pass
    them straight through."""

    @functools.wraps(verify_fn)
    async def wrapper(*args, **kwargs):
        try:
            return await verify_fn(*args, **kwargs)
        except JUDGE_FAILURES as e:
            if not (is_judge_failure(e) or is_judge_provider_stop(e)):
                raise
            kind = e.failure_kind if is_judge_provider_stop(e) else JUDGE_FAILED
            body = kwargs.get("body") or next(
                (a for a in (*kwargs.values(), *args) if hasattr(a, "model_dump") and hasattr(a, "response")),
                None,
            )
            if body is None:  # verify always has a request body; guard against an opaque 500
                raise RuntimeError("judge_failsafe: could not locate the verify request body") from e
            error = str(e)[:2000]
            data = body.model_dump()
            instance_config = data.get("instance_config")
            data |= {
                "reward": 0.0,
                "mask_sample": True,
                # The judge owns this boundary; more specific diagnostics stay on the explicit cause.
                "failure_kind": kind,
                "failure_reason": error,
                # Keep older NeMo RL consumers working during top-level field migration.
                "instance_config": {
                    **(instance_config if isinstance(instance_config, dict) else {}),
                    "mask_sample": True,
                },
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": error,
            }
            # Legacy/parser JudgeErrors also need a judge-owned event, rather than borrowing an inner kind.
            failure = e
            if not (isinstance(e, NgFailureEvent) and e.failure_kind == kind):
                failure = judge_call_failed(error, operation="judge.verify")
                failure.__cause__ = e
            data |= FailureReportRow.fields(report_for_current_request(failure))
            return JSONResponse(content=jsonable_encoder(data))

    return wrapper
