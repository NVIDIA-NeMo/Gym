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
"""Shared LLM-as-judge failure abstraction.

A failed judge call is a distinct outcome, not a wrong answer. Resources servers
issue judge calls through ``call_judge``, or wrap a bespoke call in
``reraise_judge_errors``; ``judge_failsafe`` wraps every verify endpoint so a
judge failure becomes a row tagged ``_ng_failure_class="judge_failed"``, which
rollout_collection routes to ``<output>_failures.jsonl`` — excluded from the
aggregate metrics and from the file-based re-aggregation (``gym eval
aggregate``), and retained on resume without regenerating its answer. Judge-only
recovery requires a supported reverification path; manifest-backed runs currently
reject reverification.

Boundary: a failed *call* (transport/timeout/auth/HTTP) → ``judge_failed`` event → sidecar; a
*received-but-unparseable* response is a legitimate wrong answer (let the parser
score it, don't raise), unless the benchmark explicitly treats an invalid judge
verdict as an unusable grade. FrontierScience Research uses that exception for
missing/invalid final numeric rubric verdicts: retry through the judge-failure
sidecar rather than assign zero points to the model answer; ordinary resume does
not regenerate that answer. Completion checks
must still reject a partial evaluation. Empty output is a per-benchmark call,
so servers differ.

Wrap the call, not the scoring around it: ``reraise_judge_errors`` relabels every
exception it sees, so covering a whole ``verify()`` would misfile ordinary bugs
(a KeyError in prompt assembly, say) as judge failures.
"""

import functools
import re
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Type, TypeVar

from aiohttp import ClientTimeout
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nemo_gym.client.http_client import get_response_json, raise_for_status
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.episode import VERIFIER_INVOCATION_HEADER, current_verifier_invocation_id
from nemo_gym.core.failure_kinds import (
    JUDGE_FAILED,
)
from nemo_gym.exception_inspection import bounded_text
from nemo_gym.failure_transport import FailureReportRow, observe_failure, report_for_current_request
from nemo_gym.server_runtime.episode_types import FailureBoundary, SdkErrorDetails
from nemo_gym.trajectory.trajectory_record import ModelCallRef
from nemo_gym.typed_exceptions import NgFailureEvent, sdk_error_details


@dataclass
class JudgeGrading:
    """Judge facts of one reviewed LLM-judge grading execution; set only by ``observe_llm_judge_grading``."""

    dependencies: list[ModelCallRef] = field(default_factory=list)
    unidentified_calls: int = 0
    parsed_dependencies: set[int] = field(default_factory=set)


_judge_grading: ContextVar[JudgeGrading | None] = ContextVar("judge_grading", default=None)


def record_parsed_judge_verdict() -> None:
    """Mark that the reviewed call site took its explicitly parsed-verdict branch for the active judge grading.

    Call it only where the grade comes from a valid judge verdict; fallback, no-verdict and caught-error branches
    must not call it, so their grading stays unknown.
    """
    grading = _judge_grading.get()
    if grading is not None and grading.dependencies:
        grading.parsed_dependencies.add(len(grading.dependencies) - 1)


ResponseT = TypeVar("ResponseT", bound=BaseModel)


def normalize_math_judge_verdict(text: str) -> str:
    """Canonicalize whitespace inside A=B/A!=B markers without changing other text."""
    return re.sub(
        r"\[\[\s*A\s*(!\s*)?=\s*B\s*\]\]",
        lambda match: "[[A!=B]]" if match.group(1) else "[[A=B]]",
        text,
    )


class JudgeError(Exception):
    """A parser-declared judge failure; judge_failsafe routes the row to the failures sidecar."""


def judge_call_failed(message: str, *, operation: str, sdk_error: SdkErrorDetails | None = None) -> NgFailureEvent:
    """The resources server boundary's ``judge_failed``; what was observed stays on the explicit cause."""
    return NgFailureEvent(
        boundary=FailureBoundary.RESOURCES_SERVER,
        failure_kind=JUDGE_FAILED,
        failure_reason=bounded_text(message, 2000),
        operation=operation,
        stage="verification",
        foreign_error=sdk_error,
        message_args=(message,),
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


def _judge_error(error: Exception) -> JudgeError | NgFailureEvent:
    """The judge failure for a failed judge call; the caller raises it ``from error``.

    A call that failed over HTTP, could not connect or timed out is a typed ``judge_failed``. The kind is not
    derived from the error's status: a policy that needs the status reads it on the cause.
    """
    message = f"{type(error).__name__}: {error}"
    return judge_call_failed(
        message,
        operation="judge.request",
        sdk_error=None if isinstance(error, NgFailureEvent) else sdk_error_details(error),
    )


async def reraise_judge_errors(coro: Awaitable[Any]) -> Any:
    """Await a judge call; preserve judge failures and wrap other errors as ``judge_failed`` events.

    Custom handlers catch ``JUDGE_FAILURES`` and select with ``is_judge_failure``;
    an ``except JudgeError`` handler alone catches only standalone parser failures.
    """
    try:
        return await coro
    except Exception as e:
        failure = e if is_judge_failure(e) else _judge_error(e)
        if failure is not e:
            failure.__cause__ = e  # set before recording, so the observation carries the cause chain
        # Recorded before the caller decides: a retry that then succeeds keeps it as an observation.
        if isinstance(failure, NgFailureEvent):
            observe_failure(failure)
        if failure is e:
            raise
        raise failure from e


async def call_judge(
    server_client: ServerClient,
    *,
    server_name: str,
    url_path: str,
    json: Any,
    response_model: Type[ResponseT],
    timeout: ClientTimeout | None = None,
) -> ResponseT:
    """POST to a judge model server and parse the reply; report failure as a ``judge_failed`` event.

    ``url_path`` is the judge's API surface (``/v1/responses`` or
    ``/v1/chat/completions``), paired with the matching ``response_model``. The
    status check comes before parsing so an auth or server error reports the HTTP
    failure itself, not a validation error against the error body.
    The enclosing verify execution supplies the invocation identity; all its judge
    calls and retries share that identity. Standalone calls do not invent a verifier execution.
    """

    async def _call() -> ResponseT:
        invocation_id = current_verifier_invocation_id()
        http_response = await server_client.post(
            server_name=server_name,
            url_path=url_path,
            json=json,
            **({"timeout": timeout} if timeout is not None else {}),
            **({"headers": {VERIFIER_INVOCATION_HEADER: invocation_id}} if invocation_id else {}),
        )
        await raise_for_status(http_response)
        parsed = response_model.model_validate(await get_response_json(http_response))
        grading = _judge_grading.get()
        if grading is not None:
            # Only a received, parsed reply is a dependency; failed calls stay failure observations.
            response_id = getattr(parsed, "id", None)
            if isinstance(response_id, str) and response_id:
                model_ref = ModelServerRef(type="responses_api_models", name=server_name)
                grading.dependencies.append(ModelCallRef(model_ref=model_ref, response_id=response_id))
            else:
                grading.unidentified_calls += 1
        return parsed

    return await reraise_judge_errors(_call())


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
            if not is_judge_failure(e):
                raise
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
                "failure_kind": JUDGE_FAILED,
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
            if not (isinstance(e, NgFailureEvent) and e.failure_kind == JUDGE_FAILED):
                failure = judge_call_failed(error, operation="judge.verify")
                failure.__cause__ = e
            data |= FailureReportRow.fields(report_for_current_request(failure))
            return JSONResponse(content=jsonable_encoder(data))

    return wrapper
