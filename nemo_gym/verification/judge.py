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

import asyncio
import random
import re
import uuid
from typing import Any, Awaitable, Type, TypeVar

from aiohttp import ClientConnectionError, ClientResponseError, ClientTimeout
from pydantic import BaseModel

from nemo_gym.client.http_client import get_response_json, raise_for_status
from nemo_gym.client.server_client import ServerClient
from nemo_gym.config.config_types import ModelServerRef
from nemo_gym.core.episode_context import (
    VERIFIER_INVOCATION_HEADER,
    current_verifier_invocation_id,
    note_verifier_model_request,
)
from nemo_gym.core.failure_transport import observe_failure
from nemo_gym.core.typed_exceptions import NgFailureEvent, sdk_error_details
from nemo_gym.trajectory.trajectory_record import ModelCallRef
from nemo_gym.utils.exception_inspection import bounded_text, error_to_http_status
from nemo_gym.verification.judge_contracts import JUDGE_FAILURES as JUDGE_FAILURES
from nemo_gym.verification.judge_contracts import JUDGE_PROVIDER_STOP_KINDS as JUDGE_PROVIDER_STOP_KINDS
from nemo_gym.verification.judge_contracts import JudgeError as JudgeError
from nemo_gym.verification.judge_contracts import JudgeGrading as JudgeGrading
from nemo_gym.verification.judge_contracts import _judge_grading as _judge_grading
from nemo_gym.verification.judge_contracts import is_judge_failure as is_judge_failure
from nemo_gym.verification.judge_contracts import is_judge_provider_stop as is_judge_provider_stop
from nemo_gym.verification.judge_contracts import judge_call_failed as judge_call_failed
from nemo_gym.verification.judge_contracts import judge_failsafe as judge_failsafe
from nemo_gym.verification.judge_contracts import judge_provider_stopped as judge_provider_stopped


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


def _judge_error(
    error: Exception, *, operation_id: str | None = None, attempt_id: str | None = None
) -> JudgeError | NgFailureEvent:
    """The judge failure for a failed judge call; the caller raises it ``from error``.

    A call that failed over HTTP, could not connect or timed out is a typed ``judge_failed``. The kind is not
    derived from the error's status: a policy that needs the status reads it on the cause. ``call_judge`` names
    each of its attempts with ``operation_id`` and ``attempt_id`` and records the reply's status.
    """
    message = f"{type(error).__name__}: {error}"
    content = getattr(error, "response_content", None)
    if content:
        # The reply's own body names the upstream cause (a rejected credential, say), bounded like the reason.
        text = content.decode(errors="replace") if isinstance(content, bytes) else str(content)
        message += f"; reply: {bounded_text(text, 500)}"
    return judge_call_failed(
        message,
        operation="judge.request",
        sdk_error=None if isinstance(error, NgFailureEvent) else sdk_error_details(error),
        operation_id=operation_id,
        attempt_id=attempt_id,
        http_status=_judge_http_status(error) if operation_id is not None else None,
    )


JUDGE_CALL_MAX_ATTEMPTS = 3
"""Attempts of one ``call_judge`` call (the first one included) before its failure masks the row."""
JUDGE_RETRY_BASE_WAIT_S = 2.0
JUDGE_RETRY_MAX_WAIT_S = 30.0
_RETRYABLE_JUDGE_STATUSES = frozenset({408, 429})


def _judge_http_status(error: BaseException) -> int | None:
    """The status of the judge reply that failed, from this hop's HTTP error; ``None`` when no reply came."""
    if isinstance(error, NgFailureEvent):
        error = error.http_error or error
    return error.status if isinstance(error, ClientResponseError) else error_to_http_status(error)


def _is_retryable_judge_error(error: BaseException) -> bool:
    """Whether another attempt can succeed: a 408, 429 or 5xx reply, a timeout or a lost connection.

    Other 4xx replies (auth, a bad request) and replies that do not parse fail the same way again.
    """
    status = _judge_http_status(error)
    if status is not None:
        return status in _RETRYABLE_JUDGE_STATUSES or status >= 500
    return isinstance(error, (TimeoutError, ClientConnectionError, ConnectionError))


def _judge_retry_wait_s(attempt: int) -> float:
    """Full-jitter exponential backoff after failed attempt ``attempt`` (1-based)."""
    return random.uniform(0, min(JUDGE_RETRY_MAX_WAIT_S, JUDGE_RETRY_BASE_WAIT_S * 2 ** (attempt - 1)))


async def reraise_judge_errors(coro: Awaitable[Any]) -> Any:
    """Await a judge call; preserve judge failures and wrap other errors as ``judge_failed`` events.

    Custom handlers catch ``JUDGE_FAILURES`` and select with ``is_judge_failure``;
    an ``except JudgeError`` handler alone catches only standalone parser failures.
    """
    return await _reraise_judge_errors(coro)


async def _reraise_judge_errors(
    coro: Awaitable[Any], *, operation_id: str | None = None, attempt_id: str | None = None
) -> Any:
    try:
        return await coro
    except Exception as e:
        failure = e if is_judge_failure(e) else _judge_error(e, operation_id=operation_id, attempt_id=attempt_id)
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
    max_attempts: int = JUDGE_CALL_MAX_ATTEMPTS,
) -> ResponseT:
    """POST to a judge model server and parse the reply; report failure as a ``judge_failed`` event.

    ``url_path`` is the judge's API surface (``/v1/responses`` or
    ``/v1/chat/completions``), paired with the matching ``response_model``. The
    status check comes before parsing so an auth or server error reports the HTTP
    failure itself, not a validation error against the error body.
    The enclosing verify execution supplies the invocation identity; all its judge
    calls and retries share that identity. Standalone calls do not invent a verifier execution.

    A 408, 429 or 5xx reply, a timeout or a lost connection is tried again, up to
    ``max_attempts`` attempts (default ``JUDGE_CALL_MAX_ATTEMPTS``; 1 for a caller that owns its retry loop)
    with full-jitter exponential backoff, before the
    failure masks the row; other 4xx replies and unparseable replies fail at once. Each failed
    attempt is observed as its own ``judge_failed`` event (one ``operation_id`` per call, the
    attempt in ``attempt_id``, the reply's ``http_status``), so a row a retry scored and a row a
    final failure masked both keep every attempt.
    """

    async def _call() -> ResponseT:
        invocation_id = current_verifier_invocation_id()
        note_verifier_model_request(server_name)
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

    operation_id = uuid.uuid4().hex
    for attempt in range(1, max_attempts):
        try:
            return await _reraise_judge_errors(
                _call(), operation_id=operation_id, attempt_id=f"{operation_id}-{attempt}"
            )
        except NgFailureEvent as failure:
            if not _is_retryable_judge_error(failure.__cause__ or failure):
                raise
        await asyncio.sleep(_judge_retry_wait_s(attempt))
    return await _reraise_judge_errors(_call(), operation_id=operation_id, attempt_id=f"{operation_id}-{max_attempts}")
