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
JudgeError becomes a row tagged ``_ng_failure_class="judge_failed"``, which
rollout_collection routes to ``<output>_failures.jsonl``. In training the row is
masked and retried on resume, as before.

In evaluation the rollout is never re-run for a judge failure: that would give the
model another answer and score a different one. ``retry_judge_in_place`` asks the
judge again about the same stored answer (``NEMO_GYM_JUDGE_MAX_ATTEMPTS``, default
3, with backoff), and when every attempt fails
``failure_kinds.judge_failure_after_retries`` decides the row by its cause: the
model's final 0 when the model's answer caused it, and an unresolved infra row (left
out of the score, counted, never re-dispatched) when a service did. The judge's last
error, the cause and the attempt count are recorded either way.

Boundary: a failed *call* (transport/timeout/auth/HTTP) → JudgeError → sidecar. In
evaluation, so is a *received-but-unparseable* verdict: the judge, not the model,
failed to produce a usable answer, so it is retried in place like a failed call.
A server raises it with ``raise_unparseable_verdict``
from the branch that finds no verdict, only when ``is_eval_mode()``; in training a
received-but-unparseable reply stays the wrong answer it always was. Empty output is
a per-benchmark call, so servers differ.

One judge error is the model's, in evaluation: a judge that rejects its input as too
long when the model's own answer made it so. A server that knows this (its judge input
carries no documents) passes ``overflow_is_the_models=True`` to ``call_judge``; in
evaluation the row is then ``judge_input_too_long``, a final 0. A server that cannot
tell keeps the default, and training is unchanged: the overflow stays ``judge_failed``.

Wrap the call, not the scoring around it: ``reraise_judge_errors`` relabels every
exception it sees, so covering a whole ``verify()`` would misfile ordinary bugs
(a KeyError in prompt assembly, say) as judge failures.
"""

import asyncio
import contextvars
import functools
import logging
import os
import re
from typing import Any, Awaitable, Callable, Type, TypeVar

from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from nemo_gym.failure_kinds import (
    INFRA,
    MODEL,
    FailureCause,
    JudgeNoVerdictOutcome,
    judge_failure_after_retries,
)
from nemo_gym.global_config import is_eval_mode
from nemo_gym.server_utils import ServerClient, get_response_json, raise_for_status


logger = logging.getLogger(__name__)

ResponseT = TypeVar("ResponseT", bound=BaseModel)
T = TypeVar("T")

# Evaluation only: how often one judge call (with its verdict parse) is tried on the same stored answer before
# judge_failure_after_retries decides the row. Training keeps a single attempt.
JUDGE_MAX_ATTEMPTS_ENV = "NEMO_GYM_JUDGE_MAX_ATTEMPTS"
DEFAULT_JUDGE_MAX_ATTEMPTS = 3
# Seconds to wait before the 2nd, 3rd, ... attempt; the last value repeats past the end.
JUDGE_RETRY_BACKOFF_SECONDS: tuple[float, ...] = (5.0, 30.0)
# Set while a retry_judge_in_place loop runs, so a nested one makes a single attempt and the cap stays exact.
_IN_JUDGE_RETRY: contextvars.ContextVar[bool] = contextvars.ContextVar("nemo_gym_in_judge_retry", default=False)


def normalize_math_judge_verdict(text: str) -> str:
    """Canonicalize whitespace inside A=B/A!=B markers without changing other text."""
    return re.sub(
        r"\[\[\s*A\s*(!\s*)?=\s*B\s*\]\]",
        lambda match: "[[A!=B]]" if match.group(1) else "[[A=B]]",
        text,
    )


class JudgeError(Exception):
    """A judge call failed; judge_failsafe routes the row to the failures sidecar.

    ``attempts`` is how many times the judge was tried on the same answer before this error, once known, and
    ``attempt_causes`` the cause of each of those tries.
    """

    attempts: int | None = None
    attempt_causes: tuple[FailureCause, ...] = ()


class JudgeInputTooLongError(JudgeError):
    """The judge rejected an input the model's own answer made too long: the model's final 0."""


class JudgeEmptyReplyError(JudgeError):
    """The judge answered with no text at all. A throttled judge or a provider-side filter can send an HTTP 200
    with an empty body, so this is infra, never a verdict about the answer."""


class JudgeNoVerdictError(JudgeError):
    """The judge answered with text that carries no verdict the server can read: the answer was judged.

    ``judge_response`` is the judge's reply, when the server keeps it for a game it records with no verdict.
    """

    judge_response: Any = None


# How an OpenAI-compatible engine (vLLM, SGLang, the OpenAI API) words a request that exceeds its context window.
_CONTEXT_OVERFLOW_PATTERN = re.compile(
    r"context_length_exceeded|maximum context length|context window|prompt is too long|input length"
    r"|too many (?:input )?tokens",
    re.IGNORECASE,
)


# How a Gym model server relays an upstream failure it does not propagate: a 500 whose body carries the upstream
# reply verbatim (``SimpleServer.setup_exception_middleware``).
_GYM_RELAY_MARKER = "calling an inner server:"


def _error_body(error: BaseException) -> str:
    body = getattr(error, "response_content", None)
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return body or ""


def _is_context_overflow(error: BaseException) -> bool:
    """Whether a judge call failed because the request exceeds the judge's context window.

    The judge's engine answers with a 4xx. A Gym model server in front of it that does not propagate that status
    answers with a 500 that carries the engine's reply, which says the same.
    """
    status = getattr(error, "status", None)
    body = _error_body(error)
    if not isinstance(status, int) or not (400 <= status < 500 or (status == 500 and _GYM_RELAY_MARKER in body)):
        return False
    return bool(_CONTEXT_OVERFLOW_PATTERN.search(f"{getattr(error, 'message', '')} {body}"))


def _judge_failure_cause(error: JudgeError) -> FailureCause:
    """What caused one judge try; ``judge_failure_after_retries`` decides the row by the causes of every try.

    - Text with no verdict (``JudgeNoVerdictError``): the judge answered, so the answer was judged. The model's.
    - An empty reply (``JudgeEmptyReplyError``): infra. A throttled judge or a provider-side filter can send an
      HTTP 200 with no text.
    - A failed call (any HTTP status, a timeout, a transport error) is infra: the judge never answered, so the answer
      got no verdict to be scored by. That covers a spent budget (a 422 or 402 whatever its body says), a rate
      limit, a gateway status, a bad key, and a context overflow from a server that cannot tell the answer caused
      it (the judge input carries documents too). The model's own overflow never reaches here: the server that can
      tell raises ``JudgeInputTooLongError``, read from the judge engine's context-length reply.
    - Anything else with no chain (a server's own JudgeError) is infra: there is no evidence the answer caused it.
    """
    if isinstance(error, JudgeNoVerdictError):
        return MODEL
    return INFRA


def raise_unparseable_verdict(server: str, judge_output: str) -> None:
    """Raise the JudgeError for a judge reply that carries no verdict the server can read.

    In evaluation ``retry_judge_in_place`` asks the judge again on the same answer; ``judge_failsafe`` then records
    a ``judge_failed`` row by ``judge_failure_after_retries``. A reply with text and no verdict is the model's
    (the answer was judged); an empty reply is infra (``JudgeEmptyReplyError``).
    """
    if not judge_output.strip():
        raise JudgeEmptyReplyError(f"{server}: judge reply is empty")
    raise JudgeNoVerdictError(f"{server}: judge verdict is unparseable: {judge_output[:500]!r}")


async def reraise_judge_errors(coro: Awaitable[Any]) -> Any:
    """Await a judge call; re-raise any exception as JudgeError (recorded verbatim)."""
    try:
        return await coro
    except JudgeError:
        raise
    except Exception as e:
        # In evaluation, the judge's own reply (an engine's 400 body, relayed through a model server's 500) says
        # why it failed; it goes on the row, so a context overflow and an unparseable verdict do not look alike.
        # Training keeps the row exactly as before.
        body = _error_body(e) if is_eval_mode() else ""
        reply = f" | judge reply: {body[:1000]}" if body else ""
        raise JudgeError(f"{type(e).__name__}: {e}{reply}") from e


def judge_max_attempts() -> int:
    """How many in-place attempts a judge call gets in evaluation (``NEMO_GYM_JUDGE_MAX_ATTEMPTS``, default 3)."""
    try:
        return max(1, int(os.environ.get(JUDGE_MAX_ATTEMPTS_ENV, DEFAULT_JUDGE_MAX_ATTEMPTS)))
    except ValueError:
        return DEFAULT_JUDGE_MAX_ATTEMPTS


async def retry_judge_in_place(attempt: Callable[[], Awaitable[T]]) -> T:
    """Run one judge call, with its verdict parse, and in evaluation retry it in place on a JudgeError.

    ``attempt`` must send the same stored answer every time, so a retry asks the judge again about the answer the
    model already gave and never gives the model another try. An HTTP error, a timeout, an empty reply and an
    unparseable verdict all surface as JudgeError and are retried, up to ``judge_max_attempts()`` with backoff. A
    ``JudgeInputTooLongError`` is the model's own answer and is not retried. The error that ends the loop carries
    the number of attempts made. Training makes a single attempt, as before.
    """
    if _IN_JUDGE_RETRY.get():
        # Nested inside a caller's loop (call_judge inside a call-and-parse attempt): the outer loop retries.
        return await attempt()
    max_attempts = judge_max_attempts() if is_eval_mode() else 1
    token = _IN_JUDGE_RETRY.set(True)
    try:
        return await _retry_loop(attempt, max_attempts)
    finally:
        _IN_JUDGE_RETRY.reset(token)


async def _retry_loop(attempt: Callable[[], Awaitable[T]], max_attempts: int) -> T:
    causes: list[FailureCause] = []
    for attempt_number in range(1, max_attempts + 1):
        try:
            return await attempt()
        except JudgeInputTooLongError as e:
            e.attempts = attempt_number
            raise
        except JudgeError as e:
            causes.append(_judge_failure_cause(e))
            if attempt_number == max_attempts:
                e.attempts = attempt_number
                e.attempt_causes = tuple(causes)
                raise
            delay = JUDGE_RETRY_BACKOFF_SECONDS[min(attempt_number - 1, len(JUDGE_RETRY_BACKOFF_SECONDS) - 1)]
            logger.warning(
                "Judge attempt %d of %d failed, retrying on the same answer in %gs: %s",
                attempt_number,
                max_attempts,
                delay,
                str(e)[:300],
            )
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


async def call_judge(
    server_client: ServerClient,
    *,
    server_name: str,
    url_path: str,
    json: Any,
    response_model: Type[ResponseT],
    overflow_is_the_models: bool = False,
) -> ResponseT:
    """POST to a judge model server and parse the reply; raise JudgeError on failure.

    ``url_path`` is the judge's API surface (``/v1/responses`` or
    ``/v1/chat/completions``), paired with the matching ``response_model``. The
    status check comes before parsing so an auth or server error reports the HTTP
    failure itself, not a validation error against the error body.
    ``overflow_is_the_models`` says the judge input carries nothing long but the model's answer, so a
    context-length rejection is the model's final 0 (``JudgeInputTooLongError``), not a retried judge failure.
    """

    async def _call() -> ResponseT:
        http_response = await server_client.post(server_name=server_name, url_path=url_path, json=json)
        try:
            await raise_for_status(http_response)
        except Exception as e:
            # Evaluation only: training keeps a judge overflow a masked judge_failed row, as before.
            if overflow_is_the_models and is_eval_mode() and _is_context_overflow(e):
                raise JudgeInputTooLongError(f"{type(e).__name__}: {e} | judge reply: {_error_body(e)[:1000]}") from e
            raise
        if not is_eval_mode():
            return response_model.model_validate(await get_response_json(http_response))
        # Evaluation: tell an empty 200 (a throttled judge, a provider-side filter) from text with no verdict.
        try:
            reply = await get_response_json(http_response)
        except Exception as e:
            raise JudgeEmptyReplyError(f"{type(e).__name__}: judge reply is empty or not JSON: {e}") from e
        if not reply:
            raise JudgeEmptyReplyError("judge reply is empty")
        try:
            return response_model.model_validate(reply)
        except ValidationError as e:
            raise JudgeNoVerdictError(f"judge reply carries no verdict: {str(reply)[:500]!r}") from e

    # A retry posts the same ``json`` object: the same answer, byte for byte.
    return await retry_judge_in_place(lambda: reraise_judge_errors(_call()))


def judge_failsafe(verify_fn: Callable, *, no_verdict: JudgeNoVerdictOutcome = "drop") -> Callable:
    """Wrap verify() so a JudgeError returns a sidecar-routed row (reward 0.0, the
    routing keys, the request's ``response`` carried) instead of propagating.
    functools.wraps keeps verify's signature so FastAPI injects the same params
    (``body``, and ``request`` for servers that take it); ``*args, **kwargs`` pass
    them straight through.

    ``no_verdict`` is what the benchmark's official grader does with a judge reply that has no verdict:
    ``wrong`` scores it 0, ``drop`` (the default) leaves it out of the score. Evaluation only."""

    @functools.wraps(verify_fn)
    async def wrapper(*args, **kwargs):
        try:
            return await verify_fn(*args, **kwargs)
        except JudgeError as e:
            body = kwargs.get("body") or next(
                (a for a in (*kwargs.values(), *args) if hasattr(a, "model_dump") and hasattr(a, "response")),
                None,
            )
            if body is None:  # verify always has a request body; guard against an opaque 500
                raise RuntimeError("judge_failsafe: could not locate the verify request body") from e
            error = str(e)[:2000]
            data = body.model_dump()
            if isinstance(e, JudgeInputTooLongError):
                # The model's own answer did not fit the judge: a final failure, scored 0, in training too.
                data |= {
                    "reward": 0.0,
                    "mask_sample": False,
                    "failure_kind": "judge_input_too_long",
                    "failure_reason": error,
                    "_ng_failure_class": "judge_input_too_long",
                    "_ng_failure_judge_error": error,
                    "_ng_failure_judge_attempts": e.attempts or 1,
                }
                return JSONResponse(content=jsonable_encoder(data))
            if is_eval_mode():
                # The judge already had its in-place attempts on this answer; the rollout is never re-run for it.
                data |= {"failure_kind": "judge_failed", "failure_reason": error} | judge_failure_after_retries(
                    judge_error=error,
                    cause=_judge_failure_cause(e),
                    attempts=e.attempts or 1,
                    attempt_causes=e.attempt_causes,
                    no_verdict=no_verdict,
                )
                return JSONResponse(content=jsonable_encoder(data))
            instance_config = data.get("instance_config")
            data |= {
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "judge_failed",
                "failure_reason": error,
                # Keep older NeMo RL consumers working during top-level field migration.
                "instance_config": {
                    **(instance_config if isinstance(instance_config, dict) else {}),
                    "mask_sample": True,
                },
                "_ng_failure_class": "judge_failed",
                "_ng_failure_judge_error": error,
            }
            return JSONResponse(content=jsonable_encoder(data))

    return wrapper
