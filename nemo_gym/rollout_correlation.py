# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Optional

from pydantic import BaseModel
from starlette.responses import JSONResponse

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


_ROLLOUT_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_rollout_id", default=None)
_LOGICAL_ROLLOUT_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_logical_rollout_id", default=None)
_ATTEMPT_INDEX: ContextVar[Optional[int]] = ContextVar("nemo_gym_attempt_index", default=None)


@dataclass
class _CheckpointParentState:
    source_capture_key: str
    parent_model_call_id: str
    consumed: bool = False


_CHECKPOINT_PARENT: ContextVar[Optional[_CheckpointParentState]] = ContextVar(
    "nemo_gym_checkpoint_parent",
    default=None,
)

# These headers propagate execution identity independently of capture.
# The rollout header is the stable logical ID.
# The attempt header identifies one dispatch of that rollout.
ROLLOUT_ID_HEADER = "x-nemo-gym-rollout-id"
ATTEMPT_INDEX_HEADER = "x-nemo-gym-attempt-index"
MODEL_CALL_ID_HEADER = "x-nemo-gym-model-call-id"
SOURCE_CAPTURE_KEY_HEADER = "x-nemo-gym-source-capture-key"
PARENT_MODEL_CALL_ID_HEADER = "x-nemo-gym-parent-model-call-id"

# The transport id appends ``-a{n}`` for re-dispatch attempts. The suffix is a
# capture and routing key, never the logical identity. This pattern recovers
# the split when only the transport id is available; header-carried values are
# authoritative because an explicit logical id may itself end in ``-a{n}``.
_ATTEMPT_SUFFIX_PATTERN = re.compile(r"^(?P<logical>.+)-a(?P<attempt>\d+)$")


def split_transport_rollout_id(rollout_id: Optional[str]) -> tuple[Optional[str], int]:
    """Split a transport rollout id into ``(logical_id, attempt_index)``.

    A missing or unsuffixed id has attempt index 0. The split is a fallback for
    path-only sources; callers holding explicit header values must prefer them.
    """
    if rollout_id is None:
        return None, 0
    match = _ATTEMPT_SUFFIX_PATTERN.match(rollout_id)
    if match is None:
        return rollout_id, 0
    return match.group("logical"), int(match.group("attempt"))


def capture_key_for(logical_rollout_id: str, attempt_index: int) -> str:
    """Return the attempt-qualified key used by capture paths and stores."""
    if attempt_index < 0:
        raise ValueError("attempt_index must be non-negative")
    return logical_rollout_id if attempt_index == 0 else f"{logical_rollout_id}-a{attempt_index}"


LOGICAL_REQUEST_HEADER = "x-nemo-gym-logical-request-id"

# A capture id is a path segment in ``/ng-rollout/<id>/...``.
# Restrict it to characters that survive a path round trip.
# Exclude leading dots because stores also use the id as a filename component.
# Middleware uses the same pattern.
ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


def _run_body_field(body: BaseModel | Mapping[str, Any], key: str) -> Any:
    if isinstance(body, Mapping):
        return body.get(key)
    if key == ROLLOUT_ID_KEY_NAME and hasattr(body, "capture_rollout_id"):
        return body.capture_rollout_id
    return getattr(body, key, None)


def maybe_rollout_id_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> Optional[str]:
    """Build the capture key for a run request.

    An explicit ``_ng_rollout_id`` takes precedence.
    Otherwise derive ``"{task}-{rollout}"`` from the task and rollout indices.
    Re-dispatch attempts append ``-a{n}``.
    Writers and consumers must use this same identity.
    Reused task and rollout indices produce a repeated capture key.
    Use an explicit id when numbering restarts across dispatches.
    """
    if not isinstance(body, (BaseModel, Mapping)):
        return None

    explicit = _run_body_field(body, ROLLOUT_ID_KEY_NAME)
    if explicit is not None:
        # Reject malformed explicit ids instead of sanitizing them.
        # Rewriting would create a key the caller cannot look up.
        if not (isinstance(explicit, str) and ROLLOUT_ID_PATTERN.match(explicit)):
            raise ValueError(
                f"{ROLLOUT_ID_KEY_NAME} must be a string of letters, digits, dots, dashes or "
                f"underscores starting with a letter or digit; got {explicit!r}"
            )
        rollout_id = explicit
    else:
        task = _run_body_field(body, TASK_INDEX_KEY_NAME)
        rollout = _run_body_field(body, ROLLOUT_INDEX_KEY_NAME)
        if task is None or rollout is None:
            return None
        rollout_id = f"{task}-{rollout}"

    attempt = int(_run_body_field(body, ATTEMPT_INDEX_KEY_NAME) or 0)
    return capture_key_for(rollout_id, attempt)


def execution_identity_from_run_body(
    body: BaseModel | Mapping[str, Any] | None,
) -> tuple[Optional[str], Optional[int]]:
    """Return the logical rollout ID and attempt index from a run request.

    This identity exists independently of observability and token capture.
    The logical ID never includes Gym's attempt suffix.
    """
    if not isinstance(body, (BaseModel, Mapping)):
        return None, None
    explicit = _run_body_field(body, ROLLOUT_ID_KEY_NAME)
    if explicit is not None:
        if not (isinstance(explicit, str) and ROLLOUT_ID_PATTERN.fullmatch(explicit)):
            raise ValueError(
                f"{ROLLOUT_ID_KEY_NAME} must be a string of letters, digits, dots, dashes or "
                f"underscores starting with a letter or digit; got {explicit!r}"
            )
        logical_rollout_id = explicit
    else:
        task = _run_body_field(body, TASK_INDEX_KEY_NAME)
        rollout = _run_body_field(body, ROLLOUT_INDEX_KEY_NAME)
        if task is None or rollout is None:
            return None, None
        logical_rollout_id = f"{task}-{rollout}"
    attempt_index = int(_run_body_field(body, ATTEMPT_INDEX_KEY_NAME) or 0)
    capture_key_for(logical_rollout_id, attempt_index)
    return logical_rollout_id, attempt_index


def current_rollout_id() -> Optional[str]:
    """Return the attempt-qualified capture and routing key."""
    return _ROLLOUT_ID.get()


def current_logical_rollout_id() -> Optional[str]:
    """Return the stable rollout ID shared by all attempts."""
    explicit = _LOGICAL_ROLLOUT_ID.get()
    if explicit is not None:
        return explicit
    rollout_id = _ROLLOUT_ID.get()
    return split_transport_rollout_id(rollout_id)[0]


def current_attempt_index() -> Optional[int]:
    """The attempt index of the current request context.

    Falls back to the suffix of the transport rollout id when no explicit
    attempt was set; ``None`` when there is no rollout context at all.
    """
    explicit = _ATTEMPT_INDEX.get()
    if explicit is not None:
        return explicit
    rollout_id = _ROLLOUT_ID.get()
    if rollout_id is None:
        return None
    return split_transport_rollout_id(rollout_id)[1]


def current_execution_identity() -> tuple[Optional[str], Optional[int]]:
    """Return the logical rollout ID and attempt index for this request."""
    rollout_id = _ROLLOUT_ID.get()
    if rollout_id is None:
        return _LOGICAL_ROLLOUT_ID.get(), _ATTEMPT_INDEX.get()
    return current_logical_rollout_id(), current_attempt_index()


def take_checkpoint_parent() -> tuple[Optional[str], Optional[str]]:
    """Consume the restored parent for the first policy model request."""
    state = _CHECKPOINT_PARENT.get()
    if state is None or state.consumed:
        return None, None
    state.consumed = True
    return state.source_capture_key, state.parent_model_call_id


@contextmanager
def checkpoint_parent_context(source_capture_key: str, parent_model_call_id: str) -> Iterator[None]:
    """Propagate an explicit cross-attempt parent to downstream model calls."""
    if ROLLOUT_ID_PATTERN.fullmatch(source_capture_key) is None:
        raise ValueError("invalid source capture key")
    if not parent_model_call_id or "\r" in parent_model_call_id or "\n" in parent_model_call_id:
        raise ValueError("invalid parent model-call ID")
    token = _CHECKPOINT_PARENT.set(
        _CheckpointParentState(
            source_capture_key=source_capture_key,
            parent_model_call_id=parent_model_call_id,
        )
    )
    try:
        yield
    finally:
        _CHECKPOINT_PARENT.reset(token)


@contextmanager
def rollout_context(
    rollout_id: Optional[str],
    attempt_index: Optional[int] = None,
    *,
    logical_rollout_id: Optional[str] = None,
) -> Iterator[None]:
    """Install an attempt-qualified capture key and its execution identity."""
    token = _ROLLOUT_ID.set(rollout_id)
    logical_token = _LOGICAL_ROLLOUT_ID.set(logical_rollout_id)
    attempt_token = _ATTEMPT_INDEX.set(attempt_index)
    try:
        yield
    finally:
        _ATTEMPT_INDEX.reset(attempt_token)
        _LOGICAL_ROLLOUT_ID.reset(logical_token)
        _ROLLOUT_ID.reset(token)


class RolloutContextMiddleware:
    """Strip a rollout prefix and expose it to downstream Gym calls for this request."""

    # Match the same id characters as ``ROLLOUT_ID_PATTERN``.
    # Anchor the id between the prefix and the remaining path.
    _PREFIX = re.compile(
        rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>{ROLLOUT_ID_PATTERN.pattern.strip('^$')})(?P<rest>/.*)$"
    )

    def __init__(self, app: Any) -> None:
        self._app = app

    @staticmethod
    def _identity_from_headers(
        scope: dict[str, Any],
    ) -> tuple[Optional[str], Optional[int], Optional[str], Optional[str], Optional[str]]:
        values: dict[str, set[str]] = {
            ROLLOUT_ID_HEADER: set(),
            ATTEMPT_INDEX_HEADER: set(),
            SOURCE_CAPTURE_KEY_HEADER: set(),
            PARENT_MODEL_CALL_ID_HEADER: set(),
        }
        for name, value in scope.get("headers") or ():
            key = name.decode("latin-1").lower()
            if key in values:
                values[key].add(value.decode("latin-1"))
        if any(len(entries) > 1 for entries in values.values()):
            return None, None, None, None, "conflicting duplicate correlation headers"
        source_values = values[SOURCE_CAPTURE_KEY_HEADER]
        parent_values = values[PARENT_MODEL_CALL_ID_HEADER]
        if bool(source_values) != bool(parent_values):
            return None, None, None, None, "source capture key and parent model-call ID must be sent together"
        source_capture_key = next(iter(source_values), None)
        parent_model_call_id = next(iter(parent_values), None)
        if source_capture_key is not None and ROLLOUT_ID_PATTERN.fullmatch(source_capture_key) is None:
            return None, None, None, None, "invalid source capture key header"
        if parent_model_call_id is not None and (
            not parent_model_call_id or "\r" in parent_model_call_id or "\n" in parent_model_call_id
        ):
            return None, None, None, None, "invalid parent model-call ID header"
        rollout_values = values[ROLLOUT_ID_HEADER]
        attempt_values = values[ATTEMPT_INDEX_HEADER]
        if bool(rollout_values) != bool(attempt_values):
            return None, None, None, None, "rollout ID and attempt index headers must be sent together"
        if not rollout_values:
            return None, None, source_capture_key, parent_model_call_id, None
        rollout_id = next(iter(rollout_values))
        attempt_raw = next(iter(attempt_values))
        if ROLLOUT_ID_PATTERN.fullmatch(rollout_id) is None:
            return None, None, None, None, "invalid logical rollout ID header"
        try:
            attempt_index = int(attempt_raw)
        except ValueError:
            return None, None, None, None, "invalid attempt index header"
        if attempt_index < 0:
            return None, None, None, None, "attempt index must be non-negative"
        return rollout_id, attempt_index, source_capture_key, parent_model_call_id, None

    @staticmethod
    async def _reject(scope: dict[str, Any], receive: Any, send: Any, detail: str) -> None:
        response = JSONResponse(
            status_code=409,
            content={"error": {"code": "execution_identity_mismatch", "detail": detail}},
        )
        await response(scope, receive, send)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return

        match = self._PREFIX.match(scope.get("path", ""))
        logical_rollout_id, attempt_index, source_capture_key, parent_model_call_id, identity_error = (
            self._identity_from_headers(scope)
        )
        if identity_error is not None:
            await self._reject(scope, receive, send, identity_error)
            return

        parent_context = (
            checkpoint_parent_context(source_capture_key, parent_model_call_id)
            if source_capture_key is not None and parent_model_call_id is not None
            else nullcontext()
        )
        if match is None:
            if logical_rollout_id is None or attempt_index is None:
                with parent_context:
                    await self._app(scope, receive, send)
                return
            capture_key = capture_key_for(logical_rollout_id, attempt_index)
            with parent_context:
                with rollout_context(
                    capture_key,
                    attempt_index=attempt_index,
                    logical_rollout_id=logical_rollout_id,
                ):
                    await self._app(scope, receive, send)
            return

        capture_key = match.group("rollout_id")
        if logical_rollout_id is None:
            logical_rollout_id, attempt_index = split_transport_rollout_id(capture_key)
        elif capture_key_for(logical_rollout_id, attempt_index or 0) != capture_key:
            await self._reject(
                scope,
                receive,
                send,
                "capture path does not match the logical rollout ID and attempt index headers",
            )
            return

        path = match.group("rest")
        scope = {**scope, "path": path, "raw_path": path.encode()}
        with parent_context:
            with rollout_context(
                capture_key,
                attempt_index=attempt_index,
                logical_rollout_id=logical_rollout_id,
            ):
                await self._app(scope, receive, send)
