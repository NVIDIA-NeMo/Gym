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
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from pydantic import BaseModel

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    EXECUTION_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


_EXECUTION_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_execution_id", default=None)
_CORRELATION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def new_execution_id() -> str:
    """Allocate an identity for one physical ``/run`` execution."""
    return f"execution-{uuid.uuid4().hex}"


def _validated_correlation_id(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _CORRELATION_ID.fullmatch(value) is None:
        raise ValueError(f"{field} must match {_CORRELATION_ID.pattern!r}; got {value!r}")
    return value


def _field(body: BaseModel | Mapping[str, Any], key: str) -> Any:
    if isinstance(body, Mapping):
        return body.get(key)
    model_extra = getattr(body, "model_extra", None)
    if isinstance(model_extra, Mapping) and key in model_extra:
        return model_extra[key]
    if key == EXECUTION_ID_KEY_NAME:
        execution_id = getattr(body, "_nemo_gym_execution_id", None)
        if execution_id is not None:
            return execution_id
    return getattr(body, key, None)


def maybe_legacy_rollout_id_from_run_body(
    body: BaseModel | Mapping[str, Any] | None,
) -> Optional[str]:
    """Return the pre-execution-ID task/rollout capture key."""
    if not isinstance(body, (BaseModel, Mapping)):
        return None

    task = _field(body, TASK_INDEX_KEY_NAME)
    rollout = _field(body, ROLLOUT_INDEX_KEY_NAME)
    if task is None or rollout is None:
        return None

    rollout_id = f"{task}-{rollout}"
    attempt = _field(body, ATTEMPT_INDEX_KEY_NAME)
    if attempt is not None and int(attempt) > 0:
        rollout_id = f"{rollout_id}-a{int(attempt)}"
    return rollout_id


def maybe_explicit_execution_id_from_run_body(
    body: BaseModel | Mapping[str, Any] | None,
) -> Optional[str]:
    """Read Gym's physical execution ID without inventing a legacy value."""
    if not isinstance(body, (BaseModel, Mapping)):
        return None
    execution_id = _field(body, EXECUTION_ID_KEY_NAME)
    if execution_id is not None:
        return _validated_correlation_id(execution_id, field=EXECUTION_ID_KEY_NAME)
    return None


def maybe_execution_id_from_run_body(
    body: BaseModel | Mapping[str, Any] | None,
) -> Optional[str]:
    """Read Gym's execution ID, with a legacy capture-key fallback."""
    execution_id = maybe_explicit_execution_id_from_run_body(body)
    if execution_id is not None:
        return execution_id
    return maybe_legacy_rollout_id_from_run_body(body)


def maybe_rollout_id_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> Optional[str]:
    """Compatibility alias for the capture correlation ID.

    New requests return ``_ng_execution_id``. Legacy requests retain their
    task/rollout/attempt-derived key.
    """
    return maybe_execution_id_from_run_body(body)


def current_execution_id() -> Optional[str]:
    return _EXECUTION_ID.get()


def current_rollout_id() -> Optional[str]:
    """Compatibility alias for :func:`current_execution_id`."""
    return current_execution_id()


@contextmanager
def execution_context(execution_id: Optional[str]) -> Iterator[None]:
    token = _EXECUTION_ID.set(execution_id)
    try:
        yield
    finally:
        _EXECUTION_ID.reset(token)


@contextmanager
def rollout_context(rollout_id: Optional[str]) -> Iterator[None]:
    """Compatibility alias for :func:`execution_context`."""
    with execution_context(rollout_id):
        yield


class RolloutContextMiddleware:
    """Strip a rollout prefix and expose it to downstream Gym calls for this request."""

    _PREFIX = re.compile(
        rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<rest>/.*)$"
    )

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        match = self._PREFIX.match(scope.get("path", "")) if scope.get("type") == "http" else None
        if match is None:
            await self._app(scope, receive, send)
            return

        path = match.group("rest")
        scope = {**scope, "path": path, "raw_path": path.encode()}
        with execution_context(match.group("rollout_id")):
            await self._app(scope, receive, send)
