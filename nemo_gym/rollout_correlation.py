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
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import ROLLOUT_ID_KEY_NAME


_ROLLOUT_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_rollout_id", default=None)

# A rollout id is a path segment in ``/ng-rollout/<id>/...``.
# Canonical UUIDv4 strings are safe as path and filename components.
# Middleware uses the same pattern.
ROLLOUT_ID_PATTERN = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$")


@lru_cache(maxsize=4096)
def _validate_rollout_id(value: str) -> str:
    """Validate one canonical UUIDv4 string and cache the result."""
    if ROLLOUT_ID_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{ROLLOUT_ID_KEY_NAME} must be a canonical UUIDv4 string; got {value!r}")
    try:
        parsed = UUID(value)
    except ValueError as error:
        raise ValueError(f"{ROLLOUT_ID_KEY_NAME} must be a canonical UUIDv4 string; got {value!r}") from error

    if parsed.version != 4 or str(parsed) != value:
        raise ValueError(f"{ROLLOUT_ID_KEY_NAME} must be a canonical UUIDv4 string; got {value!r}")
    return value


def validate_rollout_id(value: Any) -> str:
    """Validate and return a canonical UUIDv4 rollout id.

    Validation is cached for repeated use throughout one rollout request.
    Non-string values are rejected before consulting the cache.
    """
    if not isinstance(value, str):
        raise ValueError(f"{ROLLOUT_ID_KEY_NAME} must be a canonical UUIDv4 string; got {value!r}")
    return _validate_rollout_id(value)


def get_rollout_id_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> str:
    """Retrieve and validate the required rollout id from a run request.

    The request must carry ``_ng_rollout_id`` as a canonical UUIDv4 string.
    Task, rollout, and attempt indices are metadata and never form this identity.

    Raises:
        ValueError: If the body is unsupported, the field is missing, or the value is invalid.
    """
    if not isinstance(body, (BaseModel, Mapping)):
        raise ValueError(f"a run request body with {ROLLOUT_ID_KEY_NAME} is required; got {body!r}")

    rollout_id = (
        body.get(ROLLOUT_ID_KEY_NAME) if isinstance(body, Mapping) else getattr(body, ROLLOUT_ID_KEY_NAME, None)
    )
    if rollout_id is None:
        raise ValueError(f"{ROLLOUT_ID_KEY_NAME} is required in every run request")
    return validate_rollout_id(rollout_id)


def current_rollout_id() -> Optional[str]:
    return _ROLLOUT_ID.get()


@contextmanager
def rollout_context(rollout_id: Optional[str]) -> Iterator[None]:
    token = _ROLLOUT_ID.set(rollout_id)
    try:
        yield
    finally:
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

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        match = self._PREFIX.match(scope.get("path", "")) if scope.get("type") == "http" else None
        if match is None:
            await self._app(scope, receive, send)
            return

        path = match.group("rest")
        scope = {**scope, "path": path, "raw_path": path.encode()}
        with rollout_context(match.group("rollout_id")):
            await self._app(scope, receive, send)
