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
from typing import Any, Optional

from pydantic import BaseModel

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.global_config import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)


_ROLLOUT_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_rollout_id", default=None)

# A capture id is a path segment in ``/ng-rollout/<id>/...``.
# Restrict it to characters that survive a path round trip.
# Exclude leading dots because stores also use the id as a filename component.
# Middleware uses the same pattern.
ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


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

    def field(key: str) -> Any:
        return body.get(key) if isinstance(body, Mapping) else getattr(body, key, None)

    explicit = field(ROLLOUT_ID_KEY_NAME)
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
        task = field(TASK_INDEX_KEY_NAME)
        rollout = field(ROLLOUT_INDEX_KEY_NAME)
        if task is None or rollout is None:
            return None
        rollout_id = f"{task}-{rollout}"

    attempt = field(ATTEMPT_INDEX_KEY_NAME)
    if attempt is not None and int(attempt) > 0:
        rollout_id = f"{rollout_id}-a{int(attempt)}"
    return rollout_id


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
