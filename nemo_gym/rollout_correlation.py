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
from collections.abc import Iterable, Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional, Protocol

from fastapi.responses import JSONResponse
from pydantic import BaseModel

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX
from nemo_gym.episode_types import EpisodeId
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
ROUTED_PATH_STATE_KEY = "nemo_gym.routed_path"

# Clients that cannot change their base URL per request name the rollout in this header instead of the
# ``/ng-rollout/<id>/`` prefix. Gym servers accept either; both on one request must agree.
ROLLOUT_ID_HEADER = "x-ng-rollout-id"
_ROLLOUT_ID_HEADER_KEY = ROLLOUT_ID_HEADER.encode("latin-1")


def scope_with_routed_path(scope: dict[str, Any], path: str) -> dict[str, Any]:
    """Copy the scope for routing and share the rewritten path with outer middleware."""
    scope.setdefault("state", {})[ROUTED_PATH_STATE_KEY] = path
    return {**scope, "path": path, "raw_path": path.encode("utf-8")}


def maybe_rollout_id_from_run_body(body: BaseModel | Mapping[str, Any] | None) -> Optional[str]:
    """Build the capture key for a run request.

    An explicit ``_ng_rollout_id`` takes precedence.
    Otherwise derive ``"{task}-{rollout}"`` from the task and rollout indices.
    Re-dispatch attempts append ``-a{n}``.
    The suffix is reserved: an explicit id ending in ``-a<N>`` names attempt N of the rollout before it,
    as a training controller names a restored attempt,
    and rollout collection refuses dataset rows that use it as a plain id.
    Writers and consumers must use this same identity.
    Reused task and rollout indices produce a repeated capture key.
    Use an explicit id when numbering restarts across dispatches.
    """
    if not isinstance(body, (BaseModel, Mapping)):
        return None

    def field(key: str) -> Any:
        if isinstance(body, Mapping):
            return body.get(key)
        if key == ROLLOUT_ID_KEY_NAME and hasattr(body, "capture_rollout_id"):
            return body.capture_rollout_id
        return getattr(body, key, None)

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
        # An explicit id may carry its own attempt suffix, but not as well as an attempt index:
        # the key would not decode to one rollout and attempt.
        try:
            EpisodeId.from_capture_key(rollout_id)
        except ValueError as error:
            raise ValueError(
                f"{ROLLOUT_ID_KEY_NAME} {explicit!r} already ends with the attempt suffix '-a<N>'; "
                f"do not also set {ATTEMPT_INDEX_KEY_NAME}"
            ) from error
    return rollout_id


def current_rollout_id() -> Optional[str]:
    return _ROLLOUT_ID.get()


def current_episode_id() -> Optional[EpisodeId]:
    """Return the logical rollout and attempt carried by the current capture key."""
    capture_key = _ROLLOUT_ID.get()
    return EpisodeId.from_capture_key(capture_key) if capture_key is not None else None


@contextmanager
def rollout_context(rollout_id: Optional[str]) -> Iterator[None]:
    token = _ROLLOUT_ID.set(rollout_id)
    try:
        yield
    finally:
        _ROLLOUT_ID.reset(token)


def resolve_request_rollout_id(
    *, path_rollout_id: Optional[str], headers: Iterable[tuple[bytes, bytes]]
) -> Optional[str]:
    """Return the rollout a request names through its path prefix or ``x-ng-rollout-id``.

    Raises ``ValueError`` when the header is malformed, repeated with different values, or names a
    different rollout than the path prefix.
    """
    header_ids = {value.decode("latin-1") for key, value in headers if key.lower() == _ROLLOUT_ID_HEADER_KEY and value}
    if not header_ids:
        return path_rollout_id
    if len(header_ids) > 1:
        raise ValueError(f"{ROLLOUT_ID_HEADER} is repeated with different values: {sorted(header_ids)}")
    header_id = header_ids.pop()
    if not ROLLOUT_ID_PATTERN.match(header_id):
        raise ValueError(
            f"{ROLLOUT_ID_HEADER} must be letters, digits, dots, dashes or underscores starting with a letter "
            f"or digit; got {header_id!r}"
        )
    if path_rollout_id is not None and path_rollout_id != header_id:
        raise ValueError(
            f"The /{ROLLOUT_PATH_PREFIX}/ path names rollout {path_rollout_id!r} but {ROLLOUT_ID_HEADER} "
            f"names {header_id!r}"
        )
    return header_id


def add_rollout_id_header(request: "_ClientRequest", *, base_urls: Sequence[str]) -> None:
    """Name the current rollout on an outgoing request to a Gym server.

    Use this as an httpx request event hook, or call it from a ``requests`` transport adapter, so a
    long-lived client keeps one base URL. Only requests whose URL starts with one of ``base_urls`` are
    tagged; a header already present is kept; with no current rollout the request is left unchanged and
    the Gym model server forwards it without capture.
    """
    rollout_id = current_rollout_id()
    if rollout_id is None or ROLLOUT_ID_HEADER in request.headers:
        return
    if _is_under_any(str(request.url), base_urls):
        request.headers[ROLLOUT_ID_HEADER] = rollout_id


async def add_rollout_id_header_async(request: "_ClientRequest", *, base_urls: Sequence[str]) -> None:
    """Async form of :func:`add_rollout_id_header` for ``httpx.AsyncClient`` request event hooks."""
    add_rollout_id_header(request, base_urls=base_urls)


class RolloutContextMiddleware:
    """Strip a rollout prefix, or read ``x-ng-rollout-id``, and expose the rollout for this request."""

    # Match the same id characters as ``ROLLOUT_ID_PATTERN``.
    # Anchor the id between the prefix and the remaining path.
    _PREFIX = re.compile(
        rf"^/{re.escape(ROLLOUT_PATH_PREFIX)}/(?P<rollout_id>{ROLLOUT_ID_PATTERN.pattern.strip('^$')})(?P<rest>/.*)$"
    )

    def __init__(self, app: Any) -> None:
        self._app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self._app(scope, receive, send)
            return
        match = self._PREFIX.match(scope.get("path", ""))
        try:
            rollout_id = resolve_request_rollout_id(
                path_rollout_id=match.group("rollout_id") if match else None, headers=scope.get("headers") or []
            )
        except ValueError as error:
            await JSONResponse({"detail": str(error)}, status_code=400)(scope, receive, send)
            return
        if match is not None:
            scope = scope_with_routed_path(scope, match.group("rest"))
        if rollout_id is None:
            await self._app(scope, receive, send)
            return
        with rollout_context(rollout_id):
            await self._app(scope, receive, send)


class _ClientRequest(Protocol):
    """The part of an httpx ``Request`` or a requests ``PreparedRequest`` that tagging uses."""

    url: Any
    headers: MutableMapping[str, str]


def _is_under_any(url: str, base_urls: Sequence[str]) -> bool:
    """Whether ``url`` is one of ``base_urls`` or a path below one.

    Compares whole path segments, so ``http://host:80001`` is not under ``http://host:8000``.
    """
    return any(url == base.rstrip("/") or url.startswith(base.rstrip("/") + "/") for base in base_urls)
