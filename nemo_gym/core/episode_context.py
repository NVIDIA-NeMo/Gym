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
"""The episode that the current request or task belongs to."""

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from pydantic import BaseModel

from nemo_gym.config.keys import (
    ATTEMPT_INDEX_KEY_NAME,
    ROLLOUT_ID_KEY_NAME,
    ROLLOUT_INDEX_KEY_NAME,
    TASK_INDEX_KEY_NAME,
)
from nemo_gym.core.episode_types import EpisodeId


VERIFIER_INVOCATION_HEADER = "x-ng-verifier-invocation-id"


_ROLLOUT_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_rollout_id", default=None)
# True while handling a request whose rollout skips token capture; see ``skips_token_capture``.
_SKIP_TOKEN_CAPTURE: ContextVar[bool] = ContextVar("nemo_gym_skip_token_capture", default=False)
_VERIFIER_INVOCATION_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_verifier_invocation_id", default=None)
_VERIFIER_MODEL_REQUESTS: ContextVar[Optional[list[str]]] = ContextVar(
    "nemo_gym_verifier_model_requests", default=None
)

# A capture id is a path segment in ``/ng-rollout/<id>/...``.
# Restrict it to characters that survive a path round trip.
# Exclude leading dots because stores also use the id as a filename component.
# Middleware uses the same pattern.
ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
# The capture key, a rollout id with its attempt suffix, also names the model server's capture files, of which
# `<key>.capture.incomplete` is the longest, and a file name holds at most 255 bytes. Room is kept for `-a9999`.
MAX_ROLLOUT_ID_LENGTH = 255 - len(".capture.incomplete") - len("-a9999")


def check_rollout_id_length(rollout_id: str) -> None:
    """Refuse a rollout id whose capture files would pass the 255-byte file name limit at some attempt."""
    if len(rollout_id.encode()) > MAX_ROLLOUT_ID_LENGTH:
        raise ValueError(
            f"Rollout id {rollout_id!r} is {len(rollout_id.encode())} bytes, more than the {MAX_ROLLOUT_ID_LENGTH} "
            "its capture file names allow with an attempt suffix; use a shorter _ng_rollout_id"
        )


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


def current_session_key() -> Optional[str]:
    """The key an upstream router or endpoint selection uses to keep one conversation on one replica.

    None outside a rollout. A session is a rollout attempt: the key is the capture key (`<rollout id>[-a<N>]`),
    so a retry attempt gets its own session, and every agent that makes model calls inside the same attempt
    shares one. The outbound session header and the vLLM model server's endpoint selection both read this
    function, so they agree on what a session is. When a model call can name its agent instance, this is the
    one place that has to learn it.
    """
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


def skips_token_capture(body: BaseModel | Mapping[str, Any] | None) -> bool:
    """Return whether a run request or row skips token capture for its rollout.

    A training framework sets ``_ng_skip_token_capture: true`` on a rollout it does not train on, such as a validation
    rollout, so none of the rollout's model calls is captured. ``false``, ``null`` or no value leaves capture to the
    run's ``token_id_capture`` block: a rollout can turn capture off, never on.
    """
    # Imported here because global_config imports feature modules that import the episode identity.
    from nemo_gym.config.global_config import (
        ROLLOUT_ID_KEY_NAME,
        ROLLOUT_INDEX_KEY_NAME,
        SKIP_TOKEN_CAPTURE_KEY_NAME,
        TASK_INDEX_KEY_NAME,
    )

    if isinstance(body, Mapping):
        value = body.get(SKIP_TOKEN_CAPTURE_KEY_NAME)
    elif isinstance(body, BaseModel):
        # BaseRunRequest declares the key as ``skip_token_capture``; other models may keep it as an extra field.
        value = getattr(body, "skip_token_capture", None)
        if value is None:
            value = (body.model_extra or {}).get(SKIP_TOKEN_CAPTURE_KEY_NAME)
    else:
        return False
    # ``null`` counts as unset, as for ``_ng_rollout_id``: a table that sets the column on some rows writes it on the
    # others.
    if value is None:
        return False
    if not isinstance(value, bool):
        row = ""
        if isinstance(body, Mapping):
            identity = {
                key: body[key]
                for key in (TASK_INDEX_KEY_NAME, ROLLOUT_INDEX_KEY_NAME, ROLLOUT_ID_KEY_NAME)
                if body.get(key) is not None
            }
            row = f" (row {identity})" if identity else ""
        raise ValueError(f"{SKIP_TOKEN_CAPTURE_KEY_NAME} must be true or false{row}; got {value!r}")
    return value


def current_rollout_skips_token_capture() -> bool:
    """Return ``True`` while handling a request whose rollout skips token capture."""
    return _SKIP_TOKEN_CAPTURE.get()


@contextmanager
def skip_token_capture_context(skip: bool) -> Iterator[None]:
    """Skip token capture for the model calls made inside the block when ``skip`` is ``True``."""
    token = _SKIP_TOKEN_CAPTURE.set(skip)
    try:
        yield
    finally:
        _SKIP_TOKEN_CAPTURE.reset(token)


def current_verifier_invocation_id() -> Optional[str]:
    """Return the enclosing verify execution, if any."""
    return _VERIFIER_INVOCATION_ID.get()


@contextmanager
def verifier_invocation_context(invocation_id: str) -> Iterator[None]:
    """Carry one verify execution's identity across its concurrent judge calls."""
    token = _VERIFIER_INVOCATION_ID.set(invocation_id)
    requests = _VERIFIER_MODEL_REQUESTS.set([])
    try:
        yield
    finally:
        _VERIFIER_MODEL_REQUESTS.reset(requests)
        _VERIFIER_INVOCATION_ID.reset(token)


def note_verifier_model_request(server_name: str) -> None:
    """Note that the enclosing verify execution sent a request to a model server; outside one, do nothing."""
    requests = _VERIFIER_MODEL_REQUESTS.get()
    if requests is not None:
        requests.append(server_name)


def verifier_model_requests() -> list[str]:
    """The model servers the enclosing verify execution has sent requests to so far."""
    return list(_VERIFIER_MODEL_REQUESTS.get() or [])
