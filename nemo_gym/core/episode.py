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
"""Episode identity, and the episode that the current request or task belongs to."""

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
)
from typing_extensions import Self


_CAPTURE_KEY_ATTEMPT = re.compile(r"-a([1-9][0-9]*)$")


class EpisodeId(BaseModel):
    """Identify one physical attempt of a logical rollout."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rollout_id: str = Field(min_length=1, pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
    attempt: int = Field(default=0, ge=0)

    @field_validator("rollout_id")
    @classmethod
    def reserve_attempt_suffix(cls, rollout_id: str) -> str:
        """Keep the derived capture key injective without changing existing keys."""
        if _CAPTURE_KEY_ATTEMPT.search(rollout_id):
            raise ValueError("rollout_id must not end with the reserved attempt suffix '-a<N>'")
        return rollout_id

    @property
    def capture_key(self) -> str:
        """Return the attempt-qualified key used by capture routes."""
        return self.rollout_id if self.attempt == 0 else f"{self.rollout_id}-a{self.attempt}"

    @classmethod
    def from_capture_key(cls, capture_key: str) -> Self:
        """Invert ``capture_key``; the reserved attempt suffix makes the mapping one-to-one."""
        match = _CAPTURE_KEY_ATTEMPT.search(capture_key)
        if match is None:
            return cls(rollout_id=capture_key)
        return cls(rollout_id=capture_key[: match.start()], attempt=int(match.group(1)))


class TaskId(BaseModel):
    """Identify one task within a run-qualified taskset."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    taskset: str = Field(
        min_length=1,
        description="Run-unique taskset name; use the '<environment>:<taskset>' convention for blended runs.",
    )
    task_id: str = Field(min_length=1)


VERIFIER_INVOCATION_HEADER = "x-ng-verifier-invocation-id"


_ROLLOUT_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_rollout_id", default=None)
_VERIFIER_INVOCATION_ID: ContextVar[Optional[str]] = ContextVar("nemo_gym_verifier_invocation_id", default=None)

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
    The suffix is reserved: an explicit id ending in ``-a<N>`` names attempt N of the rollout before it,
    as a training controller names a restored attempt,
    and rollout collection refuses dataset rows that use it as a plain id.
    Writers and consumers must use this same identity.
    Reused task and rollout indices produce a repeated capture key.
    Use an explicit id when numbering restarts across dispatches.
    """
    # Imported here because global_config imports feature modules that import the episode identity above.
    from nemo_gym.config.global_config import (
        ATTEMPT_INDEX_KEY_NAME,
        ROLLOUT_ID_KEY_NAME,
        ROLLOUT_INDEX_KEY_NAME,
        TASK_INDEX_KEY_NAME,
    )

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


def current_verifier_invocation_id() -> Optional[str]:
    """Return the enclosing verify execution, if any."""
    return _VERIFIER_INVOCATION_ID.get()


@contextmanager
def verifier_invocation_context(invocation_id: str) -> Iterator[None]:
    """Carry one verify execution's identity across its concurrent judge calls."""
    token = _VERIFIER_INVOCATION_ID.set(invocation_id)
    try:
        yield
    finally:
        _VERIFIER_INVOCATION_ID.reset(token)
