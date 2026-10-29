# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Read and build the correlation prefix of a Gym request path.

A Gym component that hands a model URL to a client puts the request's identity in the path::

    /ng-rollout/<capture key>[/ng-agent-instance/<name>][/training-token-capture][/ng-trace/<traceparent>]/v1/...

The segments appear in that fixed order. A path may also carry only ``/ng-trace/<traceparent>``.
``resolve`` reads the prefix and ``encode`` builds it, so no other code parses or builds it by hand.

An agent instance name is ``<agent kind>[.<n>]``, for example ``assistant``, ``judge`` or ``debater.1``.
The name never equals a segment literal or ``v1``, so a missing name cannot swallow the next segment.

Each request is resolved once, by the outermost Gym middleware, which shares the result through
``scope["state"]``; inner middleware reads it there instead of parsing the path again.
Headers are not read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from nemo_gym.config_types import ROLLOUT_PATH_PREFIX, TOKEN_CAPTURE_PATH_SEGMENT, TRACE_CONTEXT_PATH_SEGMENT


AGENT_INSTANCE_PATH_SEGMENT = "ng-agent-instance"
AGENT_INSTANCE_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*(?:\.[a-z0-9_-]+)?$")
# A name equal to one of these would read as the segment that should have followed it.
RESERVED_AGENT_INSTANCE_NAMES = frozenset(
    {ROLLOUT_PATH_PREFIX, AGENT_INSTANCE_PATH_SEGMENT, TOKEN_CAPTURE_PATH_SEGMENT, TRACE_CONTEXT_PATH_SEGMENT, "v1"}
)
# A W3C ``traceparent``, version 00, the only form the propagator writes.
TRACEPARENT_PATTERN = re.compile(r"^00-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$")

CORRELATION_STATE_KEY = "nemo_gym.correlation"
_SEGMENT_LITERALS = frozenset(
    {ROLLOUT_PATH_PREFIX, AGENT_INSTANCE_PATH_SEGMENT, TOKEN_CAPTURE_PATH_SEGMENT, TRACE_CONTEXT_PATH_SEGMENT}
)


class CorrelationPathError(ValueError):
    """A request path carries a malformed correlation prefix. Servers answer it with 400."""

    def __init__(self, reason: str, path: str) -> None:
        super().__init__(f"{reason}: {path!r}")
        self.reason = reason
        self.path = path


def validate_agent_instance(name: str) -> str:
    """Return ``name`` if it is a valid agent instance name; raise ``ValueError`` otherwise."""
    if not isinstance(name, str) or not AGENT_INSTANCE_PATTERN.match(name) or name in RESERVED_AGENT_INSTANCE_NAMES:
        raise ValueError(
            f"invalid agent instance name {name!r}: expected <agent kind>[.<n>] in lowercase letters, digits, "
            f"'_' and '-', starting with a letter, and none of {sorted(RESERVED_AGENT_INSTANCE_NAMES)}"
        )
    return name


def agent_kind_of(agent_instance: str) -> str:
    """Return the agent kind of an agent instance name: the part before the dot."""
    return agent_instance.partition(".")[0]


@dataclass(frozen=True)
class CorrelationContext:
    """The identity a request path carries.

    ``capture_key`` is ``None`` for a trace-only path. ``agent_instance`` is ``None`` when the path names
    no agent instance, as in a single-agent episode.
    """

    capture_key: str | None = None
    agent_instance: str | None = None
    token_capture: bool = False
    trace_parent: str | None = None

    def __post_init__(self) -> None:
        if self.capture_key is not None and (not self.capture_key or "/" in self.capture_key):
            raise ValueError(f"invalid capture key {self.capture_key!r}")
        if self.capture_key is None and (self.agent_instance is not None or self.token_capture):
            raise ValueError("an agent instance or the training tag requires a capture key")
        if self.agent_instance is not None:
            validate_agent_instance(self.agent_instance)
        if self.trace_parent is not None and not TRACEPARENT_PATTERN.match(self.trace_parent):
            raise ValueError(f"invalid traceparent {self.trace_parent!r}")

    @property
    def agent_kind(self) -> str | None:
        return agent_kind_of(self.agent_instance) if self.agent_instance is not None else None


def encode(context: CorrelationContext) -> str:
    """Build the path prefix for ``context``; ``""`` when it carries nothing."""
    prefix = ""
    if context.capture_key is not None:
        prefix = f"/{ROLLOUT_PATH_PREFIX}/{context.capture_key}"
        if context.agent_instance is not None:
            prefix += f"/{AGENT_INSTANCE_PATH_SEGMENT}/{context.agent_instance}"
        if context.token_capture:
            prefix += f"/{TOKEN_CAPTURE_PATH_SEGMENT}"
    if context.trace_parent is not None:
        prefix += f"/{TRACE_CONTEXT_PATH_SEGMENT}/{context.trace_parent}"
    return prefix


def parse_path(path: str, *, allow_empty_rest: bool = False) -> tuple[CorrelationContext, str] | None:
    """Split ``path`` into the context its prefix carries and the remaining path.

    Return ``None`` when ``path`` starts with no correlation segment, or, unless ``allow_empty_rest``,
    when nothing follows the prefix. Raise ``CorrelationPathError`` for a malformed prefix:
    segments out of order or repeated, a missing value, an invalid agent instance name or traceparent.
    """
    if not path.startswith("/ng-"):
        # Fast path: almost no request carries a prefix, and every segment that can start one begins with ``ng-``.
        return None
    segments = path.split("/")
    # segments[0] is the empty string before the leading slash.
    index = 1
    capture_key = agent_instance = trace_parent = None
    token_capture = False

    def value_after(literal: str) -> str:
        if index + 1 >= len(segments) or not segments[index + 1]:
            raise CorrelationPathError(f"{literal} segment without a value", path)
        return segments[index + 1]

    if segments[index] == ROLLOUT_PATH_PREFIX:
        capture_key = value_after(ROLLOUT_PATH_PREFIX)
        index += 2
        if index < len(segments) and segments[index] == AGENT_INSTANCE_PATH_SEGMENT:
            agent_instance = value_after(AGENT_INSTANCE_PATH_SEGMENT)
            index += 2
        if index < len(segments) and segments[index] == TOKEN_CAPTURE_PATH_SEGMENT:
            token_capture = True
            index += 1
    elif segments[index] == AGENT_INSTANCE_PATH_SEGMENT:
        raise CorrelationPathError(f"{AGENT_INSTANCE_PATH_SEGMENT} segment without a preceding capture key", path)
    if index < len(segments) and segments[index] == TRACE_CONTEXT_PATH_SEGMENT:
        trace_parent = value_after(TRACE_CONTEXT_PATH_SEGMENT)
        index += 2
    if index == 1:
        # A path such as /ng-other/... carries no correlation prefix.
        return None
    if index < len(segments) and segments[index] in _SEGMENT_LITERALS:
        raise CorrelationPathError(f"{segments[index]} segment out of order or repeated", path)
    rest = "/" + "/".join(segments[index:]) if index < len(segments) else ""
    if not rest and not allow_empty_rest:
        return None
    try:
        context = CorrelationContext(
            capture_key=capture_key,
            agent_instance=agent_instance,
            token_capture=token_capture,
            trace_parent=trace_parent,
        )
    except ValueError as error:
        raise CorrelationPathError(str(error), path) from error
    return context, rest


@dataclass(frozen=True)
class _Resolved:
    path: str
    result: tuple[CorrelationContext, str] | None


def resolve(scope: dict[str, Any]) -> tuple[CorrelationContext, str] | None:
    """Return the context a request's path carries and the remaining path, or ``None`` without a prefix.

    Resolve once per request: the result is stored in ``scope["state"]`` for the path it was read from,
    and a later call for the same path returns it. Raise ``CorrelationPathError`` for a malformed prefix.
    """
    path = scope.get("path", "") if scope.get("type") == "http" else ""
    state = scope.setdefault("state", {})
    stored = state.get(CORRELATION_STATE_KEY)
    if isinstance(stored, _Resolved) and stored.path == path:
        return stored.result
    result = parse_path(path)
    state[CORRELATION_STATE_KEY] = _Resolved(path, result)
    return result


def remember(scope: dict[str, Any], result: tuple[CorrelationContext, str] | None) -> None:
    """Store ``result`` as the resolution of the scope's current path.

    A middleware that rewrites the path after resolving it calls this, so inner middleware reading the
    rewritten path finds the context without parsing it again.
    """
    scope.setdefault("state", {})[CORRELATION_STATE_KEY] = _Resolved(scope.get("path", ""), result)
