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
"""Shared names for why a rollout failed.

Gym describes failures with free text and component-local labels, so the same failure
arrives at a collector, a log line and a metric under three different names and cannot be
grouped. This module holds the one vocabulary those producers share.

A name says *what kind of thing went wrong* and nothing else. Whether to retry, whether
the request may be replayed, whether a completed result should be masked, and what to tell
a person are all properties of the occurrence, not of the name — they live on the failure
record, on the verify response, and in ``failure_reason`` respectively. Keeping them out is
deliberate: metadata attached to a name goes stale the moment one caller wants to retry
what another caller does not.

Names are low cardinality on purpose. They are safe as a metric label or a span dimension;
``failure_reason`` never is, because it carries occurrence detail and can be unbounded.

This module imports only the standard library so that data tooling can read the vocabulary
without installing any server's requirements.
"""

from __future__ import annotations

import logging
import re


logger = logging.getLogger(__name__)


# ``<domain>_<condition>``. The domain says which layer observed the failure, so a reader
# can tell an unreachable model server from an unreachable sandbox without a second field.
_NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

# An environment may use ``<server>:<kind>`` for something the shared vocabulary should not
# grow a name for. The prefix keeps it groupable without making it look registered.
_NAMESPACED_PATTERN = re.compile(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_]*$")


# --- transport: reaching another process at all ------------------------------------- #
TRANSPORT_UNREACHABLE = "transport_unreachable"
TRANSPORT_PEER_DROP = "transport_peer_drop"
TRANSPORT_TIMEOUT = "transport_timeout"
TRANSPORT_LOCAL_RESOURCE = "transport_local_resource"

# --- agent: the /run call and the harness behind it ---------------------------------- #
# Both are produced today by rollout collection: the agent answered and broke, versus
# something in front of it answered and the rollout never ran.
AGENT_RUN_ERROR = "agent_run_error"
AGENT_REQUEST_FAILED = "agent_request_failed"
AGENT_TIMEOUT = "agent_timeout"
AGENT_COMMAND_NOT_ALLOWED = "agent_command_not_allowed"

# --- verifier and judge --------------------------------------------------------------- #
# ``judge_failed`` is produced today by judge_failsafe and by reverification.
JUDGE_FAILED = "judge_failed"
JUDGE_UNPARSEABLE = "judge_unparseable"
VERIFIER_ERROR = "verifier_error"

# --- provider: sandboxes, browsers, containers ----------------------------------------- #
PROVIDER_UNAVAILABLE = "provider_unavailable"
PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"
PROVIDER_OOM_KILLED = "provider_oom_killed"

# --- resources session ------------------------------------------------------------------ #
SESSION_LOST = "session_lost"
SESSION_EXPIRED = "session_expired"
SESSION_RELEASE_FAILED = "session_release_failed"

# --- cancellation and process lifecycle -------------------------------------------------- #
# ``kill_shaped`` names what rollout collection already treats as unstorable: a SIGTERM, a
# dead Ray actor, an OOM outside the rollout's own process.
CANCELLED = "cancelled"
KILL_SHAPED = "kill_shaped"
SHUTDOWN = "shutdown"

# --- persistence and cohorts --------------------------------------------------------------- #
PERSISTENCE_FAILED = "persistence_failed"
COHORT_INCOMPLETE = "cohort_incomplete"


FAILURE_KINDS: frozenset[str] = frozenset(
    {
        TRANSPORT_UNREACHABLE,
        TRANSPORT_PEER_DROP,
        TRANSPORT_TIMEOUT,
        TRANSPORT_LOCAL_RESOURCE,
        AGENT_RUN_ERROR,
        AGENT_REQUEST_FAILED,
        AGENT_TIMEOUT,
        AGENT_COMMAND_NOT_ALLOWED,
        JUDGE_FAILED,
        JUDGE_UNPARSEABLE,
        VERIFIER_ERROR,
        PROVIDER_UNAVAILABLE,
        PROVIDER_QUOTA_EXHAUSTED,
        PROVIDER_OOM_KILLED,
        SESSION_LOST,
        SESSION_EXPIRED,
        SESSION_RELEASE_FAILED,
        CANCELLED,
        KILL_SHAPED,
        SHUTDOWN,
        PERSISTENCE_FAILED,
        COHORT_INCOMPLETE,
    }
)


def is_registered(name: str) -> bool:
    """Whether ``name`` is part of the shared vocabulary."""
    return name in FAILURE_KINDS


def is_namespaced(name: str) -> bool:
    """Whether ``name`` is an environment's own ``<server>:<kind>`` extension."""
    return bool(_NAMESPACED_PATTERN.match(name))


def validate_failure_kind(name: str | None) -> str | None:
    """Return ``name`` unchanged, warning once about anything unregistered.

    Producers call this at their boundary. It warns rather than raises so a component that
    still emits an old label stays visible during migration — rejecting it outright would
    replace an observable wrong name with an invisible dropped failure, which is worse than
    the problem this vocabulary exists to fix.
    """
    if name is None or is_registered(name) or is_namespaced(name):
        return name
    logger.warning(
        "unregistered failure_kind %r; register it in nemo_gym/failure_kinds.py or namespace it as '<server>:%s'",
        name,
        name,
    )
    return name
