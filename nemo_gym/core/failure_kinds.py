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
"""Shared failure names and episode execution stages.

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
from typing import Literal


logger = logging.getLogger(__name__)


# Where an episode failed, independently of its kind and retry policy.
# Admission precedes seed. Collector-observed transport failures leave the stage
# unset: delivery evidence and failure_kind describe the observation without
# claiming to know where execution stopped inside the Environment Server.
FailureStage = Literal["admission", "seed", "agent", "verification", "cleanup"]


# ``<domain>_<condition>``. The domain says which layer observed the failure, so a reader
# can tell an unreachable model server from an unreachable sandbox without a second field.
# ``fullmatch`` below, not ``match``: ``$`` also matches before a final newline, which
# would admit "judge_failed\n" as a second, silently different label for one failure.
_NAME_PATTERN = re.compile(r"[a-z][a-z0-9_]*")

# An environment may use ``<server>:<kind>`` for something the shared vocabulary should not
# grow a name for. The prefix keeps it groupable without making it look registered.
_NAMESPACED_PATTERN = re.compile(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*")


# --- transport: reaching another process at all ------------------------------------- #
TRANSPORT_UNREACHABLE = "transport_unreachable"
TRANSPORT_PEER_DROP = "transport_peer_drop"
TRANSPORT_TIMEOUT = "transport_timeout"
TRANSPORT_LOCAL_RESOURCE = "transport_local_resource"

# --- agent: the /run call and the harness behind it ---------------------------------- #
# Both are produced today by rollout collection: the agent answered and broke, versus
# something in front of it answered and the rollout never ran.
AGENT_RUN_ERROR = "agent_run_error"
# An agent server route failed; its reply is its cause's reply.
AGENT_ROUTE_FAILED = "agent_route_failed"
# A harness supervisor reported that termination was not confirmed (a negative cleanup receipt).
HARNESS_TERMINATION_UNCONFIRMED = "harness_termination_unconfirmed"
# An environment server handled an episode failure that names no more specific kind.
EPISODE_FAILED = "environment_episode_failed"
# Collection could not dispatch an episode; routing metadata remains separate.
COLLECTION_DISPATCH_FAILED = "collection_dispatch_failed"
AGENT_REQUEST_FAILED = "agent_request_failed"
AGENT_TIMEOUT = "agent_timeout"
AGENT_COMMAND_NOT_ALLOWED = "agent_command_not_allowed"
AGENT_LIMIT = "agent_limit"
AGENT_INVALID_OUTPUT = "agent_invalid_output"

# --- model: requests made to a model endpoint ---------------------------------------- #
# Preserve the upstream HTTP status separately; it is not a retry policy.
MODEL_REQUEST_ERROR = "model_request_error"
MODEL_TRANSPORT_ERROR = "model_transport_error"
# A policy-model operation whose observed attempts all timed out after a confirmed upload.
MODEL_POLICY_RESPONSE_TIMEOUT = "model_policy_response_timeout"
# Explicit evidence that the model's context window was exceeded, not an HTTP status alone.
MODEL_CONTEXT_LIMIT = "model_context_limit"

# --- environment: one episode attempt ------------------------------------------------ #
ENVIRONMENT_SERVER_FAILED = "environment_server_failed"
ENVIRONMENT_PROTOCOL_VIOLATION = "environment_protocol_violation"

# --- verifier and judge --------------------------------------------------------------- #
# ``judge_failed`` is produced today by judge_failsafe and by reverification.
JUDGE_FAILED = "judge_failed"
JUDGE_UNPARSEABLE = "judge_unparseable"
VERIFIER_ERROR = "verifier_error"

# --- provider: sandboxes, browsers, containers ----------------------------------------- #
# The kind describes provider evidence; the event's boundary identifies its caller-facing API.
# The create/reconnect readiness deadline expired without a confirmed Pending observation.
PROVIDER_CREATE_TIMEOUT = "provider_create_timeout"
# An attempt-owned sandbox was observed Pending when its create deadline expired.
PROVIDER_CREATE_PENDING = "provider_create_pending"
# A configured readiness probe failed, or the SDK reported the sandbox unhealthy.
PROVIDER_NOT_READY = "provider_not_ready"
# Sandbox setup or a required runtime check failed.
PROVIDER_SETUP_FAILED = "provider_setup_failed"
# An SDK internal error without a recognized transport cause.
PROVIDER_INTERNAL_ERROR = "provider_internal_error"
# PTY refusal, availability and protocol evidence do not assert whole-sandbox death.
# The PTY session is missing, detached, closed or has lost its connection.
PROVIDER_PTY_UNAVAILABLE = "provider_pty_unavailable"
# Attachment was refused because another client holds the PTY session.
PROVIDER_PTY_BUSY = "provider_pty_busy"
# An SDK response or PTY frame violates the required protocol structure.
PROVIDER_PROTOCOL_ERROR = "provider_protocol_error"
# A provider request failed without a more specific classification.
PROVIDER_REQUEST_FAILED = "provider_request_failed"
# A provider request or file transfer timed out; remote completion is unknown.
PROVIDER_IO_TIMEOUT = "provider_io_timeout"
# Gym observed its own command-session deadline expire; cleanup is handled separately.
PROVIDER_COMMAND_TIMEOUT = "provider_command_timeout"
# Command outcome fields are absent/contradictory, or the PTY ended before command completion.
PROVIDER_COMMAND_OUTCOME_UNKNOWN = "provider_command_outcome_unknown"
# The budget for obtaining an already-submitted command's status expired.
PROVIDER_STATUS_UNAVAILABLE = "provider_status_unavailable"
# A transport failure or unavailable-backend HTTP response.
PROVIDER_UNAVAILABLE = "provider_unavailable"
PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"
# Authoritative current-workload OOM evidence. OpenSandbox requires an explicit terminal
# OOMKilled lifecycle reason or current user-container inspect evidence, retaining its source.
# Exit 137, message text, historical container state and sidecar OOMs are insufficient.
PROVIDER_OOM_KILLED = "provider_oom_killed"
# The provider answered HTTP 401 or 403: the sandbox credentials were refused.
PROVIDER_CREDENTIALS_REJECTED = "provider_credentials_rejected"

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
# A Gym server process exited before it was ready, for example because its port was already in use.
SERVER_START_FAILED = "server_start_failed"

# --- persistence and cohorts --------------------------------------------------------------- #
PERSISTENCE_FAILED = "persistence_failed"
COHORT_INCOMPLETE = "cohort_incomplete"

# --- run startup: before rollout collection runs anything ------------------------------------ #
# The server supervisor waited longer than ``server_spinup_timeout_seconds`` for the Gym servers to become ready.
SERVER_SPINUP_TIMEOUT = "server_spinup_timeout"
# The driver step of an orchestrated job could not install Gym (the uv installer, the checkout or the package install).
DRIVER_INSTALL_FAILED = "driver_install_failed"
# Collection refused to resume saved rollouts whose input and configuration identity does not verify.
COLLECTION_RESUME_REFUSED = "collection_resume_refused"
# A rollout was in flight when its allocation ended (a walltime cut, a preemption, a node failure or a driver that
# ended) for a second or later time. The first such end re-runs for free; each later one spends an attempt. The
# allocation's limits are the evaluation's setup, not the model's outcome.
ROLLOUT_ALLOCATION_ENDED = "rollout_allocation_ended"

# --- producer labels older than the layer-prefix rule ------------------------------------------ #
# Shipped producers write these today: the Harbor agent's failed trial, and the terminal-bench sidecar class of an
# agent or sandbox failure. They are registered so saved rows have an owner; new producers use a prefixed name.
HARBOR_FAILED = "harbor_failed"
INFRASTRUCTURE_ERROR = "infrastructure_error"


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
        ENVIRONMENT_SERVER_FAILED,
        ENVIRONMENT_PROTOCOL_VIOLATION,
        AGENT_LIMIT,
        AGENT_INVALID_OUTPUT,
        MODEL_REQUEST_ERROR,
        MODEL_TRANSPORT_ERROR,
        MODEL_POLICY_RESPONSE_TIMEOUT,
        MODEL_CONTEXT_LIMIT,
        JUDGE_FAILED,
        JUDGE_UNPARSEABLE,
        VERIFIER_ERROR,
        PROVIDER_CREATE_TIMEOUT,
        PROVIDER_CREATE_PENDING,
        PROVIDER_NOT_READY,
        PROVIDER_SETUP_FAILED,
        PROVIDER_INTERNAL_ERROR,
        PROVIDER_PTY_UNAVAILABLE,
        PROVIDER_PTY_BUSY,
        PROVIDER_PROTOCOL_ERROR,
        PROVIDER_REQUEST_FAILED,
        PROVIDER_IO_TIMEOUT,
        PROVIDER_COMMAND_OUTCOME_UNKNOWN,
        PROVIDER_COMMAND_TIMEOUT,
        PROVIDER_STATUS_UNAVAILABLE,
        PROVIDER_UNAVAILABLE,
        PROVIDER_QUOTA_EXHAUSTED,
        PROVIDER_OOM_KILLED,
        PROVIDER_CREDENTIALS_REJECTED,
        SESSION_LOST,
        SESSION_EXPIRED,
        SESSION_RELEASE_FAILED,
        CANCELLED,
        KILL_SHAPED,
        SHUTDOWN,
        SERVER_START_FAILED,
        PERSISTENCE_FAILED,
        COHORT_INCOMPLETE,
        SERVER_SPINUP_TIMEOUT,
        DRIVER_INSTALL_FAILED,
        COLLECTION_RESUME_REFUSED,
        ROLLOUT_ALLOCATION_ENDED,
        AGENT_ROUTE_FAILED,
        HARNESS_TERMINATION_UNCONFIRMED,
        EPISODE_FAILED,
        COLLECTION_DISPATCH_FAILED,
        HARBOR_FAILED,
        INFRASTRUCTURE_ERROR,
    }
)


# --- derived by the acceptance gate from saved rows; no producer emits them ------------- #
# The trial gate names these from what a saved row shows, never from a producer's report: a scored-zero answer the
# benchmark's rule scored without a grade (cut off with a cutoff signal, empty, or malformed), a model server's 5xx
# reply whatever kind the failure names, another model's 4xx rejection, and the policy's request the model server
# rejected because its output budget alone exceeds the context window or because a parameter carries a value the
# deployment does not support (setups the deployment cannot serve, not the model's outcome). They are registered here
# so their names are shared, and kept out of ``FAILURE_KINDS`` so no producer can emit one (``require_failure_kind``
# refuses them).
TRUNCATED_ANSWER = "truncated_answer"
EMPTY_ANSWER = "empty_answer"
MALFORMED_ANSWER = "malformed_answer"
MODEL_SERVER_ERROR = "model_server_error"
NON_POLICY_MODEL_REJECTION = "non_policy_model_rejection"
# An ``agent_timeout`` whose window held a recorded server-side fault (``is_infrastructure_agent_timeout`` in
# ``nemo_gym.rollouts.rollout_recovery``): the deadline was spent on our servers, so it is not the model's result.
AGENT_TIMEOUT_INFRASTRUCTURE = "agent_timeout_infrastructure"
INVALID_OUTPUT_BUDGET = "invalid_output_budget"
INVALID_REQUEST_PARAMETER = "invalid_request_parameter"
GATE_DERIVED_KINDS: frozenset[str] = frozenset(
    {
        TRUNCATED_ANSWER,
        EMPTY_ANSWER,
        MALFORMED_ANSWER,
        MODEL_SERVER_ERROR,
        NON_POLICY_MODEL_REJECTION,
        AGENT_TIMEOUT_INFRASTRUCTURE,
        INVALID_OUTPUT_BUDGET,
        INVALID_REQUEST_PARAMETER,
    }
)


# --- fault classes: the shared default for evaluation and RL ---------------------------- #
# A model fault is the evaluated model's own outcome under the benchmark protocol: its request rejected or timed out
# (``MODEL_POLICY_RESPONSE_TIMEOUT``: a deployment's slowness is on the model's side by default; the acceptance gate's
# ``policy_timeout_counts_as_model`` setting can move it to the server class), its context limit, its agent's limit, timeout, invalid output or a command the
# harness refused (the model issued it), or an answer it left cut off, empty or malformed. The evaluation gate counts
# these by kind and never fails health on them; RL treats them as the policy's own failures. A server fault is a
# server that did not start, a model transport error, a model server's 5xx reply or an agent timeout whose window held
# such a fault (``AGENT_TIMEOUT_INFRASTRUCTURE``): evaluation-side. Any other kind is
# evaluation-side by its kind alone. A recorded HTTP status refines the class: the gate keys it on the status first.
MODEL_FAULT_KINDS: frozenset[str] = frozenset(
    {
        MODEL_REQUEST_ERROR,
        MODEL_POLICY_RESPONSE_TIMEOUT,
        MODEL_CONTEXT_LIMIT,
        AGENT_TIMEOUT,
        AGENT_COMMAND_NOT_ALLOWED,
        AGENT_LIMIT,
        AGENT_INVALID_OUTPUT,
        TRUNCATED_ANSWER,
        EMPTY_ANSWER,
        MALFORMED_ANSWER,
    }
)
SERVER_FAULT_KINDS: frozenset[str] = frozenset(
    {SERVER_START_FAILED, MODEL_TRANSPORT_ERROR, MODEL_SERVER_ERROR, AGENT_TIMEOUT_INFRASTRUCTURE}
)


def is_server_status(status: object) -> bool:
    """Whether a recorded HTTP status is a server's own failure: an integer in the 5xx class."""
    return type(status) is int and 500 <= status <= 599


def is_evaluation_side(name: str) -> bool:
    """Whether ``name`` is a registered kind whose class is evaluation-side by its kind alone: not a model fault.

    A recorded HTTP status can still refine an occurrence of it (see the fault classes above); this is the kind's own
    class, for a saved row that carries a kind and nothing else.
    """
    return is_registered(name) and name not in MODEL_FAULT_KINDS


def is_gate_derived(name: str) -> bool:
    """Whether ``name`` is a class the acceptance gate derives from saved rows (``GATE_DERIVED_KINDS``)."""
    return name in GATE_DERIVED_KINDS


def is_registered(name: str) -> bool:
    """Whether ``name`` is part of the shared vocabulary."""
    return name in FAILURE_KINDS


JUDGE_FAILURE_KINDS = ("judge_failed", "judge_invalid", JUDGE_UNPARSEABLE)
"""The judge failures whose saved answer is judged again, never regenerated: a failed judge call, a verdict that did not
parse (``judge_unparseable``), and collection's own legacy class for an invalid judge response (``judge_invalid``)."""


def is_namespaced(name: str) -> bool:
    """Whether ``name`` is an environment's own ``<server>:<kind>`` extension."""
    return bool(_NAMESPACED_PATTERN.fullmatch(name))


# Names already reported. A failing component emits the same label on every rollout, so
# warning per call would bury the one line that matters under thousands of copies.
_WARNED_UNKNOWN: set[str] = set()


def validate_failure_kind(name: str | None) -> str | None:
    """Return ``name`` unchanged, warning once per unregistered value.

    Producers call this at their boundary. It warns rather than raises so a component that
    still emits an old label stays visible during migration — rejecting it outright would
    replace an observable wrong name with an invisible dropped failure, which is worse than
    the problem this vocabulary exists to fix.
    """
    if name is None or is_registered(name) or is_namespaced(name):
        return name
    if name not in _WARNED_UNKNOWN:
        _WARNED_UNKNOWN.add(name)
        logger.warning(
            "unregistered failure_kind %r; register it in nemo_gym/core/failure_kinds.py or namespace it as "
            "'<server>:<kind>'. Reported once per name.",
            name,
        )
    return name


def require_failure_kind(name: str | None) -> str:
    """Validate a canonical event kind; legacy outcome envelopes keep their migration validator.

    A gate-derived class (``GATE_DERIVED_KINDS``) is no producer's kind and is refused like any unregistered name.
    """
    if name is not None and is_gate_derived(name):
        raise ValueError(
            f"failure_kind {name!r} is derived by the acceptance gate from saved rows; no producer emits it"
        )
    if name is None or not (is_registered(name) or is_namespaced(name)):
        raise ValueError(f"failure_kind {name!r} is not registered in nemo_gym/core/failure_kinds.py or namespaced")
    return name
