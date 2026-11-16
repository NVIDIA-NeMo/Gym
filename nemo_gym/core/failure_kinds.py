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

Classification configuration uses Pydantic, but this module imports no server code.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from enum import StrEnum
from functools import cache
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


logger = logging.getLogger(__name__)


# The single execution-stage vocabulary for failure events, sandbox observations
# and stage-specific outcome policy. Seed prepares the task; preparation done for
# verification belongs to verification. Preserve the originating stage on forwarding.
# Where an episode failed is independent of its kind and retry policy.
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
# A policy response timeout whose every attempt ended while the model server had requests waiting in its queue:
# the server's capacity, not the model, held the request back.
MODEL_SERVER_QUEUE_TIMEOUT = "model_server_queue_timeout"
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
# Verification was skipped by configuration (a smoke mode): a masked result of this kind is complete and is never
# re-judged or rerun; the run is UNVERIFIED by construction.
JUDGE_SKIPPED = "judge_skipped"

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
# Command outcome fields are absent/contradictory, the client wait expired without a final outcome,
# or the PTY ended before command completion. None establishes remote process termination.
PROVIDER_COMMAND_OUTCOME_UNKNOWN = "provider_command_outcome_unknown"
# Failed status reads exhausted the recovery budget for an already-submitted command.
PROVIDER_STATUS_UNAVAILABLE = "provider_status_unavailable"
# A transport failure, unavailable-backend HTTP response or structured endpoint-unavailability report.
PROVIDER_UNAVAILABLE = "provider_unavailable"
PROVIDER_QUOTA_EXHAUSTED = "provider_quota_exhausted"
# Authoritative current-workload OOM evidence. OpenSandbox inspect must identify the
# current user container; an unattributed lifecycle OOMKilled reason alone is insufficient.
# Exit 137, message text, historical container state and sidecar OOMs are insufficient.
PROVIDER_OOM_KILLED = "provider_oom_killed"
# The provider answered HTTP 401 or 403: the credentials were refused (a sandbox provider's, or the grading API's).
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
# A benchmark's data preparation (its script, its dependencies or its isolated environment) failed; the next
# segment would fail the same way, so a job chain stops instead of retrying it.
ENVIRONMENT_PREPARE_FAILED = "environment_prepare_failed"
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


PROVIDER_FAILURE_KINDS: frozenset[str] = frozenset(
    {
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
    }
)
"""Registered provider observations; membership does not assign responsibility."""


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
        MODEL_SERVER_QUEUE_TIMEOUT,
        MODEL_CONTEXT_LIMIT,
        JUDGE_FAILED,
        JUDGE_UNPARSEABLE,
        VERIFIER_ERROR,
        JUDGE_SKIPPED,
        *PROVIDER_FAILURE_KINDS,
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
        ENVIRONMENT_PREPARE_FAILED,
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
GRADING_WITHOUT_VERDICT = "grading_without_verdict"
GATE_DERIVED_KINDS: frozenset[str] = frozenset(
    {
        GRADING_WITHOUT_VERDICT,
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


# --- shared classification: interpretation, not observed failure facts ------------------ #
class FailureClass(StrEnum):
    """Configurable responsibility for reporting; no class implies an action or health verdict."""

    MODEL = "model"
    EVALUATION = "evaluation"
    UNKNOWN = "unknown"


# Each default contains its class and optional evaluation-side reporting subcategory.
# Keep these together so historical model/server views cannot become competing classifiers.
_DEFAULT_CLASSIFICATION: dict[str, tuple[FailureClass, str | None]] = {
    kind: (FailureClass.EVALUATION, None) for kind in FAILURE_KINDS | GATE_DERIVED_KINDS
}
_DEFAULT_CLASSIFICATION.update(
    dict.fromkeys(
        (
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
        ),
        (FailureClass.MODEL, None),
    )
)
_DEFAULT_CLASSIFICATION.update(
    dict.fromkeys(
        (
            SERVER_START_FAILED,
            MODEL_TRANSPORT_ERROR,
            MODEL_SERVER_ERROR,
            MODEL_SERVER_QUEUE_TIMEOUT,
            AGENT_TIMEOUT_INFRASTRUCTURE,
            *PROVIDER_FAILURE_KINDS,
        ),
        (FailureClass.EVALUATION, "server"),
    )
)
# OOM requires admitted execution/ownership context; its kind alone cannot assign responsibility.
_DEFAULT_CLASSIFICATION[PROVIDER_OOM_KILLED] = (FailureClass.UNKNOWN, "server")
MODEL_FAULT_KINDS: frozenset[str] = frozenset(
    kind for kind, (classification, _) in _DEFAULT_CLASSIFICATION.items() if classification == FailureClass.MODEL
)
SERVER_FAULT_KINDS: frozenset[str] = frozenset(
    kind
    for kind, (classification, category) in _DEFAULT_CLASSIFICATION.items()
    if classification == FailureClass.EVALUATION and category == "server"
)


def _validate_classification_kinds[T](rules: dict[str, T]) -> dict[str, T]:
    invalid = {
        kind for kind in rules.keys() - FAILURE_KINDS - GATE_DERIVED_KINDS - {"unknown"} if not is_namespaced(kind)
    }
    if invalid:
        raise ValueError(f"Unregistered failure kinds: {sorted(invalid)}")
    return rules


class StageClassificationConfig(BaseModel):
    """Classification overrides for an established originating stage."""

    model_config = ConfigDict(extra="forbid")
    by_failure_kind: dict[str, FailureClass] = Field(default_factory=dict)
    default_class: FailureClass | None = None

    @field_validator("by_failure_kind")
    @classmethod
    def validate_kinds(cls, value: dict[str, FailureClass]) -> dict[str, FailureClass]:
        return _validate_classification_kinds(value)


def _default_stage_classification() -> dict[FailureStage, StageClassificationConfig]:
    return {
        "agent": StageClassificationConfig(by_failure_kind={PROVIDER_OOM_KILLED: FailureClass.MODEL}),
        "seed": StageClassificationConfig(by_failure_kind={PROVIDER_OOM_KILLED: FailureClass.EVALUATION}),
        "verification": StageClassificationConfig(default_class=FailureClass.EVALUATION),
    }


class FailureClassificationConfig(BaseModel):
    """One shared stage/kind mapping and explicit HTTP evidence rules.

    Class overrides by HTTP status take precedence, followed by stage kind/default,
    HTTP-derived evidence, kind, and the unknown fallback. Stage rules require context
    admitted by the caller; a provider cannot establish agent ownership on its own.
    Partial overrides retain unspecified defaults. HTTP-derived labels are reporting
    projections only: never write them back into observed failure events.
    """

    model_config = ConfigDict(extra="forbid")
    by_failure_kind: dict[str, FailureClass] = Field(
        default_factory=lambda: {kind: value[0] for kind, value in _DEFAULT_CLASSIFICATION.items()}
    )
    by_stage: dict[FailureStage, StageClassificationConfig] = Field(default_factory=_default_stage_classification)
    by_status_code: dict[Annotated[int, Field(ge=400, le=599)], FailureClass] = Field(default_factory=dict)
    default_class: FailureClass = FailureClass.UNKNOWN
    reporting_kind_by_status_code: dict[Annotated[int, Field(ge=400, le=599)], str | None] = Field(
        default_factory=lambda: dict.fromkeys(range(500, 600), MODEL_SERVER_ERROR)
    )
    non_policy_reporting_kind_by_status_code: dict[Annotated[int, Field(ge=400, le=599)], str | None] = Field(
        default_factory=lambda: dict.fromkeys(range(400, 500), NON_POLICY_MODEL_REJECTION)
    )

    @field_validator(
        "by_failure_kind",
        "by_status_code",
        "reporting_kind_by_status_code",
        "non_policy_reporting_kind_by_status_code",
    )
    @classmethod
    def merge_rules(cls, value: dict, info: ValidationInfo) -> dict:
        return cls.model_fields[info.field_name].get_default(call_default_factory=True) | value

    @field_validator("by_failure_kind")
    @classmethod
    def validate_kinds(cls, value: dict[str, FailureClass]) -> dict[str, FailureClass]:
        return _validate_classification_kinds(value)

    @field_validator("reporting_kind_by_status_code", "non_policy_reporting_kind_by_status_code")
    @classmethod
    def validate_reporting_kinds(cls, value: dict[int, str | None]) -> dict[int, str | None]:
        _validate_classification_kinds({kind: None for kind in value.values() if kind is not None})
        return value

    @field_validator("by_stage", mode="before")
    @classmethod
    def merge_stages(cls, value: object) -> object:
        return _merge_stage_overrides(_default_stage_classification(), value, dict_fields=("by_failure_kind",))


def _merge_stage_overrides(
    defaults: dict[str, BaseModel],
    value: object,
    *,
    dict_fields: tuple[str, ...],
) -> object:
    """Merge partial stage configurations before their concrete models validate them."""
    if not isinstance(value, dict):
        return value
    merged = {stage: rules.model_dump() for stage, rules in defaults.items()}
    for stage, override in value.items():
        if isinstance(override, BaseModel):
            override = override.model_dump(exclude_unset=True)
        if not isinstance(override, dict):
            merged[stage] = override
            continue
        rules = merged.get(stage, {})
        for name, rule in override.items():
            rules[name] = rules.get(name, {}) | rule if name in dict_fields and isinstance(rule, dict) else rule
        merged[stage] = rules
    return merged


@cache
def _default_classification_config() -> FailureClassificationConfig:
    """Private read-only default for classification; caller-created configs remain independent."""
    return FailureClassificationConfig()


def reporting_failure_kind(
    kind: str | None,
    http_status: object,
    *,
    policy: bool = True,
    config: FailureClassificationConfig | None = None,
) -> str | None:
    """Project observed HTTP evidence to a reporting label without changing the event."""
    if type(http_status) is not int:
        return kind
    config = config if config is not None else _default_classification_config()
    if not policy and http_status in config.non_policy_reporting_kind_by_status_code:
        return config.non_policy_reporting_kind_by_status_code[http_status] or kind
    return config.reporting_kind_by_status_code.get(http_status) or kind


def classify_failure(
    *,
    failure_kind: str | None,
    stage: FailureStage | None = None,
    http_status: object = None,
    policy: bool = True,
    config: FailureClassificationConfig | None = None,
) -> FailureClass:
    """Interpret only supplied evidence; never walk causes or mutate the observed kind/stage."""
    config = config if config is not None else _default_classification_config()
    if type(http_status) is int and http_status in config.by_status_code:
        return config.by_status_code[http_status]
    stage_rules = config.by_stage.get(stage)
    if stage_rules is not None:
        if failure_kind in stage_rules.by_failure_kind:
            return stage_rules.by_failure_kind[failure_kind]
        if stage_rules.default_class is not None:
            return stage_rules.default_class
    reporting_kind = reporting_failure_kind(failure_kind, http_status, policy=policy, config=config)
    return config.by_failure_kind.get(reporting_kind, config.default_class)


def is_server_failure(kind: str | None, *, config: FailureClassificationConfig | None = None) -> bool:
    """Server subcategory for an already established evaluation failure, never a class decision.

    Callers check the applied class first. In particular, OOM can be model-side or
    evaluation-side depending on its originating stage and admitted ownership.
    """
    return _DEFAULT_CLASSIFICATION.get(kind, (FailureClass.UNKNOWN, None))[1] == "server" or (
        kind == MODEL_POLICY_RESPONSE_TIMEOUT
        and classify_failure(failure_kind=kind, config=config) == FailureClass.EVALUATION
    )


def is_server_status(status: object) -> bool:
    """Whether a recorded HTTP status is an integer in the 5xx class."""
    return type(status) is int and 500 <= status <= 599


def is_evaluation_side(name: str, *, config: FailureClassificationConfig | None = None) -> bool:
    """Whether a registered kind has a shared evaluation-side default without extra context."""
    return is_registered(name) and classify_failure(failure_kind=name, config=config) == FailureClass.EVALUATION


def is_gate_derived(name: str) -> bool:
    """Whether ``name`` is a class the acceptance gate derives from saved rows (``GATE_DERIVED_KINDS``)."""
    return name in GATE_DERIVED_KINDS


QUOTA_RESET_REGEX = r"resets at (\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC)"
_QUOTA_RESET_PATTERN = re.compile(QUOTA_RESET_REGEX)


def quota_reset_text(reason: object) -> str | None:
    """The UTC time a ``provider_quota_exhausted`` reason names (``resets at YYYY-MM-DD HH:MM UTC``), else None."""
    match = _QUOTA_RESET_PATTERN.search(reason) if isinstance(reason, str) else None
    return match.group(1) if match else None


def quota_reset_unix(reason: object) -> float | None:
    """``quota_reset_text`` as a unix time, else None."""
    text = quota_reset_text(reason)
    if text is None:
        return None
    return datetime.strptime(text, "%Y-%m-%d %H:%M UTC").replace(tzinfo=timezone.utc).timestamp()


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
