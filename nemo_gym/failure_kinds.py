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

A name says *what kind of thing went wrong*. Whether resume retries it is one fixed rule per
name, in ``FAILURE_RULES``, so no agent or benchmark decides it: a failure in a service the model
does not control is retried, and a failure caused by what the model produced is final and scores 0.
A name that can be either (``EVIDENCE``) is split by where the failure happened, which
``failure_cause_of`` reads from the error itself. An unknown name is final, so the model never
gets a free retry. Whether a result is masked and what to tell a person stay on the verify
response and in ``failure_reason``.

Names are low cardinality on purpose. They are safe as a metric label or a span dimension;
``failure_reason`` never is, because it carries occurrence detail and can be unbounded.

This module imports only the standard library so that data tooling can read the vocabulary
without installing any server's requirements.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator
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
            "unregistered failure_kind %r; register it in nemo_gym/failure_kinds.py or namespace it as "
            "'<server>:<kind>'. Reported once per name.",
            name,
        )
    return name


# --- retry rule ------------------------------------------------------------------------------ #
FailureRule = Literal["retry", "final", "evidence", "unscored"]
RETRY: FailureRule = "retry"
FINAL: FailureRule = "final"
EVIDENCE: FailureRule = "evidence"
# A gap in the benchmark's own data or judge eligibility: nothing to retry and nothing to score.
UNSCORED: FailureRule = "unscored"


FailureCause = Literal["infra", "model"]
INFRA: FailureCause = "infra"
MODEL: FailureCause = "model"

# Gym's exception middleware names the cause of every 500 it returns in this header, so a relay such
# as legacy_agent carries it to rollout collection unchanged.
FAILURE_CAUSE_HEADER = "X-NeMo-Gym-Failure-Cause"


FAILURE_RULES: dict[str, FailureRule] = {
    TRANSPORT_UNREACHABLE: RETRY,
    TRANSPORT_PEER_DROP: RETRY,
    TRANSPORT_TIMEOUT: RETRY,
    TRANSPORT_LOCAL_RESOURCE: RETRY,
    AGENT_RUN_ERROR: EVIDENCE,
    AGENT_REQUEST_FAILED: EVIDENCE,
    AGENT_TIMEOUT: EVIDENCE,
    AGENT_COMMAND_NOT_ALLOWED: EVIDENCE,
    JUDGE_FAILED: RETRY,
    # The judge, a service the model does not control, produced the verdict it could not parse.
    JUDGE_UNPARSEABLE: RETRY,
    VERIFIER_ERROR: EVIDENCE,
    PROVIDER_UNAVAILABLE: RETRY,
    PROVIDER_QUOTA_EXHAUSTED: RETRY,
    PROVIDER_OOM_KILLED: RETRY,
    SESSION_LOST: RETRY,
    SESSION_EXPIRED: RETRY,
    SESSION_RELEASE_FAILED: RETRY,
    CANCELLED: RETRY,
    KILL_SHAPED: RETRY,
    SHUTDOWN: RETRY,
    PERSISTENCE_FAILED: RETRY,
    COHORT_INCOMPLETE: RETRY,
    # Sidecar labels that predate this vocabulary and that Gym's own collector, servers and agents write.
    "environment_server_failed": EVIDENCE,
    "tau2_malformed_tool_call": FINAL,
    "tau2_agent_error": EVIDENCE,
    "remote_agent_error": EVIDENCE,
    "harbor_failed": EVIDENCE,
    "infrastructure_error": EVIDENCE,
    "transient": RETRY,
    "timeout_exceeded": FINAL,
    # GDPval data and judge-eligibility gaps: the same inputs fail the same way, and the model did nothing wrong.
    "reference_missing": UNSCORED,
    "eval_missing": UNSCORED,
    "transport_ineligible": UNSCORED,
}

# 401/402/403: a key, a payment or a route the model does not control; 408/425/429 and 5xx: an overloaded or
# broken service. Any other 4xx answers a request the model shaped, such as output that overflows the context
# window, unless its body reports a spent budget or quota (``_SPENT_BUDGET_PATTERN``).
_INFRA_HTTP_STATUSES = frozenset({401, 402, 403, 408, 425, 429})
# An endpoint that refuses a key whose budget or quota is spent answers with a 4xx (a proxy's 422 "Budget has been
# exceeded", OpenAI's insufficient_quota): the same request succeeds once the budget is restored.
_SPENT_BUDGET_PATTERN = re.compile(
    r"budget has been exceeded|insufficient_quota|exceeded your current quota|quota (?:has been )?exhausted"
    r"|credit balance is too low|max budget",
    re.IGNORECASE,
)
# A key or an authorization the model does not control, as providers word it, sometimes behind a 400 rather than a
# 401 or 403: retried on any endpoint, never the model's 0.
_BROKEN_CREDENTIAL_PATTERN = re.compile(
    r"malformed[_ ]authorization|(?:incorrect|invalid|missing) api[_ ]key|requires approved access",
    re.IGNORECASE,
)

# Matched against every class in an exception's MRO, so subclasses (aiohttp's ClientConnectorError,
# LiteLLM's wrappers of the OpenAI SDK errors) count too. A bare TimeoutError is not listed: it is the
# model's own time budget, unless it was raised by an HTTP or sandbox call (see ``_INFRA_TIMEOUT_MODULES``).
# CancelledError is not listed either: every ``asyncio.timeout`` expiry chains from one.
_INFRA_EXCEPTION_NAMES = frozenset(
    {
        "ClientConnectionError",
        "ClientPayloadError",
        "ConnectionError",
        "ServerTimeoutError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
        "RateLimitError",
        "ServiceUnavailableError",
        "AuthenticationError",
        "PermissionDeniedError",
        "MemoryError",
        "SandboxCreateError",
        "SandboxPtyError",
        "SandboxOperationTimeoutError",
        # The raw SDK errors of sandbox providers, which reach a row without Gym's own wrapper.
        "OpenSandboxCreateError",
        "SandboxApiException",
        "SandboxTimeoutException",
        # Ray lost the worker, the node or the object, not the task's own code.
        "OutOfMemoryError",
        "WorkerCrashedError",
        "NodeDiedError",
        "ActorDiedError",
        "RaySystemError",
        "UnserializableException",
        "ObjectLostError",
    }
)
# A bare TimeoutError raised inside one of these packages is an HTTP or sandbox call that timed out.
_INFRA_TIMEOUT_MODULES = ("aiohttp.", "nemo_gym.sandbox.")
# What an error looks like once it is only text: "ClientConnectorError: ..." or repr "ClientConnectorError(...)".
_EXCEPTION_NAME_PATTERN = re.compile(r"\b([A-Z]\w*(?:Error|Exception|Failure|Expired))\b")
# aiohttp's ClientResponseError as text: "503, message='Service Unavailable', url=...".
_RESPONSE_STATUS_PATTERN = re.compile(r"\b(\d{3}), message=")
# The API routes of Gym's model servers, which the policy and the judge are called on. A 404 there is a model that is
# not deployed or a wrong route: a deployment or config error. Only a call to one of these routes is read this way; a
# sandbox exec 404 is a provider SDK error, not an aiohttp response, and the sandbox rule decides it.
_MODEL_ROUTE_PATTERN = re.compile(r"/v1/(?:responses|chat/completions|completions|messages|embeddings)\b")
_MODEL_ROUTE_404_TEXT_PATTERN = re.compile(
    r"\b404, message=(?:'[^']*'|\"[^\"]*\"), url='[^']*/v1/(?:responses|chat/completions|completions|messages"
    r"|embeddings)\b"
)
# The infra exception classes by their subclass names, which appear in text where the MRO does not.
_INFRA_SUBCLASS_NAMES = frozenset(
    {
        "ClientConnectorError",
        "ClientOSError",
        "ServerDisconnectedError",
        "ServerConnectionError",
        "ClientConnectorDNSError",
        "ConnectionRefusedError",
        "ConnectionResetError",
        "BrokenPipeError",
        "ConnectionTimeoutError",
        "SocketTimeoutError",
        # Sandboxed harnesses (Terminal-Bench): the environment or the harness itself failed before or
        # around the agent's run, or the verifier sandbox missed its start deadline; Terminal-Bench retries
        # these. The agent's own exits (AgentTimeoutError, NonZeroAgentExitCodeError) and a run that leaves no
        # reward file (RewardFileNotFoundError) are the model's.
        "EnvironmentStartTimeoutError",
        "HealthcheckError",
        "AgentSetupTimeoutError",
        "AgentInfrastructureError",
        "AgentRecordError",
        "SeededSessionExpired",
    }
)
# Ray's wrapper of a task's own exception: "RayTaskError(ImportError): ...". The wrapped name decides.
_RAY_TASK_ERROR_PATTERN = re.compile(r"\bRayTaskError\((\w+)\)")
# aiohttp's own messages for a peer that is down, for text that kept the message and lost the class name.
_INFRA_MESSAGE_PATTERN = re.compile(
    r"Cannot connect to host|Server disconnected|Connection reset by peer|Connection refused"
)


def _is_infra_status(status: int) -> bool:
    return status in _INFRA_HTTP_STATUSES or status >= 500


def _exception_chain(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending = [error]
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        pending.extend(getattr(current, "exceptions", ()))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        # ``raise ... from None`` hides the context on purpose, so it is no evidence.
        if current.__context__ is not None and not current.__suppress_context__:
            pending.append(current.__context__)


def _raised_in_infra_module(error: BaseException) -> bool:
    traceback = error.__traceback__
    while traceback is not None:
        if traceback.tb_frame.f_globals.get("__name__", "").startswith(_INFRA_TIMEOUT_MODULES):
            return True
        traceback = traceback.tb_next
    return False


def failure_cause_of(error: BaseException) -> FailureCause:
    """Where a raised failure happened: in a service the model does not control, or in what the model produced.

    The chain is read (``raise ... from``, an unsuppressed ``__context__``, exception groups), so a model-server
    connection error wrapped in an agent's own error is still infra. A cause that an inner Gym server named in
    ``FAILURE_CAUSE_HEADER`` wins over that response's status. A timeout is infra only when an HTTP or sandbox
    call raised it; the agent's own time budget is the model's. Without infra evidence the failure is the model's.
    """
    for current in _exception_chain(error):
        headers = getattr(current, "headers", None)
        upstream = headers.get(FAILURE_CAUSE_HEADER) if hasattr(headers, "get") else None
        if upstream in (INFRA, MODEL):
            return upstream
        status = getattr(current, "status", None)
        if isinstance(status, int) and _is_infra_status(status):
            return INFRA
        if isinstance(status, int) and 400 <= status < 500 and _reports_spent_budget(current):
            return INFRA
        if isinstance(status, int) and 400 <= status < 500 and _reports_broken_credential(current):
            return INFRA
        if status == 404 and _is_model_server_call(current):
            return INFRA
        if any(cls.__name__ in _INFRA_EXCEPTION_NAMES for cls in type(current).__mro__):
            return INFRA
        if isinstance(current, TimeoutError) and _raised_in_infra_module(current):
            return INFRA
    return MODEL


def failure_cause_of_text(text: str) -> FailureCause:
    """``failure_cause_of`` for a failure that is only known as text, such as a relayed error body.

    A ``RayTaskError(X)`` takes the cause of ``X``, the task's own exception, not of the wrapper.
    """
    text = _RAY_TASK_ERROR_PATTERN.sub(r"\1", text)
    names = set(_EXCEPTION_NAME_PATTERN.findall(text))
    if names & (_INFRA_EXCEPTION_NAMES | _INFRA_SUBCLASS_NAMES) or _INFRA_MESSAGE_PATTERN.search(text):
        return INFRA
    if any(_is_infra_status(int(status)) for status in _RESPONSE_STATUS_PATTERN.findall(text)):
        return INFRA
    if _SPENT_BUDGET_PATTERN.search(text) or _BROKEN_CREDENTIAL_PATTERN.search(text):
        return INFRA
    if _MODEL_ROUTE_404_TEXT_PATTERN.search(text):
        return INFRA
    return MODEL


def _is_model_server_call(error: BaseException) -> bool:
    """Whether an HTTP error answered a call to a model server's API route (the policy or the judge)."""
    request_info = getattr(error, "request_info", None)
    url = str(getattr(request_info, "real_url", None) or getattr(request_info, "url", None) or "")
    return bool(_MODEL_ROUTE_PATTERN.search(url))


def _reports_broken_credential(error: BaseException) -> bool:
    """Whether an HTTP error's reason or body says the key or the authorization is broken."""
    body = getattr(error, "response_content", None)
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return bool(_BROKEN_CREDENTIAL_PATTERN.search(f"{getattr(error, 'message', '')} {body or ''}"))


def _reports_spent_budget(error: BaseException) -> bool:
    """Whether an HTTP error's reason or body says the key's budget or quota is spent."""
    body = getattr(error, "response_content", None)
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return bool(_SPENT_BUDGET_PATTERN.search(f"{getattr(error, 'message', '')} {body or ''}"))


def episode_failure_cause(error: BaseException, *, terminal: bool) -> FailureCause:
    """The cause an environment server records on an episode failure it builds from ``error``.

    Infra evidence in the error wins, and so does a ``model`` cause an inner Gym server named. Without either,
    the server's own ``terminal`` decides: a failure it would let collection try again is a retry.
    """
    for current in _exception_chain(error):
        headers = getattr(current, "headers", None)
        if hasattr(headers, "get") and headers.get(FAILURE_CAUSE_HEADER) == MODEL:
            return MODEL
    if failure_cause_of(error) == INFRA:
        return INFRA
    return MODEL if terminal else INFRA


def is_retryable(kind: str | None, *, cause: FailureCause | None) -> bool:
    """Whether resume retries a failure of ``kind``.

    An ``EVIDENCE`` kind, and a kind with no rule, is retried only with infra evidence. Without it the
    failure is final: a failure Gym cannot place must not hand the model another attempt, and it stays
    visible in the failure counts and the health check.
    """
    rule = FAILURE_RULES.get(kind, EVIDENCE)
    if rule == EVIDENCE:
        return cause == INFRA
    return rule == RETRY


def is_unscored(kind: str | None) -> bool:
    """Whether a failure of ``kind`` is left out of the score: never retried and never counted as 0."""
    return FAILURE_RULES.get(kind) == UNSCORED
