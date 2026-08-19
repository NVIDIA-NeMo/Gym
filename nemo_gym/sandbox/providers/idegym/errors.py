# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""IdeGYM provider exceptions and orchestrator failure classification.

The SDK collapses every HTTP failure into a plain ``RuntimeError`` whose message
embeds the status and body, and the orchestrator relays a sandbox-side failure as a
500 carrying the sandbox's own error text. The provider still has to tell "the sandbox
is gone" from "the control plane is busy" from "the command timed out", so the message
parsing that recovers that lives here, in one tested place.
"""

import re
from http import HTTPStatus
from types import ModuleType

from nemo_gym.sandbox.providers.base import SandboxCreateError, SandboxCreateVerificationError


# Both patterns are anchored to the surrounding message rather than searching it: an
# orchestrator error carries the sandbox's own output in its `body`, and matching a
# bare `status=404` or `'status_code': 404` from there would read a live sandbox as
# gone. `HTTPUtils.make_request` formats failures as
# "Request failed: url=... status=404 reason='Not Found' data='...'".
_STATUS_PATTERN = re.compile(r"Request failed: url=\S* status=(\d{3})\b")
# An orchestrator ErrorResponse is relayed by dumping the model, whose first key is
# `status_code`; `[^{]*` keeps the match on that outer dict.
_STATUS_CODE_PATTERN = re.compile(r"^[^{]*\{'status_code': (\d{3})\b")
# The sandbox's own wording, from BashCommandExecutionTimeoutError, returned as a 500
# body. Matched in full rather than on "timed out after", which ordinary tool output
# relayed in that body could also contain.
_TIMEOUT_PATTERN = re.compile(r"Command execution timed out after", re.IGNORECASE)

# Statuses that mean the sandbox no longer exists as far as the orchestrator is
# concerned: 404 for an unknown client/server, 410 for one in a terminal state
# (see the orchestrator's `validate_server`).
GONE_STATUSES = frozenset({HTTPStatus.NOT_FOUND.value, HTTPStatus.GONE.value})


class IdeGymError(RuntimeError):
    """Base class for IdeGYM provider failures."""


class IdeGymCreateError(SandboxCreateError):
    """Raised when the IdeGYM orchestrator cannot provide a ready server."""


class IdeGymCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a started IdeGYM server fails its readiness probe."""


class IdeGymOperationError(IdeGymError):
    """Raised when an IdeGYM operation on a live sandbox fails."""


class IdeGymUnknownServerError(IdeGymError):
    """Raised when a session is asked about a server it does not hold.

    Distinct from a generic failure so callers can treat it as "already stopped"
    rather than as a live problem: the session only forgets a server after stopping
    it, so this is what a second teardown of the same sandbox looks like.
    """


class IdeGymCommandTooLongError(ValueError):
    """Raised when a generated script exceeds the sandbox's shell argument limit.

    A ``ValueError`` rather than an ``IdeGymError``: nothing is wrong with the
    sandbox, the command simply cannot be delivered as written. ``exec()`` turns it
    into a failed :class:`~nemo_gym.sandbox.providers.base.SandboxExecResult` so an
    oversized model-generated command cannot end a rollout.
    """


class IdeGymTransferError(IdeGymOperationError):
    """Raised when a file upload or download through the bash tool fails."""


def orchestrator_status(exc: BaseException) -> int | None:
    """Return the HTTP status carried by an IdeGYM SDK failure, or ``None``.

    The SDK raises ``RuntimeError`` for both direct request failures and relayed
    orchestrator error responses, so both message shapes are recognized.
    """
    message = str(exc)
    for pattern in (_STATUS_PATTERN, _STATUS_CODE_PATTERN):
        match = pattern.search(message)
        if match is not None:
            return int(match.group(1))
    return None


def is_sandbox_gone(exc: BaseException) -> bool:
    """Whether a failure means the orchestrator no longer has this sandbox."""
    return orchestrator_status(exc) in GONE_STATUSES


def _httpx() -> ModuleType | None:
    """The ``httpx`` module, or ``None``. The SDK's transport errors are httpx's."""
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx ships with the idegym SDK
        return None
    return httpx


def is_command_timeout(exc: BaseException) -> bool:
    """Whether a failure means the command ran out of time.

    Covers the sandbox killing the process group and answering with a 500, and the
    client giving up waiting. ``httpx.ConnectTimeout`` is excluded: failing to reach
    the orchestrator at all is a connectivity problem, not a slow command.
    """
    if isinstance(exc, TimeoutError):
        return True
    httpx = _httpx()
    if httpx is not None and isinstance(exc, httpx.TimeoutException):
        return not isinstance(exc, httpx.ConnectTimeout)
    return bool(_TIMEOUT_PATTERN.search(str(exc)))


def is_retryable(exc: BaseException) -> bool:
    """Whether a control-plane failure is transient enough to retry.

    Covers transport failures — the SDK lets httpx's through unwrapped, and
    ``httpx.TransportError`` is not a builtin ``ConnectionError`` — plus the
    orchestrator's back-pressure and availability statuses. Anything else is more
    likely a bad request or a bug than a blip, so it is not retried.
    """
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    httpx = _httpx()
    if httpx is not None and isinstance(exc, httpx.TransportError):
        return True
    status = orchestrator_status(exc)
    if status == HTTPStatus.TOO_MANY_REQUESTS.value:
        return True
    return status in {
        HTTPStatus.REQUEST_TIMEOUT.value,
        HTTPStatus.BAD_GATEWAY.value,
        HTTPStatus.SERVICE_UNAVAILABLE.value,
        HTTPStatus.GATEWAY_TIMEOUT.value,
    }
