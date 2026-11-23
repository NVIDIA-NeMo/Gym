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

"""IdeGYM provider exceptions and SDK failure classification.

The SDK raises typed exceptions (``idegym.client.exceptions``) carrying the HTTP status
and body, and keeps a failure the sandbox itself reported (``IdeGYMSandboxError``) apart
from one about the orchestrator or the forward. The provider still has to tell "the
sandbox is gone" from "the control plane is busy" from "the command timed out"; that
mapping lives here, in one tested place. The SDK is an optional extra, so its exceptions
are imported only when a failure is classified.
"""

import json
from http import HTTPStatus
from types import ModuleType

from nemo_gym.sandbox.providers.base import SandboxCreateError, SandboxCreateVerificationError


# The sandbox's own wording, from BashCommandExecutionTimeoutError. 0.13.1 still relays it
# only as the text of a 500, so it is the one message this module reads. It is matched as
# a prefix of the error's `message`, never searched for: the body can carry tool output.
_SANDBOX_TIMEOUT_PREFIX = "Command execution timed out after"


class IdeGymError(RuntimeError):
    """Base class for IdeGYM provider failures."""


class IdeGymCreateError(SandboxCreateError):
    """Raised when the IdeGYM orchestrator cannot provide a ready server."""


class IdeGymCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a started IdeGYM server cannot run a command in ``spec.workdir``."""


class IdeGymOperationError(IdeGymError):
    """Raised when an IdeGYM operation on a live sandbox fails."""


class IdeGymUnknownServerError(IdeGymError):
    """Raised when a session is asked about a server it does not hold.

    Distinct from a generic failure so callers can treat it as "already stopped"
    rather than as a live problem: the session only forgets a server after stopping
    it, so this is what a second teardown of the same sandbox looks like.
    """


class IdeGymTransferError(IdeGymOperationError):
    """Raised when a file upload or download fails."""


class IdeGymInvalidRequestError(IdeGymOperationError):
    """Raised when a command request fails validation before it is sent.

    Kept apart from the ``ValueError`` family on purpose: a malformed orchestrator
    response surfaces as a ``JSONDecodeError`` or a pydantic ``ValidationError``, both
    ``ValueError`` subclasses, and must not read as the caller's mistake.
    """


def _sdk_exceptions() -> ModuleType | None:
    """``idegym.client.exceptions``, or ``None`` when the SDK is not installed."""
    try:
        from idegym.client import exceptions
    except ImportError:  # pragma: no cover - the provider cannot run without the SDK
        return None
    return exceptions


def _is_sandbox_error(exc: BaseException, sdk: ModuleType | None) -> bool:
    return sdk is not None and isinstance(exc, sdk.IdeGYMSandboxError)


def _caused_by_connect_timeout(exc: BaseException) -> bool:
    """Whether the SDK wrapped an ``httpx.ConnectTimeout``: the request never left."""
    try:
        from httpx import ConnectTimeout
    except ImportError:  # pragma: no cover - httpx ships with the idegym SDK
        return False
    return isinstance(exc.__cause__, ConnectTimeout)


def _sandbox_error_message(body: str | None) -> str:
    """The ``message`` of a relayed sandbox error body, or the raw body if it is not JSON."""
    if not body:
        return ""
    try:
        parsed = json.loads(body)
    except ValueError:
        return body
    if isinstance(parsed, dict) and isinstance(parsed.get("message"), str):
        return parsed["message"]
    return body


def is_sandbox_gone(exc: BaseException) -> bool:
    """Whether a failure means the sandbox is gone, for the orchestrator or for the session.

    The SDK maps an orchestrator 404 or 410 to ``IdeGYMNotFoundError``. A 404 the sandbox
    itself returned (a missing file, say) is an ``IdeGYMSandboxError`` from a live sandbox.
    The session forgets a server only once it is stopped, so ``IdeGymUnknownServerError``
    means the same thing.
    """
    if isinstance(exc, IdeGymUnknownServerError):
        return True
    sdk = _sdk_exceptions()
    if sdk is None or _is_sandbox_error(exc, sdk):
        return False
    return isinstance(exc, sdk.IdeGYMNotFoundError)


def is_deadline_exceeded(exc: BaseException) -> bool:
    """Whether a client-side deadline ran out: one the SDK enforces, or the caller's own.

    The SDK raises its request and polling deadlines as an ``IdeGYMTimeoutError`` with no
    status. A 408 or 504 is an ``IdeGYMTimeoutError`` too, but its status says the
    orchestrator or a gateway gave up on one request, and a connect timeout never reached
    the orchestrator at all. Neither spent a deadline; both are retryable instead.
    """
    if not isinstance(exc, TimeoutError) or getattr(exc, "status_code", None) is not None:
        return False
    return not _caused_by_connect_timeout(exc)


def is_command_timeout(exc: BaseException) -> bool:
    """Whether a failure means the command ran out of time.

    Either the sandbox reported its own timeout, or the client stopped waiting for the
    result (see :func:`is_deadline_exceeded`).
    """
    if isinstance(exc, TimeoutError):
        return is_deadline_exceeded(exc)
    sdk = _sdk_exceptions()
    if not _is_sandbox_error(exc, sdk) or exc.status_code != HTTPStatus.INTERNAL_SERVER_ERROR.value:
        return False
    return _sandbox_error_message(exc.body).startswith(_SANDBOX_TIMEOUT_PREFIX)


def is_retryable(exc: BaseException) -> bool:
    """Whether a control-plane failure is transient enough to retry.

    Never a sandbox-reported failure: the sandbox answered, so repeating the request would
    only repeat the answer, or run the command twice.
    """
    sdk = _sdk_exceptions()
    if _is_sandbox_error(exc, sdk):
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    if sdk is None:
        return False
    # 408 and 504 arrive as IdeGYMTimeoutError, a TimeoutError, and 429 and 503 as
    # IdeGYMBusyError; 502 has no type of its own and arrives as IdeGYMServerError.
    if isinstance(exc, (sdk.IdeGYMConnectionError, sdk.IdeGYMBusyError)):
        return True
    return isinstance(exc, sdk.IdeGYMHTTPError) and exc.status_code == HTTPStatus.BAD_GATEWAY.value


def is_invalid_request(exc: BaseException) -> bool:
    """Whether a failure means the request itself was invalid, not the sandbox.

    The sandbox answers a ``cwd`` or ``user`` it cannot honour with a 400, and the
    session rejects an env name that is not a shell identifier, or a value with a NUL in
    it, before sending anything (``IdeGymInvalidRequestError``).
    """
    if isinstance(exc, IdeGymInvalidRequestError):
        return True
    sdk = _sdk_exceptions()
    return _is_sandbox_error(exc, sdk) and exc.status_code == HTTPStatus.BAD_REQUEST.value
