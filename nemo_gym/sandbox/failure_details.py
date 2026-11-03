# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A sandbox its provider ended, read in one place from the provider's own status report.

OpenSandbox reports an ended sandbox as ``SANDBOX::ENDED`` with a fixed message, for example::

    Sandbox <id> has ended (OOMKilled): 1/1 observed pods failed; primary reason=ContainerFailed;
    sample pod=<pod>; sample detail=Pod <pod> container sandbox exited with code 137 (OOMKilled)

    Sandbox <id> has ended (Evicted): 1/1 observed pods failed; primary reason=PodFailed; sample pod=<pod>;
    sample detail=Pod <pod> entered Failed phase (Evicted): Pod ephemeral local storage usage exceeds ...

and Gym's provider adds its own notice after losing the exec backend (``Sandbox was OOM-killed. OpenSandbox status:
state=..., reason=..., message=...; sandbox_id=...``). Only these reports name a reason: an exit status 137 without one
names none.

The failure event carries what ``SandboxFailureDetails`` holds: the ids, the provider, the provider's reason code and
the container it names, as reported. ``EndedSandbox`` reads two more facts from the report beside the details: which
kind of container ended and whether an eviction was for storage. They decide the kind: a memory kill is
``provider_oom_killed`` only when the agent's own container (or none, for Gym's notice) ended; a helper container's
kill is infrastructure.
"""

import re
from dataclasses import dataclass
from typing import Literal

from nemo_gym.core.failure_kinds import PROVIDER_OOM_KILLED, PROVIDER_UNAVAILABLE, FailureStage
from nemo_gym.core.failure_records import FailureBoundary, SandboxFailureDetails
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.utils.exception_inspection import bounded_text, error_to_http_status


ENDED_CODE = "SANDBOX::ENDED"
"""OpenSandbox's error code for a sandbox it has ended."""

USER_CONTAINER = "sandbox"
"""The container an OpenSandbox sandbox runs its commands in."""

_ENDED = re.compile(r"Sandbox (?P<sandbox>[\w.-]+) has ended \((?P<reason>[A-Za-z]+)\)")
_NOTICE = re.compile(
    r"OpenSandbox status: state=[^,]*, reason=(?:'(?P<reason>[^']*)'|None), message=(?P<message>.*?); "
    r"sandbox_id='(?P<sandbox>[^']*)'",
    re.DOTALL,
)
_CONTAINER_EXIT = re.compile(r"\bcontainer (?P<name>[\w.-]+) exited with code (?P<code>-?\d+)")
_INIT_CONTAINER = re.compile(r"\binit container\b", re.IGNORECASE)
_STORAGE = re.compile(r"ephemeral local storage|local ephemeral storage", re.IGNORECASE)
_OOM = re.compile(r"\boom[\s_-]*killed\b|\bout of memory\b", re.IGNORECASE)


@dataclass(frozen=True)
class EndedSandbox:
    """A sandbox its provider reported ended.

    ``details`` is what the failure event carries. ``cause`` reads the provider's reason (``OOMKilled`` is ``oom``;
    ``Evicted`` is ``storage`` when the report names exceeded storage, else ``evicted``; any other reason is
    ``unknown``) and ``container`` the kind of container the report names (the user container is ``agent``, an init
    container ``init``, any other ``sidecar``; a report that names none, such as a pod-level eviction or Gym's own
    notice, is ``unknown``).
    """

    details: SandboxFailureDetails
    cause: Literal["oom", "evicted", "storage", "unknown"]
    container: Literal["agent", "sidecar", "init", "unknown"]

    @property
    def failure_kind(self) -> str:
        """``provider_oom_killed`` exactly when the provider reported a memory kill of the agent's own container (or
        named none); any other ending, including a helper container's memory kill, is ``provider_unavailable``."""
        oom = self.cause == "oom" and self.container in ("agent", "unknown")
        return PROVIDER_OOM_KILLED if oom else PROVIDER_UNAVAILABLE


def provider_report(error: BaseException) -> str:
    """The provider's own text for ``error``: an SDK error's ``error.message``, its ``message``, else ``str``."""
    for text in (getattr(getattr(error, "error", None), "message", None), getattr(error, "message", None)):
        if isinstance(text, str) and text:
            return text
    return str(error)


def ended_sandbox(
    error: BaseException,
    *,
    provider: str | None = None,
    sandbox_id: str | None = None,
    execution_id: str | None = None,
) -> EndedSandbox | None:
    """The sandbox ``error`` reports as ended by its provider, else None. The ids come from the caller when it knows
    them; the reported sandbox id fills in otherwise."""
    text = provider_report(error)
    code = getattr(getattr(error, "error", None), "code", None)
    notice = _NOTICE.search(text)
    ended = _ENDED.search(text)
    if not (ended or notice or code == ENDED_CODE):
        return None
    reason = ended["reason"] if ended else (notice["reason"] if notice else None)
    if notice and not reason and _OOM.search(notice["message"]):
        reason = "OOMKilled"
    if reason and reason.lower() == "oomkilled":
        cause = "oom"
    elif reason and reason.lower() == "evicted":
        cause = "storage" if _STORAGE.search(text) else "evicted"
    else:
        cause = "unknown"
    exited = _CONTAINER_EXIT.search(text)
    if _INIT_CONTAINER.search(text):
        container = "init"
    elif exited:
        container = "agent" if exited["name"] == USER_CONTAINER else "sidecar"
    else:
        container = "unknown"
    reported_id = ended["sandbox"] if ended else (notice["sandbox"] if notice else None)
    sandbox = sandbox_id or reported_id
    return EndedSandbox(
        details=SandboxFailureDetails(
            sandbox_id=bounded_text(sandbox, 128) if sandbox else None,
            execution_id=bounded_text(execution_id, 128) if execution_id else None,
            provider=provider or "opensandbox",
            # The provider's reason code as reported: short, so a whole event stays inside one failure header.
            provider_status_reason=bounded_text(reason, 128) if reason else None,
            # The container's name as the provider reported it; short, for the same reason.
            container=bounded_text(exited["name"], 128) if exited else None,
        ),
        cause=cause,
        container=container,
    )


def sandbox_failure_event(
    error: BaseException,
    *,
    stage: FailureStage | None,
    provider: str | None = None,
    sandbox_id: str | None = None,
    execution_id: str | None = None,
    operation: str | None = None,
) -> NgFailureEvent | None:
    """The ``sandbox`` failure ``error`` reports, with ``error`` as its cause; None when ``error`` reports no
    provider-ended sandbox.

    The kind follows the provider's report (``EndedSandbox.failure_kind``): ``provider_oom_killed`` exactly when it
    reported a memory kill of the agent's own container, else ``provider_unavailable``; a helper container's
    memory kill is infrastructure. ``stage`` says when it ended (``seed``, ``agent``, ``verification``). Producers observe
    it beside the failure they already report, so the reply, the kind and the retry decision stay theirs.
    """
    ended = ended_sandbox(error, provider=provider, sandbox_id=sandbox_id, execution_id=execution_id)
    if ended is None:
        return None
    event = NgFailureEvent(
        boundary=FailureBoundary.SANDBOX,
        failure_kind=ended.failure_kind,
        failure_reason=f"The provider ended the sandbox ({ended.details.provider_status_reason or 'no reason given'})"
        + (f" during the {stage} stage" if stage else ""),
        http_status=error_to_http_status(error),
        details=ended.details,
        operation=operation,
        stage=stage,
    )
    event.__cause__ = error
    return event
