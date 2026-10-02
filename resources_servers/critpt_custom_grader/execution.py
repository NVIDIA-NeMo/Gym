# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Remote execution unit. Three separately owned Daytona sandboxes per job. Comparator is sole verdict authority.

Job order: bounds check, reference run, comparator preflight of every reference outcome against stored
expectations, then a candidate run judged by the same comparator, then delete every owned sandbox. The candidate
sandbox receives only the candidate source, normalized inputs and the public runner package. A result file is
read only behind a process exit the provider independently reports, is shape-checked without decoding any value,
and must carry the paired status and the nonce this host wrote. None of these checks authenticates a result.
Before a candidate runs, a reference or comparator fault is nonterminal uncertainty, not fault attribution. After
a completed preflight, a candidate source error, suite timeout, exception or encoding-error observation is a
candidate-attributable fault scored 0 with the category preserved. A candidate can forge such a fault only into
its own failure, never an undeserved pass. Importing this module executes nothing remote and no task content.

Ownership lifecycle. A domain is ``deleted`` only after the provider accepted the delete and a bounded exact
lookup reported authoritative absence. Reconciliation looks a domain up by exact id or exact stable name,
verifies identity and the complete owned label set, deletes through the recovered handle and confirms absence.
Absence by exact id is authoritative. Absence by stable name for an ambiguous create is not authoritative until a
wall-clock settle deadline passes, so a create the provider never made clears on its own. One job runs at a time
and lifecycle operations never overlap job execution.
"""

import asyncio
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator

from nemo_gym import failure_kinds
from nemo_gym.sandbox.providers.base import (
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
)

from .runner import (
    CLEANUP_BUDGET_S,
    COMPARE_MODE,
    COMPARE_STATUSES,
    EXIT_COMPLETED,
    EXIT_RUNNER_ERROR,
    EXIT_SOURCE_ERROR,
    EXIT_TIMEOUT,
    MAX_CODE_LENGTH,
    MAX_NONCE_LENGTH,
    PROTOCOL_VERSION,
    RESULT_BYTES_LIMIT,
    RESULT_EXIT_CODES,
    RUN_MODE,
    RUN_STATUSES,
    RUNNER_ERROR_CODES,
    RUNTIME_PACKAGE,
    SOURCE_ERROR_CODES,
    VERDICT_STATUSES,
    WORKER_ABORT_EXIT_CODES,
)
from .task_data import MAX_CASES, MAX_TEXT_BYTES, WIRE_FORMAT, TaskData, bounded_json


PACKAGE_DIR = Path(__file__).resolve().parent
RUNTIME_FILES = ("task_data.py", "codec.py", "symbolic.py", "runner.py")
COMPARATOR_FILES = RUNTIME_FILES + ("comparator.py",)
DOMAINS = ("reference", "comparator", "candidate")
LABEL_OWNER, LABEL_JOB, LABEL_DOMAIN = "ng-grader-owner", "ng-grader-job", "ng-grader-domain"
RESOLVED_STATES = frozenset({"planned", "create_failed_clean", "deleted"})
UNRESOLVED_STATES = frozenset({"creating", "created", "create_uncertain", "delete_failed", "delete_unconfirmed"})
# Why a lookup could not establish presence or absence. Fixed vocabulary. Provider text never leaves the backend.
LOOKUP_ERROR_KINDS = ("permission", "transport", "timeout", "malformed", "provider")
# Per-domain lifecycle outcomes reported by ``GradeResult.cleanup``, ``Grader.reconcile`` and ``Grader.shutdown``.
CLEANUP_STATUSES = frozenset(
    {
        "not_created",
        "created",
        "create_uncertain",
        "create_failed_clean",
        "deleted",
        "absent",
        "delete_failed",
        "delete_unconfirmed",
        "foreign_resource",
        *(f"lookup_{kind}" for kind in LOOKUP_ERROR_KINDS),
    }
)
_ABSENCE_POLL_INTERVAL_S = 1.0
_ABSENCE_POLL_MAX_S = 8.0
# Bounded re-lookups by exact stable name for an ambiguous create that may still be settling remotely. A single
# absent name lookup does not prove the create failed.
_CREATE_SETTLE_POLLS = 3
# Settle margin above the create timeout before an absent stable-name lookup becomes authoritative. Generous on
# purpose: resolving a live create as deleted orphans a sandbox, while waiting only delays a failed create.
_CREATE_SETTLE_MARGIN_FACTOR = 2.0
_CREATE_SETTLE_MARGIN_FLOOR_S = 60.0


def _now() -> float:
    """Wall-clock seconds since the epoch. Indirected so tests can inject a deterministic clock without sleeping."""
    return time.time()


def _monotonic() -> float:
    """Monotonic seconds. It never moves backward, so it anchors the settle deadline. Indirected like ``_now``."""
    return time.monotonic()


_SAFE_PATH = re.compile(r"/[A-Za-z0-9_./-]{1,255}\Z")
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/-]{0,255}\Z")
# Compiled so pydantic uses Python's ``re`` (its default Rust engine has no ``\Z``). pydantic searches rather
# than fullmatches, so ``\A`` anchors the start explicitly.
_OS_USER = re.compile(r"\A[a-z_][a-z0-9_-]*\Z")
_OWNER_ID = re.compile(r"\A[a-z0-9-]+\Z")
# Result transport: one download, including the SDK generator's closure, ends within ``timeout_s`` plus this floor.
_STREAM_CLOSE_FLOOR_S = 1.0
# Every file transfer is awaited for ``transfer_timeout_s`` plus this grace for the wrapper itself.
_TRANSFER_GRACE_S = 5.0
# A blocking ``process.exec`` through the Daytona proxy stops returning past about ten minutes, so a run domain
# sends its command as an asynchronous session command and polls the exit. Each session request is short. The
# runner command never calls ``exit``, which would end the persistent shell before the exit code arrives.
_SESSION_POLL_INTERVAL_S = 5.0  # wall time between two exit reads
_SESSION_CALL_TIMEOUT_S = 30.0  # client-side bound on any one session request: create, execute, poll or delete
# Bounded transfers one job awaits: each run domain uploads runtime files, source and job and downloads one
# result. The comparator uploads its files once, then uploads a job and downloads a result twice.
_JOB_TRANSFERS = 2 * (len(RUNTIME_FILES) + 3) + (len(COMPARATOR_FILES) + 4)
# Result envelope shapes (runner.py). Verdicts are shaped by ``comparator._result``. Wire values are never decoded.
_ENVELOPE_KEYS = frozenset({"version", "mode", "nonce", "status", "code", "timed_out_case"})
_WIRE_KEYS = frozenset({"format", "value"})
_VERDICT_KEYS = frozenset({"version", "status", "equal", "side", "code", "path"})
_VERDICT_VERSION = 1
_MAX_VERDICT_LABEL = 256  # ``side`` and ``code`` come from short fixed comparator vocabularies
_MAX_VERDICT_PATH = 1024  # the comparator truncates ``path`` to this many characters
# Status/exit pairing. A shape-valid result is accepted only behind the one process exit its status maps to, as
# the provider independently reports it. Necessary, not sufficient: same-UID code can exit with any code and
# write any file.
_STATUS_EXIT_CODES = {
    "completed": EXIT_COMPLETED,
    "source_error": EXIT_SOURCE_ERROR,
    "runner_error": EXIT_RUNNER_ERROR,
    "timeout": EXIT_TIMEOUT,
}

# Outcome categories (bounded public vocabulary). Failed and passed categories map to a reward. Unscorable
# categories carry a Gym failure class for the app to project into ``_ng_failure_class``.
PASSED = "passed"
# Candidate-attributable faults, observed only after a completed preflight. A forged fault cannot win an
# undeserved pass, only its own failure, so each scores a reward-0 failure with the category preserved. The
# category names what the candidate domain reported, not a proven cause. Reference-side and comparator-side
# faults are never here.
CANDIDATE_FAULT_CATEGORIES = {
    "candidate_source_error": "candidate process reported a source error after a completed preflight",
    "candidate_timeout": "candidate process exceeded the suite timeout after a completed preflight",
    "candidate_exception": "candidate produced an exception observation after a completed preflight",
    "candidate_encoding_error": "candidate produced an unencodable observation after a completed preflight",
    "candidate_worker_abort": "candidate worker aborted or broke the runner protocol after a completed preflight",
}
FAILED_CATEGORIES = {
    "candidate_mismatch": "comparator found a mismatch for an observed value",
    "candidate_source_limit": "candidate source violated the host size or encoding contract",
    "candidate_invalid_output": "comparator validated an observed value as invalid",
    **CANDIDATE_FAULT_CATEGORIES,
}
UNSCORABLE_CATEGORIES = {
    # category: (failure class, terminal)
    "task_invalid": (failure_kinds.VERIFIER_ERROR, True),
    "reference_mismatch": (failure_kinds.VERIFIER_ERROR, True),
    # These name unauthenticated observations, not established reference faults.
    "reference_error": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "reference_timeout": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "comparator_uncertain": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "provider_create": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "provider_exec": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "transfer_limit": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "result_invalid": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    # An absent result file, or an exit the protocol pairs with no result, proves nothing about the domain that
    # should have written it. Never terminal.
    "candidate_no_result": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "reference_no_result": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "comparator_no_result": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "job_deadline": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "ownership_unresolved": (failure_kinds.PROVIDER_UNAVAILABLE, False),
    "busy": (failure_kinds.PROVIDER_UNAVAILABLE, False),
}
_NO_RESULT = {domain: f"{domain}_no_result" for domain in DOMAINS}


class ExecutionPolicy(BaseModel):
    """Operator-owned runtime policy. Tasks cannot select any field. Unknown fields are rejected."""

    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)

    provider: Literal["daytona"] = "daytona"
    snapshot: str = Field(min_length=1, max_length=256, description="Pinned operator-owned snapshot identity.")
    os_user: str = Field(min_length=1, max_length=32, pattern=_OS_USER)
    network_block_all: Literal[True] = True
    network_allow_list: None = None
    interpreter: str = "/usr/bin/python3"
    workdir: str = "/tmp/ng-grader"
    memory_limit_mib: int = Field(default=2048, ge=64, le=65536)
    cpu_time_limit_s: int = Field(default=1800, ge=1, le=86400)
    max_processes: int = Field(default=64, ge=1, le=4096)
    create_timeout_s: float = Field(default=120.0, gt=0, le=3600)
    transfer_timeout_s: float = Field(default=120.0, gt=0, le=3600)
    exec_timeout_margin_s: float = Field(default=60.0, gt=0, le=3600)
    suite_timeout_s: int = Field(default=1800, ge=1, le=86400, description="Whole-suite deadline per run.")
    compare_timeout_s: int = Field(default=600, ge=1, le=86400)
    job_timeout_s: float | None = Field(default=None, gt=0, le=7 * 86400)
    cleanup_timeout_s: float = Field(default=120.0, gt=0, le=3600)
    max_candidate_source_bytes: int = Field(default=262_144, ge=1, le=1_048_576)
    max_concurrent_jobs: int = Field(
        default=1, ge=1, le=1, description="Exactly one job: cleanup and reconciliation never race execution."
    )
    owner_id: str = Field(min_length=1, max_length=32, pattern=_OWNER_ID)
    journal_dir: str = Field(min_length=1, max_length=1024)
    api_url: str | None = Field(default=None, max_length=1024)
    target: str | None = Field(default=None, max_length=256)

    @model_validator(mode="after")
    def bounded_policy(self) -> "ExecutionPolicy":
        if self.os_user == "root":
            raise ValueError("os_user must not be root")
        for name in ("interpreter", "workdir"):
            if not _SAFE_PATH.fullmatch(getattr(self, name)):
                raise ValueError(f"{name} must be an absolute path without quoting characters")
        if not _SAFE_ID.fullmatch(self.snapshot):
            raise ValueError("snapshot must be an immutable identity without control characters")
        if self.cpu_time_limit_s > self.suite_timeout_s + self.exec_timeout_margin_s:
            raise ValueError("cpu_time_limit_s must not exceed the suite deadline plus margin")
        if self.exec_timeout_margin_s < CLEANUP_BUDGET_S:
            raise ValueError(
                f"exec_timeout_margin_s must be at least the runner cleanup budget ({CLEANUP_BUDGET_S:.0f} s)"
            )
        minimum = self.minimum_job_timeout_s()
        if self.job_timeout_s is not None and self.job_timeout_s < minimum:
            raise ValueError(f"job_timeout_s must be at least {minimum:.0f}")
        return self

    def minimum_job_timeout_s(self) -> float:
        """Sum of every bound awaited inside one job, so a legal but slow job never trips ``job_deadline``.

        Covers two runs and two comparisons with margins, three creates, ``_JOB_TRANSFERS`` file transfers, and
        three deletes each with a bounded absence confirmation.
        """
        margin = 2 * self.exec_timeout_margin_s
        return (
            2 * (self.suite_timeout_s + margin)
            + 2 * (self.compare_timeout_s + margin)
            + 3 * (self.create_timeout_s + 4 * self.transfer_timeout_s)
            + _JOB_TRANSFERS * (self.transfer_timeout_s + _TRANSFER_GRACE_S)
            + 3 * 2 * self.cleanup_timeout_s
        )

    def effective_job_timeout_s(self) -> float:
        return self.job_timeout_s if self.job_timeout_s is not None else self.minimum_job_timeout_s()

    def domain_config(self) -> dict[str, Any]:
        """Operator constructor settings: category-only provider diagnostics and no create or command retries."""
        auto_stop = int(self.effective_job_timeout_s() // 60) + 5
        connection: dict[str, Any] = {"otel_enabled": False}
        if self.api_url is not None:
            connection["api_url"] = self.api_url
        if self.target is not None:
            connection["target"] = self.target
        return {
            "category_only_diagnostics": True,
            "connection": connection,
            "create": {
                "timeout_s": self.create_timeout_s,
                "retries": 0,
                "os_user": self.os_user,
                "auto_stop_interval": auto_stop,
                "auto_delete_interval": 0,
                "ephemeral": True,
                "network_block_all": True,
                "network_allow_list": None,
            },
            "operations": {
                "retries": 1,
                "command_retries": 0,
                "command_timeout_margin_s": self.exec_timeout_margin_s,
                "file_timeout_s": int(self.transfer_timeout_s) or 1,
                "close_timeout_s": self.cleanup_timeout_s,
            },
        }

    def sandbox_spec(self, *, stable_name: str, labels: dict[str, str]) -> SandboxSpec:
        return SandboxSpec(
            image=None,
            ttl_s=None,
            workdir=self.workdir,
            metadata=dict(labels),
            provider_options={"snapshot_id": self.snapshot, "extensions": {"daytona.name": stable_name}},
        )


@dataclass(frozen=True)
class GradeResult:
    """Public verdict metadata only: no outputs, expectations, source or provider text."""

    outcome: Literal["passed", "failed", "unscorable"]
    reward: float | None
    category: str
    failure_class: str | None
    terminal: bool | None
    job_id: str
    case_count: int
    cases_attempted: int
    cases_equal: int
    first_failed_case: int | None
    cleanup: dict[str, str]


class TransferLimitExceeded(Exception):
    pass


class MissingRemoteFile(Exception):
    pass


class SandboxCreateRejected(Exception):
    """A backend proved a create request was refused before it left the host, so no sandbox exists.

    Only this signal marks a create ``create_failed_clean``. A backend raises it only when it can prove
    pre-dispatch rejection. Every other create exception is ambiguous. The request may have reached the provider
    and a sandbox may exist, so the ownership intent is kept for lookup and reconciliation. A plain ``ValueError``
    or ``TypeError`` is not proof, because an SDK can raise either while decoding a create that succeeded remotely.
    """


class LookupFailed(Exception):
    """Presence could not be established. ``kind`` is one of ``LOOKUP_ERROR_KINDS``. Provider text is dropped."""

    def __init__(self, kind: str):
        if kind not in LOOKUP_ERROR_KINDS:
            raise ValueError("unknown lookup error kind")
        super().__init__(kind)
        self.kind = kind


class GraderBusy(RuntimeError):
    """Raised by ``Grader.reconcile`` while a job holds the lifecycle lock: deletes never race live execution."""


@dataclass(frozen=True)
class SandboxLookup:
    """One live provider resource resolved by an exact key: the handle that can delete it plus its identity."""

    handle: SandboxHandle
    name: str | None
    labels: dict[str, str]
    state: str | None


class SandboxBackend(Protocol):
    """Asynchronous per-domain control surface. Each domain owns exactly one backend instance."""

    async def create(self, spec: SandboxSpec) -> SandboxHandle: ...
    async def exec(self, handle: SandboxHandle, command: str, *, cwd: str, timeout_s: int) -> SandboxExecResult: ...
    async def write_file(self, handle: SandboxHandle, target_path: str, data: bytes) -> None: ...
    async def download_bounded(
        self, handle: SandboxHandle, remote_path: str, *, limit: int, timeout_s: float
    ) -> bytes:
        """Return at most ``limit`` bytes of one remote file, within a bound that covers the whole operation.

        Raises ``MissingRemoteFile`` only on the provider's verified absence signal. Raises
        ``TransferLimitExceeded`` past ``limit``.
        """
        ...

    async def close(self, handle: SandboxHandle) -> None:
        """Request deletion through a live handle. Returning means the provider accepted the request, no more."""
        ...

    async def lookup(self, key: str) -> SandboxLookup | None:
        """Resolve exactly one sandbox by exact id or exact stable name (never prefix or list scans).

        ``None`` is authoritative absence (the provider says the resource does not exist or is destroyed).
        Anything less certain raises ``LookupFailed`` with a fixed kind.
        """
        ...

    async def aclose(self) -> None: ...


def _caused_by_validation(exc: BaseException) -> bool:
    """Whether the SDK error wraps a pydantic validation failure, i.e. the response shape was malformed."""
    from pydantic import ValidationError

    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, ValidationError):
            return True
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return False


def _daytona_lookup_types() -> dict[str, Any]:
    """Pinned SDK exception types and the terminal state value, imported lazily like the provider does.

    The 403 class is named differently across SDK generations, so ``DaytonaForbiddenError`` is imported softly
    and the permission tuple carries whichever 403 classes the installed SDK exposes beside the 401 class.
    """
    from daytona import (
        DaytonaAuthenticationError,
        DaytonaAuthorizationError,
        DaytonaConnectionError,
        DaytonaError,
        DaytonaNotFoundError,
        DaytonaTimeoutError,
        SandboxState,
    )

    try:
        from daytona import DaytonaForbiddenError
    except ImportError:
        permission: tuple[type[BaseException], ...] = (DaytonaAuthenticationError, DaytonaAuthorizationError)
    else:
        permission = (DaytonaAuthenticationError, DaytonaAuthorizationError, DaytonaForbiddenError)

    return {
        "not_found": DaytonaNotFoundError,
        "permission": permission,
        "timeout": DaytonaTimeoutError,
        "connection": DaytonaConnectionError,
        "error": DaytonaError,
        "destroyed": str(SandboxState.DESTROYED.value),
    }


def _daytona_file_types() -> dict[str, Any]:
    """SDK error types that carry the daemon's verdict that a single downloaded file is absent.

    ``not_found`` is the generic 404 class, exported by every supported SDK. ``file_not_found`` is the dedicated
    file-absent subclass, present only on newer SDKs and ``None`` on the pinned 0.183.0. It is imported softly.
    """
    from daytona import DaytonaNotFoundError

    try:
        from daytona import DaytonaFileNotFoundError as file_not_found
    except ImportError:
        file_not_found = None
    return {"file_not_found": file_not_found, "not_found": DaytonaNotFoundError}


def _missing_file_error(exc: BaseException) -> bool:
    """Whether the SDK itself reported the requested file absent. Exception text is never consulted.

    The signal is the dedicated ``DaytonaFileNotFoundError`` where the SDK has it, otherwise the generic
    not-found class carrying a 404 status. The direction is fail-closed. Without the SDK, or without one of
    these structured signals, the function returns ``False`` and the caller keeps the download as uncertainty.
    """
    try:
        types = _daytona_file_types()
    except ImportError:
        return False  # without the SDK no exception can be one of its types
    file_not_found = types["file_not_found"]
    if file_not_found is not None and isinstance(exc, file_not_found):
        return True
    return type(exc) is types["not_found"] and getattr(exc, "status_code", None) == 404


async def _close_stream(stream: Any, timeout_s: float) -> None:
    """Bounded closure of the SDK download generator. A failed or slow close never replaces the transfer outcome."""
    aclose = getattr(stream, "aclose", None)
    if aclose is None:
        return
    try:
        await asyncio.wait_for(aclose(), timeout=timeout_s)
    except asyncio.CancelledError:
        raise
    except Exception:
        pass


def _describe_sandbox(handle: SandboxHandle, destroyed: str) -> SandboxLookup | None:
    """Project the documented ``id``/``name``/``labels``/``state`` of a live SDK sandbox. Reject odd shapes."""
    raw = handle.raw
    sandbox_id = getattr(raw, "id", None)
    name = getattr(raw, "name", None)
    labels = getattr(raw, "labels", None)
    state = getattr(raw, "state", None)
    state = getattr(state, "value", state)
    if type(sandbox_id) is not str or not sandbox_id or sandbox_id != handle.sandbox_id:
        raise LookupFailed("malformed")
    if name is not None and type(name) is not str:
        raise LookupFailed("malformed")
    if labels is None:
        labels = {}
    if type(labels) is not dict or any(
        type(key) is not str or type(value) is not str for key, value in labels.items()
    ):
        raise LookupFailed("malformed")
    if state is not None and type(state) is not str:
        raise LookupFailed("malformed")
    if state == destroyed:
        return None
    return SandboxLookup(handle=handle, name=name, labels=dict(labels), state=state)


class DaytonaBackend:
    """Thin adapter over Gym's DaytonaProvider async interface plus a bounded streaming download."""

    def __init__(self, policy: ExecutionPolicy):
        from .provider_support import CategoryOnlyDaytonaProvider

        self._provider = CategoryOnlyDaytonaProvider(**policy.domain_config())
        self._lookup_timeout_s = policy.cleanup_timeout_s

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        return await self._provider.create(spec)

    async def exec(self, handle: SandboxHandle, command: str, *, cwd: str, timeout_s: int) -> SandboxExecResult:
        """Run one command through the Daytona session API and return its process exit as a ``SandboxExecResult``.

        The command starts as an asynchronous background session command and its exit is polled, so no single
        provider request outlives ``_SESSION_CALL_TIMEOUT_S``. The command runs exactly as the caller built it. A
        run that never reports an integer exit within the bound raises ``asyncio.TimeoutError``, which the caller
        maps to nonterminal provider uncertainty.
        """
        from daytona import SessionExecuteRequest

        process = handle.raw.process
        # A fresh unguessable session id for this one invocation. This id is a control-plane handle only, never an
        # authentication token. The result file stays correlated by the runner nonce downstream.
        session_id = f"critpt-grade-{secrets.token_hex(16)}"
        # ``SessionExecuteRequest`` carries no working directory, so the command applies the caller's cwd. The
        # redirection binds to the runner invocation alone, so the result gate is unchanged.
        script = f'cd "{cwd}" && {command}' if cwd else command
        loop = asyncio.get_running_loop()
        deadline = loop.time() + float(timeout_s)
        created = False
        try:
            await self._session_call(process.create_session(session_id))
            created = True
            request = SessionExecuteRequest(command=script, run_async=True)
            started = await self._session_call(
                process.execute_session_command(session_id, request, timeout=int(_SESSION_CALL_TIMEOUT_S))
            )
            exit_code = await self._poll_session_exit(process, session_id, started.cmd_id, deadline)
        finally:
            if created:
                await self._delete_session_quietly(process, session_id)
        return SandboxExecResult(stdout="", stderr="", return_code=exit_code)

    async def _session_call(self, awaitable: Any) -> Any:
        """Await one session request under a short client-side bound. A hang raises ``asyncio.TimeoutError``.

        The pinned SDK takes no request-timeout argument on these session calls, so the bound is applied here.
        """
        return await asyncio.wait_for(awaitable, timeout=_SESSION_CALL_TIMEOUT_S)

    async def _poll_session_exit(self, process: Any, session_id: str, command_id: str, deadline: float) -> int:
        """Poll the background command until it reports an integer exit or ``deadline`` passes.

        A transient read failure is retried on the next interval, so one blip cannot discard a finished run. The
        exit type is checked exactly, so a bool can never stand in for an exit code. A deadline with no integer
        exit raises ``asyncio.TimeoutError``.
        """
        loop = asyncio.get_running_loop()
        while True:
            try:
                command = await asyncio.wait_for(
                    process.get_session_command(session_id, command_id), timeout=_SESSION_CALL_TIMEOUT_S
                )
                exit_code = getattr(command, "exit_code", None)
                if type(exit_code) is int:
                    return exit_code
            except asyncio.CancelledError:
                raise
            except Exception:
                pass
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise asyncio.TimeoutError()
            await asyncio.sleep(min(_SESSION_POLL_INTERVAL_S, remaining))

    async def _delete_session_quietly(self, process: Any, session_id: str) -> None:
        """Delete the session on every non-cancelled exit path. Any delete error is swallowed.

        ``close`` deletes the whole sandbox regardless. Cancellation is re-raised to honor the caller's cancel path.
        """
        try:
            await asyncio.wait_for(process.delete_session(session_id), timeout=_SESSION_CALL_TIMEOUT_S)
        except asyncio.CancelledError:
            raise
        except Exception:
            pass

    async def write_file(self, handle: SandboxHandle, target_path: str, data: bytes) -> None:
        await self._provider.write_file(handle, target_path, data)

    async def download_bounded(
        self, handle: SandboxHandle, remote_path: str, *, limit: int, timeout_s: float
    ) -> bytes:
        """Stream one remote file into memory, never past ``limit`` bytes, within ``timeout_s`` plus a closure floor.

        The SDK's ``download_file_stream`` is lazy, so the call and the iteration share one deadline. Only the
        SDK's own not-found types become ``MissingRemoteFile``. The transfer limit and cancellation propagate
        unchanged, and the generator is always closed with a bound. No task is spawned, so nothing outlives a timeout.
        """
        cancel = threading.Event()  # The SDK only polls ``is_set()`` and closes the response when it is set.
        chunks: list[bytes] = []
        total = 0
        stream = None
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        try:
            async with asyncio.timeout_at(deadline):
                stream = await handle.raw.fs.download_file_stream(remote_path, timeout=timeout_s, cancel_event=cancel)
                async for chunk in stream:
                    total += len(chunk)
                    if total > limit:
                        raise TransferLimitExceeded()
                    chunks.append(chunk)
        except (asyncio.CancelledError, asyncio.TimeoutError, TransferLimitExceeded):
            raise
        except Exception as exc:
            if _missing_file_error(exc):
                raise MissingRemoteFile() from None
            raise
        finally:
            cancel.set()
            if stream is not None:
                await _close_stream(stream, max(deadline - loop.time(), _STREAM_CLOSE_FLOOR_S))
        return b"".join(chunks)

    async def close(self, handle: SandboxHandle) -> None:
        # The provider hands ``handle.raw`` to the SDK delete call. A bare id cannot be deleted through it.
        if handle.raw is None:
            raise TypeError("delete needs the live SDK sandbox object carried by the handle")
        await self._provider.close(handle, delete=True)

    async def lookup(self, key: str) -> SandboxLookup | None:
        """Exact ``get`` by id or name through the provider's ``connect``. Absence is a 404 or a destroyed state."""
        # Reconcile passes a stable name here, not an id. Daytona 0.183.0 ``AsyncDaytona.get`` takes
        # ``sandbox_id_or_name`` and resolves either, so a name lookup finds a sandbox created under it.
        types = _daytona_lookup_types()
        try:
            handle = await asyncio.wait_for(self._provider.connect(key), timeout=self._lookup_timeout_s)
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise LookupFailed("timeout") from None
        except types["not_found"]:
            return None
        except types["permission"]:
            raise LookupFailed("permission") from None
        except types["timeout"]:
            raise LookupFailed("timeout") from None
        except types["connection"]:
            raise LookupFailed("transport") from None
        except types["error"] as exc:
            raise LookupFailed("malformed" if _caused_by_validation(exc) else "provider") from None
        except Exception:
            raise LookupFailed("provider") from None
        return _describe_sandbox(handle, types["destroyed"])

    async def aclose(self) -> None:
        await self._provider.aclose()


class OwnershipJournal:
    """Atomic per-job records with stable names, labels and lifecycle states. No task content."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.directory, 0o700)

    def path(self, job_id: str) -> Path:
        return self.directory / f"{job_id}.json"

    def open_job(self, job_id: str, owner_id: str, names: dict[str, str], labels: dict[str, dict[str, str]]) -> dict:
        record = {
            "version": 1,
            "job_id": job_id,
            "owner_id": owner_id,
            "created_at": time.time(),
            "domains": {
                domain: {"name": names[domain], "labels": labels[domain], "state": "planned", "sandbox_id": None}
                for domain in DOMAINS
            },
        }
        self._write(record)
        return record

    def update(self, record: dict, domain: str, state: str, sandbox_id: str | None = None) -> None:
        entry = record["domains"][domain]
        entry["state"] = state
        if sandbox_id is not None:
            entry["sandbox_id"] = sandbox_id
        self._write(record)

    def resolve(self, record: dict) -> bool:
        """Delete the record only when every domain is provably resolved."""
        if all(entry["state"] in RESOLVED_STATES for entry in record["domains"].values()):
            self.path(record["job_id"]).unlink(missing_ok=True)
            self._sync_directory()
            return True
        self._write(record)
        return False

    def unresolved(self) -> list[dict]:
        records = []
        for path in sorted(self.directory.glob("*.json")):
            try:
                with open(path, "rb") as handle:
                    data = handle.read(65_536 + 1)
                record = json.loads(data.decode("utf-8")) if len(data) <= 65_536 else None
            except (OSError, ValueError):
                record = None
            if (
                type(record) is not dict
                or record.get("version") != 1
                or type(record.get("domains")) is not dict
                or record.get("job_id") != path.stem
                or not all(_well_formed_entry(entry) for entry in record["domains"].values())
            ):
                records.append({"job_id": path.stem, "corrupt": True})
            elif any(entry["state"] not in RESOLVED_STATES for entry in record["domains"].values()):
                records.append(record)
        return records

    def _write(self, record: dict) -> None:
        target = self.path(record["job_id"])
        temporary = target.with_suffix(".tmp")
        data = json.dumps(record, separators=(",", ":")).encode("utf-8")
        if len(data) > 65_536:
            raise ValueError("journal record limit")
        with open(temporary, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        self._sync_directory()

    def _sync_directory(self) -> None:
        """Make the rename or unlink itself durable: the intent must survive a crash right after ``create``."""
        fd = os.open(self.directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def _well_formed_entry(entry: Any) -> bool:
    """A journal domain entry carries exactly the identity reconciliation relies on: name, labels, state, id."""
    return (
        type(entry) is dict
        and type(entry.get("name")) is str
        and type(entry.get("labels")) is dict
        and all(type(key) is str and type(value) is str for key, value in entry["labels"].items())
        and entry.get("state") in RESOLVED_STATES | UNRESOLVED_STATES
        and (entry.get("sandbox_id") is None or type(entry["sandbox_id"]) is str)
    )


class _Unscorable(Exception):
    def __init__(self, category: str, attempted: int = 0):
        super().__init__(category)
        self.category = category
        self.attempted = attempted


class _CandidateWorkerAbort(Exception):
    """A runner-authenticated candidate worker abort. The candidate step catches it and scores 0."""


class _Domain:
    """One owned sandbox: exact create/delete bookkeeping against the journal entry of the same name."""

    def __init__(
        self,
        name: str,
        policy: ExecutionPolicy,
        backend: SandboxBackend,
        journal: OwnershipJournal,
        record: dict,
        dispatch_anchors: dict[str, float] | None = None,
    ):
        self.name = name
        self.policy = policy
        self.backend = backend
        self.journal = journal
        self.record = record
        self.handle: SandboxHandle | None = None
        # Per-job monotonic dispatch times, shared with the ``Grader`` so a rebuilt cleanup client keeps the anchor.
        self._dispatch_anchors = dispatch_anchors
        # Monotonic dispatch anchor for a create this process made. None for a disk-recovered record, whose only
        # anchor is the wall-clock ``dispatched_at`` persisted before the crash.
        self._dispatched_monotonic: float | None = None if dispatch_anchors is None else dispatch_anchors.get(name)
        state = self.entry["state"]
        self.cleanup = "not_created" if state == "planned" else state

    @property
    def entry(self) -> dict:
        return self.record["domains"][self.name]

    @property
    def resolved(self) -> bool:
        return self.entry["state"] in RESOLVED_STATES

    def _mark(self, state: str, sandbox_id: str | None = None) -> None:
        self.journal.update(self.record, self.name, state, sandbox_id)
        self.cleanup = state

    def release(self) -> None:
        """Drop the live handle once its client is closed. The journal keeps the exact identity for retries."""
        self.handle = None

    async def create(self) -> None:
        spec = self.policy.sandbox_spec(stable_name=self.entry["name"], labels=self.entry["labels"])
        self.entry["dispatched_at"] = _now()  # the create's wall-clock dispatch time, the disk-recovery anchor
        self._dispatched_monotonic = _monotonic()  # the in-process anchor, immune to a wall-clock jump
        if self._dispatch_anchors is not None:
            # Keep the anchor for this job so a cleanup client rebuilt in ``reconcile`` or ``shutdown`` reuses it.
            self._dispatch_anchors[self.name] = self._dispatched_monotonic
        self.journal.update(self.record, self.name, "creating")
        try:
            self.handle = await asyncio.wait_for(
                self.backend.create(spec), timeout=self.policy.create_timeout_s + 4 * self.policy.transfer_timeout_s
            )
        except asyncio.CancelledError as exc:
            # A delete cancelled during create cleanup can still name a live sandbox. Keep the id so cleanup
            # deletes it exactly, the same as the ``BaseException`` branch below.
            leaked = getattr(exc, "sandbox_id", None)
            leaked = leaked if isinstance(leaked, str) and leaked else None
            self._mark("delete_failed" if leaked is not None else "create_uncertain", leaked)
            raise
        except BaseException as exc:
            leaked = getattr(exc, "sandbox_id", None)
            leaked = leaked if isinstance(leaked, str) and leaked else None
            if leaked is not None:
                state = "delete_failed"  # the provider names the sandbox its own cleanup could not delete
            elif isinstance(exc, SandboxCreateRejected):
                state = "create_failed_clean"  # the backend proved the request never left the host
            else:
                # Ambiguous: the request may have reached the provider, so a sandbox may exist. This includes a
                # SandboxCreateVerificationError (its delete was accepted, not confirmed absent) and any
                # ValueError/TypeError, which an SDK can raise while decoding a response to a create that succeeded.
                state = "create_uncertain"
            self._mark(state, leaked)
            if not isinstance(exc, Exception):
                raise
            raise _Unscorable("provider_create") from None
        self._mark("created", self.handle.sandbox_id)

    async def upload(self, files: dict[str, bytes]) -> None:
        for relative, data in files.items():
            await self._transfer(self.backend.write_file(self.handle, f"{self.policy.workdir}/{relative}", data))

    async def run(self, mode: str, deadline_s: int, nonce: str) -> dict:
        """Run the runner in this domain and return its shape-valid, correlated result, never a verdict.

        The result file is read only behind a process exit the provider independently reports in
        ``RESULT_EXIT_CODES``, then held to the exact paired exit and to the nonce this host wrote. These are
        necessary checks and no more. Same-UID code can produce an accepted exit, write the file and copy the
        nonce, so an accepted result stays an unauthenticated observation and attributes nothing. A missing exit,
        an unpaired or unknown exit, an absent file or an uncorrelated file all end as nonterminal uncertainty.
        """
        workdir = self.policy.workdir
        runner = f"{workdir}/{RUNTIME_PACKAGE}/runner.py"
        # The provider exec runs every command as root, and the runner refuses to run as root, so this command
        # drops privileges to the operator-configured os_user with runuser. The "--" stops runuser from parsing
        # the interpreter flags. Every path and the os_user are operator-fixed and pre-validated as quote-free.
        command = (
            f"runuser -u {self.policy.os_user} -- "
            f'"{self.policy.interpreter}" -I -B "{runner}" {mode} "{workdir}/job.json" "{workdir}/result.json" '
            ">/dev/null 2>&1"
        )
        # The provider must wait for the runner to reap the worker and remove its descendants after the wall
        # deadline, or a clean timeout run is cut short and mislabeled provider_exec. bounded_policy keeps
        # exec_timeout_margin_s at or above CLEANUP_BUDGET_S, so this wait always covers the deadline plus budget.
        timeout_s = deadline_s + int(self.policy.exec_timeout_margin_s)
        try:
            result = await asyncio.wait_for(
                self.backend.exec(self.handle, command, cwd=workdir, timeout_s=timeout_s),
                timeout=timeout_s + self.policy.exec_timeout_margin_s,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            raise _Unscorable("provider_exec") from None
        except Exception:
            raise _Unscorable("provider_exec") from None
        exit_code = _reported_exit(result)
        if exit_code is None:
            raise _Unscorable("provider_exec")  # the provider reported no process exit: a sentinel, or nothing
        if exit_code not in RESULT_EXIT_CODES:
            # The runner's own no-result exits (containment failed, worker aborted, protocol violation), usage, a
            # signal death or any unknown status: by protocol no result exists, so the file is never read.
            if self.name == "candidate" and exit_code in WORKER_ABORT_EXIT_CODES:
                # A worker abort the supervisor ran to completion and reported. The candidate runs its suite once
                # and last, so this can only follow a completed preflight. Attribute it to the candidate as a
                # reward-0 fault, not provider-ambiguous no-result.
                raise _CandidateWorkerAbort()
            raise _Unscorable(_NO_RESULT[self.name])
        try:
            data = await self._transfer(
                self.backend.download_bounded(
                    self.handle,
                    f"{workdir}/result.json",
                    limit=RESULT_BYTES_LIMIT,
                    timeout_s=self.policy.transfer_timeout_s,
                )
            )
        except MissingRemoteFile:
            # An absent file behind an accepted exit proves nothing about the domain that should have written it:
            # a lost sandbox, a killed process and a slow provider all look the same from here.
            raise _Unscorable(_NO_RESULT[self.name]) from None
        parsed = _parse_result(data, mode)
        _correlate_result(parsed, exit_code, nonce)
        return parsed

    async def _transfer(self, awaitable: Any) -> Any:
        try:
            return await asyncio.wait_for(awaitable, timeout=self.policy.transfer_timeout_s + _TRANSFER_GRACE_S)
        except (asyncio.CancelledError, MissingRemoteFile):
            raise
        except TransferLimitExceeded:
            raise _Unscorable("transfer_limit") from None
        except Exception:
            raise _Unscorable("provider_exec") from None

    async def close(self) -> str:
        """Resolve this domain exactly and return a ``CLEANUP_STATUSES`` value. A no-op once resolved.

        Delete is requested through the live handle, then absence is confirmed by bounded exact lookups. Every
        failure, timeout or cancellation leaves the journal with the id, stable name and owned labels needed to
        retry the same deletion. Cancellation is re-raised after that. Nothing is shielded.
        """
        if self.resolved:
            return self.cleanup
        handle = self.handle
        if handle is None:
            handle = await self._recover()
            if handle is None:
                return self.cleanup
        try:
            await asyncio.wait_for(self.backend.close(handle), timeout=self.policy.cleanup_timeout_s)
        except asyncio.CancelledError:
            self._mark("delete_failed", handle.sandbox_id)
            raise
        except BaseException as exc:
            self._mark("delete_failed", handle.sandbox_id)
            if not isinstance(exc, Exception):
                raise
            return self.cleanup
        try:
            absent = await self._confirm_absent(handle.sandbox_id)
        except asyncio.CancelledError:
            self._mark("delete_unconfirmed", handle.sandbox_id)
            raise
        if not absent:
            self._mark("delete_unconfirmed", handle.sandbox_id)
            return self.cleanup
        self.handle = None
        self._mark("deleted", handle.sandbox_id)
        return self.cleanup

    async def _recover(self) -> SandboxHandle | None:
        """Recover the live handle by exact id, or by exact stable name when no id was ever learned.

        A learned id is authoritative. Absence by exact id means the sandbox is gone, so the domain resolves
        ``deleted``. An ambiguous create may still be settling on the provider, so a single absent name lookup is
        not proof until the settle deadline passes. Before the deadline the name gets a bounded re-lookup window,
        and the domain keeps its unresolved intent so a later reconciliation can still delete a create that
        settles afterwards. At or after the deadline an absent exact-name lookup is authoritative.
        """
        entry = self.entry
        # Both states are ambiguous: a sandbox may still be settling remotely, so a single absent-name lookup is
        # not proof before the deadline.
        ambiguous_create = entry["state"] in {"create_uncertain", "creating"}
        try:
            found = await self._lookup(entry["sandbox_id"] or entry["name"])
            if found is None and ambiguous_create and not self._create_settle_deadline_passed():
                found = await self._await_created(entry["name"])
        except LookupFailed as exc:
            self.cleanup = f"lookup_{exc.kind}"
            return None
        if found is None:
            if ambiguous_create and not self._create_settle_deadline_passed():
                # Within the settle window a point-in-time absence proves neither cancellation nor terminal
                # failure of the create, so keep the tombstone for reconciliation.
                self.cleanup = "create_uncertain"  # unresolved
                return None
            self._mark("deleted")
            self.cleanup = "absent"
            return None
        if not self._owned(found):
            self.cleanup = "foreign_resource"
            return None
        return found.handle

    def _create_settle_deadline_passed(self) -> bool:
        """Whether an absent ambiguous create has waited past its settle deadline.

        The deadline is the dispatch time plus the create timeout plus a margin. A create this process dispatched
        is measured on the monotonic clock, so a wall-clock jump cannot trip it early. A disk-recovered record
        falls back to the persisted wall-clock ``dispatched_at``, then to the record's creation time, and a record
        with neither is treated as already past the deadline.
        """
        margin = max(_CREATE_SETTLE_MARGIN_FACTOR * self.policy.create_timeout_s, _CREATE_SETTLE_MARGIN_FLOOR_S)
        deadline = self.policy.create_timeout_s + margin
        if self._dispatched_monotonic is not None:
            return _monotonic() >= self._dispatched_monotonic + deadline
        dispatched_at = self.entry.get("dispatched_at")
        if not isinstance(dispatched_at, (int, float)):
            dispatched_at = self.record.get("created_at")
        if not isinstance(dispatched_at, (int, float)):
            return True
        return _now() >= dispatched_at + deadline

    async def _await_created(self, name: str) -> SandboxLookup | None:
        """Bounded re-lookup by exact stable name: give an ambiguous create time to become visible remotely."""
        delay = _ABSENCE_POLL_INTERVAL_S
        for _ in range(_CREATE_SETTLE_POLLS):
            await asyncio.sleep(delay)
            delay = min(delay * 2, _ABSENCE_POLL_MAX_S)
            found = await self._lookup(name)
            if found is not None:
                return found
        return None

    async def _lookup(self, key: str) -> SandboxLookup | None:
        try:
            return await asyncio.wait_for(self.backend.lookup(key), timeout=self.policy.cleanup_timeout_s)
        except (asyncio.CancelledError, LookupFailed):
            raise
        except asyncio.TimeoutError:
            raise LookupFailed("timeout") from None
        except Exception:
            raise LookupFailed("provider") from None

    def _owned(self, found: SandboxLookup) -> bool:
        """Exact identity plus the complete owned label set. Anything else is refused, never deleted."""
        entry = self.entry
        if entry["sandbox_id"] is not None and found.handle.sandbox_id != entry["sandbox_id"]:
            return False
        if found.name != entry["name"]:
            return False
        return all(found.labels.get(key) == value for key, value in entry["labels"].items())

    async def _confirm_absent(self, sandbox_id: str) -> bool:
        """Poll the exact id until the provider reports authoritative absence, within ``cleanup_timeout_s``."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.policy.cleanup_timeout_s
        delay = _ABSENCE_POLL_INTERVAL_S
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            try:
                found = await asyncio.wait_for(self.backend.lookup(sandbox_id), timeout=remaining)
            except asyncio.TimeoutError:
                return False
            except LookupFailed as exc:
                if exc.kind in ("permission", "malformed"):
                    return False
            except Exception:
                pass  # transient, keep polling until the deadline
            else:
                if found is None:
                    return True
            remaining = deadline - loop.time()
            if remaining <= 0:
                return False
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 2, _ABSENCE_POLL_MAX_S)


def _bounded_text(value: Any, limit: int) -> bool:
    return type(value) is str and len(value) <= limit


def _check_outcome(item: Any) -> bool:
    """One run outcome in one of the runner's three shapes. A value is a single accepted wire envelope, undecoded."""
    if type(item) is not dict or type(item.get("kind")) is not str:
        return False
    kind = item["kind"]
    if kind == "value":
        wire = item.get("value")
        if set(item) != {"kind", "value"} or type(wire) is not dict or set(wire) != _WIRE_KEYS:
            return False
        if wire["format"] != WIRE_FORMAT or type(wire["value"]) is not list:
            return False
        try:
            bounded_json(wire)  # the byte, node and depth limits every accepted task wire value already meets
        except (ValueError, UnicodeError, RecursionError, TypeError):
            return False
        return True
    if kind == "exception":
        names = set(item) - {"kind"}
        return names <= {"type", "message"} and all(
            item[name] is None or _bounded_text(item[name], MAX_TEXT_BYTES) for name in names
        )
    if kind == "encoding_error":
        return set(item) == {"kind", "code"} and _bounded_text(item["code"], MAX_CODE_LENGTH)
    return False


def _check_verdict(item: Any) -> bool:
    """One comparator verdict exactly as ``comparator._result`` shapes it. ``equal`` must agree with ``status``."""
    if type(item) is not dict or set(item) != _VERDICT_KEYS:
        return False
    status = item["status"]
    if type(item["version"]) is not int or item["version"] != _VERDICT_VERSION:
        return False
    if type(status) is not str or status not in VERDICT_STATUSES:
        return False
    if item["equal"] is not (True if status == "equal" else False if status == "mismatch" else None):
        return False
    if not all(item[name] is None or _bounded_text(item[name], _MAX_VERDICT_LABEL) for name in ("side", "code")):
        return False
    return _bounded_text(item["path"], _MAX_VERDICT_PATH)


def _validate_verdict_batch(results: list[dict], role: str, attempted: int = 0) -> None:
    """Check every role/status pair before disposition, without interpreting values or diagnostic codes.

    These sides mirror ``comparator.compare_request``'s ``_result`` calls. Expectation defects use ``expected``.
    Uncertainty can concern the observed role, the comparator or the request, but never an unrelated role.
    """
    sides = {
        "equal": {role},
        f"invalid_{role}": {role},
        "invalid_expected": {"expected"},
        "uncertain": {role, "comparator", "request"},
    }
    if role == "candidate":
        sides["mismatch"] = {role}
    if any(verdict["side"] not in sides.get(verdict["status"], ()) for verdict in results):
        raise _Unscorable("result_invalid", attempted)


def _parse_result(data: bytes, mode: str) -> dict:
    """Shape-validate an untrusted result file without decoding any value. Contents stay untrusted observations.

    Bounds apply in cost order: total bytes, then the fixed envelope and item count, then each item on its own. A
    ``value`` outcome carries one wire value held to the limits an accepted task wire value must meet, so no
    single item can exceed that budget. A malformed file is ``result_invalid`` and attributes nothing.
    """
    if type(data) is not bytes or len(data) > RESULT_BYTES_LIMIT:
        raise _Unscorable("result_invalid")
    try:
        result = json.loads(data.decode("utf-8"))
    except (ValueError, UnicodeError, RecursionError, TypeError):
        raise _Unscorable("result_invalid") from None
    key = "outcomes" if mode == RUN_MODE else "results"
    if type(result) is not dict or set(result) != _ENVELOPE_KEYS | {key}:
        raise _Unscorable("result_invalid")
    status, code, timed_out, items = result["status"], result["code"], result["timed_out_case"], result[key]
    statuses = RUN_STATUSES if mode == RUN_MODE else COMPARE_STATUSES
    codes = (
        SOURCE_ERROR_CODES if status == "source_error" else RUNNER_ERROR_CODES if status == "runner_error" else None
    )
    code_ok = code is None if codes is None else _bounded_text(code, MAX_CODE_LENGTH) and code in codes
    timed_out_ok = timed_out is None or (status == "timeout" and type(timed_out) is int and 0 <= timed_out < MAX_CASES)
    if (
        type(result["version"]) is not int
        or result["version"] != PROTOCOL_VERSION
        or result["mode"] != mode
        or not _bounded_text(result["nonce"], MAX_NONCE_LENGTH)
        or type(status) is not str
        or status not in statuses
        or not code_ok
        or not timed_out_ok
        or type(items) is not list
        or len(items) > MAX_CASES
    ):
        raise _Unscorable("result_invalid")
    check = _check_outcome if mode == RUN_MODE else _check_verdict
    if not all(check(item) for item in items):
        raise _Unscorable("result_invalid")
    return result


def _reported_exit(result: Any) -> int | None:
    """The process exit the provider independently reported, or ``None`` when it reported no process exit.

    A set ``error_type`` marks a runtime-failure sentinel rather than an exit, and a non-integer ``return_code``
    is no exit either. The type is exact, so ``True`` can never stand in for ``EXIT_SOURCE_ERROR``.
    """
    if getattr(result, "error_type", None) is not None:
        return None
    exit_code = getattr(result, "return_code", None)
    return exit_code if type(exit_code) is int else None


def _correlate_result(result: dict, exit_code: int, nonce: str) -> None:
    """Necessary cross-checks between a shape-valid result and what this host knows independently of the file.

    The provider-reported exit must be exactly the paired one, and the payload must echo this invocation's nonce.
    The nonce is correlation only, never authentication, since any same-UID code in the domain can read the job
    file. A mismatch is ``result_invalid`` and attributes nothing.
    """
    paired = _STATUS_EXIT_CODES.get(result["status"])
    if type(exit_code) is not int or exit_code != paired or result["nonce"] != nonce:
        raise _Unscorable("result_invalid")


def _runtime_files(names: tuple[str, ...]) -> dict[str, bytes]:
    return {f"{RUNTIME_PACKAGE}/{name}": (PACKAGE_DIR / name).read_bytes() for name in names}


def _limits(policy: ExecutionPolicy, deadline_s: int) -> dict[str, int]:
    return {
        "suite_deadline_s": deadline_s,
        "memory_mib": policy.memory_limit_mib,
        "cpu_time_s": min(policy.cpu_time_limit_s, deadline_s + int(policy.exec_timeout_margin_s)),
        "processes": policy.max_processes,
    }


def _dump(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def build_run_job(task: TaskData, entrypoint: str, nonce: str, policy: ExecutionPolicy) -> dict:
    """Normalized invocation inputs only: no expectations, tolerances or other cases' material."""
    return {
        "version": PROTOCOL_VERSION,
        "mode": RUN_MODE,
        "nonce": nonce,
        "entrypoint": entrypoint,
        "source_path": f"{policy.workdir}/source.py",
        # Always present as a list, possibly empty: the runner requires the key and applies the per-position
        # symbolic conversions before the call. None resolves to [], meaning no conversion for any position.
        "input_conversions": task.input_conversions or [],
        "cases": [
            {
                "args": [value.model_dump(mode="json") for value in case.args],
                "kwargs": {key: value.model_dump(mode="json") for key, value in case.kwargs.items()},
            }
            for case in task.test_cases
        ],
        "limits": _limits(policy, policy.suite_timeout_s),
    }


def build_compare_job(
    task: TaskData,
    role: str,
    outcomes: list[dict],
    nonce: str,
    policy: ExecutionPolicy,
    default_rtol: str | None = None,
    default_atol: str | None = None,
) -> dict:
    """Comparator input: trusted role, stored expectations and projected policy per case."""
    return {
        "version": PROTOCOL_VERSION,
        "mode": COMPARE_MODE,
        "nonce": nonce,
        "observed_role": role,
        "cases": [
            {
                "observed": outcomes[index],
                "expected": case.expected.model_dump(mode="json"),
                "policy": task.comparison_policy(index, default_rtol, default_atol),
            }
            for index, case in enumerate(task.test_cases[: len(outcomes)])
        ],
        "limits": _limits(policy, policy.compare_timeout_s),
    }


def validate_candidate_source(source: Any, policy: ExecutionPolicy) -> str | None:
    """Host-side contract check only. Never parses or executes. Returns a failed category or None."""
    if type(source) is not str or not source.strip() or "\x00" in source:
        return "candidate_source_limit"
    try:
        if len(source.encode("utf-8", errors="strict")) > policy.max_candidate_source_bytes:
            return "candidate_source_limit"
    except UnicodeEncodeError:
        return "candidate_source_limit"
    return None


class Grader:
    """One grading job at a time. Refuses to start while owned resources are unresolved.

    ``grade``, ``reconcile`` and ``shutdown`` share one lifecycle lock, so a delete never races live execution. A
    second job reports ``busy``, ``reconcile`` raises ``GraderBusy`` and ``shutdown`` drains with a bound. A
    returned verdict requires resolved final cleanup and journal resolution, otherwise ownership is unscorable.
    """

    def __init__(
        self,
        policy: ExecutionPolicy,
        *,
        backend_factory: Callable[[ExecutionPolicy], SandboxBackend] | None = None,
        journal: OwnershipJournal | None = None,
        default_rtol: str | None = None,
        default_atol: str | None = None,
    ):
        self.policy = policy
        # The operator's server default for a silent numeric leaf, transported verbatim to the comparator per
        # case. None leaves the comparator on its own SILENT_DEFAULT constants.
        self._default_rtol = default_rtol
        self._default_atol = default_atol
        self._backend_factory = backend_factory or DaytonaBackend
        self.journal = journal or OwnershipJournal(policy.journal_dir)
        self._lifecycle = asyncio.Lock()
        self._draining = False
        self._leftover: dict[str, dict] = {}  # records this process could not fully resolve at the end of a job
        # Per-job monotonic dispatch anchors, held only in memory. A record this process dispatched measures the
        # settle deadline on the monotonic clock. A disk-recovered record falls back to the wall clock.
        self._monotonic_anchors: dict[str, dict[str, float]] = {}

    async def grade(self, task: TaskData, candidate_source: str, *, job_id: str | None = None) -> GradeResult:
        if not isinstance(task, TaskData):
            raise TypeError("task must be a validated TaskData")
        job_id = job_id if job_id is not None and re.fullmatch(r"[a-z0-9]{8,32}", job_id) else secrets.token_hex(8)
        counts = {"attempted": 0, "equal": 0, "first_failed": None}
        cleanup: dict[str, str] = {}

        def unscorable(category: str) -> GradeResult:
            failure_class, terminal = UNSCORABLE_CATEGORIES[category]
            return GradeResult(
                "unscorable",
                None,
                category,
                failure_class,
                terminal,
                job_id,
                len(task.test_cases),
                counts["attempted"],
                counts["equal"],
                counts["first_failed"],
                cleanup,
            )

        contract = validate_candidate_source(candidate_source, self.policy)
        if contract is not None:
            return GradeResult("failed", 0.0, contract, None, None, job_id, len(task.test_cases), 0, 0, None, cleanup)
        if self._draining or self._lifecycle.locked():
            return unscorable("busy")
        if self._leftover or self.journal.unresolved():
            # A record left unresolved may already be gone on the provider. A transient delete-confirm delay,
            # or a create the provider never completed, clears on a re-check a moment later. Reconcile once
            # before refusing, so one slow round does not wedge a long-running server until a restart.
            # reconcile() merges this process's in-memory leftovers and the disk journal and runs under the
            # free lifecycle lock, so both sources are retried on this job. A reconcile that itself fails, such
            # as a journal write error, must not crash the job: swallow it and refuse below, so the gate still
            # fails closed. Refuse the job only if something is still unresolved after that one attempt.
            try:
                await self.reconcile()
            except Exception:
                pass
            if self._leftover or self.journal.unresolved():
                return unscorable("ownership_unresolved")
        async with self._lifecycle:
            names = {domain: f"ngcg-{job_id[:12]}-{domain}" for domain in DOMAINS}
            labels = {
                domain: {LABEL_OWNER: self.policy.owner_id, LABEL_JOB: job_id, LABEL_DOMAIN: domain}
                for domain in DOMAINS
            }
            record = self.journal.open_job(job_id, self.policy.owner_id, names, labels)
            anchors = self._monotonic_anchors.setdefault(job_id, {})
            domains = {
                name: _Domain(name, self.policy, self._backend_factory(self.policy), self.journal, record, anchors)
                for name in DOMAINS
            }
            try:
                try:
                    verdict = await asyncio.wait_for(
                        self._run_job(task, candidate_source, domains, counts),
                        timeout=self.policy.effective_job_timeout_s(),
                    )
                except asyncio.TimeoutError:
                    verdict = unscorable("job_deadline")
                except _Unscorable as exc:
                    counts["attempted"] = max(counts["attempted"], exc.attempted)
                    verdict = unscorable(exc.category)
            finally:
                try:
                    await self._close_domains(domains.values(), cleanup)
                finally:
                    try:
                        resolved = self.journal.resolve(record)
                    except Exception:
                        resolved = False  # Keep the exact record even if the unlink or its durability check failed.
                    resolved = resolved and all(domain.resolved for domain in domains.values())
                    if not resolved:
                        self._leftover[job_id] = record  # keep the anchors so a later cleanup reuses them
                    else:
                        self._monotonic_anchors.pop(job_id, None)
        if not resolved:
            return unscorable("ownership_unresolved")
        return GradeResult(**{**verdict.__dict__, "cleanup": cleanup})

    async def _run_job(self, task: TaskData, source: str, domains: dict[str, _Domain], counts: dict) -> GradeResult:
        policy, cases = self.policy, len(task.test_cases)
        reference, comparator, candidate = domains["reference"], domains["comparator"], domains["candidate"]
        # Nonces correlate individual invocations, not the whole job. They do not authenticate observations.
        reference_nonce = secrets.token_hex(16)

        # 1. Reference run in its own domain.
        await reference.create()
        await reference.upload(_runtime_files(RUNTIME_FILES))
        await reference.upload({"source.py": task.reference_source.encode("utf-8")})
        await reference.upload(
            {"job.json": _dump(build_run_job(task, task.effective_reference_entrypoint, reference_nonce, policy))}
        )
        reference_result = await reference.run(RUN_MODE, policy.suite_timeout_s, reference_nonce)
        await reference.close()
        if reference_result["status"] == "timeout":
            raise _Unscorable("reference_timeout")
        if reference_result["status"] != "completed":
            raise _Unscorable("reference_error")
        if len(reference_result["outcomes"]) != cases:
            raise _Unscorable("result_invalid")  # a completed run reports exactly one outcome per case
        # A reference observation must be a value, or an exception for a case whose stored expectation is an
        # exception, matching the historical grader. An exception for a value-expected case, or any encoding
        # error, is a broken reference and refuses the whole batch before preflight. A value for an
        # exception-expected case is admitted here and caught by preflight as a reference defect.
        for outcome, case in zip(reference_result["outcomes"], task.test_cases):
            if outcome["kind"] == "value":
                continue
            if outcome["kind"] == "exception" and case.expected.kind == "exception":
                continue
            raise _Unscorable("reference_error")

        # 2. Preflight: the comparator checks every reference value against stored expectations. A value that
        #    contradicts or cannot be represented is ``invalid_reference``. Candidate-role verdicts cannot occur
        #    here and mark a malformed result. Expectations come from the task only, never this run.
        await comparator.create()
        await comparator.upload(_runtime_files(COMPARATOR_FILES))
        preflight_nonce = secrets.token_hex(16)
        await comparator.upload(
            {
                "job.json": _dump(
                    build_compare_job(
                        task,
                        "reference",
                        reference_result["outcomes"],
                        preflight_nonce,
                        policy,
                        self._default_rtol,
                        self._default_atol,
                    )
                )
            }
        )
        preflight = await comparator.run(COMPARE_MODE, policy.compare_timeout_s, preflight_nonce)
        if preflight["status"] != "completed":
            raise _Unscorable("comparator_uncertain")
        if len(preflight["results"]) != cases:
            raise _Unscorable("result_invalid")
        _validate_verdict_batch(preflight["results"], "reference")
        # An earlier apparent defect cannot hide unresolved uncertainty elsewhere in the batch.
        if any(verdict["status"] == "uncertain" for verdict in preflight["results"]):
            raise _Unscorable("comparator_uncertain")
        for verdict in preflight["results"]:
            status = verdict["status"]
            if status == "equal":
                continue
            if status == "invalid_reference":
                raise _Unscorable("reference_mismatch")
            if status == "invalid_expected":
                raise _Unscorable("task_invalid")
            raise _Unscorable("comparator_uncertain" if status == "uncertain" else "result_invalid")

        # 3. Candidate run, only after a complete successful preflight.
        job_id = domains["reference"].record["job_id"]

        def failed(category: str, first: int | None) -> GradeResult:
            counts["first_failed"] = first
            return GradeResult(
                "failed", 0.0, category, None, None, job_id, cases, counts["attempted"], counts["equal"], first, {}
            )

        await candidate.create()
        await candidate.upload(_runtime_files(RUNTIME_FILES))
        await candidate.upload({"source.py": source.encode("utf-8")})
        candidate_nonce = secrets.token_hex(16)
        await candidate.upload({"job.json": _dump(build_run_job(task, task.entrypoint, candidate_nonce, policy))})
        try:
            candidate_result = await candidate.run(RUN_MODE, policy.suite_timeout_s, candidate_nonce)
        except _CandidateWorkerAbort:
            # The candidate worker aborted or broke the runner protocol after a completed preflight. No result was
            # written, so no case was attempted and none failed. Score it 0 like the other candidate faults.
            return failed("candidate_worker_abort", None)
        await candidate.close()
        outcomes = candidate_result["outcomes"]
        counts["attempted"] = min(len(outcomes), cases)
        status = candidate_result["status"]

        # A candidate source error or suite timeout after a completed preflight is a candidate-attributable fault,
        # scored 0 with the category preserved. The bounded code and correlated exit do not authenticate a cause,
        # so the category names what the domain reported, not a proof. A whole-run fault leaves first_failed unset.
        if status == "source_error":
            return failed("candidate_source_error", None)
        if status == "timeout":
            return failed("candidate_timeout", None)
        if status != "completed":
            raise _Unscorable("provider_exec", counts["attempted"])
        if len(outcomes) != cases:
            raise _Unscorable("result_invalid", counts["attempted"])
        # Inspect every observation before any verdict can score. An exception for a case whose stored expectation
        # is an exception is not a fault, matching the historical grader, and reaches the comparator to be judged
        # equal. An exception for a value-expected case, or any encoding error, is a candidate-attributable fault
        # scored 0 at the first offending case, never an authenticated pass.
        for index, outcome in enumerate(outcomes):
            if outcome["kind"] == "exception" and task.test_cases[index].expected.kind != "exception":
                return failed("candidate_exception", index)
            if outcome["kind"] == "encoding_error":
                return failed("candidate_encoding_error", index)

        # 4. Candidate value verdicts from the same comparator domain. Expectations never left it.
        comparison_nonce = secrets.token_hex(16)
        await comparator.upload(
            {
                "job.json": _dump(
                    build_compare_job(
                        task,
                        "candidate",
                        outcomes,
                        comparison_nonce,
                        policy,
                        self._default_rtol,
                        self._default_atol,
                    )
                )
            }
        )
        judged = await comparator.run(COMPARE_MODE, policy.compare_timeout_s, comparison_nonce)
        await comparator.close()
        # The comparator handled the reference side cleanly at preflight, so a judge-stage timeout is
        # candidate-attributable. Score it 0 as candidate_invalid_output at the case the runner timed out on.
        # Every other non-completed judge status is unattributed comparator uncertainty.
        if judged["status"] == "timeout":
            timed_out_case = judged["timed_out_case"]
            return failed("candidate_invalid_output", timed_out_case if type(timed_out_case) is int else None)
        if judged["status"] != "completed":
            raise _Unscorable("comparator_uncertain", cases)
        if len(judged["results"]) != cases:
            raise _Unscorable("result_invalid", cases)
        _validate_verdict_batch(judged["results"], "candidate", cases)
        if any(verdict["status"] in {"uncertain", "invalid_expected"} for verdict in judged["results"]):
            raise _Unscorable("comparator_uncertain", cases)
        for index, verdict in enumerate(judged["results"]):
            status = verdict["status"]
            if status == "equal":
                counts["equal"] += 1
                continue
            if status == "mismatch":
                return failed("candidate_mismatch", index)
            if status == "invalid_candidate":
                return failed("candidate_invalid_output", index)
            if status == "invalid_reference":
                raise _Unscorable("result_invalid", cases)  # a reference-role verdict cannot judge the candidate
            raise _Unscorable("comparator_uncertain", cases)
        return GradeResult("passed", 1.0, PASSED, None, None, job_id, cases, cases, counts["equal"], None, {})

    async def _close_domains(self, domains: Iterable[_Domain], statuses: dict[str, str]) -> None:
        """Give every domain one bounded close and one bounded client close, in order.

        A cancellation arriving mid-way is remembered and re-raised only after the remaining domains had their
        turn, so a cancelled job still deletes what it can. An ordinary close failure is remembered the same way
        and never aborts cleanup of the remaining domains, so one domain's failure cannot strand another's live
        sandbox. Cancellation still wins over a remembered ordinary error.
        """
        pending: BaseException | None = None
        for domain in domains:
            try:
                await domain.close()
            except asyncio.CancelledError as exc:
                pending = exc
            except Exception as exc:
                if not isinstance(pending, asyncio.CancelledError):
                    pending = exc
            finally:
                statuses[domain.name] = domain.cleanup
            try:
                await asyncio.wait_for(domain.backend.aclose(), timeout=self.policy.cleanup_timeout_s)
            except asyncio.CancelledError as exc:
                pending = exc
            except Exception:
                pass
            domain.release()
        if pending is not None:
            raise pending

    async def _resolve_record(self, record: dict, report: list[dict[str, str]]) -> bool:
        """Retry exact cleanup of one record's unresolved domains with fresh clients. True once fully resolved.

        A record this process dispatched keeps its monotonic anchors, so a rebuilt client measures the settle
        deadline on the monotonic clock. A disk-recovered record falls back to the wall clock.
        """
        anchors = self._monotonic_anchors.get(record["job_id"])
        domains = [
            _Domain(name, self.policy, self._backend_factory(self.policy), self.journal, record, anchors)
            for name in DOMAINS
            if name in record["domains"] and record["domains"][name]["state"] not in RESOLVED_STATES
        ]
        statuses: dict[str, str] = {}
        try:
            await self._close_domains(domains, statuses)
        finally:
            for name, status in statuses.items():
                report.append({"job_id": record["job_id"], "domain": name, "status": status})
            resolved = self.journal.resolve(record)
        return resolved

    async def reconcile(self) -> list[dict[str, str]]:
        """Exact reconciliation of every unresolved journal record.

        Per domain: lookup by exact id (or exact stable name when no id was learned), identity and complete owned
        label verification, delete through the recovered live handle, then bounded absence confirmation. Report
        statuses are ``CLEANUP_STATUSES`` values plus ``corrupt_record``. Raises ``GraderBusy`` while a job runs.
        """
        if self._lifecycle.locked():
            raise GraderBusy("a grading job holds the lifecycle lock")
        async with self._lifecycle:
            report: list[dict[str, str]] = []
            # A failed final journal resolution can leave only the in-memory record, even after all deletes.
            records = dict(self._leftover)
            records.update((record["job_id"], record) for record in self.journal.unresolved())
            for record in records.values():
                if record.get("corrupt"):
                    report.append({"job_id": record["job_id"], "status": "corrupt_record"})
                    continue
                if await self._resolve_record(record, report):
                    self._leftover.pop(record["job_id"], None)
                    self._monotonic_anchors.pop(record["job_id"], None)
            return report

    async def shutdown(self) -> list[dict[str, str]]:
        """Drain: refuse new jobs, wait a bounded time for a running job to finish its own cleanup, then retry
        exact cleanup of whatever this process left unresolved. Never deletes under a running job. A job that
        outlives the bound keeps its journal record for a later ``reconcile``."""
        self._draining = True
        try:
            await asyncio.wait_for(self._lifecycle.acquire(), timeout=self.policy.cleanup_timeout_s)
        except asyncio.TimeoutError:
            return [{"status": "busy"}]
        try:
            report: list[dict[str, str]] = []
            # Scan disk too, the way reconcile does, so a disk-recovered record whose first delete failed
            # gets another attempt at shutdown rather than waiting for the next restart.
            records = dict(self._leftover)
            records.update((record["job_id"], record) for record in self.journal.unresolved())
            for job_id, record in list(records.items()):
                if record.get("corrupt"):
                    report.append({"job_id": job_id, "status": "corrupt_record"})
                    continue
                if await self._resolve_record(record, report):
                    self._leftover.pop(job_id, None)
                    self._monotonic_anchors.pop(job_id, None)
            return report
        finally:
            self._lifecycle.release()


__all__ = [
    "CANDIDATE_FAULT_CATEGORIES",
    "CLEANUP_STATUSES",
    "DaytonaBackend",
    "ExecutionPolicy",
    "FAILED_CATEGORIES",
    "GradeResult",
    "Grader",
    "GraderBusy",
    "LOOKUP_ERROR_KINDS",
    "LookupFailed",
    "OwnershipJournal",
    "PASSED",
    "SandboxBackend",
    "SandboxCreateRejected",
    "SandboxLookup",
    "UNSCORABLE_CATEGORIES",
    "build_compare_job",
    "build_run_job",
    "validate_candidate_source",
]
