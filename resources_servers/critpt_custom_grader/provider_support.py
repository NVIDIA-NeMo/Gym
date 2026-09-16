# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A local Daytona provider subclass, so this grader runs against the unmodified shared provider.

The shared ``DaytonaProvider`` on its own does two things this grader does not want:

1. An exec whose SDK response carries no process exit code is returned as ``return_code=0``, i.e.
   success. This grader must never read an unknown exit status as success.
2. Its runtime failure text and its lifecycle log lines quote sandbox ids and SDK error reprs.

``CategoryOnlyDaytonaProvider`` fixes both for the exec path, without editing shared code:

- A missing or non-integer exit code becomes a typed sandbox failure (``return_code`` set to the
  runtime sentinel, ``error_type='sandbox'``), never an implicit success.
- With ``category_only_diagnostics=True`` the provider's own exec text and log lines are fixed
  categories, so no provider-authored free text is emitted on the exec path.

Scope. Only ``_exec`` is overridden. That is the one shared method whose result-building this
grader must change, and it is the one path where the shared provider would otherwise assume
success. This grader does not run the provider's create-verify probe (it configures none) and does
not use the provider's own ``exec`` for command execution (it runs commands through the Daytona
session API in ``DaytonaBackend.exec``), so ``_exec`` is reached only through a configured probe.
The override is therefore defensive and directly tested.

Not reproduced from the shared provider's category-only mode: the fixed-category text and logs on
the create, batch-create, cleanup, retry and close paths. Those paths carry sandbox ids and SDK
error reprs, never the task statement or the candidate response, and this grader never puts a
provider exception into an HTTP response or a log (it converts every provider outcome to a fixed
category at its own boundary and suppresses provider exception text with ``raise ... from None``).
So the grader's privacy contract does not depend on them. Reproducing them would mean re-vendoring
most of the shared provider, which is the shared change this rebuild drops.
"""

import asyncio

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.sandbox.providers.daytona.provider import (
    LOGGER,
    DaytonaCreateError,
    DaytonaProvider,
    _daytona_exec_error_message,
    _daytona_exec_error_result,
    _is_daytona_command_exec_error,
)


# Envelope value for an exec that produced no process exit code. Meaningful only together with
# ``SandboxExecResult.error_type``. Matches the shared providers' runtime sentinel.
SANDBOX_RUNTIME_RETURN_CODE = 125
SANDBOX_TIMEOUT_RETURN_CODE = 124


class DaytonaCreateCleanupError(DaytonaCreateError):
    """A create failure whose sandbox could not be deleted, carrying the leaked ``sandbox_id``.

    The grader's create path reads ``sandbox_id`` off a failed-create exception by attribute
    (``getattr(exc, "sandbox_id", None)``), never by type, and journals a ``delete_failed`` state
    when it is present. This local error is the grader-owned stand-in for that contract; nothing on
    the grader's own configured path raises it, because the grader configures no create-verify
    probe.
    """

    def __init__(self, message: str, *, sandbox_id: str) -> None:
        super().__init__(message)
        self.sandbox_id = sandbox_id


def _missing_exit_code_result(stdout: str | None, stderr: str | None) -> SandboxExecResult:
    """Typed sandbox failure for an SDK response that carries no process exit code.

    The output that did arrive is kept, but the result must not read as success: ``return_code`` is
    the runtime sentinel and ``error_type`` says why.
    """
    detail = "Daytona process.exec returned no process exit code; the command's exit status is unknown"
    return SandboxExecResult(
        stdout=stdout,
        stderr=f"{stderr}\n{detail}" if stderr else detail,
        return_code=SANDBOX_RUNTIME_RETURN_CODE,
        error_type="sandbox",
    )


class CategoryOnlyDaytonaProvider(DaytonaProvider):
    """Shared ``DaytonaProvider`` with a missing-exit-code guard and category-only exec diagnostics."""

    def __init__(self, *, category_only_diagnostics: bool = False, **kwargs) -> None:
        if type(category_only_diagnostics) is not bool:
            raise TypeError("category_only_diagnostics must be a boolean")
        self._category_only_diagnostics = category_only_diagnostics
        super().__init__(**kwargs)

    def _warn(self, category: str, message: str, *args) -> None:
        if self._category_only_diagnostics:
            LOGGER.warning(category)
        else:
            LOGGER.warning(message, *args)

    def _exec_error_result(self, exception: BaseException) -> SandboxExecResult:
        """The exec-error result, category-only when the flag is set, else the shared provider's text."""
        if not self._category_only_diagnostics:
            return _daytona_exec_error_result(exception)
        message = _daytona_exec_error_message(exception)
        is_timeout = "timeout" in message.lower() or "timed out" in message.lower()
        return SandboxExecResult(
            stdout=None,
            stderr="daytona.command_exec_timeout" if is_timeout else "daytona.command_exec_failed",
            return_code=SANDBOX_TIMEOUT_RETURN_CODE if is_timeout else SANDBOX_RUNTIME_RETURN_CODE,
            error_type="timeout" if is_timeout else "sandbox",
        )

    async def _exec(
        self,
        handle,
        command,
        *,
        cwd=None,
        env=None,
        timeout_s=None,
        user=None,
        retries=None,
    ) -> SandboxExecResult:
        # A faithful reimplementation of the shared provider's ``_exec`` tail, changed in two ways:
        # a missing or non-integer exit code becomes a typed sandbox failure, and the provider's own
        # exec text and log lines are fixed categories under ``category_only_diagnostics``.
        try:
            response = await self._await_operation(
                lambda: handle.raw.process.exec(
                    self._effective_command(command, user),
                    cwd=cwd,
                    env=env,
                    timeout=timeout_s,
                ),
                operation="process.exec",
                sandbox_id=handle.sandbox_id,
                timeout_s=float(timeout_s) + self._operations.command_timeout_margin_s
                if timeout_s is not None
                else None,
                retries=self._command_retry_count() if retries is None else retries,
            )
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if not _is_daytona_command_exec_error(e):
                raise
            self._warn(
                "daytona.command_exec_failed",
                "Daytona process.exec failed without a process result; sandbox_id=%s; error=%r",
                handle.sandbox_id,
                e,
            )
            return self._exec_error_result(e)
        artifacts = getattr(response, "artifacts", None)
        stdout = getattr(response, "result", None)
        if stdout is None and artifacts is not None:
            stdout = getattr(artifacts, "stdout", None)
        stderr = getattr(response, "stderr", None)
        if stderr is None and artifacts is not None:
            stderr = getattr(artifacts, "stderr", None)
        return_code = getattr(response, "exit_code", None)
        if isinstance(return_code, bool) or not isinstance(return_code, int):
            # No process exit status arrived. Success cannot be assumed; the caller gets a typed
            # sandbox failure with whatever output was returned.
            self._warn(
                "daytona.command_exit_missing",
                "Daytona process.exec returned no process exit code; sandbox_id=%s",
                handle.sandbox_id,
            )
            return _missing_exit_code_result(stdout, stderr)
        return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=return_code)
