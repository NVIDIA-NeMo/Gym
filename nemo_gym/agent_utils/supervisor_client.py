# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side transport for the shared sandbox process supervisor.

Session ownership and cancellation live in sandbox_session.py; output parsing stays with the adapter.
Unlike process_supervisor.sh, this module is not uploaded to the task sandbox.

Session control files (all paths are relative to session_dir):

===================== ==================== ===========================================
Filename              Writer               When
===================== ==================== ===========================================
process_supervisor.sh Session              Uploaded before launch
bin/gym-runner        Launch shell         Interpreter link for a standard-library runner
launch.claim          Launch or stop shell Atomically claims launch or fences it
supervisor.pid        Launch shell         After claiming launch, before exec (see below)
stop.request          Stop shell           Polled by the supervisor to stop the harness
cleanup.json          Supervisor or stop   After cleanup, or when stop fences launch
output.log            Launch shell         Captures supervisor and harness diagnostics
===================== ==================== ===========================================

Agents in the task sandbox often clean up with ``pkill -f python3``, ``pkill python3`` or
``killall python3``. These must not stop the supervisor or a runner, which would end the
harness. The supervisor and the shared harness runner are bash scripts, so their names
never mention python. A standard-library Python runner that opts in (Codex) runs under an
interpreter link with a neutral name, from a file without a ``.py`` suffix, so neither its
process name nor the supervisor's command line mentions python; a harness argv that keeps
its own python (Hermes, Stirrup) still appears in the supervisor's command line. Gym stops
the supervisor through the stop file, which does not depend on these names.

``supervisor.pid`` holds the PID that executes the supervisor. When the provider did not
make the launch shell a process group leader, that process starts the supervisor in a new
group, passes TERM on to it and exits with its status.
"""

import json
import logging
from pathlib import Path
from shlex import join, quote
from typing import TypedDict
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.utils import read_text


SUPERVISOR_FILE = "process_supervisor.sh"
SUPERVISOR_SCRIPT = Path(__file__).with_name(SUPERVISOR_FILE)
RUNNER_INTERPRETER = "bin/gym-runner"
LAUNCH_CLAIM_FILE = "launch.claim"
SUPERVISOR_PID_FILE = "supervisor.pid"
STOP_REQUEST_FILE = "stop.request"
CLEANUP_RECEIPT_FILE = "cleanup.json"
OUTPUT_LOG_FILE = "output.log"

DEFAULT_CLEANUP_TIMEOUT = 10.0
CLEANUP_PHASE_COUNT = 3


class CleanupReceipt(TypedDict):
    """Outcome after bounded harness termination, as written by process_supervisor.sh.

    ``cleanup_confirmed`` means no live harness-group members remain. After a forced
    stop it also means that no process the stop reached still runs: the harness's
    descendants, including tools in their own sessions, and processes that inherited
    the harness's ``NEMO_GYM_SUPERVISOR`` marker without having left the harness
    before the stop. Processes that left the harness before the stop, and a process
    that left between two scans without the marker, remain until the owner closes the
    sandbox; this is not a sandbox-empty check. ``return_code`` is the shell status:
    128+N for a harness killed by signal N, and 126 or 127 when the command could not
    be run.
    """

    cleanup_confirmed: bool
    return_code: int | None
    error: str | None
    timed_out: bool


def exec_timeout(*, timeout: float, cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT) -> float:
    """Reserve TERM grace, direct-process reaping, and process-group reaping.

    The extra 30 seconds cover startup, polling, and receipt I/O beyond those
    bounded cleanup phases. Keep their count shared with the controller budget.
    """
    return timeout + CLEANUP_PHASE_COUNT * cleanup_timeout + 30


_CLEANUP_RECEIPT = TypeAdapter(CleanupReceipt)
# The interpreter binary itself, past virtual-environment links and launcher shims.
_RESOLVE_INTERPRETER = 'import os; print(os.readlink("/proc/self/exe"))'
LOG = logging.getLogger(__name__)


class HarnessTerminationError(RuntimeError):
    """A final supervisor receipt could not confirm the harness stopped.

    The sandbox owner must tear down the sandbox. Re-reading this terminal
    receipt cannot recover execution, but a borrowed client can release it.
    """

    def __init__(self, *, harness: str, receipt: CleanupReceipt) -> None:
        self.receipt: CleanupReceipt = receipt
        super().__init__(f"{harness} harness termination was not confirmed: {receipt['error']}")


def _raise_terminal_failure(payload: object, *, harness: str) -> None:
    if not isinstance(payload, dict) or payload.get("cleanup_confirmed") is not False:
        return
    try:
        receipt = _CLEANUP_RECEIPT.validate_python(payload, strict=True, extra="forbid")
    except ValidationError:
        return  # Malformed evidence remains unknown and retryable.
    raise HarnessTerminationError(harness=harness, receipt=receipt)


def parse_cleanup_receipt(payload: object) -> CleanupReceipt:
    """Require positive cleanup evidence, tolerating malformed optional diagnostics."""
    if not isinstance(payload, dict) or payload.get("cleanup_confirmed") is not True:
        raise ValueError("Harness termination was not confirmed")
    try:
        return _CLEANUP_RECEIPT.validate_python(payload, strict=True, extra="forbid")
    except ValidationError:
        LOG.warning("Harness termination was confirmed, but its diagnostic fields did not match the receipt schema")
    return {
        "cleanup_confirmed": True,
        "return_code": payload.get("return_code") if type(payload.get("return_code")) is int else None,
        "timed_out": payload.get("timed_out") is True,
        "error": payload.get("error") if isinstance(payload.get("error"), str) else None,
    }


def supervision_timeouts(*, timeout: float, close_timeout: float) -> tuple[float, float]:
    """Return the per-cleanup-phase and provider execution deadlines.

    Divide the close budget across TERM grace, direct-process and group reaping.
    The provider deadline reserves all phases after
    the harness timeout, plus 30 seconds for startup, polling, and receipt I/O.
    """
    cleanup_timeout = close_timeout / CLEANUP_PHASE_COUNT
    return cleanup_timeout, exec_timeout(timeout=timeout, cleanup_timeout=cleanup_timeout)


def _seconds(value: float) -> str:
    """Seconds as the supervisor parses them: fixed-point, at least one centisecond."""
    return str(value) if 0.01 <= value < 1e16 else f"{max(value, 0.01):.2f}"


def check_stdlib_runner(command: list[str], *, stdlib_runner: bool) -> None:
    """Reject ``stdlib_runner`` for a harness argv without an interpreter and a runner to rename."""
    if stdlib_runner and len(command) < 2:
        raise ValueError(f"stdlib_runner requires a harness argv that starts with its interpreter; got {command!r}")


def supervised_launch_command(
    *,
    session_dir: str,
    command: list[str],
    timeout: float,
    cleanup_timeout: float,
    stdlib_runner: bool = False,
) -> str:
    """Fence delayed launches and run a harness command under the shared supervisor.

    The supervisor needs only bash. Check that bash can parse it before claiming a
    launch: a failed bootstrap cannot write a cleanup receipt. Supervisor and harness
    diagnostics share output.log. The supervisor leaves the provider's process group and
    stops the harness group, and after a forced stop the processes it reached; the
    sandbox owner stops services that left the harness earlier. Stops reach the
    supervisor through the stop file, not a signal, so the harness keeps the default
    TERM disposition.

    With ``stdlib_runner``, the harness argv starts with a Python interpreter, and the
    runner runs through ``bin/gym-runner``, a link to the binary behind that interpreter,
    so that python-matching kills in the task skip it. The link resolves a virtual
    environment to its base interpreter, so only standard-library runners may opt in.
    When the link cannot be made, the argv runs unchanged.
    """
    check_stdlib_runner(command, stdlib_runner=stdlib_runner)
    supervisor = quote(f"{session_dir}/{SUPERVISOR_FILE}")
    claim_path = quote(f"{session_dir}/{LAUNCH_CLAIM_FILE}")
    harness = join(command)
    runner = ""
    if stdlib_runner:
        python = quote(command[0])
        bin_dir = quote(f"{session_dir}/bin")
        runner_link = quote(f"{session_dir}/{RUNNER_INTERPRETER}")
        # Create only bin, so a launch delayed past close cannot recreate the session directory.
        runner = (
            f"run={python}; real=$({python} -I -c {quote(_RESOLVE_INTERPRETER)} 2>/dev/null) && "
            f'{{ [ -d {bin_dir} ] || mkdir {bin_dir}; }} && ln -sfn "$real" {runner_link} && run={runner_link}; '
        )
        harness = f'"$run" {join(command[1:])}'
    return (
        f"[ -d {quote(session_dir)} ] && [ ! -L {claim_path} ] || exit 0; "
        f"bash -n {supervisor} || exit $?; "
        f"{runner}ln -s launch {claim_path} 2>/dev/null || exit 0; "
        f"echo $$ > {quote(f'{session_dir}/{SUPERVISOR_PID_FILE}')} && "
        f"exec bash {supervisor} "
        f"--timeout {_seconds(timeout)} --cleanup-timeout {_seconds(cleanup_timeout)} "
        f"--stop-file {quote(f'{session_dir}/{STOP_REQUEST_FILE}')} "
        f"--receipt {quote(f'{session_dir}/{CLEANUP_RECEIPT_FILE}')} -- {harness} "
        f">{quote(f'{session_dir}/{OUTPUT_LOG_FILE}')} 2>&1"
    )


async def stop_and_confirm_cleanup(
    sandbox: AsyncSandbox, *, session_dir: str, workdir: str | None, timeout: float, harness: str
) -> CleanupReceipt:
    """Fence a pending launch or require explicit supervisor acknowledgement before teardown.

    A stop that wins the launch claim writes the same receipt shape as the
    supervisor, with no return code because no harness process ran.
    """
    receipt_path = f"{session_dir}/{CLEANUP_RECEIPT_FILE}"
    try:
        receipt = json.loads(await read_text(sandbox, path=receipt_path))
    except Exception:
        receipt = {}
    _raise_terminal_failure(receipt, harness=harness)
    if not isinstance(receipt, dict) or receipt.get("cleanup_confirmed") is not True:
        stop_path = quote(f"{session_dir}/{STOP_REQUEST_FILE}")
        claim_path = quote(f"{session_dir}/{LAUNCH_CLAIM_FILE}")
        temporary = quote(f"{receipt_path}.{uuid4().hex}.tmp")
        stopped = quote(
            json.dumps({"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": False})
        )
        script = (
            f"[ -f {quote(receipt_path)} ] && exit 0; "
            f"touch {stop_path} || exit 1; "
            f"ln -s stop {claim_path} 2>/dev/null || true; "
            f'if [ "$(readlink {claim_path})" = stop ]; then '
            f"printf '%s' {stopped} > {temporary} && mv {temporary} {quote(receipt_path)}; exit $?; fi; "
            f"for _ in $(seq 1 {max(1, int(timeout))}); do "
            f"[ -f {quote(receipt_path)} ] && exit 0; sleep 1; done; exit 1"
        )
        await sandbox.exec(script, cwd=workdir, timeout_s=timeout + 5)
        try:
            receipt = json.loads(await read_text(sandbox, path=receipt_path))
        except Exception as error:
            raise RuntimeError(f"{harness} launch outcome is unknown; cannot confirm termination") from error
        if not isinstance(receipt, dict) or receipt.get("cleanup_confirmed") is not True:
            detail = receipt.get("error") if isinstance(receipt, dict) else "cleanup receipt is not a JSON object"
            _raise_terminal_failure(receipt, harness=harness)
            raise RuntimeError(f"{harness} harness termination was not confirmed: {detail}")
    return parse_cleanup_receipt(receipt)


async def remove_session_directory(
    sandbox: AsyncSandbox, *, session_dir: str, workdir: str | None, timeout: float, harness: str
) -> None:
    """Remove adapter-owned files after cleanup, keeping delayed launches fenced.

    Retire the directory atomically before unlinking its claim. Otherwise a
    delayed launch could win the claim while recursive deletion is in progress.
    A failed removal leaves the retired path available for a close retry.
    """
    retired = f"{session_dir}.closed"
    result = await sandbox.exec(
        f"if [ -d {quote(session_dir)} ]; then "
        f"mv {quote(session_dir)} {quote(retired)} || exit 1; fi; rm -rf -- {quote(retired)}",
        cwd=workdir,
        timeout_s=timeout,
    )
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f"Could not remove {harness} session files")
