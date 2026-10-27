# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side transport for the shared sandbox process supervisor.

Session ownership and cancellation live in sandbox_session.py; output parsing stays with the adapter.
Unlike process_supervisor.py, this module is not uploaded to the task sandbox.

Session control files (all paths are relative to session_dir):

===================== ==================== ===========================================
Filename              Writer               When
===================== ==================== ===========================================
process_supervisor    Session              Uploaded before launch
bin/gym-supervisor    Launch shell         Interpreter link for the supervisor
bin/gym-runner        Launch shell         Interpreter link for a standard-library runner
launch.claim          Launch or stop shell Atomically claims launch or fences it
supervisor.pid        Launch shell         After claiming launch, before exec
stop.request          Stop shell           Before signalling the supervisor
cleanup.json          Supervisor or stop   After cleanup, or when stop fences launch
output.log            Launch shell         Captures supervisor and harness diagnostics
===================== ==================== ===========================================

Agents in the task sandbox often clean up with ``pkill -f python3``, ``pkill python3`` or
``killall python3``. In a reference harness whose agent CLI is a native binary these
commands are harmless; here they would stop the supervisor, which then ends the
harness. So the supervisor, and a standard-library runner that opts in, run under
interpreter links with neutral names, from files without a ``.py`` suffix. Their process
names never mention python, and neither do their command lines when the runner opts in; a
harness argv that keeps its own python (Hermes, Stirrup) still appears in the supervisor's
command line. Gym's own stop path signals the recorded supervisor PID and does not depend
on these names.
"""

import json
import logging
from shlex import join, quote
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from nemo_gym.agent_utils.process_supervisor import CLEANUP_PHASE_COUNT, CleanupReceipt, exec_timeout
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.utils import read_text


SUPERVISOR_FILE = "process_supervisor"
SUPERVISOR_INTERPRETER = "bin/gym-supervisor"
RUNNER_INTERPRETER = "bin/gym-runner"
LAUNCH_CLAIM_FILE = "launch.claim"
SUPERVISOR_PID_FILE = "supervisor.pid"
STOP_REQUEST_FILE = "stop.request"
CLEANUP_RECEIPT_FILE = "cleanup.json"
OUTPUT_LOG_FILE = "output.log"

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


def check_stdlib_runner(command: list[str], *, python: str, stdlib_runner: bool) -> None:
    """Reject ``stdlib_runner`` for a harness argv that the runner link cannot replace."""
    if stdlib_runner and command[:1] != [python]:
        raise ValueError(
            f"stdlib_runner requires a harness argv that starts with the interpreter {python!r}; got {command[:1]!r}"
        )


def supervised_launch_command(
    *,
    session_dir: str,
    command: list[str],
    timeout: float,
    cleanup_timeout: float,
    python: str,
    stdlib_runner: bool = False,
) -> str:
    """Fence delayed launches and run a harness command under the shared supervisor.

    The adapter must install or select the interpreter explicitly. Check that it
    can load the supervisor before claiming a launch: a failed bootstrap cannot
    write a cleanup receipt. Supervisor and harness diagnostics share output.log.
    The supervisor stops the harness group; the sandbox owner stops escaped services.

    The supervisor runs through ``bin/gym-supervisor``, a link to the binary behind
    ``python``, so that python-matching kills in the task skip it. With ``stdlib_runner``,
    the harness command, which must start with ``python``, runs through ``bin/gym-runner``
    the same way. The links resolve a virtual environment to its base interpreter, so only
    standard-library runners may opt in. When the link cannot be made or cannot load
    the supervisor, both processes keep ``python``.
    """
    check_stdlib_runner(command, python=python, stdlib_runner=stdlib_runner)
    supervisor = quote(f"{session_dir}/{SUPERVISOR_FILE}")
    claim_path = quote(f"{session_dir}/{LAUNCH_CLAIM_FILE}")
    link = quote(f"{session_dir}/{SUPERVISOR_INTERPRETER}")
    harness = join(command)
    runner = ""
    if stdlib_runner:
        runner_link = quote(f"{session_dir}/{RUNNER_INTERPRETER}")
        runner = f'run={quote(python)}; [ "$py" = {link} ] && ln -sfn "$real" {runner_link} && run={runner_link}; '
        harness = f'"$run" {join(command[1:])}'
    return (
        f"[ -d {quote(session_dir)} ] && [ ! -L {claim_path} ] || exit 0; "
        f"py={quote(python)}; "
        f"real=$({quote(python)} -I -c {quote(_RESOLVE_INTERPRETER)} 2>/dev/null) && "
        f'{{ [ -d {quote(f"{session_dir}/bin")} ] || mkdir {quote(f"{session_dir}/bin")}; }} && ln -sfn "$real" {link} && '
        f"{link} -I {supervisor} --help >/dev/null 2>&1 && py={link}; "
        f'[ "$py" = {link} ] || {quote(python)} -I {supervisor} --help >/dev/null || exit $?; '
        f"{runner}trap '' TERM; ln -s launch {claim_path} 2>/dev/null || exit 0; "
        f"echo $$ > {quote(f'{session_dir}/{SUPERVISOR_PID_FILE}')} && "
        f'exec "$py" -I {supervisor} '
        f"--timeout {timeout} --cleanup-timeout {cleanup_timeout} "
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
        pid_path = quote(f"{session_dir}/{SUPERVISOR_PID_FILE}")
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
            f'if [ -s {pid_path} ]; then kill -TERM "$(cat {pid_path})" 2>/dev/null || true; fi; '
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
