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
launch.claim          Launch or stop shell Atomically claims launch or fences it
supervisor.pid        Launch shell         After claiming launch, before exec (see below)
stop.request          Stop shell           Polled by the supervisor to stop the harness
expire.request        Session              Polled by the supervisor to time out the harness
cleanup.json          Supervisor or stop   After cleanup, or when stop fences launch
output.log            Launch shell         Captures supervisor and harness diagnostics
pause.request         Pause shell          Holds sequence number N to pause the harness
paused.json           Supervisor           Acknowledges pause N once every reached process stopped
continue.request      Continue shell       Holds sequence number N to continue after pause N
continued.json        Supervisor           Acknowledges continue N once the processes run again
===================== ==================== ===========================================

Pause and continue are re-armable: each checkpoint writes a sequence number greater than the last,
and an acknowledgement names the number it answers, so a request is never mistaken for an earlier
one and no file has to be removed. A pause written before the launch holds the launch until its
continue; a pause the supervisor never acknowledges is answered by ``cleanup.json``.

Agents in the task sandbox often clean up with ``pkill -f python3``, ``pkill python3`` or
``killall python3``. These must not stop the supervisor or a runner, which would end the
harness. The supervisor and the shared harness runner are bash scripts, so their names
never mention python; a harness argv that runs its own python (Hermes, NOOA, Stirrup) still
appears in the supervisor's command line. Gym stops the supervisor through the stop file,
which does not depend on these names.

``supervisor.pid`` holds the PID that executes the supervisor. When the provider did not
make the launch shell a process group leader, that process starts the supervisor in a new
group, passes TERM on to it and exits with its status.
"""

import json
import logging
from pathlib import Path
from shlex import join, quote
from typing import Literal, NotRequired, TypedDict
from uuid import uuid4

from pydantic import TypeAdapter, ValidationError

from nemo_gym.core.failure_kinds import HARNESS_TERMINATION_UNCONFIRMED
from nemo_gym.core.failure_records import FailureBoundary
from nemo_gym.core.typed_exceptions import NgFailureEvent
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.utils import read_text


SUPERVISOR_FILE = "process_supervisor.sh"
SUPERVISOR_SCRIPT = Path(__file__).with_name(SUPERVISOR_FILE)
LAUNCH_CLAIM_FILE = "launch.claim"
SUPERVISOR_PID_FILE = "supervisor.pid"
STOP_REQUEST_FILE = "stop.request"
EXPIRE_REQUEST_FILE = "expire.request"
CLEANUP_RECEIPT_FILE = "cleanup.json"
OUTPUT_LOG_FILE = "output.log"
PAUSE_REQUEST_FILE = "pause.request"
PAUSED_FILE = "paused.json"
CONTINUE_REQUEST_FILE = "continue.request"
CONTINUED_FILE = "continued.json"
# How long a pause or continue waits for the supervisor's acknowledgement by default. The supervisor
# waits up to 5 seconds for reached processes to stop, and 25 more while some are in uninterruptible
# sleep.
DEFAULT_PAUSE_TIMEOUT = 45.0

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
    be run. ``stopped``, absent from receipts written before pausing existed, says the
    stop request, not the harness's own exit or the deadline, ended the harness or
    fenced its launch.
    """

    cleanup_confirmed: bool
    return_code: int | None
    error: str | None
    timed_out: bool
    stopped: NotRequired[bool]


class PauseAcknowledgement(TypedDict):
    """A supervisor's answer to pause or continue request ``seq``, from ``paused.json`` or ``continued.json``."""

    seq: int
    confirmed: bool
    pids: list[int]
    error: str | None


# What a pause or continue request found:
# ``acknowledged`` - the supervisor answered (see the acknowledgement's ``confirmed``);
# ``not_launched`` - no supervisor runs yet, and the launch will honour the request when it starts;
# ``finished`` - the cleanup receipt exists, so the harness has ended;
# ``timeout`` - nothing answered in time.
SupervisorRequestOutcome = Literal["acknowledged", "not_launched", "finished", "timeout"]


def exec_timeout(
    *, timeout: float, cleanup_timeout: float = DEFAULT_CLEANUP_TIMEOUT, pause_allowance: float = 0.0
) -> float:
    """Reserve TERM grace, direct-process reaping, and process-group reaping.

    The extra 30 seconds cover startup, polling, and receipt I/O beyond those
    bounded cleanup phases. Keep their count shared with the controller budget.
    ``pause_allowance`` covers time the harness may spend paused for checkpoints,
    which the supervisor does not count against ``timeout``.
    """
    return timeout + CLEANUP_PHASE_COUNT * cleanup_timeout + 30 + pause_allowance


_CLEANUP_RECEIPT = TypeAdapter(CleanupReceipt)
LOG = logging.getLogger(__name__)


def harness_termination_failure(*, harness: str, receipt: CleanupReceipt) -> NgFailureEvent:
    """The harness boundary's failure for a negative cleanup receipt.

    Its event names the harness but not the receipt's text, which stays in ``str()`` and on ``receipt``
    (runtime evidence for teardown, not part of the serialized event).
    """
    failure = NgFailureEvent(
        boundary=FailureBoundary.HARNESS,
        failure_kind=HARNESS_TERMINATION_UNCONFIRMED,
        failure_reason=f"{harness} harness termination was not confirmed",
        operation="harness.terminate",
        message_args=(f"{harness} harness termination was not confirmed: {receipt['error']}",),
    )
    failure.receipt = receipt
    return failure


def is_harness_termination_failure(error: BaseException) -> bool:
    """Whether ``error`` reports that a harness's termination was not confirmed."""
    return (
        isinstance(error, NgFailureEvent)
        and error.boundary == FailureBoundary.HARNESS
        and error.failure_kind == HARNESS_TERMINATION_UNCONFIRMED
    )


def _raise_terminal_failure(payload: object, *, harness: str) -> None:
    if not isinstance(payload, dict) or payload.get("cleanup_confirmed") is not False:
        return
    try:
        receipt = _CLEANUP_RECEIPT.validate_python(payload, strict=True, extra="forbid")
    except ValidationError:
        return  # Malformed evidence remains unknown and retryable.
    raise harness_termination_failure(harness=harness, receipt=receipt)


def parse_cleanup_receipt(payload: object) -> CleanupReceipt:
    """Require positive cleanup evidence, tolerating malformed optional diagnostics."""
    if not isinstance(payload, dict) or payload.get("cleanup_confirmed") is not True:
        raise ValueError("Harness termination was not confirmed")
    try:
        return _CLEANUP_RECEIPT.validate_python(payload, strict=True, extra="forbid")
    except ValidationError:
        LOG.warning("Harness termination was confirmed, but its diagnostic fields did not match the receipt schema")
    receipt: CleanupReceipt = {
        "cleanup_confirmed": True,
        "return_code": payload.get("return_code") if type(payload.get("return_code")) is int else None,
        "timed_out": payload.get("timed_out") is True,
        "error": payload.get("error") if isinstance(payload.get("error"), str) else None,
    }
    if isinstance(payload.get("stopped"), bool):
        receipt["stopped"] = payload["stopped"]
    return receipt


def supervision_timeouts(*, timeout: float, close_timeout: float, pause_allowance: float = 0.0) -> tuple[float, float]:
    """Return the per-cleanup-phase and provider execution deadlines.

    Divide the close budget across TERM grace, direct-process and group reaping.
    The provider deadline reserves all phases after
    the harness timeout, plus 30 seconds for startup, polling, and receipt I/O,
    plus ``pause_allowance`` for time paused at checkpoints.
    """
    cleanup_timeout = close_timeout / CLEANUP_PHASE_COUNT
    return cleanup_timeout, exec_timeout(
        timeout=timeout, cleanup_timeout=cleanup_timeout, pause_allowance=pause_allowance
    )


def _seconds(value: float) -> str:
    """Seconds as the supervisor parses them: fixed-point, at least one centisecond."""
    return str(value) if 0.01 <= value < 1e16 else f"{max(value, 0.01):.2f}"


def supervised_launch_command(
    *,
    session_dir: str,
    command: list[str],
    timeout: float,
    cleanup_timeout: float,
    pausable: bool = False,
) -> str:
    """Fence delayed launches and run a harness command under the shared supervisor.

    The supervisor needs only bash. Check that bash can parse it before claiming a
    launch: a failed bootstrap cannot write a cleanup receipt. Supervisor and harness
    diagnostics share output.log. The supervisor leaves the provider's process group and
    stops the harness group, and after a forced stop the processes it reached; the
    sandbox owner stops services that left the harness earlier. Stops reach the
    supervisor through the stop file, not a signal, so the harness keeps the default
    TERM disposition.

    With ``pausable``, the supervisor also serves pause and continue requests in
    ``session_dir`` (see ``request_pause`` and ``request_continue``).
    """
    supervisor = quote(f"{session_dir}/{SUPERVISOR_FILE}")
    pause_options = (
        f"--pause-file {quote(f'{session_dir}/{PAUSE_REQUEST_FILE}')} "
        f"--paused {quote(f'{session_dir}/{PAUSED_FILE}')} "
        f"--continue-file {quote(f'{session_dir}/{CONTINUE_REQUEST_FILE}')} "
        f"--continued {quote(f'{session_dir}/{CONTINUED_FILE}')} "
        if pausable
        else ""
    )
    claim_path = quote(f"{session_dir}/{LAUNCH_CLAIM_FILE}")
    return (
        f"[ -d {quote(session_dir)} ] && [ ! -L {claim_path} ] || exit 0; "
        f"bash -n {supervisor} || exit $?; "
        f"ln -s launch {claim_path} 2>/dev/null || exit 0; "
        f"echo $$ > {quote(f'{session_dir}/{SUPERVISOR_PID_FILE}')} && "
        f"exec bash {supervisor} "
        f"--timeout {_seconds(timeout)} --cleanup-timeout {_seconds(cleanup_timeout)} "
        f"--stop-file {quote(f'{session_dir}/{STOP_REQUEST_FILE}')} "
        f"--expire-file {quote(f'{session_dir}/{EXPIRE_REQUEST_FILE}')} "
        f"--receipt {quote(f'{session_dir}/{CLEANUP_RECEIPT_FILE}')} {pause_options}-- {join(command)} "
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
            json.dumps(
                {"cleanup_confirmed": True, "return_code": None, "error": None, "timed_out": False, "stopped": True}
            )
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


_PAUSE_ACKNOWLEDGEMENT = TypeAdapter(PauseAcknowledgement)


async def _request(
    sandbox: AsyncSandbox,
    *,
    session_dir: str,
    workdir: str | None,
    request_file: str,
    acknowledgement_file: str,
    seq: int,
    timeout: float,
) -> tuple[SupervisorRequestOutcome, PauseAcknowledgement | None]:
    """Write request ``seq`` and wait for the supervisor's acknowledgement or the cleanup receipt, in one exec.

    A session directory without a launch claim has no supervisor yet: the request stays for the launch to read.
    A launch the stop fenced counts as finished.
    """
    receipt = quote(f"{session_dir}/{CLEANUP_RECEIPT_FILE}")
    request = quote(f"{session_dir}/{request_file}")
    temporary = quote(f"{session_dir}/{request_file}.{uuid4().hex}.tmp")
    acknowledgement = quote(f"{session_dir}/{acknowledgement_file}")
    claim = quote(f"{session_dir}/{LAUNCH_CLAIM_FILE}")
    # The receipt is read first: a finished harness is never paused. The request is written before the claim is
    # read, so a supervisor that claims the launch afterwards reads the request when it starts.
    script = (
        f"[ -f {receipt} ] && {{ echo finished; exit 0; }}; "
        f"[ -d {quote(session_dir)} ] || {{ echo not_launched; exit 0; }}; "
        f"printf '%s' {seq} > {temporary} && mv -f {temporary} {request} || exit 1; "
        f'claim=$(readlink {claim} 2>/dev/null); [ -n "$claim" ] || {{ echo not_launched; exit 0; }}; '
        '[ "$claim" = stop ] && { echo finished; exit 0; }; '
        f"for _ in $(seq 1 {max(1, int(timeout * 20))}); do "
        f"[ -f {receipt} ] && {{ echo finished; exit 0; }}; "
        f"if [ -f {acknowledgement} ]; then ack=$(cat {acknowledgement}); "
        f'case "$ack" in *{quote(f'"seq": {seq},')}*) echo acknowledged; printf "%s\\n" "$ack"; exit 0;; esac; fi; '
        "sleep 0.05; done; echo timeout"
    )
    result = await sandbox.exec(script, cwd=workdir, timeout_s=timeout + 10)
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f"supervisor request {request_file} failed: {result.error_type or result.stderr}")
    lines = (result.stdout or "").splitlines()
    outcome = lines[0].strip() if lines else ""
    if outcome == "acknowledged":
        try:
            return "acknowledged", _PAUSE_ACKNOWLEDGEMENT.validate_python(json.loads(lines[1]), strict=True)
        except (IndexError, ValueError, ValidationError) as error:
            raise RuntimeError(f"malformed supervisor acknowledgement in {acknowledgement_file}") from error
    if outcome not in ("not_launched", "finished", "timeout"):
        raise RuntimeError(f"unexpected supervisor request outcome {outcome!r}")
    return outcome, None  # type: ignore[return-value]


async def request_pause(
    sandbox: AsyncSandbox, *, session_dir: str, workdir: str | None, seq: int, timeout: float = DEFAULT_PAUSE_TIMEOUT
) -> tuple[SupervisorRequestOutcome, PauseAcknowledgement | None]:
    """Ask the supervisor in ``session_dir`` to pause the harness, as request ``seq``, and wait for its answer."""
    return await _request(
        sandbox,
        session_dir=session_dir,
        workdir=workdir,
        request_file=PAUSE_REQUEST_FILE,
        acknowledgement_file=PAUSED_FILE,
        seq=seq,
        timeout=timeout,
    )


async def request_continue(
    sandbox: AsyncSandbox, *, session_dir: str, workdir: str | None, seq: int, timeout: float = DEFAULT_PAUSE_TIMEOUT
) -> tuple[SupervisorRequestOutcome, PauseAcknowledgement | None]:
    """Ask the supervisor in ``session_dir`` to continue after pause ``seq`` and wait for its answer."""
    return await _request(
        sandbox,
        session_dir=session_dir,
        workdir=workdir,
        request_file=CONTINUE_REQUEST_FILE,
        acknowledgement_file=CONTINUED_FILE,
        seq=seq,
        timeout=timeout,
    )
