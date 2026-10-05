# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side transport for the shared sandbox process supervisor.

Session ownership and cancellation live in session.py; output parsing stays with the adapter.
Unlike process_supervisor.py, this module is not uploaded to the task sandbox.
"""

import json
from shlex import join, quote
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.process_supervisor import CleanupReceipt


_CLEANUP_RECEIPT = TypeAdapter(CleanupReceipt)


class HarnessProcessInfo(BaseModel):
    """Identity of the supervised worker, which may be a shim that spawns the harness CLI.

    This is independent of supervisor cleanup and harness output. The PID is
    diagnostic metadata, not the supervisor PID used to request cleanup.
    """

    model_config = ConfigDict(extra="forbid", strict=True)
    hostname: str
    pid: int
    python: str | None = None


def parse_cleanup_receipt(payload: object) -> CleanupReceipt:
    """Validate the supervisor's existing schema without adding worker dependencies.

    Older stop-before-launch receipts contain only acknowledgement and error.
    Normalize that form without inventing a successful worker exit code.
    """
    if isinstance(payload, dict) and payload.keys() == {"cleanup_confirmed", "error"}:
        payload = {**payload, "return_code": None, "timed_out": False}
    return _CLEANUP_RECEIPT.validate_python(payload, strict=True, extra="forbid")


def supervised_launch_command(
    *,
    directory: str,
    command: list[str],
    timeout: float,
    cleanup_timeout: float,
    python: str = "python3",
    supervisor_path: str | None = None,
) -> str:
    """Fence delayed launches and run a harness command under the shared supervisor.

    Use a private interpreter and supervisor path when the harness installs its
    own runtime. Supervisor and worker diagnostics are combined in runner.log.
    """
    return (
        f"trap '' TERM; ln -s launch {quote(directory + '/launch.claim')} 2>/dev/null || exit 0; "
        f"echo $$ > {quote(directory + '/runner.pid')} && "
        f"exec {quote(python)} -I {quote(supervisor_path or directory + '/process_supervisor.py')} "
        f"--timeout {timeout} --cleanup-timeout {cleanup_timeout} "
        f"--stop-file {quote(directory + '/runner.stop')} "
        f"--receipt {quote(directory + '/cleanup.json')} -- {join(command)} "
        f">{quote(directory + '/runner.log')} 2>&1"
    )


async def stop_and_confirm_cleanup(
    sandbox: AsyncSandbox, *, directory: str, workdir: str | None, timeout: float, harness: str
) -> CleanupReceipt:
    """Fence a pending launch or require explicit supervisor acknowledgement before teardown.

    A stop that wins the launch claim writes a minimal receipt: no worker ran, so
    runtime metadata and a return code are intentionally absent. Worker metadata
    and harness output are validated separately by the adapter.
    """
    receipt_path = f"{directory}/cleanup.json"
    try:
        receipt = json.loads(await sandbox.read_text(receipt_path))
    except Exception:
        receipt = {}
    if receipt.get("cleanup_confirmed") is not True:
        pid_path = quote(f"{directory}/runner.pid")
        stop_path = quote(f"{directory}/runner.stop")
        claim_path = quote(f"{directory}/launch.claim")
        temporary = quote(f"{receipt_path}.{uuid4().hex}.tmp")
        stopped = quote(json.dumps({"cleanup_confirmed": True, "error": None}))
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
            receipt = json.loads(await sandbox.read_text(receipt_path))
        except Exception as error:
            raise RuntimeError(f"{harness} launch outcome is unknown; cannot confirm termination") from error
        if receipt.get("cleanup_confirmed") is not True:
            raise RuntimeError(f"{harness} sandbox cleanup was not confirmed: {receipt.get('error')}")
    return parse_cleanup_receipt(receipt)


async def remove_session_directory(
    sandbox: AsyncSandbox, *, directory: str, workdir: str | None, timeout: float, harness: str
) -> None:
    """Remove adapter-owned files after cleanup, keeping delayed launches fenced.

    Retire the directory atomically before unlinking its claim. Otherwise a
    delayed launch could win the claim while recursive deletion is in progress.
    A failed removal leaves the retired path available for a close retry.
    """
    retired = f"{directory}.closed"
    result = await sandbox.exec(
        f"if [ -d {quote(directory)} ]; then "
        f"mv {quote(directory)} {quote(retired)} || exit 1; fi; rm -rf -- {quote(retired)}",
        cwd=workdir,
        timeout_s=timeout,
    )
    if result.return_code != 0 or result.error_type:
        raise RuntimeError(f"Could not remove {harness} session files")
