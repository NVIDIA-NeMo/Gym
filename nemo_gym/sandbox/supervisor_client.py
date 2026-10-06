# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side transport for the shared sandbox process supervisor.

Session ownership and cancellation live in session.py; output parsing stays with the adapter.
Unlike process_supervisor.py, this module is not uploaded to the task sandbox.
"""

import json
import logging
from shlex import join, quote
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.process_supervisor import CleanupReceipt
from nemo_gym.sandbox.utils import read_text


_CLEANUP_RECEIPT = TypeAdapter(CleanupReceipt)
LOG = logging.getLogger(__name__)


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
    """Require positive cleanup evidence, tolerating malformed optional diagnostics.

    Older stop-before-launch receipts contain only acknowledgement and error.
    Normalize that form without inventing a successful worker exit code.
    """
    if not isinstance(payload, dict) or payload.get("cleanup_confirmed") is not True:
        raise ValueError("Sandbox cleanup was not confirmed")
    if payload.keys() == {"cleanup_confirmed", "error"}:
        payload = {**payload, "return_code": None, "timed_out": False}
    try:
        return _CLEANUP_RECEIPT.validate_python(payload, strict=True, extra="forbid")
    except ValidationError:
        LOG.warning("Cleanup was confirmed, but its diagnostic fields did not match the receipt schema")
    return {
        "cleanup_confirmed": True,
        "return_code": payload.get("return_code") if type(payload.get("return_code")) is int else None,
        "timed_out": payload.get("timed_out") is True,
        "error": payload.get("error") if isinstance(payload.get("error"), str) else None,
    }


def parse_runtime_info(payload: object) -> HarnessProcessInfo | None:
    """Read optional worker diagnostics without failing an otherwise valid episode."""
    try:
        return HarnessProcessInfo.model_validate(payload)
    except ValidationError:
        LOG.warning("Harness runtime metadata is missing or malformed")
        return None


def supervised_launch_command(
    *,
    directory: str,
    command: list[str],
    timeout: float,
    cleanup_timeout: float,
    python: str,
    supervisor_path: str | None = None,
) -> str:
    """Fence delayed launches and run a harness command under the shared supervisor.

    The adapter must install or select the interpreter explicitly. Check that it
    can load the supervisor before claiming a launch: a failed bootstrap cannot
    write a cleanup receipt. Supervisor and worker diagnostics share runner.log.
    """
    supervisor = quote(supervisor_path or directory + "/process_supervisor.py")
    return (
        f"[ -d {quote(directory)} ] && [ ! -L {quote(directory + '/launch.claim')} ] || exit 0; "
        f"{quote(python)} -I {supervisor} --help >/dev/null || exit $?; "
        f"trap '' TERM; ln -s launch {quote(directory + '/launch.claim')} 2>/dev/null || exit 0; "
        f"echo $$ > {quote(directory + '/runner.pid')} && "
        f"exec {quote(python)} -I {supervisor} "
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
        receipt = json.loads(await read_text(sandbox, path=receipt_path))
    except Exception:
        receipt = {}
    if not isinstance(receipt, dict) or receipt.get("cleanup_confirmed") is not True:
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
            receipt = json.loads(await read_text(sandbox, path=receipt_path))
        except Exception as error:
            raise RuntimeError(f"{harness} launch outcome is unknown; cannot confirm termination") from error
        if not isinstance(receipt, dict) or receipt.get("cleanup_confirmed") is not True:
            detail = receipt.get("error") if isinstance(receipt, dict) else "cleanup receipt is not a JSON object"
            raise RuntimeError(f"{harness} sandbox cleanup was not confirmed: {detail}")
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
