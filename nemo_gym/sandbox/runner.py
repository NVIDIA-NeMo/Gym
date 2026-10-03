# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Controller-side transport for the shared sandbox process supervisor.

Session ownership, cancellation, and harness output parsing stay with the adapter.
Unlike process_supervisor.py, this module is not uploaded to the task sandbox.
"""

import json
import tempfile
from pathlib import Path
from shlex import join, quote
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, TypeAdapter

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.process_supervisor import CleanupReceipt


_CLEANUP_RECEIPT = TypeAdapter(CleanupReceipt)


class RunnerRuntimeInfo(BaseModel):
    """Worker identity, independent of supervisor cleanup and harness output."""

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


async def upload_text(sandbox: AsyncSandbox, *, path: str, text: str) -> None:
    """Upload adapter-owned text without interpolating its contents into a shell command."""
    with tempfile.TemporaryDirectory(prefix="sandbox-runner-upload-") as directory:
        source = Path(directory) / "payload"
        source.write_text(text)
        await sandbox.upload(source, path)


async def read_text(sandbox: AsyncSandbox, *, path: str) -> str:
    """Download an adapter-owned artifact, preserving non-UTF8 output with replacement."""
    with tempfile.TemporaryDirectory(prefix="sandbox-runner-download-") as directory:
        destination = Path(directory) / "payload"
        await sandbox.download(path, destination)
        return destination.read_text(errors="replace")


def supervisor_command(*, directory: str, command: list[str], timeout: float, cleanup_timeout: float) -> str:
    """Fence delayed launches and run a harness command under the shared supervisor."""
    return (
        f"trap '' TERM; ln -s launch {quote(directory + '/launch.claim')} 2>/dev/null || exit 0; "
        f"echo $$ > {quote(directory + '/runner.pid')} && "
        f"exec python3 -I {quote(directory + '/process_supervisor.py')} "
        f"--timeout {timeout} --cleanup-timeout {cleanup_timeout} "
        f"--stop-file {quote(directory + '/runner.stop')} "
        f"--receipt {quote(directory + '/cleanup.json')} -- {join(command)} "
        f">{quote(directory + '/runner.log')} 2>&1"
    )


async def confirm_runner_cleanup(
    sandbox: AsyncSandbox, *, directory: str, workdir: str, timeout: float, harness: str
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
            receipt = json.loads(await read_text(sandbox, path=receipt_path))
        except Exception as error:
            raise RuntimeError(f"{harness} launch outcome is unknown; cannot confirm termination") from error
        if receipt.get("cleanup_confirmed") is not True:
            raise RuntimeError(f"{harness} sandbox cleanup was not confirmed: {receipt.get('error')}")
    return parse_cleanup_receipt(receipt)
