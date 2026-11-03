# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move a frozen task filesystem between isolated sandboxes through shared storage."""

import json
from pathlib import Path
from shlex import quote
from time import monotonic
from uuid import uuid4

from environment_servers.swe_together.evaluation import JudgeMeasurementError
from environment_servers.swe_together.runtime import PYTHON, SUBMISSION_ROOT, RuntimeProvisioner
from environment_servers.swe_together.submission_worker import task_roots
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult, read_text


TRANSFER_TIMEOUT_S = 600
RESOURCE_PROBE = "cat /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.peak 2>&1; df -k -- " + quote(SUBMISSION_ROOT)


def record_command(
    result: SandboxExecResult,
    *,
    action: str,
    artifact_dir: Path,
    timeout_s: float | None = None,
    elapsed_s: float | None = None,
) -> None:
    """Keep both streams: some providers put worker stderr in merged stdout."""
    details = {key: getattr(result, key) for key in ("return_code", "error_type", "stdout", "stderr")}
    timing = ""
    if timeout_s is not None and elapsed_s is not None:
        # Evidence only: OpenSandbox execd records a kill at its command timeout and an OOM kill alike
        # (exit -1, "signal: killed"), so reaching the timeout does not prove that the timeout killed it.
        details |= {
            "timeout_seconds": timeout_s,
            "elapsed_seconds": round(elapsed_s, 3),
            "reached_timeout": elapsed_s >= timeout_s,
        }
        timing = f", elapsed_seconds={elapsed_s:.1f}, timeout_seconds={timeout_s:g}"
    (artifact_dir / f"{action}-command.json").write_text(json.dumps(details, indent=2))
    if result.return_code != 0 or result.error_type:
        if result.error_type == "timeout":
            cause = "submission_timeout"
        elif result.return_code == -1 or "signal: killed" in (result.stderr or ""):
            cause = "submission_killed"
        else:
            cause = "submission_failed"
        raise JudgeMeasurementError(
            cause,
            f"Submission {action} failed ({cause}): return_code={result.return_code}, "
            f"error_type={result.error_type}{timing}\n"
            f"stderr: {(result.stderr or '')[-4000:]}\nstdout: {(result.stdout or '')[-8000:]}",
        )


async def run_transfer(sandbox: AsyncSandbox, command: str, *, action: str, artifact_dir: Path) -> SandboxExecResult:
    """Run a freeze or restore step; after a failure, keep timing, memory and disk state as kill evidence."""
    started = monotonic()
    result = await sandbox.exec(command, timeout_s=TRANSFER_TIMEOUT_S)
    elapsed = monotonic() - started
    try:
        record_command(
            result, action=action, artifact_dir=artifact_dir, timeout_s=TRANSFER_TIMEOUT_S, elapsed_s=elapsed
        )
    except JudgeMeasurementError:
        try:
            probe = await sandbox.exec(RESOURCE_PROBE, timeout_s=15)
            resources = {key: getattr(probe, key) for key in ("return_code", "error_type", "stdout", "stderr")}
        except Exception as error:
            resources = {"probe_error": f"{type(error).__name__}: {error}"}
        (artifact_dir / f"{action}-resources.json").write_text(json.dumps(resources, indent=2))
        raise
    return result


async def replicate_submission(
    candidate: AsyncSandbox,
    judge: AsyncSandbox,
    *,
    roots: list[str],
    artifact_dir: Path,
    storage: RuntimeProvisioner,
    session_id: str,
) -> dict:
    """Freeze after agent close, seal through shared storage, and restore in a fresh sandbox."""
    roots = task_roots(roots)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    remote = "/tmp/gym-swet-submission-" + uuid4().hex + ".py"
    worker = Path(__file__).with_name("submission_worker.py")
    try:
        await candidate.upload(worker, remote)
        command = " ".join(
            quote(value) for value in [PYTHON, "-I", remote, "freeze", SUBMISSION_ROOT, json.dumps(roots)]
        )
        await run_transfer(candidate, command, action="freeze", artifact_dir=artifact_dir)
        raw = await read_text(candidate, path=SUBMISSION_ROOT + "/manifest.json")
        manifest = json.loads(raw)
        if manifest["roots"] != roots:
            raise ValueError("Frozen submission roots differ from the requested task roots")
        (artifact_dir / "manifest.json").write_text(raw)
        await storage.seal_submission(session_id, manifest)
        await judge.upload(worker, remote)
        result = await run_transfer(
            judge,
            " ".join(quote(value) for value in [PYTHON, "-I", remote, "restore", SUBMISSION_ROOT]),
            action="restore",
            artifact_dir=artifact_dir,
        )
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        if receipt["archive_sha256"] != manifest["archive_sha256"]:
            raise ValueError("Restored submission checksum differs")
        receipt["transfer"] = "shared_storage"
        receipt["bytes"] = manifest["bytes"]
        (artifact_dir / "restore.json").write_text(json.dumps(receipt, indent=2))
        return receipt
    finally:
        # Per-episode shared directories are removed by their owner after both sandboxes stop.
        for role, sandbox in (("candidate", candidate), ("judge", judge)):
            try:
                result = await sandbox.exec("rm -f -- " + quote(remote), timeout_s=15)
                record_command(result, action="cleanup-" + role, artifact_dir=artifact_dir)
            except Exception:
                pass
