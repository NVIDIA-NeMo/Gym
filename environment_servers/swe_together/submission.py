# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Move a frozen task filesystem between isolated sandboxes through shared storage."""

import json
from pathlib import Path
from shlex import quote
from uuid import uuid4

from environment_servers.swe_together.runtime import PYTHON, SUBMISSION_ROOT, RuntimeProvisioner
from environment_servers.swe_together.submission_worker import task_roots
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.utils import read_text


def record_command(result: SandboxExecResult, *, action: str, artifact_dir: Path) -> None:
    """Keep both streams: some providers put worker stderr in merged stdout."""
    details = {key: getattr(result, key) for key in ("return_code", "error_type", "stdout", "stderr")}
    (artifact_dir / f"{action}-command.json").write_text(json.dumps(details, indent=2))
    if result.return_code or result.error_type:
        raise RuntimeError(
            f"Submission {action} failed: return_code={result.return_code}, error_type={result.error_type}\n"
            f"stderr: {(result.stderr or '')[-4000:]}\nstdout: {(result.stdout or '')[-8000:]}"
        )


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
        command = " ".join(quote(value) for value in [PYTHON, remote, "freeze", SUBMISSION_ROOT, json.dumps(roots)])
        result = await candidate.exec(command, timeout_s=600)
        record_command(result, action="freeze", artifact_dir=artifact_dir)
        raw = await read_text(candidate, path=SUBMISSION_ROOT + "/manifest.json")
        manifest = json.loads(raw)
        if manifest["roots"] != roots:
            raise ValueError("Frozen submission roots differ from the requested task roots")
        (artifact_dir / "manifest.json").write_text(raw)
        await storage.seal_submission(session_id, manifest)
        await judge.upload(worker, remote)
        result = await judge.exec(
            " ".join(quote(value) for value in [PYTHON, remote, "restore", SUBMISSION_ROOT]), timeout_s=600
        )
        record_command(result, action="restore", artifact_dir=artifact_dir)
        receipt = json.loads(result.stdout.strip().splitlines()[-1])
        if receipt["archive_sha256"] != manifest["archive_sha256"]:
            raise ValueError("Restored submission checksum differs")
        receipt["transfer"] = "shared_storage"
        receipt["bytes"] = manifest["bytes"]
        (artifact_dir / "restore.json").write_text(json.dumps(receipt, indent=2))
        return receipt
    finally:
        # Per-episode shared directories are removed by their owner after both sandboxes stop.
        for sandbox in (candidate, judge):
            try:
                await sandbox.exec("rm -f -- " + quote(remote), timeout_s=15)
            except Exception:
                pass
