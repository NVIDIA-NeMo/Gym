# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-task sandbox the GDPVal resources server owns and lends to the agent.

The server starts one sandbox per session with the task's reference files under the working
directory, checks the files the agent submits with ``finish``, and copies them out at ``/verify``
into the persisted layout the scorer and the multistage tools read.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shlex
import shutil
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional

from fastapi import HTTPException

from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxConfig, resolve_provider_config
from resources_servers.gdpval import persisted_layout


LOGGER = logging.getLogger(__name__)

WORKDIR = "/root"


def sandbox_path(path: str) -> str:
    """Resolve a path the agent reported against the sandbox working directory."""
    return path if path.startswith("/") else f"{WORKDIR}/{path}"


class GDPValSandboxConfig(SandboxConfig):
    # An OCI image built from containers/Dockerfile; see README.md.
    image: Optional[str] = None


async def start_task_sandbox(
    sandbox_provider: str,
    sandbox_config: GDPValSandboxConfig,
    *,
    task_id: str,
    reference_dir: Path,
    reference_files: list[str],
) -> AsyncSandbox:
    """Start a sandbox and stage the task's reference files under the working directory."""
    if not sandbox_config.image:
        raise ValueError(
            "Set sandbox_config.image (or GDPVAL_SANDBOX_IMAGE) to an OCI image built from "
            "resources_servers/gdpval/containers/Dockerfile"
        )
    global_config_dict = get_global_config_dict()
    spec = sandbox_config.spec(
        sandbox_provider,
        image=sandbox_config.image,
        workdir=WORKDIR,
        metadata={"task_id": task_id},
        named_configs=global_config_dict,
    )
    sandbox = AsyncSandbox(resolve_provider_config(sandbox_provider, global_config_dict))
    await sandbox.start(spec)
    try:
        for rel_path in reference_files:
            await sandbox.upload(reference_dir / rel_path, sandbox_path(rel_path))
    except BaseException:
        await sandbox.stop()
        raise
    return sandbox


async def missing_files(sandbox: AsyncSandbox, paths: list[str]) -> list[str]:
    """Return the reported paths that are not regular files in the sandbox, in their given order."""
    if not paths:
        return []
    script = "; ".join(f"test -f {shlex.quote(sandbox_path(p))} || echo {i}" for i, p in enumerate(paths))
    result = await sandbox.exec(script, timeout_s=60)
    if result.error_type is not None:
        # An infrastructure failure, not something the model should react to.
        raise HTTPException(status_code=503, detail=f"Could not check the submitted files: {result.stderr}")
    missing = {int(index) for index in (result.stdout or "").split()}
    return [path for i, path in enumerate(paths) if i in missing]


async def collect_deliverables(
    sandbox: AsyncSandbox, finish_call: Optional[Mapping[str, Any]], reference_dir: Path, task_dir: Path
) -> None:
    """Copy the submitted files, the finish arguments and the reference files into ``task_dir``.

    Files land under their base names, so a later path with the same name replaces an earlier one, and a reference
    file replaces a submitted file at the same path. A file
    that cannot be downloaded raises, since `/finish` confirmed it exists. ``task_dir`` is built beside its
    final place and renamed into it, so a failure never leaves a partial directory, and removes ``task_dir``
    too. ``finish_params.json``
    holds ``null`` when the agent never finished.
    """
    staging = task_dir.with_name(f".{task_dir.name}.{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        for path in (finish_call or {}).get("paths") or []:
            await sandbox.download(sandbox_path(path), staging / Path(path).name)
        finish_params = None if finish_call is None else {k: v for k, v in finish_call.items() if k != "tool"}
        (staging / persisted_layout.FINISH_PARAMS_FILE).write_text(json.dumps(finish_params, indent=2, default=str))
        await asyncio.to_thread(shutil.copytree, reference_dir, staging, dirs_exist_ok=True)
        await asyncio.to_thread(_make_group_readable, staging)
        await asyncio.to_thread(shutil.rmtree, task_dir, ignore_errors=True)
        await asyncio.to_thread(staging.rename, task_dir)
    except BaseException:
        # Synchronous so a second cancellation cannot interrupt the cleanup.
        shutil.rmtree(staging, ignore_errors=True)
        # A multistage retry reuses the persisted tree, so leave none rather than an earlier run's.
        shutil.rmtree(task_dir, ignore_errors=True)
        raise


def _make_group_readable(task_dir: Path) -> None:
    for path in [task_dir.parent, task_dir, *task_dir.rglob("*")]:
        try:
            os.chmod(path, path.stat().st_mode | 0o755)
        except OSError as error:
            LOGGER.warning("Could not relax permissions on %s: %s", path, error)
