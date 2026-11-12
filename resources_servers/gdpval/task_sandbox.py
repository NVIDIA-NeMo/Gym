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
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from fastapi import HTTPException

from nemo_gym.config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxConfig, resolve_provider_config
from resources_servers.gdpval import persisted_layout
from resources_servers.gdpval.preconvert import (
    OFFICE_EXTENSIONS,
    OOXML_EXTENSIONS,
    _normalize_ooxml_zip,
    _ooxml_has_ns0_prefix,
    _safe_basename,
    libreoffice_pdf_argv,
    roundtrip_ooxml_copy,
    sidecar_pdf,
)


LOGGER = logging.getLogger(__name__)
# One LibreOffice run, and every run of one rollout together (the default of sandbox_render_budget_s).
RENDER_TIMEOUT_S = 150
DEFAULT_RENDER_BUDGET_S = 600.0

WORKDIR = "/root"


def sandbox_path(path: str, workdir: Optional[str] = None) -> str:
    """Resolve a path the agent reported against the sandbox working directory."""
    return path if path.startswith("/") else f"{workdir or WORKDIR}/{path}"


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
    workdir: Optional[str] = None,
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
        workdir=workdir or WORKDIR,
        metadata={"task_id": task_id},
        named_configs=global_config_dict,
    )
    sandbox = AsyncSandbox(resolve_provider_config(sandbox_provider, global_config_dict))
    await sandbox.start(spec)
    try:
        for rel_path in reference_files:
            await sandbox.upload(reference_dir / rel_path, sandbox_path(rel_path, workdir))
    except BaseException:
        await sandbox.stop()
        raise
    return sandbox


async def missing_files(sandbox: AsyncSandbox, paths: list[str], workdir: Optional[str] = None) -> list[str]:
    """Return the reported paths that are not regular files in the sandbox, in their given order."""
    if not paths:
        return []
    script = "; ".join(f"test -f {shlex.quote(sandbox_path(p, workdir))} || echo {i}" for i, p in enumerate(paths))
    result = await sandbox.exec(script, timeout_s=60)
    if result.error_type is not None:
        # An infrastructure failure, not something the model should react to.
        raise HTTPException(status_code=503, detail=f"Could not check the submitted files: {result.stderr}")
    missing = {int(index) for index in (result.stdout or "").split()}
    return [path for i, path in enumerate(paths) if i in missing]


async def collect_deliverables(
    sandbox: AsyncSandbox,
    finish_call: Optional[Mapping[str, Any]],
    reference_dir: Path,
    task_dir: Path,
    workdir: Optional[str] = None,
    *,
    render_libreoffice_command: Optional[Sequence[str]] = None,
    render_budget_s: float = DEFAULT_RENDER_BUDGET_S,
) -> None:
    """Copy the submitted files, the finish arguments and the reference files into ``task_dir``.

    Files land under their base names, so a later path with the same name replaces an earlier one, and a reference
    file replaces a submitted file at the same path. A file
    that cannot be downloaded raises, since `/finish` confirmed it exists. ``task_dir`` is built beside its
    final place and renamed into it, so a failure never leaves a partial directory, and removes ``task_dir``
    too. ``finish_params.json``
    holds ``null`` when the agent never finished. With ``render_libreoffice_command``, every Office file also gets
    a ``<name>.pdf`` sidecar rendered by that command inside the sandbox; see :func:`render_office_in_sandbox`.
    """
    staging = task_dir.with_name(f".{task_dir.name}.{uuid.uuid4().hex}")
    staging.mkdir(parents=True)
    try:
        for path in (finish_call or {}).get("paths") or []:
            await sandbox.download(sandbox_path(path, workdir), staging / Path(path).name)
        finish_params = None if finish_call is None else {k: v for k, v in finish_call.items() if k != "tool"}
        (staging / persisted_layout.FINISH_PARAMS_FILE).write_text(json.dumps(finish_params, indent=2, default=str))
        await asyncio.to_thread(shutil.copytree, reference_dir, staging, dirs_exist_ok=True)
        if render_libreoffice_command:
            await render_office_in_sandbox(sandbox, staging, render_libreoffice_command, budget_s=render_budget_s)
        await asyncio.to_thread(_make_group_readable, staging)
        await asyncio.to_thread(shutil.rmtree, task_dir, ignore_errors=True)
        await asyncio.to_thread(staging.rename, task_dir)
    except BaseException:
        # Synchronous so a second cancellation cannot interrupt the cleanup.
        shutil.rmtree(staging, ignore_errors=True)
        # A multistage retry reuses the persisted tree, so leave none rather than an earlier run's.
        shutil.rmtree(task_dir, ignore_errors=True)
        raise


async def render_office_in_sandbox(
    sandbox: AsyncSandbox,
    directory: Path,
    libreoffice_command: Sequence[str],
    *,
    budget_s: float = DEFAULT_RENDER_BUDGET_S,
) -> None:
    """Write the injective ``<name>.pdf`` sidecar beside every Office file under ``directory``.

    LibreOffice runs in the task sandbox rather than in the server, so the server neither needs LibreOffice nor
    renders model-written files itself. The steps mirror :func:`preconvert.convert_to_pdf`: a whitespace-free
    name, the ``ns0`` package fix and one OOXML round-trip retry. A file that cannot be rendered keeps no sidecar
    and is judged as the server would judge it without LibreOffice. ``budget_s`` bounds the whole rendering, so
    files that hang LibreOffice cannot hold /verify for long: the files left when it runs out are not rendered.
    """
    deadline = time.monotonic() + budget_s
    paths = sorted(p for p in directory.rglob("*") if p.is_file() and p.suffix.lower() in OFFICE_EXTENSIONS)
    for index, path in enumerate(paths):
        if sidecar_pdf(path).exists():
            continue
        if time.monotonic() >= deadline:
            left = [p.name for p in paths[index:] if not sidecar_pdf(p).exists()]
            LOGGER.warning("Sandbox render budget of %ss used up; not rendering %s", budget_s, left)
            return
        with tempfile.TemporaryDirectory(prefix="gdpval-sandbox-render-") as local:
            first = Path(local) / _safe_basename(path.name)
            if _ooxml_has_ns0_prefix(path):
                await asyncio.to_thread(_normalize_ooxml_zip, path, first)
            else:
                await asyncio.to_thread(shutil.copy2, path, first)
            error = await _render_one(sandbox, first, sidecar_pdf(path), libreoffice_command, deadline)
            if error and path.suffix.lower() in OOXML_EXTENSIONS:
                retry = Path(local) / "retry" / first.name
                retry.parent.mkdir()
                try:
                    await asyncio.to_thread(roundtrip_ooxml_copy, path, retry)
                    error = await _render_one(sandbox, retry, sidecar_pdf(path), libreoffice_command, deadline)
                except Exception as exc:
                    error = f"{error}; round-trip retry failed: {exc!r}"
        if error:
            LOGGER.warning("Could not render %s in the sandbox: %s", path.name, error)


async def _render_one(
    sandbox: AsyncSandbox, source: Path, output_pdf: Path, libreoffice_command: Sequence[str], deadline: float
) -> Optional[str]:
    """Render ``source`` to ``output_pdf`` in the sandbox; return the error, or None on success.

    Rendering is optional, so a sandbox operation that raises is a render error too: the collected file is then
    judged without a render instead of failing /verify.
    """
    remote = Path(f"/tmp/gdpval-render-{uuid.uuid4().hex}")
    remote_source = remote / "in" / source.name
    try:
        prepared = await sandbox.exec(f"mkdir -p {shlex.quote(str(remote / 'in'))}", timeout_s=60)
        if prepared.return_code != 0:
            return f"mkdir failed: {prepared.stderr}"
        await sandbox.upload(source, str(remote_source))
        argv = libreoffice_pdf_argv(remote_source, remote / "out", remote / "profile", libreoffice_command)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return "sandbox render budget used up"
        result = await sandbox.exec(shlex.join(argv), timeout_s=min(RENDER_TIMEOUT_S, remaining))
        # A timed-out or otherwise inconclusive run may still have written a partial PDF; never accept it.
        if result.error_type is not None:
            return f"LibreOffice execution failed ({result.error_type}): {(result.stderr or '').strip()[-300:]}"
        remote_pdf = remote / "out" / (source.stem + ".pdf")
        try:
            await sandbox.download(str(remote_pdf), output_pdf)
        except Exception as exc:
            output_pdf.unlink(missing_ok=True)
            return f"rc={result.return_code} produced no PDF ({exc!r}): {(result.stderr or '').strip()[-300:]}"
        return None
    except Exception as exc:
        output_pdf.unlink(missing_ok=True)
        return f"sandbox render failed: {exc!r}"
    finally:
        try:
            cleanup = await sandbox.exec(f"rm -rf {shlex.quote(str(remote))}", timeout_s=60)
            if cleanup.error_type is not None or cleanup.return_code != 0:
                LOGGER.warning("Could not remove render scratch %s: %s", remote, cleanup.stderr)
        except Exception:
            LOGGER.warning("Could not remove render scratch %s", remote, exc_info=True)


def _make_group_readable(task_dir: Path) -> None:
    for path in [task_dir.parent, task_dir, *task_dir.rglob("*")]:
        try:
            os.chmod(path, path.stat().st_mode | 0o755)
        except OSError as error:
            LOGGER.warning("Could not relax permissions on %s: %s", path, error)
