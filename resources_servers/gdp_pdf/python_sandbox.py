# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The per-rollout sandbox behind the gdp_pdf ``python`` tool.

The task's source PDF is uploaded as ``document.pdf`` into the sandbox's working directory, where a persistent
IPython kernel (``ipython_kernel.py``) runs. The image needs ``python3`` with ``ipykernel`` and ``jupyter_client``;
``containers/Dockerfile`` builds one with common PDF and data libraries.
"""

from __future__ import annotations

import contextlib
import shlex
from pathlib import Path
from typing import Optional

from fastapi import HTTPException

from nemo_gym.global_config import get_global_config_dict
from nemo_gym.sandbox import AsyncSandbox, SandboxExecResult
from nemo_gym.sandbox.config import SandboxConfig, resolve_provider_config
from nemo_gym.sandbox.utils import upload_text


KERNEL_SCRIPT = Path(__file__).with_name("ipython_kernel.py")


class PythonSandboxConfig(SandboxConfig):
    image: Optional[str] = None
    # /sandbox is also the per-sandbox mount of the enroot and apptainer providers, so it is writable there.
    workdir: str = "/sandbox"
    # Only a backstop: the sandbox is stopped when its rollout is verified.
    ttl_s: Optional[float] = 21600


def _kernel(config: PythonSandboxConfig, command: str) -> str:
    return f"python3 {shlex.quote(f'{config.workdir}/.ipython_kernel/{KERNEL_SCRIPT.name}')} {command}"


def _stdout(result: SandboxExecResult) -> str:
    if result.error_type is not None or result.return_code != 0:
        # An infrastructure failure, not something the model should react to.
        reason = result.error_type or f"exit code {result.return_code}"
        raise HTTPException(503, f"Python sandbox command failed ({reason}): {(result.stderr or '').strip()[-2000:]}")
    return result.stdout or ""


async def start_python_sandbox(
    sandbox_provider: str, config: PythonSandboxConfig, pdf: Path, *, task_id: str
) -> AsyncSandbox:
    """Start a sandbox with ``pdf`` as ``document.pdf`` in its working directory and a ready IPython kernel."""
    global_config = get_global_config_dict()
    spec = config.spec(
        sandbox_provider,
        image=config.image,
        workdir=config.workdir,
        metadata={"task_id": task_id[:63]},
        files={f"{config.workdir}/.ipython_kernel/{KERNEL_SCRIPT.name}": KERNEL_SCRIPT.read_text()},
        named_configs=global_config,
    )

    async def setup(sandbox: AsyncSandbox) -> None:
        await sandbox.upload(pdf, f"{config.workdir}/document.pdf")
        _stdout(await sandbox.exec(_kernel(config, "start"), timeout_s=120, preserve_background_services=True))

    sandbox = AsyncSandbox(resolve_provider_config(sandbox_provider, global_config))
    return await sandbox.start_with_setup(spec, setup)


async def run_python(
    sandbox: AsyncSandbox, config: PythonSandboxConfig, code: str, *, timeout_s: float, max_chars: int
) -> str:
    """Run one cell and return its text output; the cell's own exceptions are part of that output."""
    await upload_text(sandbox, path=f"{config.workdir}/.ipython_kernel/cell.py", text=code)
    command = _kernel(config, f"run --timeout {timeout_s:g} --max-chars {max_chars}")
    # Room for a kernel restart on top of the cell's own timeout.
    return _stdout(await sandbox.exec(command, timeout_s=timeout_s + 150))


async def stop_python_sandbox(sandbox: AsyncSandbox, config: PythonSandboxConfig) -> None:
    # Container providers end the kernel with the sandbox; the local provider does not, so end it first.
    with contextlib.suppress(Exception):
        await sandbox.exec(_kernel(config, "stop"), timeout_s=30)
    await sandbox.stop()
