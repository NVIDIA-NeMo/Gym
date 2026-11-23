# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA's borrowed-session fixture, backed by the suite's local subprocess provider."""

import asyncio
import hashlib
import shlex
from collections.abc import Mapping
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from omegaconf import OmegaConf

from nemo_gym.sandbox import AsyncSandbox, LocalProvider
from nemo_gym.sandbox.providers import SandboxHandle, SandboxSpec


OBSERVER = Path(__file__).with_name("nooa_observer.py")


def observer_module_name() -> str:
    """Bind the fixture module import to its exact source bytes."""
    return "nooa_conformance_observer_" + hashlib.sha256(OBSERVER.read_bytes()).hexdigest()[:16]


OmegaConf.register_new_resolver("harness_conformance.nooa_observer", observer_module_name, replace=True)


async def install_observer(sandbox: AsyncSandbox) -> None:
    """Stage an additive fixture module in the disposable private runtime."""
    from responses_api_agents.nooa_agent.sandbox_runtime import prepare_nooa_runtime

    python = await prepare_nooa_runtime(sandbox)
    site = await sandbox.exec(
        shlex.join([python, "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]),
        timeout_s=30,
    )
    if site.return_code != 0 or site.error_type:
        raise RuntimeError(f"Cannot locate NOOA fixture runtime: {site.stderr}")
    module = observer_module_name()
    await sandbox.upload(OBSERVER, f"{site.stdout.strip()}/{module}.py")
    checked = await sandbox.exec(
        shlex.join([python, "-I", "-c", f"from {module} import invoke_with_witness"]), timeout_s=30
    )
    if checked.return_code != 0 or checked.error_type:
        raise RuntimeError(f"Cannot import NOOA execution witness: {checked.stderr}")


class BorrowableLocalProvider(LocalProvider):
    """Reconnect to the Resources fixture's local workspace without acquiring ownership."""

    name = "conformance_local"

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        return {"sandbox_id": handle.sandbox_id, "workspace": str(handle.raw["workspace"])}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        workspace = Path(descriptor["workspace"])
        if not workspace.is_dir():
            raise FileNotFoundError(workspace)
        return SandboxHandle(
            sandbox_id=descriptor["sandbox_id"],
            provider_name=self.name,
            raw={"workspace": workspace, "env": {}, "owned": False},
        )


async def prepare_runtime() -> None:
    """Warm NOOA's ordinary sandbox runtime inside the disposable CI job container."""
    from nemo_gym.client import get_global_aiohttp_client

    with TemporaryDirectory(prefix="nooa-conformance-setup-") as workdir:
        sandbox = AsyncSandbox(BorrowableLocalProvider(concurrency=1), SandboxSpec(workdir=workdir))
        try:
            await sandbox.start()
            await install_observer(sandbox)
        finally:
            await sandbox.stop()
            await get_global_aiohttp_client().close()


if __name__ == "__main__":
    asyncio.run(prepare_runtime())
