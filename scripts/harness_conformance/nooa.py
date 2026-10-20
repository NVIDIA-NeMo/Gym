# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NOOA's borrowed-session fixture, backed by the suite's local subprocess provider."""

import asyncio
import hashlib
import json
import shlex
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers import SandboxHandle, SandboxSpec, register_provider
from nemo_gym.sandbox.providers.local.provider import LocalProvider

from .provider import Probe


OBSERVER = Path(__file__).with_name("nooa_observer.py")


def observer_module_name() -> str:
    """Bind the fixture module import to its exact source bytes."""
    return "nooa_conformance_observer_" + hashlib.sha256(OBSERVER.read_bytes()).hexdigest()[:16]


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


@dataclass
class BorrowedSandboxProbe(Probe):
    """Seed and close the sandbox through Resources; NOOA runs its ordinary borrower path."""

    sandbox: AsyncSandbox | None = field(default=None, init=False)
    native_tool_calls: list[dict] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        register_provider("conformance_local", BorrowableLocalProvider, override=True)

    def save(self) -> None:
        super().save()
        path = self.directory / "witness.json"
        witness = json.loads(path.read_text())
        witness["native_tool_calls"] = self.native_tool_calls
        path.write_text(json.dumps(witness, indent=2) + "\n")

    def _observe_tool_results(self, body: dict) -> None:
        super()._observe_tool_results(body)
        # CodeActV2 acknowledges a cell first and reports its execution in a later
        # user-side PythonOutput event. Witness the feedback the model actually sees.
        execution_path = self.directory / "native_executions.jsonl"
        executions = (
            [json.loads(line) for line in execution_path.read_text().splitlines()] if execution_path.exists() else []
        )
        scripted_ids = {call["id"] for call in self.tool_calls}
        if self.finish_call:
            scripted_ids.add(self.finish_call["id"])
        calls = {
            item["call_id"]: item
            for item in body.get("input", [])
            if isinstance(item, dict) and item.get("type") == "function_call"
        }
        self.native_tool_calls = []
        for execution in executions:
            if execution["id"] in scripted_ids:
                continue
            request = calls.get(execution["id"], {})
            self.native_tool_calls.append({**execution, "name": request.get("name"), "outputs": []})
        for call in self.tool_calls + self.native_tool_calls:
            outputs = [
                item["content"]
                for item in body.get("input", [])
                if isinstance(item, dict)
                and item.get("role") == "user"
                and isinstance(item.get("content"), str)
                and f"PythonOutput(tool_call_id={call['id']!r}," in item["content"]
            ]
            if outputs:
                call["outputs"] = outputs
                call["result_seen"] = any(call["token"] in output for output in outputs) if "token" in call else True

    async def seed_sandbox(self) -> dict:
        if self.sandbox is None:
            if (self.directory / "native_executions.jsonl").exists():
                raise FileExistsError("NOOA execution witness must belong to a fresh episode")
            self.sandbox = AsyncSandbox(
                BorrowableLocalProvider(concurrency=1), SandboxSpec(workdir=str(self.directory / "workspace"))
            )
            await self.sandbox.start()
            await install_observer(self.sandbox)
        return {
            "sandbox_access": {
                "connection": {
                    "kind": "direct",
                    "provider_config_ref": "probe_sandbox",
                    "descriptor": await self.sandbox.serialize(),
                },
                "workdir": str(self.directory / "workspace"),
            }
        }

    async def close_sandbox(self) -> None:
        if self.sandbox is not None:
            await self.sandbox.stop()


async def prepare_runtime() -> None:
    """Warm NOOA's ordinary sandbox runtime inside the disposable CI job container."""
    from nemo_gym.server_utils import get_global_aiohttp_client

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
