# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage a private Fabric interpreter without using the task's Python or venv."""

import asyncio
import hashlib
import io
import json
import logging
import shlex
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox
from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download


LOG = logging.getLogger(__name__)
FABRIC_VERSION = "0.5.0"
# Astral python-build-standalone 20260805, also used by Gym's NOOA adapter.
PYTHON_SHA256 = {
    "x86_64/gnu": "9f9ebf776b2d4913fe5a91508ab42e6ce17d266138f76a7782c05fa1c3cba1d1",  # pragma: allowlist secret
    "aarch64/gnu": "4777d7df2edb47b96e53abad5e1b9df1b2a1a9b2f7bdba12b5c0122163b3fed9",  # pragma: allowlist secret
}
HEALTH = (
    "import sys, importlib.metadata as m; assert sys.version_info[:3] == (3,13,14); "
    "assert m.version('nemo-fabric') == '0.5.0'; from nemo_fabric import Fabric"
)


def runtime_requirements(adapter_id: str, *, relay: bool) -> list[str]:
    """Install the selected Fabric extra; CLI adapters still need their native runtime."""
    adapter = adapter_id.removeprefix("nvidia.fabric.")
    adapter = {"langchain.deepagents": "deepagents"}.get(adapter, adapter)
    extra = {"hermes": "hermes-agent"}.get(adapter, adapter)
    available = {"claude", "codex", "deepagents", "mini-swe-agent", "nooa", "openclaw", "openhands", "hermes-agent"}
    if not adapter_id.startswith("nvidia.fabric.") or extra not in available:
        raise ValueError(f"No sandbox runtime profile for Fabric adapter {adapter_id!r}")
    extras = [extra]
    if relay:
        extras.append("relay")
    suffix = f"[{','.join(sorted(extras))}]" if extras else ""
    requirements = [f"nemo-fabric{suffix}=={FABRIC_VERSION}", "litellm<1.103"]
    if adapter == "openhands":
        requirements += ["openhands-sdk==1.50.0", "openhands-tools==1.50.0"]
    if adapter == "hermes":
        requirements += [
            "hermes-agent @ https://github.com/NousResearch/hermes-agent/archive/0ee05d72f2b30385d4861fd8e30511c6d51100e4.tar.gz",
            "tokenizers>=0.22.2",
        ]
    return requirements


def runtime_archive(requirements: list[str]) -> tuple[bytes, str]:
    """Fingerprint installer, dependency specification and interpreter together."""
    contents = ("\n".join(requirements) + "\n").encode()
    installer = Path(__file__).with_name("install_fabric_runtime.sh").read_bytes()
    fingerprint = hashlib.sha256(
        contents + installer + json.dumps(PYTHON_SHA256, sort_keys=True).encode() + HEALTH.encode()
    )
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w:gz") as archive:
        entry = tarfile.TarInfo("runtime-requirements.txt")
        entry.size = len(contents)
        archive.addfile(entry, io.BytesIO(contents))
    return output.getvalue(), fingerprint.hexdigest()


async def prepare_fabric_runtime(sandbox: AsyncSandbox, *, adapter_id: str, relay: bool, timeout: float) -> str:
    """Reuse or atomically install Python + Fabric outside the task worktree.

    Requires Linux glibc, bash, tar, sha256sum, flock, and a writable /opt and /tmp.
    Python is downloaded and verified on the controller; pip runs in the private
    interpreter. The task's Python, PATH, HOME and dependencies are not modified.
    """
    source, fingerprint = runtime_archive(runtime_requirements(adapter_id, relay=relay))
    scratch = f"/tmp/nemo-gym-fabric-setup-{uuid4().hex}"
    script = scratch + ".sh"
    completed = False
    try:
        await sandbox.upload(Path(__file__).with_name("install_fabric_runtime.sh"), script)
        probe = await sandbox.exec(shlex.join(["bash", script, "probe"]), cwd="/", timeout_s=30)
        target = (probe.stdout or "").strip()
        if probe.return_code != 0 or probe.error_type or target not in PYTHON_SHA256:
            raise RuntimeError(f"Unsupported Fabric sandbox platform: {target!r}; {probe.stderr}")
        arch, libc = target.split("/")
        runtime = f"/opt/nemo-gym-fabric/{fingerprint}-{arch}-{libc}"
        python = f"{runtime}/python/bin/python3"
        cached = await sandbox.exec(
            shlex.join(["bash", script, "check", runtime, fingerprint, HEALTH]), cwd="/", timeout_s=60
        )
        if cached.return_code == 0 and not cached.error_type:
            completed = True
            return python
        expected = PYTHON_SHA256[target]
        url = (
            "https://github.com/astral-sh/python-build-standalone/releases/download/20260805/"
            f"cpython-3.13.14%2B20260805-{arch}-unknown-linux-{libc}-install_only.tar.gz"
        )
        cached_python = await cached_download(url, sha256=expected)
        with tempfile.TemporaryDirectory(prefix="fabric-runtime-") as temporary:
            directory = Path(temporary)
            python_archive = directory / "python.tar.gz"
            await asyncio.to_thread(python_archive.write_bytes, cached_python.read_bytes())
            if hashlib.sha256(python_archive.read_bytes()).hexdigest() != expected:
                raise RuntimeError("Fabric Python upload snapshot failed checksum verification")
            source_archive = directory / "source.tar.gz"
            source_archive.write_bytes(source)
            await sandbox.upload(python_archive, f"{scratch}/python.tar.gz")
            await sandbox.upload(source_archive, f"{scratch}/source.tar.gz")
        result = await sandbox.exec(
            shlex.join(
                ["bash", script, "install", runtime, fingerprint, hashlib.sha256(source).hexdigest(), expected, HEALTH]
            ),
            cwd="/",
            timeout_s=timeout,
        )
        if result.return_code != 0 or result.error_type:
            raise RuntimeError(
                f"Fabric runtime installation failed; diagnostics in {scratch}: {(result.stderr or result.stdout or '')[-8000:]}"
            )
        completed = True
        return python
    finally:
        # Retain failed setup artifacts: cancellation does not prove remote pip exited.
        if completed:
            try:
                result = await sandbox.exec(
                    f"rm -rf -- {shlex.quote(scratch)} {shlex.quote(script)}", cwd="/", timeout_s=30
                )
                if result.return_code != 0 or result.error_type:
                    LOG.warning("Fabric setup scratch cleanup failed: %s", scratch)
            except Exception:
                LOG.warning("Fabric setup scratch cleanup failed: %s", scratch, exc_info=True)
