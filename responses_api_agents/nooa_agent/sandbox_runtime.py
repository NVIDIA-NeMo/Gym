# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Package and stage a private NOOA runtime for the target sandbox."""

import gzip
import hashlib
import io
import json
import logging
import re
import shlex
import tarfile
import tempfile
from pathlib import Path
from uuid import uuid4

from nemo_gym.sandbox import AsyncSandbox


LOGGER = logging.getLogger(__name__)
_RUNTIME_ROOT = "/opt/nemo-gym-nooa"
_PYTHON_VERSION = "3.13.14"
_PYTHON_RELEASE = "20260805"
_INSTALLER_PATH = "responses_api_agents/nooa_agent/install_nooa_runtime.sh"
# Expected digests published with the immutable Astral PBS 20260805 release.
# https://github.com/astral-sh/python-build-standalone/releases/tag/20260805
_PYTHON_SHA256 = {
    "x86_64/gnu": "9f9ebf776b2d4913fe5a91508ab42e6ce17d266138f76a7782c05fa1c3cba1d1",  # pragma: allowlist secret
    "aarch64/gnu": "4777d7df2edb47b96e53abad5e1b9df1b2a1a9b2f7bdba12b5c0122163b3fed9",  # pragma: allowlist secret
}
_IMPORT_CHECK = (
    "import sys; assert sys.version_info[:3] == (3,13,14); "
    "import nooa, nooa_cli; from nooa_bench.bench_agent import BenchAgent; "
    "import responses_api_agents.nooa_agent.runner; "
    "from responses_api_agents.nooa_agent.resource_tools import ResourceToolDispatcher; "
    "from responses_api_agents.nooa_agent.observability import GymTraceHooks"
)


def _runtime_archive(root: Path) -> tuple[bytes, str]:
    """Include current source, never repository metadata, caches, or user artifacts."""
    files = {name: (root / name).read_bytes() for name in ("pyproject.toml", "README.md", "LICENSE")}
    for path in (root / "nemo_gym").rglob("*.py"):
        if not path.is_symlink() and not any(part.startswith(".") for part in path.relative_to(root).parts):
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    agent_dir = root / "responses_api_agents/nooa_agent"
    files[_INSTALLER_PATH] = (root / _INSTALLER_PATH).read_bytes()
    for path in agent_dir.glob("*.py"):
        if not path.is_symlink():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    requirements = (agent_dir / "requirements.txt").read_text()
    runtime_requirements = []
    for line in requirements.splitlines():
        if not line.strip() or line.startswith(("#", "-e ")):
            continue
        # Same immutable source pin, without a dependency on the task image's git.
        line = re.sub(
            r"git\+https://github\.com/([^@ ]+)\.git@([0-9a-f]{40})$",
            r"https://github.com/\1/archive/\2.tar.gz",
            line,
        )
        runtime_requirements.append(line)
    files["runtime-requirements.txt"] = ("\n".join(runtime_requirements) + "\n").encode()
    digest = hashlib.sha256(
        json.dumps([_PYTHON_VERSION, _PYTHON_RELEASE, _PYTHON_SHA256, _IMPORT_CHECK], sort_keys=True).encode()
    )
    output = io.BytesIO()
    with gzip.GzipFile(fileobj=output, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, contents in sorted(files.items()):
                digest.update(name.encode() + b"\0" + contents + b"\0")
                entry = tarfile.TarInfo(name)
                entry.size = len(contents)
                entry.mode = 0o644
                archive.addfile(entry, io.BytesIO(contents))
    return output.getvalue(), digest.hexdigest()


async def prepare_nooa_runtime(sandbox: AsyncSandbox) -> str:
    """Stage verified artifacts and return the target's healthy private Python.

    Cold setup requires Linux x86_64/aarch64 with glibc, bash, tar, gzip,
    sha256sum, flock, tee, mv with -T support, and writable /opt/nemo-gym-nooa
    and /tmp. No root or system
    package manager is invoked. The controller supplies Python; the sandbox still
    needs outbound package/build dependency access. This is not an offline install.
    A healthy preinstalled or mounted runtime is reused without writing into it.
    """
    root = Path(__file__).resolve().parents[2]
    source, fingerprint = _runtime_archive(root)
    scratch = f"/tmp/nemo-gym-nooa-setup-{uuid4().hex}"
    script = scratch + ".sh"
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix="nooa-runtime-") as directory:
            local = Path(directory)
            # Execute precisely the installer included in this source fingerprint.
            with tarfile.open(fileobj=io.BytesIO(source), mode="r:gz") as archive:
                member = archive.extractfile(_INSTALLER_PATH)
                assert member is not None
                installer = local / "install_nooa_runtime.sh"
                installer.write_bytes(member.read())
            await sandbox.upload(installer, script)
            probe = await sandbox.exec(shlex.join(["bash", script, "probe"]), cwd="/", timeout_s=30)
            target = (probe.stdout or "").strip()
            if probe.return_code != 0 or target not in _PYTHON_SHA256:
                raise RuntimeError(
                    f"Unsupported NOOA sandbox target: {target!r}; exit={probe.return_code}; "
                    f"stderr={(probe.stderr or '')[-2000:]}"
                )
            arch, libc = target.split("/")
            runtime = f"{_RUNTIME_ROOT}/{fingerprint}-{arch}-{libc}"
            python = f"{runtime}/python/bin/python3"
            check = shlex.join(["bash", script, "check", runtime, fingerprint, _IMPORT_CHECK])
            cached = await sandbox.exec(check, cwd="/", timeout_s=60)
            if cached.return_code == 0:
                completed = True
                return python

            # Reuse mini-SWE's checksum-aware, atomic controller download cache.
            # This controller-only helper is deliberately not needed in the worker.
            from responses_api_agents.miniswe_sandboxed_agent.bootstrap import cached_download

            expected = _PYTHON_SHA256[target]
            url = (
                "https://github.com/astral-sh/python-build-standalone/releases/download/"
                f"{_PYTHON_RELEASE}/cpython-{_PYTHON_VERSION}%2B{_PYTHON_RELEASE}-"
                f"{arch}-unknown-linux-{libc}-install_only.tar.gz"
            )
            cached_archive = await cached_download(url, sha256=expected)
            python_archive = local / "python.tar.gz"
            python_archive.write_bytes(cached_archive.read_bytes())
            # Check the private upload snapshot too, so cache replacement cannot
            # change the bytes between verification and sandbox transfer.
            if hashlib.sha256(python_archive.read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"NOOA Python artifact checksum mismatch for {target}")
            source_archive = local / "source.tar.gz"
            source_archive.write_bytes(source)
            await sandbox.upload(source_archive, f"{scratch}/source.tar.gz")
            await sandbox.upload(python_archive, f"{scratch}/python.tar.gz")
            command = shlex.join(
                [
                    "bash",
                    script,
                    "install",
                    runtime,
                    fingerprint,
                    hashlib.sha256(source).hexdigest(),
                    expected,
                    _IMPORT_CHECK,
                ]
            )
            result = await sandbox.exec(command, cwd="/", timeout_s=1800)
            if result.return_code != 0:
                raise RuntimeError(
                    f"NOOA runtime preparation failed: exit={result.return_code}; command={command}; "
                    f"diagnostics={scratch}; stdout={(result.stdout or '')[-2000:]} "
                    f"stderr={(result.stderr or '')[-2000:]}"
                )
            completed = True
            return python
    finally:
        # Failed/cancelled setup retains its unique uploads and installer logs.
        # It may still be executing remotely; a retry never modifies those paths.
        if completed:
            try:
                cleanup = await sandbox.exec(
                    f"rm -rf -- {shlex.quote(scratch)} {shlex.quote(script)}", cwd="/", timeout_s=30
                )
                if cleanup.return_code != 0:
                    LOGGER.warning("Could not remove NOOA setup files at %s", scratch)
            except Exception:
                # The runtime is already healthy; a scratch cleanup transport
                # failure must not turn successful installation into seed failure.
                LOGGER.warning("Could not remove NOOA setup files at %s", scratch, exc_info=True)
