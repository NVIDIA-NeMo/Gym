# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run each model-written pytest invocation in a fresh, offline Gym sandbox."""

import asyncio
import base64
import json
import math
import shlex
import stat
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path, PurePosixPath
from typing import Any, AsyncIterator

from anyio import CancelScope

from nemo_gym.sandbox import AsyncSandbox, SandboxSpec, resolve_provider_config, resolve_provider_metadata


WORKDIR = "/workspace/episode"
MAX_FILES_BYTES = 16 * 1024 * 1024
# Run with -I so an episode's sitecustomize.py cannot affect serialization. Only
# regular files are returned: a symlink/device created by pytest never reaches Gym.
COLLECT_FILES = """
import base64, json, pathlib, stat
root = pathlib.Path('/workspace/episode')
files = {}
size = 0
for path in root.rglob('*'):
    mode = path.lstat().st_mode
    if stat.S_ISDIR(mode):
        continue
    if not stat.S_ISREG(mode):
        raise RuntimeError('pytest produced a non-regular file')
    size += path.stat().st_size
    if size > 16 * 1024 * 1024:
        raise RuntimeError('pytest files exceed transfer limit')
    data = path.read_bytes()
    files[path.relative_to(root).as_posix()] = {'data': base64.b64encode(data).decode(), 'mode': stat.S_IMODE(mode) & 0o777}
pathlib.Path('/tmp/gym-pytest-files.json').write_text(json.dumps(files))
"""


def episode_files(directory: Path) -> dict[str, bytes]:
    """Read only regular files inside the requested episode directory."""
    files: dict[str, bytes] = {}
    size = 0
    for path in directory.rglob("*"):
        mode = path.lstat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ValueError("Episode files must not contain symlinks or special files")
        size += path.stat().st_size
        if size > MAX_FILES_BYTES:
            raise ValueError("Episode files exceed transfer limit")
        data = path.read_bytes()
        files[path.relative_to(directory).as_posix()] = data
    return files


def restore_files(directory: Path, files: dict[str, dict[str, Any]]) -> None:
    """Validate the complete returned tree before changing any episode file."""
    decoded: dict[str, tuple[bytes, int]] = {}
    size = 0
    for name, value in files.items():
        path = PurePosixPath(name)
        if not name or name == "." or path.is_absolute() or ".." in path.parts or path.as_posix() != name:
            raise ValueError("Sandbox returned a path outside the episode")
        mode = value["mode"]
        if isinstance(mode, bool) or not isinstance(mode, int) or not 0 <= mode <= 0o777:
            raise ValueError("Sandbox returned an invalid regular-file mode")
        data = base64.b64decode(value["data"], validate=True)
        size += len(data)
        if size > MAX_FILES_BYTES:
            raise ValueError("Sandbox files exceed transfer limit")
        decoded[name] = (data, mode)
    # Reject file/directory collisions before writing any part of the result.
    if any(parent.as_posix() in decoded for name in decoded for parent in PurePosixPath(name).parents):
        raise ValueError("Sandbox returned conflicting file paths")
    previous = episode_files(directory)
    for name in previous:
        (directory / name).unlink()
    for path in sorted(directory.rglob("*"), reverse=True):
        if path.is_dir():
            path.rmdir()
    for name, (data, mode) in decoded.items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        path.chmod(mode)


def isolated_provider(provider: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Require a provider whose offline, mount-free execution we can enforce."""
    name, options = next(iter(provider.items()))
    options = dict(options or {})
    if name == "docker":
        options["create"] = dict(options.get("create") or {}) | {
            "network": "none",
            "extra_run_args": [],
            "cap_drop": ["ALL"],
            "security_opt": ["no-new-privileges"],
        }
        return {name: options}, {}
    if name == "opensandbox":
        return provider, {"network_policy": {"defaultAction": "deny", "egress": []}}
    raise ValueError(f"Provider {name!r} has no supported offline pytest isolation policy")


class PytestIsolation:
    """Own one phase's private bridge and all of its short-lived pytest sandboxes."""

    def __init__(
        self,
        root: Path,
        provider_ref: str | dict[str, Any],
        named_configs: dict[str, Any],
        image: str | None,
    ) -> None:
        self.root = root.resolve()
        self.image = image
        self.provider_ref = provider_ref
        self.named_configs = named_configs
        self._provider: dict[str, Any] | None = None
        self.provider_options: dict[str, Any] = {}
        self.metadata: dict[str, Any] = {}
        self.tasks: set[asyncio.Task[None]] = set()
        self.lock = asyncio.Lock()
        self.failure: Exception | None = None

    @property
    def provider(self) -> dict[str, Any]:
        """Resolve isolation only when a coding scenario actually requests pytest."""
        if self._provider is None:
            if not self.image:
                raise ValueError("pytest_sandbox_image is required; host execution is disabled")
            self._provider, self.provider_options = isolated_provider(
                resolve_provider_config(self.provider_ref, self.named_configs)
            )
            self.metadata = resolve_provider_metadata(self.provider_ref, self.named_configs)
        return self._provider

    async def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute the unchanged pytest arguments with just the current directory."""
        directory = Path(request["directory"])
        # Never follow a symlink supplied as the directory or one of its ancestors.
        if directory.is_symlink() or directory.resolve() != directory.absolute():
            raise ValueError("pytest directory must not traverse symlinks")
        directory = directory.resolve()
        if not directory.is_relative_to(self.root) or not directory.is_dir():
            raise ValueError("pytest directory is outside this phase")
        targets = request["targets"]
        if not isinstance(targets, list) or any(
            not isinstance(t, str)
            or t.startswith("-")
            or PurePosixPath(t).is_absolute()
            or ".." in PurePosixPath(t).parts
            for t in targets
        ):
            raise ValueError("pytest targets must stay inside the episode")
        timeout_s = request["timeout_s"]
        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not math.isfinite(timeout_s)
            or not 0 < timeout_s <= 7200
        ):
            raise ValueError("Invalid pytest timeout")
        provider = self.provider
        files = episode_files(directory)
        modes = {name: stat.S_IMODE((directory / name).stat().st_mode) & 0o777 for name in files}
        spec = SandboxSpec(
            image=self.image,
            workdir=WORKDIR,
            ttl_s=timeout_s + 120,
            files={f"{WORKDIR}/{name}": value for name, value in files.items()},
            metadata=self.metadata,
            provider_options=self.provider_options,
            env={"HOME": WORKDIR, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        sandbox = AsyncSandbox(provider, spec)
        try:
            await sandbox.start()
            args = ["python", "-I", "-m", "pytest", "-q"]
            if request["exitfirst"]:
                args.append("-x")
            args += ["--no-header", "-p", "no:cacheprovider"]
            if request["noconftest"]:
                args += ["--noconftest", "--rootdir", WORKDIR, "-c", "/dev/null"]
            # Uploads may not retain modes. Restore only regular-file permissions before executing pytest.
            setup = [
                shlex.join(["chmod", format(mode, "o"), "--", f"{WORKDIR}/{name}"]) for name, mode in modes.items()
            ]
            command = " && ".join([*setup, shlex.join(args + targets)])
            result = await sandbox.exec(command, cwd=WORKDIR, timeout_s=timeout_s)
            if result.error_type == "timeout":
                return {"return_code": None, "output": ""}
            if result.error_type is not None or result.return_code is None:
                raise RuntimeError("Isolated pytest did not report a process exit")
            collected = await sandbox.exec("python -I -c " + shlex.quote(COLLECT_FILES), timeout_s=30)
            if collected.error_type is not None or collected.return_code != 0:
                raise RuntimeError("Could not safely collect isolated pytest files")
            with tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "files.json"
                await sandbox.download("/tmp/gym-pytest-files.json", output)
                info = output.lstat()
                if not stat.S_ISREG(info.st_mode):
                    raise ValueError("Sandbox result must be a regular file")
                if info.st_size > 2 * MAX_FILES_BYTES:
                    raise ValueError("Sandbox result exceeds transfer limit")
                restore_files(directory, json.loads(output.read_text()))
            return {"return_code": result.return_code, "output": (result.stdout or "") + (result.stderr or "")}
        finally:
            with CancelScope(shield=True):
                await sandbox.stop()

    async def handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        """Service a trusted harness request; never expose this socket to pytest."""
        task = asyncio.current_task()
        self.tasks.add(task)
        try:
            try:
                async with self.lock:
                    request = json.loads(await asyncio.wait_for(reader.readline(), 10))
                    response = await self.run(request)
            except Exception as error:
                self.failure = error
                response = {"error": "Isolated pytest failed; local execution is disabled"}
            writer.write(json.dumps(response).encode() + b"\n")
            await writer.drain()
        finally:
            writer.close()
            self.tasks.discard(task)
            with CancelScope(shield=True):
                await writer.wait_closed()

    @asynccontextmanager
    async def serve(self) -> AsyncIterator[str]:
        """Close the socket and stop in-flight sandboxes on every exit path."""
        with tempfile.TemporaryDirectory(prefix="gym-pytest-control-") as temporary:
            endpoint = str(Path(temporary) / "runner.sock")
            server = await asyncio.start_unix_server(self.handle, path=endpoint)
            try:
                yield endpoint
            finally:
                server.close()
                with CancelScope(shield=True):
                    tasks = list(self.tasks)
                    for task in tasks:
                        task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    await server.wait_closed()
            if self.failure is not None:
                raise RuntimeError("Isolated pytest failed") from self.failure
