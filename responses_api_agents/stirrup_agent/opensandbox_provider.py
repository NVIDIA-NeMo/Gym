# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""OpenSandbox-backed code execution provider for Stirrup.

Runs each tool call in an OpenSandbox sandbox through ``nemo_gym.sandbox.AsyncSandbox`` and returns what
``ApptainerCodeExecToolProvider`` would return for the same command, so the model sees the same tool results.
"""

from __future__ import annotations

import logging
import os
import shlex
import tempfile
import uuid
from pathlib import Path

from stirrup.core.models import ImageContentBlock, Tool, ToolUseCountMetadata
from stirrup.tools.code_backends.base import (
    SHELL_TIMEOUT,
    CodeExecToolProvider,
    CodeExecutionParams,
    CommandResult,
    SavedFile,
    SaveOutputFilesResult,
    UploadedFile,
    UploadFilesResult,
)


logger = logging.getLogger(__name__)

# Per-call stdout, stderr and exit code are written here and read back through the file API: execd merges the
# streams of background commands and strips line endings from streamed ones.
IO_DIR = "/tmp/.stirrup_exec"

_OPENSANDBOX_CONFIG = Path("sandbox/providers/opensandbox/configs/opensandbox.yaml")


def _provider_config(poll_interval_s: float) -> dict:
    """Gym's shipped OpenSandbox provider config, with connection settings resolved from the environment."""
    from omegaconf import OmegaConf

    import nemo_gym

    path = Path(nemo_gym.__file__).parent / _OPENSANDBOX_CONFIG
    cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)["sandbox"]["opensandbox"]
    cfg["operations"]["background_exec"] = True
    cfg["operations"]["background_poll_interval_s"] = poll_interval_s
    return {"opensandbox": cfg}


def _image_auth(image: str) -> dict:
    if image.startswith("nvcr.io/") and os.environ.get("NGC_API_KEY"):
        return {"image_auth": {"username": "$oauthtoken", "password": os.environ["NGC_API_KEY"]}}
    return {}


def shell_stdout(raw: bytes) -> bytes:
    """Stdout as stirrup's Apptainer shell returns it: always ending in exactly one added or kept newline."""
    return raw if raw.endswith(b"\n") else raw + b"\n"


class OpenSandboxCodeExecToolProvider(CodeExecToolProvider):
    """Execute Stirrup tool calls in an OpenSandbox sandbox started from an OCI image."""

    def __init__(
        self,
        image: str,
        *,
        working_dir: str = "/root",
        cpu: float = 4,
        memory_mib: int = 16384,
        arch: str = "arm64",
        poll_interval_s: float = 2.0,
        ready_timeout_s: int = 1200,
        allowed_commands: list[str] | None = None,
    ) -> None:
        super().__init__(allowed_commands=allowed_commands)
        self._image = image
        self._working_dir = working_dir.rstrip("/")
        self._cpu = cpu
        self._memory_mib = memory_mib
        self._arch = arch
        self._poll_interval_s = poll_interval_s
        self._ready_timeout_s = ready_timeout_s

        self._sandbox = None
        self._stale_io_dirs: list[str] = []

    def _serializable_kwargs(self) -> dict:
        """Return constructor kwargs that can be passed through Ray."""
        return {
            "image": self._image,
            "working_dir": self._working_dir,
            "cpu": self._cpu,
            "memory_mib": self._memory_mib,
            "arch": self._arch,
            "poll_interval_s": self._poll_interval_s,
            "ready_timeout_s": self._ready_timeout_s,
            "allowed_commands": None,
        }

    @property
    def patch(self) -> str | None:
        return None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def __aenter__(self) -> Tool[CodeExecutionParams, ToolUseCountMetadata]:
        from nemo_gym.sandbox.api import AsyncSandbox
        from nemo_gym.sandbox.providers import SandboxSpec

        spec = SandboxSpec(
            image=self._image,
            workdir=self._working_dir,
            ready_timeout_s=self._ready_timeout_s,
            # execd holds every response for 1 s after the command ends unless told otherwise.
            env={"EXECD_API_GRACE_SHUTDOWN": "50ms"},
            resources={"cpu": self._cpu, "memory_mib": self._memory_mib},
            provider_options={"platform": {"os": "linux", "arch": self._arch}} | _image_auth(self._image),
        )
        sandbox = AsyncSandbox(_provider_config(self._poll_interval_s))
        await sandbox.start(spec)
        self._sandbox = sandbox
        try:
            # Same setup as the Apptainer provider: git trusts every directory (this writes /root/.gitconfig).
            await self._exec("git config --global --add safe.directory '*'", timeout=10)
        except BaseException:
            await self._stop()
            raise
        logger.info("Started OpenSandbox sandbox from %s", self._image)
        return self.get_code_exec_tool()

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self._stop()

    async def _stop(self) -> None:
        if self._sandbox is not None:
            try:
                await self._sandbox.stop()
            except Exception as exc:
                logger.warning("Failed to stop OpenSandbox sandbox: %s", exc)
            self._sandbox = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_sandbox(self):
        if self._sandbox is None:
            raise RuntimeError("OpenSandbox sandbox not started.")
        return self._sandbox

    async def _download(self, remote_path: str) -> bytes:
        with tempfile.TemporaryDirectory(prefix="stirrup_osb_") as tmp:
            local = Path(tmp) / "f"
            await self._require_sandbox().download(remote_path, local)
            return local.read_bytes()

    async def _exec(self, cmd: str, *, timeout: int = SHELL_TIMEOUT) -> tuple[int, bytes, bytes]:
        """Run a command the way the Apptainer shell does and return (exit code, stdout, stderr)."""
        sandbox = self._require_sandbox()
        io_dir = f"{IO_DIR}/{uuid.uuid4().hex[:12]}"
        cleanup = f"rm -rf {' '.join(self._stale_io_dirs)}; " if self._stale_io_dirs else ""
        self._stale_io_dirs = []
        wrapped_cmd = f"timeout -k 10 {timeout} bash -c {shlex.quote(cmd)}" if timeout > 0 else cmd
        script = (
            f"{cleanup}mkdir -p {io_dir} && "
            f"{{ cd {self._working_dir} && ( {wrapped_cmd} ); }} >{io_dir}/stdout 2>{io_dir}/stderr </dev/null; "
            f"_rc=$?; "
            f'echo "$_rc $(stat -c %s {io_dir}/stdout) $(stat -c %s {io_dir}/stderr)"'
        )
        try:
            result = await sandbox.exec(script, timeout_s=timeout + 30)
        except TimeoutError:
            self._stale_io_dirs.append(io_dir)
            return 1, b"", f"Command timed out after {timeout} seconds".encode()
        self._stale_io_dirs.append(io_dir)

        try:
            rc_text, stdout_size, stderr_size = (result.stdout or "").strip().splitlines()[-1].split()
            rc, stdout_size, stderr_size = int(rc_text), int(stdout_size), int(stderr_size)
        except (IndexError, ValueError) as exc:
            raise RuntimeError(
                f"OpenSandbox exec returned no status (rc={result.return_code}, error={result.error_type}): "
                f"{(result.stdout or '')[-500:]} {(result.stderr or '')[-500:]}"
            ) from exc

        stdout = await self._download(f"{io_dir}/stdout") if stdout_size else b""
        stderr = await self._download(f"{io_dir}/stderr") if stderr_size else b""
        stdout = shell_stdout(stdout)

        if rc in (124, 137) and timeout > 0:
            timeout_msg = f"Command timed out after {timeout} seconds".encode()
            return 1, stdout, stderr + b"\n" + timeout_msg if stderr else timeout_msg
        return rc, stdout, stderr

    def _resolve_path(self, path: str) -> str:
        """Resolve a relative or absolute path to a sandbox-absolute path."""
        if path.startswith("/"):
            return path
        return f"{self._working_dir}/{path}"

    # ------------------------------------------------------------------
    # CodeExecToolProvider interface
    # ------------------------------------------------------------------

    async def run_command(self, cmd: str, *, timeout: int | None = None) -> CommandResult:
        timeout = SHELL_TIMEOUT if timeout is None else timeout
        if not self._check_allowed(cmd):
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=f"Command not allowed: '{cmd}' does not match any allowed patterns",
                error_kind="command_not_allowed",
                advice="Only commands matching the allowlist patterns are permitted.",
            )

        try:
            rc, stdout, stderr = await self._exec(cmd, timeout=timeout)
            return CommandResult(
                exit_code=rc,
                stdout=stdout.decode("utf-8", errors="replace"),
                stderr=stderr.decode("utf-8", errors="replace"),
            )
        except Exception as exc:
            return CommandResult(
                exit_code=1,
                stdout="",
                stderr=str(exc),
                error_kind="execution_error",
            )

    async def read_file_bytes(self, path: str) -> bytes:
        try:
            return await self._download(self._resolve_path(path))
        except Exception as exc:
            raise FileNotFoundError(f"Cannot read {path}: {exc}") from exc

    async def write_file_bytes(self, path: str, content: bytes) -> None:
        sandbox_path = self._resolve_path(path)
        rc, _, stderr = await self._exec(f"mkdir -p $(dirname {shlex.quote(sandbox_path)})", timeout=30)
        if rc != 0:
            raise OSError(f"Failed to write {path}: {stderr.decode('utf-8', errors='replace')}")
        with tempfile.TemporaryDirectory(prefix="stirrup_osb_") as tmp:
            local = Path(tmp) / "f"
            local.write_bytes(content)
            await self._require_sandbox().upload(local, sandbox_path)

    async def file_exists(self, path: str) -> bool:
        try:
            rc, _, _ = await self._exec(f"test -f {shlex.quote(self._resolve_path(path))}", timeout=10)
            return rc == 0
        except RuntimeError:
            return False

    async def is_directory(self, path: str) -> bool:
        try:
            rc, _, _ = await self._exec(f"test -d {shlex.quote(self._resolve_path(path))}", timeout=10)
            return rc == 0
        except RuntimeError:
            return False

    async def list_files(self, path: str) -> list[str]:
        sandbox_path = self._resolve_path(path)
        rc, stdout, _ = await self._exec(f"find {shlex.quote(sandbox_path)} -type f 2>/dev/null", timeout=30)
        if rc != 0:
            return []

        base = sandbox_path.rstrip("/")
        files = []
        for line in stdout.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            files.append(line[len(base) + 1 :] if line.startswith(base + "/") else line)
        return files

    async def view_image(self, path: str) -> ImageContentBlock:
        return ImageContentBlock(data=await self.read_file_bytes(path))

    # ------------------------------------------------------------------
    # File transfer helpers
    # ------------------------------------------------------------------

    async def save_output_files(
        self,
        paths: list[str],
        output_dir: Path | str,
        dest_env: CodeExecToolProvider | None = None,
    ) -> SaveOutputFilesResult:
        if dest_env is not None:
            return await super().save_output_files(paths, output_dir, dest_env)

        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        result = SaveOutputFilesResult()
        for src_path in paths:
            try:
                sandbox_path = self._resolve_path(src_path)
                rc, _, stderr = await self._exec(f"test -f {shlex.quote(sandbox_path)}", timeout=30)
                if rc != 0:
                    result.failed[src_path] = stderr.decode("utf-8", errors="replace") or "File not found"
                    continue
                dest_path = output_dir_path / Path(src_path).name
                dest_path.write_bytes(await self._download(sandbox_path))
                result.saved.append(
                    SavedFile(source_path=src_path, output_path=dest_path, size=dest_path.stat().st_size)
                )
            except Exception as exc:
                result.failed[src_path] = str(exc)
                logger.exception("Failed to save file: %s", src_path)
        return result

    async def upload_files(
        self,
        *paths: Path | str,
        source_env: CodeExecToolProvider | None = None,
        dest_dir: str | None = None,
    ) -> UploadFilesResult:
        if source_env is not None:
            return await super().upload_files(*paths, source_env=source_env, dest_dir=dest_dir)

        sandbox = self._require_sandbox()
        sandbox_dest = f"{self._working_dir}/{dest_dir}" if dest_dir else self._working_dir
        result = UploadFilesResult()
        for source in paths:
            source = Path(source).resolve()
            if not source.exists():
                result.failed[str(source)] = "File or directory does not exist"
                continue
            try:
                files = (
                    [(source, source.name)]
                    if source.is_file()
                    else [(f, str(f.relative_to(source))) for f in sorted(source.rglob("*")) if f.is_file()]
                )
                dirs = sorted({str(Path(f"{sandbox_dest}/{rel}").parent) for _, rel in files} | {sandbox_dest})
                rc, _, stderr = await self._exec(f"mkdir -p {' '.join(shlex.quote(d) for d in dirs)}", timeout=60)
                if rc != 0:
                    result.failed[str(source)] = stderr.decode("utf-8", errors="replace")
                    continue
                for local, rel in files:
                    await sandbox.upload(local, f"{sandbox_dest}/{rel}")
                    result.uploaded.append(
                        UploadedFile(source_path=local, dest_path=f"{sandbox_dest}/{rel}", size=local.stat().st_size)
                    )
            except Exception as exc:
                result.failed[str(source)] = str(exc)
                logger.exception("Failed to upload: %s", source)
        return result
