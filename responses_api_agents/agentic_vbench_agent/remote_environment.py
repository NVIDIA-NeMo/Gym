# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Upstream Harbor Docker environment backed by the cluster's rootless API."""

import asyncio
import hashlib
import json
import shlex
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from harbor.environments.base import ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment


def _adapt_dockerfile(original: bytes) -> bytes:
    # The pinned material downloads are GETs saved with curl -o, so retries
    # discard partial output. Default --retry misses connection resets/EOFs;
    # HTTP/1.1 also avoids the observed HTTP/2 stream cancellations.
    return original.replace(b"apt-get ", b"apt-get -o APT::Sandbox::User=root ").replace(
        b"curl --fail --silent --show-error --location --retry 5 --retry-delay 3",
        b"curl --http1.1 --retry-all-errors --fail --silent --show-error --location --retry 5 --retry-delay 3",
    )


class RemoteEnvironment(DockerEnvironment):
    _DOCKER_COMPOSE_BASE_PATH = Path(__file__).with_name("remote-compose.yaml")

    def __init__(
        self, *, environment_dir: Path, runtime_root: str, opencode_session_headers_path: str | None = None, **kwargs
    ):
        self._opencode_session_headers_path = opencode_session_headers_path
        if (environment_dir / "docker-compose.yaml").exists():
            raise ValueError("Refusing to replace a task's existing compose definition")
        staged = Path(runtime_root) / "build-context"
        shutil.copytree(environment_dir, staged)
        dockerfile = staged / "Dockerfile"
        original = dockerfile.read_bytes()
        adapted = _adapt_dockerfile(original)
        dockerfile.write_bytes(adapted)
        (staged.parent / "rootless-build.json").write_text(
            json.dumps(
                {
                    "source": str(environment_dir),
                    "original_dockerfile_sha256": hashlib.sha256(original).hexdigest(),
                    "build_dockerfile_sha256": hashlib.sha256(adapted).hexdigest(),
                    "adaptation": (
                        "APT sandbox uses mapped container root; material GETs retry transport errors over HTTP/1.1; "
                        "task inputs and verifiers unchanged"
                    ),
                },
                indent=2,
            )
            + "\n"
        )
        super().__init__(environment_dir=staged, **kwargs)

    async def _run_docker_compose_command(
        self, command: list[str], check: bool = True, timeout_sec: int | None = None
    ) -> ExecResult:
        retries = 0
        while True:
            try:
                return await super()._run_docker_compose_command(command, check=check, timeout_sec=timeout_sec)
            except RuntimeError as error:
                detail = str(error)
                # Rootless teardown can fail after the task exits because its
                # runtime or network namespace state has already disappeared.
                # Repeating this session's idempotent down completes removal.
                missing_runtime_state = "crun delete --force" in detail or (
                    "netavark" in detail and "No such file or directory" in detail
                )
                if command[:1] != ["down"] or not missing_runtime_state or retries == 2:
                    raise
                retries += 1
                self.logger.warning("Retrying task-container teardown (%s/2): %s", retries, error)
                await asyncio.sleep(retries)

    async def start(self, force_build: bool) -> None:
        await super().start(force_build)
        # Agent installers also run apt-get in the live container, where the
        # same single-UID namespace prevents apt's _apt privilege drop.
        result = await self.exec(
            "if [ -d /etc/apt/apt.conf.d ]; then "
            "printf 'APT::Sandbox::User \"root\";\\n' > /etc/apt/apt.conf.d/99-agentic-vbench-rootless; fi",
            user="root",
        )
        if result.return_code != 0:
            raise RuntimeError(f"Failed to configure rootless APT sandbox: {result.stdout} {result.stderr}")
        if self._opencode_session_headers_path is not None:
            await self.upload_file(
                Path(__file__).with_name("opencode_session_headers.mjs"), self._opencode_session_headers_path
            )

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        # Native installers extract archives with foreign UIDs (Node uses
        # UID 1000). Only container root is mapped by this rootless backend.
        effective_env = self._merge_env(env) or {}
        effective_env.setdefault("TAR_OPTIONS", "--no-same-owner")
        return await super().exec(command, cwd=cwd, env=effective_env, timeout_sec=timeout_sec, user=user)

    async def _container_tar(self, arguments: list[str], archive: BinaryIO, *, upload: bool) -> None:
        """Stream files without Docker cp preserving unmapped host UIDs."""
        for attempt in range(6):
            if upload:
                archive.seek(0)
            else:
                archive.seek(0)
                archive.truncate(0)

            result = await self._run_docker_compose_command(["ps", "-q", "main"])
            container_id = (result.stdout or "").strip()
            if not container_id or len(container_id.splitlines()) != 1:
                raise RuntimeError("Expected one running Harbor main container for file transfer")
            process = await asyncio.create_subprocess_exec(
                "docker",
                "exec",
                "--interactive",
                "--user",
                "0",
                container_id,
                "tar",
                *arguments,
                stdin=archive if upload else asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.DEVNULL if upload else archive,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                _, stderr = await asyncio.wait_for(process.communicate(), timeout=600)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                process.kill()
                await process.wait()
                raise
            if process.returncode == 0:
                return

            detail = stderr.decode(errors="replace")
            transient_api_error = "unable to upgrade to tcp, received 500" in detail
            if not transient_api_error or attempt == 5:
                raise RuntimeError(f"Harbor file transfer failed: {detail}")
            delay = min(2**attempt, 10)
            self.logger.warning(
                "Retrying transient Harbor file-transfer API failure (%s/5) in %ss: %s",
                attempt + 1,
                delay,
                detail.strip(),
            )
            await asyncio.sleep(delay)

    @staticmethod
    def _write_upload_archive(archive: BinaryIO, source: Path, name: str) -> None:
        def mapped_owner(member: tarfile.TarInfo) -> tarfile.TarInfo:
            return member.replace(uid=0, gid=0, uname="root", gname="root", deep=False)

        with tarfile.open(fileobj=archive, mode="w") as output:
            output.add(source, arcname=name, filter=mapped_owner)
        archive.seek(0)

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        target = PurePosixPath(target_path)
        result = await self.exec(f"mkdir -p -- {shlex.quote(str(target.parent))}", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Cannot create file transfer destination: {result.stdout} {result.stderr}")
        with tempfile.TemporaryFile() as archive:
            await asyncio.to_thread(self._write_upload_archive, archive, Path(source_path), target.name)
            await self._container_tar(["-xf", "-", "--no-same-owner", "-C", str(target.parent)], archive, upload=True)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        result = await self.exec(f"mkdir -p -- {shlex.quote(target_dir)}", user="root")
        if result.return_code != 0:
            raise RuntimeError(f"Cannot create directory transfer destination: {result.stdout} {result.stderr}")
        with tempfile.TemporaryFile() as archive:
            await asyncio.to_thread(self._write_upload_archive, archive, Path(source_dir), ".")
            await self._container_tar(["-xf", "-", "--no-same-owner", "-C", target_dir], archive, upload=True)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        source = PurePosixPath(source_path)
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as archive:
            await self._container_tar(["-cf", "-", "-C", str(source.parent), "--", source.name], archive, upload=False)
            archive.seek(0)
            with tempfile.TemporaryDirectory(dir=target.parent) as staging:
                with tarfile.open(fileobj=archive) as incoming:
                    incoming.extractall(staging, filter="data")
                shutil.move(str(Path(staging) / source.name), target)

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile() as archive:
            await self._container_tar(["-cf", "-", "-C", source_dir, "."], archive, upload=False)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as incoming:
                incoming.extractall(target, filter="data")

    async def _chown_to_host_user(self, path: str, recursive: bool = False) -> None:
        # This environment has no host bind mounts. Extraction assigns host
        # ownership; chown inside a single-UID container cannot map host IDs.
        return None

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return super().capabilities.model_copy(update={"mounted": False})
