# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned EnterpriseOps service artifacts for the managed Apptainer runtime."""

import asyncio
import contextlib
import fcntl
import hashlib
import os
import platform
import shutil
import socket
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, TextIO
from urllib.parse import urlparse
from zipfile import ZipFile

from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxSpec


@dataclass(frozen=True)
class EnterpriseOpsService:
    domain: str
    gym_name: str
    port: int
    image: str
    app_target: str = "main:app"
    environment: tuple[tuple[str, str], ...] = ()


def _image(domain: str, digest: str) -> str:
    return f"shivakrishnareddyma225/enterpriseops-gym-mcp-{domain}@{digest}"


SERVICES = {
    "csm": EnterpriseOpsService(
        "csm",
        "sn-csm-server",
        8001,
        _image("csm", "sha256:eaa456ac9aa85728426e7d3813a0bbca0949d6a8695be30e26f03894e6e6b189"),
        environment=(("API_BASE_URL", "http://127.0.0.1:8001"),),
    ),
    "teams": EnterpriseOpsService(
        "teams",
        "gym-teams-mcp",
        8002,
        _image("teams", "sha256:602655e46f6501885540c36dc9b12114cb173c75063d7f25c17ed0652695fa78"),
        environment=(("API_PORT", "8002"),),
    ),
    "calendar": EnterpriseOpsService(
        "calendar",
        "gym-calendar",
        8003,
        _image("calendar", "sha256:994c5421a6dd065861bc7f813a177f6d408875e9df60fe8d012959bc4510da02"),
        environment=(("API_PORT", "8003"),),
    ),
    "email": EnterpriseOpsService(
        "email",
        "gym-email-mcp",
        8004,
        _image("email", "sha256:69c2081fe4ab0962b86233f9fb52b307b8ad0019f6746ba64ce75851036201cd"),
        environment=(("API_PORT", "8004"),),
    ),
    "itsm": EnterpriseOpsService(
        "itsm",
        "gym-itsm-mcp",
        8006,
        _image("itsm", "sha256:a234ae3fb7cee196ba25e6b9957969dea829919b6e8271dddae128f065aaf39f"),
        environment=(("ITSM_API_BASE_URL", "http://127.0.0.1:8006"),),
    ),
    "hr": EnterpriseOpsService(
        "hr",
        "sn-hr-internal",
        8008,
        _image("hr", "sha256:1ea1c1d64d4be35e8062e56f00b8318e9e6c09289cfa56bcfd0595bfa59ac64d"),
        environment=(("HR_API_BASE_URL", "http://127.0.0.1:8008"),),
    ),
    "drive": EnterpriseOpsService(
        "drive",
        "gym-google-drive-mcp",
        8009,
        _image("drive", "sha256:3475962fcf6da7675e194dbf138de01fa3e96134a302ad47316e4111a5e63f32"),
        app_target="app.main:app",
        environment=(
            ("FASTAPI_BASE_URL", "http://127.0.0.1:8009"),
            ("MCP_SERVER_HOST", "127.0.0.1"),
            ("MCP_SERVER_PORT", "8009"),
        ),
    ),
}

ENTERPRISEOPS_REPOSITORY = "https://github.com/ServiceNow/EnterpriseOps-Gym.git"
ENTERPRISEOPS_REVISION = "271f2c357f763376997dfd16807fcde2474ae41b"  # pragma: allowlist secret
DATABASE_ARCHIVE_SHA256 = (
    # pragma: allowlist nextline secret
    "d947543d4fba1aabc4aade73d3df955114187b7a94da7ac825c4c31169ddab47"
)
ARM64_MACHINE_NAMES = {"aarch64", "arm64"}


def is_arm64_host() -> bool:
    return platform.machine().lower() in ARM64_MACHINE_NAMES


class EnterpriseOpsAssets:
    """Materialize digest-pinned EnterpriseOps images as reusable SIF files."""

    repository = ENTERPRISEOPS_REPOSITORY
    revision = ENTERPRISEOPS_REVISION
    database_archive_sha256: str | None = DATABASE_ARCHIVE_SHA256

    def __init__(self, cache_dir: Path | str, native_sif_dir: Path | str | None = None) -> None:
        self.cache_dir = Path(cache_dir)
        self.native_sif_dir = Path(native_sif_dir) if native_sif_dir is not None else None

    def sif_path(self, service: EnterpriseOpsService) -> Path:
        if is_arm64_host() and self.native_sif_dir is not None:
            return self.native_sif_dir / f"{service.domain}-arm64.sif"
        if is_arm64_host():
            return self.cache_dir / "images" / f"{service.domain}-arm64.sif"
        digest = service.image.rsplit("@", 1)[1].removeprefix("sha256:")
        return self.cache_dir / "images" / f"{service.domain}-{digest}.sif"

    def ensure_native_arm64_sifs(self) -> None:
        """Fail before downloads when a native host lacks its prebuilt service images."""
        if not is_arm64_host():
            return
        missing = [self.sif_path(service) for service in SERVICES.values() if not self.sif_path(service).is_file()]
        if not missing:
            return
        cache_dir = self.native_sif_dir or self.cache_dir / "images"
        missing_paths = "\n".join(f"  - {path}" for path in missing)
        raise RuntimeError(
            "missing native ARM64 EnterpriseOps SIFs before startup. "
            f"Expected cache directory: {cache_dir}\n{missing_paths}\n"
            "Build them with: python -m resources_servers.enterpriseops_gym.arm64_images "
            f"--all --output-dir {cache_dir}"
        )

    @property
    def source_root(self) -> Path:
        return self.cache_dir / "source" / self.revision

    async def ensure_seed_root(self) -> Path:
        return await asyncio.to_thread(self._ensure_seed_root_sync)

    def _ensure_seed_root_sync(self) -> Path:
        source_root = self.source_root
        with self._cache_lock(f"source-{self.revision}"):
            archive = source_root / "gym_dbs.zip"
            if not self._archive_is_valid(archive):
                temporary = source_root.with_name(f".{source_root.name}.{uuid.uuid4().hex}.tmp")
                shutil.rmtree(temporary, ignore_errors=True)
                try:
                    temporary.parent.mkdir(parents=True, exist_ok=True)
                    self._checkout_source(temporary)
                    temporary_archive = temporary / "gym_dbs.zip"
                    if not self._archive_is_valid(temporary_archive):
                        raise RuntimeError("pinned EnterpriseOps checkout does not contain the expected gym_dbs.zip")
                    if source_root.exists():
                        shutil.rmtree(source_root)
                    source_root.parent.mkdir(parents=True, exist_ok=True)
                    os.replace(temporary, source_root)
                finally:
                    shutil.rmtree(temporary, ignore_errors=True)
                archive = source_root / "gym_dbs.zip"
            self._extract_archive(archive, source_root)
        return source_root

    @contextlib.contextmanager
    def _cache_lock(self, name: str) -> Iterator[TextIO]:
        lock_path = self.cache_dir / ".locks" / f"{name}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock_file:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield lock_file
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _archive_is_valid(self, archive: Path) -> bool:
        if not archive.is_file():
            return False
        if self.database_archive_sha256 is None:
            return True
        return hashlib.sha256(archive.read_bytes()).hexdigest() == self.database_archive_sha256

    def _checkout_source(self, target: Path) -> None:
        subprocess.run(["git", "clone", self.repository, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "--detach", self.revision], check=True)

    def _extract_archive(self, archive: Path, destination: Path) -> None:
        destination = destination.resolve()
        with ZipFile(archive) as zip_file:
            for member in zip_file.infolist():
                if destination not in (destination / member.filename).resolve().parents:
                    raise RuntimeError(f"unsafe path in EnterpriseOps database archive: {member.filename}")
            zip_file.extractall(destination)

    async def ensure_sif(self, service: EnterpriseOpsService) -> Path:
        target = self.sif_path(service)
        if target.is_file():
            return target
        if is_arm64_host():
            raise RuntimeError(f"native ARM64 EnterpriseOps SIF is missing: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        await self._pull_sif(service, target)
        return target

    async def _pull_sif(self, service: EnterpriseOpsService, target: Path) -> None:
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            process = await asyncio.create_subprocess_exec(
                "apptainer",
                "pull",
                str(temporary),
                f"docker://{service.image}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _stdout, stderr = await process.communicate()
            if process.returncode != 0:
                raise RuntimeError(
                    f"failed to pull EnterpriseOps image {service.image}: {stderr.decode(errors='replace')}"
                )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)


class EnterpriseOpsServiceRuntime:
    """Own the fixed EnterpriseOps service set for one resources-server process."""

    def __init__(
        self,
        assets: EnterpriseOpsAssets,
        sandbox_factory: Callable[[SandboxSpec], Any] | None = None,
        readiness_probe: Callable[[str], Any] | None = None,
        readiness_timeout_seconds: float = 60.0,
    ) -> None:
        self.assets = assets
        self.sandbox_factory = sandbox_factory or self._create_sandbox
        self.readiness_probe = readiness_probe or self._wait_for_endpoint
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self.seed_root: Path | None = None
        self.urls: dict[str, str] = {}
        self.sandboxes: list[Any] = []
        self._port_locks: list[TextIO] = []

    @staticmethod
    def _create_sandbox(spec: SandboxSpec) -> AsyncSandbox:
        return AsyncSandbox(
            {
                "apptainer": {
                    "create": {
                        "extra_start_args": [
                            "--writable-tmpfs",
                            "--contain",
                            "--cleanenv",
                            "--no-home",
                            "--no-mount",
                            "hostfs,bind-paths",
                        ]
                    }
                }
            },
            spec,
        )

    async def _wait_for_endpoint(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.hostname is None or parsed.port is None:
            raise ValueError(f"EnterpriseOps service endpoint must include a host and port: {url}")

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.readiness_timeout_seconds
        while True:
            try:
                _reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(parsed.hostname, parsed.port), timeout=3.0
                )
                writer.close()
                await writer.wait_closed()
                return
            except (OSError, TimeoutError):
                if loop.time() >= deadline:
                    raise RuntimeError(f"EnterpriseOps service did not become reachable: {url}")
                await asyncio.sleep(0.5)

    async def start(self) -> None:
        if self.sandboxes:
            return
        self._reserve_service_ports()
        try:
            self.assets.ensure_native_arm64_sifs()
            self.seed_root = await self.assets.ensure_seed_root()
            for service in SERVICES.values():
                sif_path = await self.assets.ensure_sif(service)
                sandbox = self.sandbox_factory(
                    SandboxSpec(
                        image=str(sif_path),
                        ports=(service.port,),
                    )
                )
                await sandbox.start()
                self.sandboxes.append(sandbox)
                environment = " ".join(f"{name}={value}" for name, value in service.environment)
                start_result = await sandbox.exec(
                    f"nohup env {environment} python -m uvicorn {service.app_target} --host 127.0.0.1 --port {service.port} "
                    f">/sandbox/{service.domain}.log 2>&1 &",
                    cwd="/app",
                    timeout_s=30.0,
                )
                launch_timed_out = getattr(start_result, "error_type", None) == "timeout"
                if start_result.return_code != 0 and not launch_timed_out:
                    raise RuntimeError(
                        f"failed to start EnterpriseOps service {service.domain}: {start_result.stderr}"
                    )
                endpoint = await sandbox.endpoint(service.port)
                self.urls[service.gym_name] = endpoint.endpoint.rstrip("/")
                await self.readiness_probe(self.urls[service.gym_name])
        except BaseException:
            with contextlib.suppress(Exception):
                await self.stop()
            raise

    async def stop(self) -> None:
        sandboxes, self.sandboxes = self.sandboxes, []
        self.urls = {}
        errors = []
        try:
            stopping = asyncio.gather(
                *(sandbox.stop() for sandbox in reversed(sandboxes)),
                return_exceptions=True,
            )
            try:
                results = await asyncio.shield(stopping)
            except asyncio.CancelledError:
                await stopping
                raise
            errors = [result for result in results if isinstance(result, Exception)]
        finally:
            self._release_service_ports()
        if errors:
            failures = "; ".join(str(error) for error in errors)
            raise RuntimeError(f"failed to stop EnterpriseOps services: {failures}")

    def _reserve_service_ports(self) -> None:
        try:
            for service in SERVICES.values():
                lock_path = self.assets.cache_dir / ".locks" / f"service-{service.port}.lock"
                lock_path.parent.mkdir(parents=True, exist_ok=True)
                lock_file = lock_path.open("a+")
                try:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    lock_file.close()
                    raise RuntimeError(f"EnterpriseOps service port {service.port} is already reserved") from error
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                        probe.bind(("127.0.0.1", service.port))
                except OSError as error:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                    lock_file.close()
                    raise RuntimeError(f"EnterpriseOps service port {service.port} is already in use") from error
                self._port_locks.append(lock_file)
        except Exception:
            self._release_service_ports()
            raise

    def _release_service_ports(self) -> None:
        port_locks, self._port_locks = self._port_locks, []
        for lock_file in reversed(port_locks):
            with contextlib.suppress(OSError):
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            with contextlib.suppress(OSError):
                lock_file.close()
