# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Harbor environment backed by a private rootless Podman store."""

import asyncio
import hashlib
import os
import shlex
import tempfile
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities


class PodmanEnvironment(BaseEnvironment):
    """Execute the benchmark's original Dockerfile without a Docker daemon."""

    def __init__(self, *args, runtime_root: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._runtime = Path(runtime_root).resolve() / hashlib.sha256(self.session_id.encode()).hexdigest()[:16]
        self._runtime.mkdir(parents=True, exist_ok=False)
        self._runroot = Path(tempfile.mkdtemp(prefix="avb-podman-", dir="/tmp"))
        self._name = "avb-" + self._runtime.name
        self._image = "localhost/" + self._name
        self._started = False
        self._process_env = dict(os.environ)
        storage = self._runtime / "storage.conf"
        storage.write_text(
            '[storage]\ndriver="vfs"\n'
            f'runroot="{self._runroot}"\ngraphroot="{self._runtime}/graph"\n'
            '[storage.options.vfs]\nignore_chown_errors="true"\n'
        )
        self._process_env["CONTAINERS_STORAGE_CONF"] = str(storage)
        self._process_env["XDG_RUNTIME_DIR"] = str(self._runroot)
        self._runtime.chmod(0o700)

    @staticmethod
    def type() -> str:
        return "agentic-vbench-podman"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=True, disable_internet=True)

    def _validate_definition(self) -> None:
        if (self.environment_dir / "docker-compose.yaml").exists():
            raise ValueError("This environment requires a single-container Dockerfile")
        if not (self.environment_dir / "Dockerfile").is_file():
            raise FileNotFoundError(self.environment_dir / "Dockerfile")

    async def _podman(self, args: list[str], *, check: bool = True, timeout: int | float | None = None) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            "podman",
            "--cgroup-manager=cgroupfs",
            "--root",
            str(self._runtime / "graph"),
            "--runroot",
            str(self._runroot),
            "--storage-driver=vfs",
            "--storage-opt=vfs.ignore_chown_errors=true",
            *args,
            env=self._process_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.wait()
            raise
        result = ExecResult(
            stdout=stdout.decode(errors="replace"),
            stderr=stderr.decode(errors="replace"),
            return_code=process.returncode,
        )
        if check and result.return_code:
            # Arguments may contain verifier credentials; never include them in the error.
            raise RuntimeError(f"Podman {args[0]} failed ({result.return_code}): {result.stderr}")
        return result

    async def start(self, force_build: bool) -> None:
        # Cluster rootless namespaces map only the invoking user. apt's default
        # _apt privilege drop therefore fails; use mapped container root while
        # building. The build-only mount leaves the task Dockerfile unchanged.
        apt_config = self._runtime / "apt-rootless.conf"
        apt_config.write_text('APT::Sandbox::User "root";\n')
        await self._podman(
            [
                "build",
                "--format=docker",
                "--isolation=rootless",
                "--volume",
                f"{apt_config}:/etc/apt/apt.conf.d/99-agentic-vbench-rootless:ro",
                "-t",
                self._image,
                str(self.environment_dir),
            ],
            timeout=self.task_env_config.build_timeout_sec,
        )
        command = [
            "run",
            # Rootless Podman's parent namespace maps container root to this user.
            # Override the cluster-wide keep-id default, which requires subuids.
            "--userns=host",
            "--pid=private",
            "--init",
            "--init-path=/usr/bin/tini",
            "--detach",
            "--name",
            self._name,
            "--cpus",
            str(self.task_env_config.cpus),
            "--memory",
            f"{self.task_env_config.memory_mb}m",
            "--network",
            "slirp4netns:allow_host_loopback=true" if self.task_env_config.allow_internet else "none",
        ]
        for host, target in [
            (self.trial_paths.agent_dir, self.env_paths.agent_dir),
            (self.trial_paths.verifier_dir, self.env_paths.verifier_dir),
            (self.trial_paths.artifacts_dir, self.env_paths.artifacts_dir),
        ]:
            host.mkdir(parents=True, exist_ok=True)
            command.extend(["--volume", f"{host.resolve()}:{target}"])
        command.extend(["--entrypoint", "sh", self._image, "-c", "exec sleep infinity"])
        await self._podman(command)
        self._started = True

    async def stop(self, delete: bool) -> None:
        if self._started:
            await self._podman(["rm", "--force", self._name])
            self._started = False
        if delete:
            result = await self._podman(["image", "exists", self._image], check=False)
            if result.return_code == 0:
                await self._podman(["rmi", self._image])
            elif result.return_code != 1:
                raise RuntimeError(f"Cannot inspect Podman image: {result.stderr}")

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._podman(["cp", str(source_path), f"{self._name}:{target_path}"])

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        result = await self.exec(f"mkdir -p {shlex.quote(target_dir)}", user="root")
        if result.return_code:
            raise RuntimeError(f"Cannot create upload directory: {result.stderr}")
        await self._podman(["cp", str(source_dir) + "/.", f"{self._name}:{target_dir}"])

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        await self._podman(["cp", f"{self._name}:{source_path}", str(target_path)])

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        Path(target_dir).mkdir(parents=True, exist_ok=True)
        await self._podman(["cp", f"{self._name}:{source_dir}/.", str(target_dir)])

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        args = ["exec"]
        effective_cwd = cwd or self.task_env_config.workdir
        if effective_cwd:
            args.extend(["--workdir", effective_cwd])
        effective_user = self._resolve_user(user)
        if effective_user is not None:
            args.extend(["--user", str(effective_user)])
        for key, value in (self._merge_env(env) or {}).items():
            args.extend(["--env", f"{key}={value}"])
        args.extend([self._name, "bash", "-lc", command])
        return await self._podman(args, check=False, timeout=timeout_sec)
