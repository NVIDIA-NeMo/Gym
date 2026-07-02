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

"""Local Docker-backed ``SandboxProvider`` implementation.

Implements the ``nemo_gym.sandbox`` provider Protocol via the ``docker`` CLI so
SWE environments can be provisioned and graded on any machine with Docker
installed, making end-to-end SWE-bench verification runnable on a single
workstation.
"""

from __future__ import annotations

import asyncio
import posixpath
import shlex
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


class DockerSandboxProvider:
    """Run sandboxes as long-lived Docker containers via the ``docker`` CLI."""

    name = "docker"

    def __init__(
        self,
        *,
        docker_bin: str = "docker",
        default_user: str | int | None = None,
        network: str | None = None,
        run_args: list[str] | None = None,
        keep_alive_command: str = "sleep infinity",
        concurrency: int = 32,
        default_exec_timeout_s: int | float | None = 3600,
        **_: Any,
    ) -> None:
        """Configure the Docker sandbox provider.

        Args:
            docker_bin: Name or path of the ``docker`` executable to invoke.
            default_user: Default user (name or UID) to run ``exec`` commands as
                when no per-call user is given; None leaves the image default.
            network: Docker network to attach containers to; None uses the
                Docker default.
            run_args: Extra arguments appended to every ``docker run``
                invocation.
            keep_alive_command: Command run as the container's entrypoint to keep
                it alive for subsequent ``exec`` calls.
            concurrency: Maximum number of concurrent ``docker`` CLI subprocesses,
                bounded by a shared semaphore (matches the apptainer provider).
            default_exec_timeout_s: Default per-``exec`` timeout (seconds) applied when a
                caller passes none, so a hung in-container command cannot block a rollout
                forever (e.g. the git-diff extraction in self_drive). None = no default.
            **_: Additional keyword arguments are accepted and ignored.

        Raises:
            ValueError: If ``concurrency`` is less than 1.
        """
        if concurrency < 1:
            raise ValueError("concurrency must be >= 1")
        self._bin = docker_bin
        self._default_user = default_user
        self._network = network
        self._run_args = list(run_args or [])
        self._keep_alive = keep_alive_command
        self._semaphore = asyncio.Semaphore(concurrency)
        self._default_exec_timeout_s = default_exec_timeout_s

    async def _run(self, *args: str, timeout_s: int | float | None = None) -> tuple[int, str, str]:
        """Run the ``docker`` CLI with the given arguments and capture output.

        Concurrency is bounded by the provider's shared semaphore so a busy SWE hot
        path (one sandbox per rollout, many ``exec`` each) cannot spawn unbounded
        ``docker`` subprocesses.

        Args:
            *args: Arguments passed to the ``docker`` executable.
            timeout_s: Optional timeout in seconds; the process is killed and the
                timeout error re-raised if it is exceeded.

        Returns:
            A tuple of ``(return_code, stdout, stderr)`` with output decoded as
            text using ``errors="replace"``.
        """
        async with self._semaphore:
            try:
                proc = await asyncio.create_subprocess_exec(
                    self._bin,
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )
            except FileNotFoundError as exc:
                # Surface a clear, actionable error instead of a bare FileNotFoundError deep in
                # create()/exec() (parity with apptainer's _require_apptainer up-front check).
                raise SandboxCreateError(
                    f"docker executable {self._bin!r} not found on PATH — install Docker or set docker_bin"
                ) from exc
            try:
                out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
            except (asyncio.TimeoutError, TimeoutError):
                proc.kill()
                await proc.wait()
                raise
            return (
                proc.returncode if proc.returncode is not None else -1,
                out.decode(errors="replace"),
                err.decode(errors="replace"),
            )

    @staticmethod
    def _resources(spec: SandboxSpec) -> SandboxResources:
        """Coerce a spec's resource request into a ``SandboxResources``.

        Args:
            spec: Sandbox spec whose ``resources`` field is a
                ``SandboxResources`` or a mapping.

        Returns:
            The spec's ``SandboxResources`` if already one, otherwise a
            ``SandboxResources`` built from the mapping (or empty defaults).
        """
        if isinstance(spec.resources, SandboxResources):
            return spec.resources
        return SandboxResources.from_mapping(spec.resources if isinstance(spec.resources, Mapping) else {})

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Start a detached container and return a handle to it.

        Applies resource limits, network, working directory, environment, and
        extra run args from the spec, then launches the image running the
        keep-alive command so the container persists for later ``exec`` calls.

        Args:
            spec: Sandbox spec describing the image, resources, workdir, env, and
                readiness timeout.

        Returns:
            A ``SandboxHandle`` whose ``sandbox_id`` is the container id.

        Raises:
            SandboxCreateError: If no image is given, ``docker run`` times out or
                fails, or no container id is returned.
        """
        if not spec.image:
            raise SandboxCreateError("DockerSandboxProvider requires spec.image")
        # Pre-assign a unique name so a container the daemon may have started can still be reaped
        # if the CLI client dies (e.g. on timeout) before we capture its id (mirrors apptainer's
        # uuid-named instances).
        name = f"nemo-gym-{uuid.uuid4().hex}"
        args = ["run", "-d", "--init", "--name", name]
        if self._network:
            args += ["--network", self._network]
        res = self._resources(spec)
        if res.memory_mib:
            args.append(f"--memory={int(res.memory_mib)}m")
        if res.cpu:
            args.append(f"--cpus={res.cpu}")
        if res.gpu:
            args.append("--gpus=all")
        if spec.workdir:
            args += ["-w", spec.workdir]
        for key, value in (spec.env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += self._run_args
        args += [spec.image, "bash", "-c", self._keep_alive]
        try:
            rc, out, err = await self._run(*args, timeout_s=spec.ready_timeout_s or 600)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            await self._reap_orphan(name)
            raise SandboxCreateError(f"docker run timed out for image {spec.image!r}") from exc
        if rc != 0:
            await self._reap_orphan(name)
            raise SandboxCreateError(f"docker run failed (rc={rc}) for {spec.image!r}: {err.strip() or out.strip()}")
        lines = out.strip().splitlines()
        container_id = lines[-1].strip() if lines else ""
        if not container_id:
            await self._reap_orphan(name)
            raise SandboxCreateError("docker run did not return a container id")
        return SandboxHandle(
            sandbox_id=container_id,
            provider_name=self.name,
            raw={"image": spec.image, "workdir": spec.workdir},
        )

    async def _reap_orphan(self, name: str) -> None:
        """Best-effort force-remove a container by its pre-assigned name.

        Used to clean up a ``docker run`` that may have started a container on the daemon even
        though the CLI client failed (timeout / non-zero rc / no id returned) before a handle was
        captured. Swallows all errors and bounds itself with a short timeout — a missing or
        already-gone container is fine.

        Args:
            name: The pre-assigned ``--name`` of the container to remove.
        """
        try:
            await self._run("rm", "-f", name, timeout_s=30)
        except Exception:
            pass

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Run a shell command inside the container.

        Args:
            handle: Handle identifying the target container.
            command: Shell command executed via ``bash -c``.
            cwd: Working directory for the command; falls back to the workdir
                recorded at create time.
            env: Extra environment variables for the command.
            timeout_s: Optional timeout in seconds; falls back to the provider's
                ``default_exec_timeout_s`` when None. On expiry a result with return
                code 124 and ``error_type="timeout"`` is returned.
            user: User (name or UID) to run as; falls back to the provider's
                default user.

        Returns:
            A ``SandboxExecResult`` with stdout, stderr, return code, and an
            ``error_type`` of ``"sandbox"`` for a docker-daemon failure (rc 125 with
            no stdout), ``"timeout"`` on timeout, or None otherwise.
        """
        args = ["exec"]
        workdir = cwd or handle.raw.get("workdir")
        if workdir:
            args += ["-w", workdir]
        eff_user = user if user is not None else self._default_user
        if eff_user is not None:
            args += ["-u", str(eff_user)]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        args += [handle.sandbox_id, "bash", "-c", command]
        eff_timeout = timeout_s if timeout_s is not None else self._default_exec_timeout_s
        try:
            rc, out, err = await self._run(*args, timeout_s=eff_timeout)
        except (asyncio.TimeoutError, TimeoutError):
            # Only the local `docker exec` client is killed here; the in-container process is
            # reaped when the sandbox is closed (`docker rm -f`), which acquire_sandbox always does.
            return SandboxExecResult(
                stdout=None,
                stderr=f"command timed out after {eff_timeout}s",
                return_code=124,
                error_type="timeout",
            )
        # rc 125 is a docker-daemon-level failure (container gone / daemon error). 126 (not
        # executable) and 127 (command not found) are legitimate *user*-command exit codes when
        # run via `bash -c`, so only rc 125 with no stdout is classified as an infra failure.
        error_type = "sandbox" if rc == 125 and not out else None
        return SandboxExecResult(stdout=out, stderr=err, return_code=rc, error_type=error_type)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Copy a host file into the container, creating parent dirs as needed.

        Args:
            handle: Handle identifying the target container.
            source_path: Path to the file on the host.
            target_path: Destination path inside the container.

        Raises:
            RuntimeError: If the ``docker cp`` upload fails.
        """
        parent = posixpath.dirname(target_path)
        if parent:
            await self.exec(handle, f"mkdir -p {shlex.quote(parent)}")
        rc, out, err = await self._run("cp", str(source_path), f"{handle.sandbox_id}:{target_path}", timeout_s=300)
        if rc != 0:
            raise RuntimeError(f"docker cp upload failed: {err.strip() or out.strip()}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Copy a file out of the container to the host.

        Args:
            handle: Handle identifying the source container.
            source_path: Path to the file inside the container.
            target_path: Destination path on the host; parent dirs are created.

        Raises:
            RuntimeError: If the ``docker cp`` download fails.
        """
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        rc, out, err = await self._run("cp", f"{handle.sandbox_id}:{source_path}", str(target), timeout_s=300)
        if rc != 0:
            raise RuntimeError(f"docker cp download failed: {err.strip() or out.strip()}")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Report whether the container is running.

        Args:
            handle: Handle identifying the container to inspect.

        Returns:
            ``RUNNING`` or ``STOPPED`` based on the container's running state,
            or ``UNKNOWN`` if the inspect command fails.
        """
        rc, out, _ = await self._run("inspect", "-f", "{{.State.Running}}", handle.sandbox_id)
        if rc != 0:
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if out.strip() == "true" else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        """Force-remove the container.

        Args:
            handle: Handle identifying the container to remove.
        """
        # Best-effort + bounded: teardown runs in acquire_sandbox's finally, so a wedged daemon
        # must not hang (or crash) it after the result is already in hand.
        try:
            await self._run("rm", "-f", handle.sandbox_id, timeout_s=120)
        except Exception:
            pass

    async def aclose(self) -> None:
        """Release provider-level resources; this provider holds none."""
        return None
