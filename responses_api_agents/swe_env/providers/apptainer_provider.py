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

"""Apptainer-backed ``SandboxProvider`` for ``.sif`` images.

Implements the ``nemo_gym.sandbox`` provider Protocol using a long-lived
``apptainer instance`` so repository edits persist across exec calls, and a
bind-mounted host scratch directory for transferring files in and out of the
sandbox.
"""

from __future__ import annotations

import asyncio
import glob
import os
import posixpath
import shlex
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from nemo_gym.sandbox import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


_IO_MOUNT = "/sandbox_io"


class ApptainerSandboxProvider:
    """Run sandboxes as ``apptainer instance`` processes from ``.sif`` images."""

    name = "apptainer"

    def __init__(
        self,
        *,
        apptainer_bin: str = "apptainer",
        image_root: str | None = None,
        scratch_root: str | None = None,
        instance_args: list[str] | None = None,
        exec_args: list[str] | None = None,
        **_: Any,
    ) -> None:
        """Configure the Apptainer sandbox provider.

        Args:
            apptainer_bin: Name or path of the ``apptainer`` executable.
            image_root: Optional root directory searched for ``.sif`` images when
                a spec does not point at an existing file.
            scratch_root: Optional parent directory for the per-sandbox scratch
                directory used as the bind-mounted I/O area.
            instance_args: Flags passed to ``apptainer instance start``. Defaults
                to ``--writable-tmpfs --cleanenv --pid --no-mount
                home,tmp,bind-paths`` so the nested harness gets its own PID
                namespace and the host ``$HOME``/``tmp`` plus the image's
                declared bind points stay out of the sandbox, letting the
                prebuilt venv resolve against the explicit ``--bind`` mounts.
            exec_args: Extra flags passed to every ``apptainer exec``.
            **_: Additional keyword arguments are accepted and ignored.
        """
        self._bin = apptainer_bin
        self._image_root = image_root
        self._scratch_root = scratch_root
        self._instance_args = list(
            instance_args
            if instance_args is not None
            else ["--writable-tmpfs", "--cleanenv", "--pid", "--no-mount", "home,tmp,bind-paths"]
        )
        self._exec_args = list(exec_args or [])

    async def _run(self, *args: str, timeout_s: int | float | None = None) -> tuple[int, str, str]:
        """Run the ``apptainer`` CLI with the given arguments and capture output.

        Args:
            *args: Arguments passed to the ``apptainer`` executable.
            timeout_s: Optional timeout in seconds; the process is killed and the
                timeout error re-raised if it is exceeded.

        Returns:
            A tuple of ``(return_code, stdout, stderr)`` with output decoded as
            text using ``errors="replace"``.
        """
        proc = await asyncio.create_subprocess_exec(
            self._bin, *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            proc.kill()
            await proc.wait()
            raise
        rc = proc.returncode if proc.returncode is not None else -1
        return rc, out.decode(errors="replace"), err.decode(errors="replace")

    def _resolve_sif(self, spec: SandboxSpec) -> str:
        """Resolve a ``.sif`` image path from provider options or by globbing.

        Prefers ``provider_options['sif_path']`` (or ``spec.image``) when it
        names an existing file; otherwise globs under the configured image
        roots, restricting matches to ``*.sif`` and also trying a lowercased
        search term.

        Args:
            spec: Sandbox spec carrying the image name and provider options
                (``sif_path``, ``image_glob``, ``image_root``).

        Returns:
            The resolved ``.sif`` path; when several candidates match, the last
            in sorted order is chosen.

        Raises:
            SandboxCreateError: If no matching ``.sif`` file is found.
        """
        sif = spec.provider_options.get("sif_path") or spec.image
        if sif and os.path.isfile(sif):
            return sif
        # Glob under the image roots. Restrict the fuzzy term to ``*.sif`` so an
        # unrelated host file can never be picked up as a container, and add a
        # lowercased search term to match case-folded candidates.
        pattern = spec.provider_options.get("image_glob")
        roots = [r for r in (self._image_root, spec.provider_options.get("image_root")) if r]
        candidates: list[str] = []
        for root in roots:
            if pattern:
                candidates += glob.glob(os.path.join(root, pattern))
            elif sif:
                terms = [sif] if sif == sif.lower() else [sif, sif.lower()]
                for term in terms:
                    candidates += glob.glob(os.path.join(root, f"*{term}*.sif"))
        if not candidates:
            raise SandboxCreateError(f"No .sif found for image={spec.image!r} (roots={roots}, glob={pattern!r})")
        return sorted(candidates)[-1]

    @staticmethod
    def _mount_binds(spec: SandboxSpec) -> list[str]:
        """Translate ``provider_options['mounts']`` into apptainer ``--bind`` args.

        Each mount is ``{"src": host_path, "dst": container_path[, "ro": bool]}``.
        A read-only flag appends ``:ro`` to the bind.

        Args:
            spec: Sandbox spec whose ``provider_options['mounts']`` lists the
                bind mounts to apply.

        Returns:
            A flat list of ``--bind`` arguments suitable for the apptainer CLI.
        """
        binds: list[str] = []
        for mount in spec.provider_options.get("mounts", []) or []:
            src = mount.get("src")
            dst = mount.get("dst")
            if not src or not dst:
                continue
            # Skip a self-bind whose source does not exist on the host: when no
            # real host dataset path is provided the caller falls back to
            # ``src == dst ==`` the in-container dataset path, which is not a host
            # path. Binding a missing host src would make apptainer create an
            # empty dir there and shadow the real dataset. Real host binds
            # (src != dst, or an existing self-bind) are unaffected.
            if src == dst and not os.path.exists(src):
                continue
            bind = f"{src}:{dst}"
            if mount.get("ro"):
                bind += ":ro"
            binds += ["--bind", bind]
        return binds

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Start an apptainer instance and return a handle to it.

        Resolves the ``.sif`` image, creates a host scratch directory bound at
        the in-container I/O mount, applies the configured instance args, extra
        binds, and environment, and launches a named instance.

        Args:
            spec: Sandbox spec describing the image, env, workdir, provider
                options, and readiness timeout.

        Returns:
            A ``SandboxHandle`` whose ``sandbox_id`` is the instance name; its
            ``raw`` records the resolved sif, scratch dir, and workdir.

        Raises:
            SandboxCreateError: If starting the instance times out or fails; the
                scratch directory is removed in that case.
        """
        sif = self._resolve_sif(spec)
        scratch = tempfile.mkdtemp(prefix="swe-apptainer-io-", dir=self._scratch_root)
        instance_name = f"swe-{(spec.metadata.get('instance_id') or 'task')[:24]}-{uuid.uuid4().hex[:8]}"
        args = ["instance", "start", *self._instance_args, "--bind", f"{scratch}:{_IO_MOUNT}"]
        args += self._mount_binds(spec)
        for key, value in (spec.env or {}).items():
            args += ["--env", f"{key}={value}"]
        args += spec.provider_options.get("instance_args", [])
        args += [sif, instance_name]
        try:
            rc, out, err = await self._run(*args, timeout_s=spec.ready_timeout_s or 600)
        except (asyncio.TimeoutError, TimeoutError) as exc:
            shutil.rmtree(scratch, ignore_errors=True)
            raise SandboxCreateError(f"apptainer instance start timed out for {sif!r}") from exc
        if rc != 0:
            shutil.rmtree(scratch, ignore_errors=True)
            raise SandboxCreateError(f"apptainer instance start failed (rc={rc}): {err.strip() or out.strip()}")
        return SandboxHandle(
            sandbox_id=instance_name,
            provider_name=self.name,
            raw={"sif": sif, "scratch": scratch, "workdir": spec.workdir},
        )

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
        """Run a shell command inside the apptainer instance.

        Args:
            handle: Handle identifying the target instance.
            command: Shell command executed via ``bash -c``.
            cwd: Working directory for the command; falls back to the workdir
                recorded at create time.
            env: Extra environment variables for the command.
            timeout_s: Optional timeout in seconds; on expiry a result with
                return code 124 and ``error_type="timeout"`` is returned.
            user: Accepted for interface compatibility; not applied by this
                provider.

        Returns:
            A ``SandboxExecResult`` with stdout, stderr, and the return code, or
            a timeout result if the command exceeds ``timeout_s``.
        """
        args = ["exec", *self._exec_args]
        workdir = cwd or handle.raw.get("workdir")
        if workdir:
            args += ["--pwd", workdir]
        for key, value in (env or {}).items():
            args += ["--env", f"{key}={value}"]
        args += [f"instance://{handle.sandbox_id}", "bash", "-c", command]
        try:
            rc, out, err = await self._run(*args, timeout_s=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            return SandboxExecResult(
                stdout=None, stderr=f"command timed out after {timeout_s}s", return_code=124, error_type="timeout"
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=rc)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Copy a host file into the instance via the scratch I/O mount.

        The file is staged into the host scratch directory, then copied to the
        target path inside the instance, creating parent dirs as needed.

        Args:
            handle: Handle identifying the target instance.
            source_path: Path to the file on the host.
            target_path: Destination path inside the instance.

        Raises:
            RuntimeError: If the in-container copy fails.
        """
        scratch = handle.raw["scratch"]
        base = posixpath.basename(target_path)
        shutil.copy(str(source_path), os.path.join(scratch, base))
        parent = posixpath.dirname(target_path)
        mkdir = f"mkdir -p {shlex.quote(parent)} && " if parent else ""
        result = await self.exec(handle, f"{mkdir}cp {_IO_MOUNT}/{shlex.quote(base)} {shlex.quote(target_path)}")
        if result.return_code != 0:
            raise RuntimeError(f"apptainer upload copy failed: {result.stderr}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Copy a file out of the instance to the host via the scratch I/O mount.

        The file is copied inside the instance to the scratch I/O mount, then
        copied from the host scratch directory to the target path.

        Args:
            handle: Handle identifying the source instance.
            source_path: Path to the file inside the instance.
            target_path: Destination path on the host; parent dirs are created.

        Raises:
            RuntimeError: If the in-container copy fails.
        """
        scratch = handle.raw["scratch"]
        base = posixpath.basename(source_path)
        result = await self.exec(handle, f"cp {shlex.quote(source_path)} {_IO_MOUNT}/{shlex.quote(base)}")
        if result.return_code != 0:
            raise RuntimeError(f"apptainer download copy failed: {result.stderr}")
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(os.path.join(scratch, base), str(target))

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Report whether the instance is running.

        Args:
            handle: Handle identifying the instance to query.

        Returns:
            ``RUNNING`` if the instance name appears in ``apptainer instance
            list`` output, ``STOPPED`` if not, or ``UNKNOWN`` if the command
            fails.
        """
        rc, out, _ = await self._run("instance", "list", handle.sandbox_id)
        if rc != 0:
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if handle.sandbox_id in out else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        """Stop the instance and remove its scratch directory.

        Args:
            handle: Handle identifying the instance to stop.
        """
        try:
            await self._run("instance", "stop", handle.sandbox_id)
        finally:
            shutil.rmtree(handle.raw.get("scratch", ""), ignore_errors=True)

    async def aclose(self) -> None:
        """Release provider-level resources; this provider holds none."""
        return None
