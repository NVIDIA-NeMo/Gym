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

"""Enroot provider implementation.

Enroot (https://github.com/NVIDIA/enroot) is an unprivileged container runtime.
Unlike Apptainer it has no persistent daemon/instance concept, so this provider
emulates a long-lived sandbox by launching a detached ``enroot start`` running a
sleeping init and then re-entering it with ``enroot exec <pid>``. See README.md
for the enroot-specific design notes (non-daemonizing ``start``, PID-based
liveness, ``#``-registry URIs, pinned ``ENROOT_*`` paths).
"""

import asyncio
import contextlib
import hashlib
import logging
import os
import posixpath
import shlex
import shutil
import signal
import tempfile
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.utils import coerce_config as _coerce_config
from nemo_gym.sandbox.providers.utils import path_under_mount as _path_under_mount


LOGGER = logging.getLogger(__name__)

DEFAULT_MOUNT_POINT = "/sandbox"
CONTAINER_NAME_PREFIX = "nemo-gym-"
# Portable init: keep the container alive without relying on `sleep infinity`,
# which busybox `sleep` rejects. A shell loop works on any image with `sh`.
DEFAULT_INIT_COMMAND = "while true; do sleep 86400; done"
READY_PROBE_COMMAND = (
    f"printf enroot-sandbox-ready > {DEFAULT_MOUNT_POINT}/.nemo-gym-ready && printf enroot-sandbox-ready"
)
READY_PROBE_EXPECTED = "enroot-sandbox-ready"
SANDBOX_RUNTIME_RETURN_CODE = 125
# Best-effort stderr markers indicating enroot itself (not the user's command)
# failed to run the command. Enroot prints its own errors prefixed with "[ERROR]".
ENROOT_RUNTIME_ERROR_MARKERS = (
    "[error]",
    "no such process",
    "does not exist",
    "no such file or directory: /proc",
    "failed to",
    "nsenter",
)
ENROOT_MISSING_CONTAINER_MARKERS = ("does not exist", "no such", "not found")
# Docker Hub canonical hostnames. These are dropped from image references (rather
# than passed as enroot's REGISTRY fragment) so enroot uses its configured Hub
# default — the real Hub registry API is registry-1.docker.io, not "docker.io".
DOCKER_HUB_HOSTS = frozenset({"docker.io", "index.docker.io", "registry-1.docker.io"})


class EnrootCreateError(SandboxCreateError):
    """Raised when Enroot cannot create a sandbox."""


class EnrootCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly-created sandbox cannot execute a probe command."""


def _read_proc_stat_ppid(pid: int) -> int | None:
    """Return the parent PID from /proc/<pid>/stat, or None if unreadable.

    /proc/<pid>/stat is: ``pid (comm) state ppid pgrp session ...``. comm may contain
    spaces/parens, so parse the fields AFTER the last ')'.
    """
    try:
        with open(f"/proc/{pid}/stat", "rb") as f:
            data = f.read().decode(errors="replace")
        after = data[data.rfind(")") + 2 :].split()
        return int(after[1])  # state=after[0], ppid=after[1]
    except (OSError, ValueError, IndexError):
        return None


def _read_proc_cmdline(pid: int) -> str:
    """Return /proc/<pid>/cmdline as a space-joined string, or '' if unreadable."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            return f.read().decode(errors="replace").replace("\x00", " ").strip()
    except OSError:
        return ""


def _find_pid_in_tree(root_pid: int | None, init_command: str) -> int | None:
    """Find the container init PID under the ``enroot start`` process (root_pid).

    Nested inside pyxis, enroot runs as real root without a per-container user
    namespace, so ``enroot list`` cannot map a PID to the container name. Since the
    detached ``enroot start`` stays in the foreground, the container init is a
    descendant of it — walk the process tree from root_pid and return the descendant
    whose cmdline contains the (per-start) init command. root_pid uniquely scopes the
    search to this one container even under high concurrency.
    """
    if root_pid is None:
        return None
    # Build ppid -> children over all current processes.
    children: dict[int, list[int]] = {}
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        ppid = _read_proc_stat_ppid(pid)
        if ppid is not None:
            children.setdefault(ppid, []).append(pid)
    # BFS from root_pid's children; return the first descendant running init_command.
    stack = list(children.get(root_pid, []))
    seen: set[int] = set()
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        seen.add(pid)
        if init_command and init_command in _read_proc_cmdline(pid):
            return pid
        stack.extend(children.get(pid, []))
    return None


def _require_enroot() -> str:
    """Return the enroot binary path or hard-error if it is not installed."""
    path = shutil.which("enroot")
    if path is None:
        raise RuntimeError(
            "The 'enroot' binary is required for the enroot sandbox provider. "
            "Install enroot (https://github.com/NVIDIA/enroot) before selecting the enroot provider."
        )
    return path


@dataclass(frozen=True)
class EnrootCreateConfig:
    """Settings for creating an Enroot sandbox container."""

    mount_point: str = DEFAULT_MOUNT_POINT
    # Provider-scoped enroot paths. When None they are resolved under a
    # provider-managed base dir (see EnrootProvider.__init__). Pinning these and
    # passing them to every subprocess call is a correctness requirement: if
    # ENROOT_RUNTIME_PATH differs between `start` and later `exec`/`list`, the
    # running container cannot be found.
    base_dir: str | None = None
    data_path: str | None = None
    cache_path: str | None = None
    runtime_path: str | None = None
    sqsh_cache_dir: str | None = None
    rw: bool = True
    remap_root: bool = False
    init_command: str = DEFAULT_INIT_COMMAND
    import_timeout_s: float | None = 1800
    create_timeout_s: float | None = 600
    start_timeout_s: float | None = 600
    start_poll_s: float = 0.5
    extra_import_args: list[str] = field(default_factory=list)
    extra_create_args: list[str] = field(default_factory=list)
    extra_start_args: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        for attr in ("import_timeout_s", "create_timeout_s", "start_timeout_s"):
            value = getattr(self, attr)
            if value is not None and value <= 0:
                raise ValueError(f"create.{attr} must be > 0")
        if self.start_poll_s <= 0:
            raise ValueError("create.start_poll_s must be > 0")
        if not self.mount_point.startswith("/"):
            raise ValueError("create.mount_point must be an absolute path")


@dataclass(frozen=True)
class EnrootExecConfig:
    """Settings for running commands inside an Enroot sandbox."""

    default_timeout_s: float | None = 180
    default_mounts: list[str] = field(default_factory=list)
    extra_exec_args: list[str] = field(default_factory=list)
    concurrency: int = 32

    def __post_init__(self) -> None:
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")


@dataclass(frozen=True)
class EnrootProbeConfig:
    """Post-create probe settings: a test command confirming the sandbox is usable."""

    command: str | None = READY_PROBE_COMMAND
    expected_stdout: str | None = READY_PROBE_EXPECTED
    timeout_s: int = 30
    deadline_s: float | None = None
    stable_count: int = 1
    stable_delay_s: float = 0.0

    def __post_init__(self) -> None:
        if self.command is not None and self.timeout_s <= 0:
            raise ValueError("probe.timeout_s must be > 0")
        if self.deadline_s is not None and self.deadline_s <= 0:
            raise ValueError("probe.deadline_s must be > 0")
        if self.stable_count < 1:
            raise ValueError("probe.stable_count must be >= 1")
        if self.stable_delay_s < 0:
            raise ValueError("probe.stable_delay_s must be >= 0")


@dataclass
class _EnrootInstance:
    """Provider-private state stashed on SandboxHandle.raw."""

    name: str  # enroot container (rootfs) name
    sqsh_path: Path  # image squashfs backing the rootfs
    staging_dir: Path  # shared host folder bind-mounted into the container
    mount_point: str  # where the staging folder shows up inside
    image: str  # original spec.image
    env: dict[str, str] = field(default_factory=dict)
    container_pid: int | None = None  # host PID of the container init (for exec)
    start_pgid: int | None = None  # process group of the detached `enroot start`
    proc: Any = None  # asyncio subprocess handle for the detached start


def _translate_docker_uri(image: str) -> str:
    """Translate a docker image reference into an enroot import URI.

    Enroot's docker scheme separates the registry host with ``#`` rather than
    ``/`` (``docker://[USER@][REGISTRY#]IMAGE[:TAG]``). A leading component that
    looks like a registry host (contains a ``.`` or ``:``, or is ``localhost``)
    is treated as the registry; otherwise the reference is a Docker Hub name.

    The Docker Hub canonical hostnames are dropped rather than passed as the
    ``REGISTRY`` fragment: the public registry API is served from
    ``registry-1.docker.io``, not the literal ``docker.io`` alias, so forwarding
    ``docker://docker.io#repo`` points enroot at an endpoint that returns
    non-JSON and the import fails. Dropping them lets enroot use its configured
    Hub default (e.g. ``docker://swebench/foo:tag``).
    """
    first, sep, rest = image.partition("/")
    if sep and first in DOCKER_HUB_HOSTS:
        return f"docker://{rest}"
    if sep and ("." in first or ":" in first or first == "localhost"):
        return f"docker://{first}#{rest}"
    return f"docker://{image}"


def _resource_gpu_env(resources: SandboxResources) -> dict[str, str]:
    """Map a neutral GPU request onto NVIDIA_VISIBLE_DEVICES for the enroot hook."""
    if not resources.gpu:
        return {}
    return {"NVIDIA_VISIBLE_DEVICES": ",".join(str(i) for i in range(resources.gpu))}


def _is_runtime_failure(stderr: str) -> bool:
    """Best-effort: did enroot itself fail to run the command (vs the command failing)?"""
    low = stderr.lower()
    return any(marker in low for marker in ENROOT_RUNTIME_ERROR_MARKERS)


def _is_missing_container(stderr: str) -> bool:
    low = stderr.lower()
    return any(marker in low for marker in ENROOT_MISSING_CONTAINER_MARKERS)


def _coerce_mounts(value: Any) -> list[str]:
    """Normalize ``spec.provider_options['mounts']`` into a list of enroot fstab entries.

    Accepts a single ``"src:dst[:type:opts]"`` string or a list of them. These are
    extra per-sandbox mounts, added on top of the staging mount and the
    provider-level ``exec.default_mounts``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise EnrootCreateError(f"provider_options['mounts'] must be a string or list, got {type(value).__name__}")


class EnrootProvider:
    """Sandbox provider backed by the local Enroot CLI."""

    name = "enroot"

    def __init__(
        self,
        *,
        exec: EnrootExecConfig | Mapping[str, Any] | None = None,
        create: EnrootCreateConfig | Mapping[str, Any] | None = None,
        probe: EnrootProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._exec_config = _coerce_config(exec, EnrootExecConfig)
        self._create_config = _coerce_config(create, EnrootCreateConfig)
        self._probe = _coerce_config(probe, EnrootProbeConfig)
        self._binary = _require_enroot()
        self._semaphore = asyncio.Semaphore(self._exec_config.concurrency)

        # Resolve and pin provider-scoped enroot paths. Falling back to a
        # provider-managed base dir keeps the provider working when XDG_* /
        # ENROOT_* are unset (common in server/daemon contexts).
        cfg = self._create_config
        base = Path(cfg.base_dir) if cfg.base_dir else Path(tempfile.gettempdir()) / f"nemo-gym-enroot-{os.getuid()}"
        data_path = cfg.data_path or os.environ.get("ENROOT_DATA_PATH") or str(base / "data")
        cache_path = cfg.cache_path or os.environ.get("ENROOT_CACHE_PATH") or str(base / "cache")
        runtime_path = cfg.runtime_path or os.environ.get("ENROOT_RUNTIME_PATH") or str(base / "runtime")
        self._sqsh_cache_dir = Path(cfg.sqsh_cache_dir or (base / "sqsh"))
        for directory in (data_path, cache_path, runtime_path, self._sqsh_cache_dir):
            Path(directory).mkdir(parents=True, exist_ok=True)

        self._enroot_env = {
            **os.environ,
            "ENROOT_DATA_PATH": data_path,
            "ENROOT_CACHE_PATH": cache_path,
            "ENROOT_RUNTIME_PATH": runtime_path,
        }
        # Serializes concurrent imports of the same image within this process.
        self._import_locks: dict[str, asyncio.Lock] = {}

    async def _run(
        self,
        argv: list[str],
        *,
        timeout_s: float | None,
        stdin: bytes | None = None,
    ) -> tuple[int, str, str]:
        """Run an enroot CLI command. Returns (return_code, stdout, stderr).

        Enforces timeout via asyncio.wait_for and kills the whole process group
        on timeout so child processes do not linger. Bounds concurrency with a
        shared semaphore. Decodes output with errors="replace". Every call runs
        with the pinned ENROOT_* environment.
        """
        async with self._semaphore:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                env=self._enroot_env,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=stdin),
                    timeout=timeout_s,
                )
            except asyncio.TimeoutError as e:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await proc.wait()
                raise TimeoutError(f"enroot command timed out after {timeout_s:g}s: {argv}") from e

            return_code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
            return return_code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    async def _start_detached(self, argv: list[str]) -> tuple[Any, IO[bytes], IO[bytes]]:
        """Launch the long-lived ``enroot start`` init without awaiting its exit.

        ``enroot start`` does not daemonize — it stays in the foreground for the
        whole container lifetime. We launch it detached in its own session
        (start_new_session=True), capture output to temp files (so an early exit
        leaves diagnosable stderr), and return the process handle plus the temp
        files. The caller confirms readiness by polling ``enroot list``.
        """
        out_f = tempfile.TemporaryFile()
        err_f = tempfile.TemporaryFile()
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=out_f,
            stderr=err_f,
            start_new_session=True,
            env=self._enroot_env,
        )
        return proc, out_f, err_f

    def _resolve_image(self, image: str) -> tuple[str | None, Path | None]:
        """Return (import_uri, sqsh_path). Exactly one is non-None.

        A local ``.sqsh`` file is used directly (no import). Anything else is
        translated to an enroot import URI.
        """
        if image.endswith(".sqsh") or (("://" not in image) and Path(image).exists()):
            return None, Path(image)
        if "://" in image:
            return image, None
        return _translate_docker_uri(image), None

    async def _ensure_image(self, image: str) -> Path:
        """Return a local squashfs path for `image`, importing (and caching) if needed."""
        import_uri, sqsh_path = self._resolve_image(image)
        if sqsh_path is not None:
            if not sqsh_path.exists():
                raise EnrootCreateError(f"image squashfs {str(sqsh_path)!r} does not exist")
            return sqsh_path

        assert import_uri is not None
        key = hashlib.sha256(image.encode()).hexdigest()[:16]
        target = self._sqsh_cache_dir / f"{key}.sqsh"

        lock = self._import_locks.setdefault(key, asyncio.Lock())
        async with lock:
            if target.exists() and target.stat().st_size > 0:
                return target
            # Import to a unique temp path then atomically rename so concurrent
            # (cross-process) creates never observe a half-written squashfs.
            tmp = self._sqsh_cache_dir / f".{key}.{uuid.uuid4().hex}.tmp"
            argv = [self._binary, "import", "-o", str(tmp), *self._create_config.extra_import_args, import_uri]
            try:
                code, _out, err = await self._run(argv, timeout_s=self._create_config.import_timeout_s)
            except TimeoutError as e:
                tmp.unlink(missing_ok=True)
                raise EnrootCreateError(f"enroot import timed out for image={image!r}: {e}") from e
            if code != 0 or not tmp.exists():
                tmp.unlink(missing_ok=True)
                raise EnrootCreateError(f"enroot import failed (code={code}) for image={image!r}: {err.strip()}")
            os.replace(tmp, target)
            return target

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Import/create the rootfs, launch a detached init, and return a ready handle."""
        if spec.ttl_s is not None:
            LOGGER.warning("ttl_s is not supported by the enroot provider; it will be ignored.")
        resources = spec.resources
        if resources.cpu is not None or resources.memory_mib is not None or resources.disk_gib is not None:
            LOGGER.warning(
                "cpu/memory_mib/disk_gib are not enforced by standalone enroot (that is pyxis/Slurm's job); "
                "these resource limits will be ignored."
            )

        if spec.image is None:
            raise EnrootCreateError("spec.image is required for the enroot provider")

        # Extra per-sandbox mounts (validated before we allocate anything).
        extra_mounts = _coerce_mounts(spec.provider_options.get("mounts"))

        sqsh_path = await self._ensure_image(spec.image)

        mount_point = self._create_config.mount_point
        name = CONTAINER_NAME_PREFIX + uuid.uuid4().hex

        # Unpack the rootfs from the squashfs.
        try:
            code, _out, err = await self._run(
                [self._binary, "create", "-n", name, *self._create_config.extra_create_args, str(sqsh_path)],
                timeout_s=self._create_config.create_timeout_s,
            )
        except TimeoutError as e:
            raise EnrootCreateError(f"enroot create timed out for image={spec.image!r}: {e}") from e
        if code != 0:
            raise EnrootCreateError(f"enroot create failed (code={code}) for image={spec.image!r}: {err.strip()}")

        # Host staging dir (bind-mounted in); must exist before start.
        staging_dir = Path(tempfile.mkdtemp(prefix="nemo-gym-enroot-"))

        # Build the `enroot start` command line for the long-lived init.
        argv: list[str] = [self._binary, "start"]
        if self._create_config.rw:
            argv.append("--rw")
        if self._create_config.remap_root:
            argv.append("--root")
        argv += ["-m", f"{staging_dir}:{mount_point}"]
        for mount in self._exec_config.default_mounts:
            argv += ["-m", mount]
        for mount in extra_mounts:
            argv += ["-m", mount]
        start_env = {**_resource_gpu_env(resources), **spec.env}
        for key, value in start_env.items():
            argv += ["-e", f"{key}={value}"]
        argv += list(self._create_config.extra_start_args)
        argv += [name, "sh", "-c", self._create_config.init_command]

        proc, out_f, err_f = await self._start_detached(argv)
        instance = _EnrootInstance(
            name=name,
            sqsh_path=sqsh_path,
            staging_dir=staging_dir,
            mount_point=mount_point,
            image=spec.image,
            env=dict(spec.env),
            start_pgid=proc.pid,
            proc=proc,
        )
        handle = SandboxHandle(sandbox_id=name, provider_name=self.name, raw=instance)

        try:
            instance.container_pid = await self._await_container_pid(instance, err_f)
            await self._verify_created_handle(handle)
        except Exception:
            await self._cleanup_failed_create_handle(handle)
            raise
        finally:
            with contextlib.suppress(Exception):
                out_f.close()
            with contextlib.suppress(Exception):
                err_f.close()

        return handle

    async def _await_container_pid(self, instance: _EnrootInstance, err_f: IO[bytes]) -> int:
        """Poll ``enroot list`` until the container's init PID appears, or raise."""
        loop = asyncio.get_running_loop()
        deadline = (
            loop.time() + self._create_config.start_timeout_s
            if self._create_config.start_timeout_s is not None
            else None
        )
        while True:
            # If the detached start already exited, it failed to launch the init.
            if instance.proc.returncode is not None:
                stderr = self._read_temp(err_f)
                raise EnrootCreateError(
                    f"enroot start exited early (code={instance.proc.returncode}) for {instance.name!r}: {stderr}"
                )
            pid, _running = await self._lookup_container(instance.name)
            if pid is None:
                # Nested-in-pyxis fallback: when enroot runs as real root
                # (ENROOT_ALLOW_SUPERUSER) it does not create a per-container user
                # namespace, so `enroot list` shows the container as present but cannot
                # map a PID to it. The detached `enroot start` stays in the foreground
                # for the container lifetime, so its init is a descendant — find it by
                # walking the start process's tree for the init command.
                pid = await loop.run_in_executor(
                    None,
                    _find_pid_in_tree,
                    instance.start_pgid,
                    self._create_config.init_command,
                )
            if pid is not None:
                return pid
            if deadline is not None and loop.time() >= deadline:
                stderr = self._read_temp(err_f)
                raise EnrootCreateError(
                    f"enroot container {instance.name!r} did not start within "
                    f"{self._create_config.start_timeout_s:g}s: {stderr}"
                )
            await asyncio.sleep(self._create_config.start_poll_s)

    @staticmethod
    def _read_temp(handle: IO[bytes]) -> str:
        with contextlib.suppress(Exception):
            handle.seek(0)
            return handle.read().decode(errors="replace").strip()
        return ""

    async def _lookup_container(self, name: str) -> tuple[int | None, bool]:
        """Return (pid, present). pid is the running init PID or None; present is
        whether the rootfs exists at all (running or not)."""
        try:
            code, out, _err = await self._run(
                [self._binary, "list", "-f"],
                timeout_s=self._exec_config.default_timeout_s,
            )
        except TimeoutError:
            return None, False
        if code != 0:
            return None, False

        lines = out.splitlines()
        for line in lines[1:]:  # skip the header row
            parts = line.split()
            if parts and parts[0] == name:
                if len(parts) >= 2 and parts[1].isdigit():
                    return int(parts[1]), True
                return None, True
        return None, False

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        """Run the readiness probe until the sandbox responds, or raise.

        - probe.command is None      -> skip (no verification).
        - probe.deadline_s is None   -> single attempt; a failure raises immediately.
        - probe.deadline_s is set    -> poll until the sandbox passes the probe
          `stable_count` consecutive times, or the deadline elapses.
        """
        probe = self._probe
        if probe.command is None:
            return

        loop = asyncio.get_running_loop()
        deadline = loop.time() + probe.deadline_s if probe.deadline_s is not None else None
        consecutive = 0
        last_detail = "no probe attempt completed"

        while True:
            result = await self.exec(handle, probe.command, timeout_s=probe.timeout_s)
            passed = result.return_code == 0 and (
                probe.expected_stdout is None or probe.expected_stdout in (result.stdout or "")
            )
            if passed:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                last_detail = f"return_code={result.return_code}, stderr={(result.stderr or '').strip()!r}"
                if deadline is None:
                    raise EnrootCreateVerificationError(
                        f"sandbox {handle.sandbox_id!r} failed readiness probe: {last_detail}"
                    )

            if deadline is not None and loop.time() >= deadline:
                raise EnrootCreateVerificationError(
                    f"sandbox {handle.sandbox_id!r} did not pass readiness probe within "
                    f"{probe.deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    def _kill_start_group(self, instance: _EnrootInstance) -> None:
        """Best-effort SIGTERM then SIGKILL of the detached start's process group."""
        if instance.start_pgid is None:
            return
        for sig in (signal.SIGTERM, signal.SIGKILL):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(instance.start_pgid, sig)

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        """Best-effort teardown of a sandbox that failed to start or verify."""
        instance = handle.raw
        self._kill_start_group(instance)
        if instance.proc is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(instance.proc.wait(), timeout=self._exec_config.default_timeout_s)
        with contextlib.suppress(Exception):
            await self._run(
                [self._binary, "remove", "-f", instance.name],
                timeout_s=self._exec_config.default_timeout_s,
            )
        shutil.rmtree(instance.staging_dir, ignore_errors=True)

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: int | float | None = None,
        user: str | int | None = None,
        stdin: bytes | None = None,
    ) -> SandboxExecResult:
        """Run a command inside the container via ``enroot exec <pid>``.

        Maps the neutral ``user`` parameter onto enroot:
        - None            -> run as the default (launching) user, or root if the
          container was started with remap_root.
        - "root" / 0      -> run directly (requires create.remap_root=true to be
          root inside the container).
        - other user/uid  -> wrap in ``su`` to switch to that user (requires root
          inside the container).

        ``stdin``, when given, is piped to the command's standard input.
        """
        instance = handle.raw
        if instance.container_pid is None:
            return SandboxExecResult(
                stdout=None,
                stderr=f"enroot container {instance.name!r} has no running init PID",
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="sandbox",
            )

        merged_env = dict(getattr(instance, "env", {}))
        if env:
            merged_env.update(env)

        flags: list[str] = []
        for key, value in merged_env.items():
            flags += ["-e", f"{key}={value}"]
        flags += list(self._exec_config.extra_exec_args)

        effective_command = command
        if cwd is not None:
            effective_command = f"cd {shlex.quote(cwd)} && {effective_command}"
        is_root = user == "root" or user == 0
        if user is not None and not is_root:
            effective_command = f"su -s /bin/sh -c {shlex.quote(effective_command)} {shlex.quote(str(user))}"

        argv = [self._binary, "exec", *flags, str(instance.container_pid), "sh", "-c", effective_command]
        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s

        try:
            code, out, err = await self._run(argv, timeout_s=effective_timeout, stdin=stdin)
        except TimeoutError as e:
            return SandboxExecResult(
                stdout=None,
                stderr=str(e),
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="timeout",
            )

        if code != 0 and _is_runtime_failure(err):
            return SandboxExecResult(
                stdout=out,
                stderr=err,
                return_code=SANDBOX_RUNTIME_RETURN_CODE,
                error_type="sandbox",
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=code, error_type=None)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one host file into the sandbox.

        Fast path (target under the bind mount): write directly to the host side
        of the shared folder. Fallback (arbitrary path): stage into the shared
        folder, then cp inside the container.
        """
        instance = handle.raw

        rel = _path_under_mount(instance.mount_point, target_path)
        if rel is not None:
            dest = instance.staging_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source_path.read_bytes())
            return

        tmp_name = uuid.uuid4().hex
        host_tmp = instance.staging_dir / tmp_name
        host_tmp.write_bytes(source_path.read_bytes())
        try:
            container_tmp = f"{instance.mount_point.rstrip('/')}/{tmp_name}"
            parent = posixpath.dirname(target_path)
            script = f"mkdir -p {shlex.quote(parent)} && cp {shlex.quote(container_tmp)} {shlex.quote(target_path)}"
            result = await self.exec(handle, script)
            if result.return_code != 0:
                raise RuntimeError(f"enroot upload to {target_path!r} failed: {result.stderr}")
        finally:
            host_tmp.unlink(missing_ok=True)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the host.

        Fast path (source under the bind mount): read directly from the host side
        of the shared folder. Fallback (arbitrary path): cp inside the container
        into the shared folder, then read the host side.
        """
        instance = handle.raw
        target_path.parent.mkdir(parents=True, exist_ok=True)

        rel = _path_under_mount(instance.mount_point, source_path)
        if rel is not None:
            target_path.write_bytes((instance.staging_dir / rel).read_bytes())
            return

        tmp_name = uuid.uuid4().hex
        host_tmp = instance.staging_dir / tmp_name
        try:
            container_tmp = f"{instance.mount_point.rstrip('/')}/{tmp_name}"
            script = f"cp {shlex.quote(source_path)} {shlex.quote(container_tmp)}"
            result = await self.exec(handle, script)
            if result.return_code != 0:
                raise RuntimeError(f"enroot download from {source_path!r} failed: {result.stderr}")
            target_path.write_bytes(host_tmp.read_bytes())
        finally:
            host_tmp.unlink(missing_ok=True)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the container's lifecycle status via ``enroot list -f``.

        Liveness is keyed off the PID column, not name presence: enroot lists
        created-but-not-running rootfs with an empty PID. Running -> RUNNING;
        present without a PID, or absent -> STOPPED; timeout/parse error -> UNKNOWN.
        """
        instance = handle.raw
        try:
            code, out, _err = await self._run(
                [self._binary, "list", "-f"],
                timeout_s=self._exec_config.default_timeout_s,
            )
        except TimeoutError:
            return SandboxStatus.UNKNOWN
        if code != 0:
            return SandboxStatus.UNKNOWN

        for line in out.splitlines()[1:]:
            parts = line.split()
            if parts and parts[0] == instance.name:
                if len(parts) >= 2 and parts[1].isdigit():
                    return SandboxStatus.RUNNING
                return SandboxStatus.STOPPED
        return SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        """Kill the container init, remove the rootfs, and clean up the staging dir."""
        instance = handle.raw

        self._kill_start_group(instance)
        if instance.proc is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(instance.proc.wait(), timeout=self._exec_config.default_timeout_s)

        remove_error: Exception | None = None
        try:
            code, _out, err = await self._run(
                [self._binary, "remove", "-f", instance.name],
                timeout_s=self._exec_config.default_timeout_s,
            )
            if code != 0 and not _is_missing_container(err):
                remove_error = RuntimeError(f"enroot remove failed (code={code}) for {instance.name!r}: {err.strip()}")
        except TimeoutError as e:
            remove_error = e

        try:
            shutil.rmtree(instance.staging_dir, ignore_errors=False)
        except OSError as e:
            LOGGER.warning("failed to remove staging dir %s: %s", instance.staging_dir, e)

        if remove_error is not None:
            raise remove_error

    async def aclose(self) -> None:
        """No provider-wide resources to close."""
        return None
