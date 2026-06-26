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

"""Apptainer provider implementation."""

import asyncio
import contextlib
import json
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
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_MOUNT_POINT = "/sandbox"
INSTANCE_NAME_PREFIX = "nemo-gym-"
READY_PROBE_COMMAND = "printf apptainer-sandbox-ready"
READY_PROBE_EXPECTED = "apptainer-sandbox-ready"
SANDBOX_RUNTIME_RETURN_CODE = 125
# Best-effort stderr markers indicating apptainer itself (not the user's command)
# failed to run the command. Apptainer prefixes its own fatal errors with "FATAL:".
APPTAINER_RUNTIME_ERROR_MARKERS = ("fatal:", "no instance found", "instance not found", "does not exist")
APPTAINER_MISSING_INSTANCE_MARKERS = ("no instance found", "instance not found", "does not exist")


class ApptainerCreateError(SandboxCreateError):
    """Raised when Apptainer cannot create a sandbox."""


class ApptainerCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a newly-created sandbox cannot execute a probe command."""


def _require_apptainer() -> str:
    """Return the apptainer binary path or hard-error if it is not installed."""
    path = shutil.which("apptainer")
    if path is None:
        raise RuntimeError(
            "The 'apptainer' binary is required for the apptainer sandbox provider. "
            "Install Apptainer before using env.sandbox.provider.name=apptainer."
        )
    return path


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    """Accept either a config dataclass instance or a plain mapping (Hydra YAML)."""
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


@dataclass(frozen=True)
class ApptainerCreateConfig:
    """Settings for creating an Apptainer sandbox instance."""

    mount_point: str = DEFAULT_MOUNT_POINT
    start_timeout_s: float | None = 600
    extra_start_args: list[str] = field(default_factory=list)
    apply_resource_limits: bool = True

    def __post_init__(self) -> None:
        if self.start_timeout_s is not None and self.start_timeout_s <= 0:
            raise ValueError("create.start_timeout_s must be > 0")
        if not self.mount_point.startswith("/"):
            raise ValueError("create.mount_point must be an absolute path")


@dataclass(frozen=True)
class ApptainerExecConfig:
    """Settings for running commands inside an Apptainer sandbox."""

    default_timeout_s: float | None = 180
    fakeroot_for_root: bool = True
    default_binds: list[str] = field(default_factory=list)
    extra_exec_args: list[str] = field(default_factory=list)
    concurrency: int = 32

    def __post_init__(self) -> None:
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")


@dataclass(frozen=True)
class ApptainerProbeConfig:
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
class _ApptainerInstance:
    """Provider-private state stashed on SandboxHandle.raw."""

    name: str  # what the instance is called
    staging_dir: Path  # the shared folder on the host
    mount_point: str  # where the folder shows up inside
    image: str  # what it was built from
    env: dict[str, str] = field(default_factory=dict)


def _resource_flags(resources: SandboxResources) -> list[str]:
    """Translate neutral resources into apptainer CLI flags."""
    return _resource_limit_flags(resources) + _resource_passthrough_flags(resources)


def _resource_limit_flags(resources: SandboxResources) -> list[str]:
    flags: list[str] = []
    if resources.cpu is not None:
        flags += ["--cpus", str(resources.cpu)]
    if resources.memory_mib is not None:
        flags += ["--memory", f"{resources.memory_mib}m"]
    return flags


def _resource_passthrough_flags(resources: SandboxResources) -> list[str]:
    flags: list[str] = []
    if resources.gpu:
        flags.append("--nv")
    return flags


def _resolve_image(image: str) -> str:
    if "://" in image or image.startswith(("/", ".")) or image.endswith(".sif"):
        return image
    return f"docker://{image}"


def _to_sandbox_status(state: str | None) -> SandboxStatus:
    """Map an apptainer-reported state string to the neutral status enum."""
    normalized = str(state or "").lower()
    if normalized in {"running", "active", "ready"}:
        return SandboxStatus.RUNNING
    if normalized in {"starting", "creating", "pending"}:
        return SandboxStatus.STARTING
    if normalized in {"stopped", "exited", "terminated"}:
        return SandboxStatus.STOPPED
    if normalized in {"error", "failed", "unhealthy"}:
        return SandboxStatus.ERROR
    return SandboxStatus.UNKNOWN


def _path_under_mount(mount_point: str, path: str) -> str | None:
    """If `path` is inside the mount, return its path relative to the mount; else None."""
    if not path.startswith("/"):
        return None
    mp = posixpath.normpath(mount_point.rstrip("/") or "/")
    normalized = posixpath.normpath(path)
    if normalized == mp:
        return ""
    try:
        if posixpath.commonpath([mp, normalized]) != mp:
            return None
    except ValueError:
        return None
    if mp == "/":
        return normalized.lstrip("/")
    return normalized[len(mp) + 1 :]


def _is_runtime_failure(stderr: str) -> bool:
    """Best-effort: did apptainer itself fail to run the command (vs the command failing)?"""
    low = stderr.lower()
    return any(marker in low for marker in APPTAINER_RUNTIME_ERROR_MARKERS)


def _is_missing_instance(stderr: str) -> bool:
    low = stderr.lower()
    return any(marker in low for marker in APPTAINER_MISSING_INSTANCE_MARKERS)


def _coerce_binds(value: Any) -> list[str]:
    """Normalize ``spec.provider_options['binds']`` into a list of bind strings.

    Accepts a single ``"src:dst[:opts]"`` string or a list of them. These are
    extra per-sandbox bind mounts, added on top of the staging mount and the
    provider-level ``exec.default_binds``.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise ApptainerCreateError(f"provider_options['binds'] must be a string or list, got {type(value).__name__}")


class ApptainerProvider:
    """Sandbox provider backed by the local Apptainer CLI."""

    name = "apptainer"

    def __init__(
        self,
        *,
        exec: ApptainerExecConfig | Mapping[str, Any] | None = None,
        create: ApptainerCreateConfig | Mapping[str, Any] | None = None,
        probe: ApptainerProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._exec_config = _coerce_config(exec, ApptainerExecConfig)
        self._create_config = _coerce_config(create, ApptainerCreateConfig)
        self._probe = _coerce_config(probe, ApptainerProbeConfig)
        self._binary = _require_apptainer()
        self._semaphore = asyncio.Semaphore(self._exec_config.concurrency)

    async def _run(
        self,
        argv: list[str],
        *,
        timeout_s: float | None,
        stdin: bytes | None = None,
        daemonize: bool = False,
    ) -> tuple[int, str, str]:
        """Run an apptainer CLI command. Returns (return_code, stdout, stderr).

        Enforces timeout via asyncio.wait_for and kills the whole process group
        on timeout so child processes do not linger. Bounds concurrency with a
        shared semaphore. Decodes output with errors="replace".

        Set ``daemonize=True`` for commands that fork a long-lived background
        process (``apptainer instance start``). Such commands hand the started
        instance a copy of the child's stdout/stderr, so reading those pipes to
        EOF (``communicate()``) blocks until the *instance* exits — i.e. the call
        appears to hang until ``timeout_s`` even though the foreground process
        finished in under a second. In that mode we capture output to temp files
        (which the instance may inherit harmlessly) and only wait for the
        foreground process to exit.
        """
        async with self._semaphore:
            if daemonize:
                return await self._run_daemonizing(argv, timeout_s=timeout_s)

            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
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
                raise TimeoutError(f"apptainer command timed out after {timeout_s:g}s: {argv}") from e

            return_code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
            return return_code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    async def _run_daemonizing(self, argv: list[str], *, timeout_s: float | None) -> tuple[int, str, str]:
        """Run a command that daemonizes a child (e.g. ``apptainer instance start``).

        Captures stdout/stderr to temp files instead of pipes so the long-lived
        instance inheriting those descriptors cannot wedge the read, then waits
        only for the foreground process to exit.
        """
        with tempfile.TemporaryFile() as out_f, tempfile.TemporaryFile() as err_f:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=out_f,
                stderr=err_f,
                start_new_session=True,
            )
            try:
                await asyncio.wait_for(proc.wait(), timeout=timeout_s)
            except asyncio.TimeoutError as e:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await proc.wait()
                raise TimeoutError(f"apptainer command timed out after {timeout_s:g}s: {argv}") from e

            out_f.seek(0)
            err_f.seek(0)
            stdout_b = out_f.read()
            stderr_b = err_f.read()

        return_code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
        return return_code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Start an apptainer instance and return a ready handle.

        Steps:
        1. Warn once if spec.ttl_s is set (unsupported by apptainer).
        2. Resolve the image source (local .sif path or remote docker://, oras://,
           library:// URI) from spec.image. -- just use as is
        3. Make a host staging dir (tempfile.mkdtemp), pick
           mount_point = self._create_config.mount_point, generate a unique
           name = INSTANCE_NAME_PREFIX + uuid4().hex.
        4. Build argv: [binary, "instance", "start", <--bind staging:mount_point>,
           <config default_binds>, <spec.provider_options["binds"]>, <--env ...>,
           _resource_flags(spec.resources), <extra_start_args>, image, name].
        5. await self._run(argv, timeout_s=self._create_config.start_timeout_s);
           on non-zero return, clean up the staging dir and raise
           ApptainerCreateError(stderr).
        6. Build the handle:
           SandboxHandle(sandbox_id=name, provider_name=self.name,
               raw=_ApptainerInstance(name, staging_dir, mount_point, image)).
        7. Verify readiness via self._verify_created_handle(handle); on failure
           clean up and raise ApptainerCreateVerificationError.
        8. Return the handle.
        """
        # ttl_s has no apptainer equivalent; warn once, then ignore it.
        if spec.ttl_s is not None:
            LOGGER.warning("ttl_s is not supported by the apptainer provider; it will be ignored.")

        if spec.image is None:
            raise ApptainerCreateError("spec.image is required for the apptainer provider")
        image = _resolve_image(spec.image)

        # Extra per-sandbox bind mounts (validated before we allocate anything).
        extra_binds = _coerce_binds(spec.provider_options.get("binds"))

        # host staging dir (bind-mounted in), mount point, unique name.
        mount_point = self._create_config.mount_point
        staging_dir = Path(
            tempfile.mkdtemp(prefix="nemo-gym-apptainer-")
        )  # create a new empty temp directory on the host and returns that path
        name = INSTANCE_NAME_PREFIX + uuid.uuid4().hex

        # build the `apptainer instance start` command line.
        argv: list[str] = [self._binary, "instance", "start"]
        argv += ["--bind", f"{staging_dir}:{mount_point}"]
        for bind in self._exec_config.default_binds:
            argv += ["--bind", bind]
        for bind in extra_binds:
            argv += ["--bind", bind]
        for key, value in spec.env.items():
            argv += ["--env", f"{key}={value}"]
        start_args = list(self._create_config.extra_start_args)
        resource_limit_flags = _resource_limit_flags(spec.resources)
        if resource_limit_flags and self._create_config.apply_resource_limits:
            if "--fakeroot" in start_args:
                LOGGER.warning(
                    "Skipping apptainer CPU/memory resource flags because create.extra_start_args contains --fakeroot."
                )
            else:
                argv += resource_limit_flags
        argv += _resource_passthrough_flags(spec.resources)
        argv += start_args
        argv += [image, name]

        # start the instance; clean up the staging dir on any failure.
        try:
            code, _out, err = await self._run(argv, timeout_s=self._create_config.start_timeout_s, daemonize=True)
        except TimeoutError as e:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ApptainerCreateError(f"apptainer instance start timed out for image={image!r}: {e}") from e
        if code != 0:
            shutil.rmtree(staging_dir, ignore_errors=True)
            raise ApptainerCreateError(
                f"apptainer instance start failed (code={code}) for image={image!r}: {err.strip()}"
            )

        # wrap provider-private state on the handle.
        handle = SandboxHandle(
            sandbox_id=name,
            provider_name=self.name,
            raw=_ApptainerInstance(
                name=name,
                staging_dir=staging_dir,
                mount_point=mount_point,
                image=image,
                env=dict(spec.env),
            ),
        )

        # Verify the sandbox can actually run a command before handing it back.
        # On any failure, tear down the half-created sandbox so we don't leak a
        # running instance / staging dir.
        try:
            await self._verify_created_handle(handle)
        except Exception:
            await self._cleanup_failed_create_handle(handle)
            raise

        return handle

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
                    raise ApptainerCreateVerificationError(
                        f"sandbox {handle.sandbox_id!r} failed readiness probe: {last_detail}"
                    )

            if deadline is not None and loop.time() >= deadline:
                raise ApptainerCreateVerificationError(
                    f"sandbox {handle.sandbox_id!r} did not pass readiness probe within "
                    f"{probe.deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        """Best-effort teardown of a sandbox that failed verification."""
        inst = handle.raw
        with contextlib.suppress(Exception):
            await self._run(
                [self._binary, "instance", "stop", inst.name],
                timeout_s=self._exec_config.default_timeout_s,
            )
        shutil.rmtree(inst.staging_dir, ignore_errors=True)

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
        """Run a command inside the instance.

        Maps the neutral ``user`` parameter onto apptainer:
        - None            -> run as the default (launching) user.
        - "root" / 0      -> add --fakeroot (root inside the container).
        - other user/uid  -> --fakeroot + wrap in ``su`` to switch to that user.

        ``stdin``, when given, is piped to the command's standard input. This is an
        apptainer-provider extension to the base protocol, useful for feeding large
        inputs (e.g. prompts) that would exceed the kernel's argv length limit.
        """
        inst = handle.raw

        flags: list[str] = []
        if cwd is not None:
            flags += ["--pwd", cwd]
        merged_env = dict(getattr(inst, "env", {}))
        if env:
            merged_env.update(env)
        if merged_env:
            for key, value in merged_env.items():
                flags += ["--env", f"{key}={value}"]

        effective_command = command
        is_root = user == "root" or user == 0
        if is_root:
            if self._exec_config.fakeroot_for_root:
                flags.append("--fakeroot")
        elif user is not None:
            # Need root inside the container to switch users, then su to the target.
            flags.append("--fakeroot")
            effective_command = f"su -s /bin/sh -c {shlex.quote(command)} {shlex.quote(str(user))}"

        flags += list(self._exec_config.extra_exec_args)

        argv = [self._binary, "exec", *flags, f"instance://{inst.name}", "sh", "-c", effective_command]
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
        inst = handle.raw

        rel = _path_under_mount(inst.mount_point, target_path)
        if rel is not None:
            dest = inst.staging_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(source_path.read_bytes())
            return

        tmp_name = uuid.uuid4().hex
        host_tmp = inst.staging_dir / tmp_name
        host_tmp.write_bytes(source_path.read_bytes())
        try:
            container_tmp = f"{inst.mount_point.rstrip('/')}/{tmp_name}"
            parent = posixpath.dirname(target_path)
            script = f"mkdir -p {shlex.quote(parent)} && cp {shlex.quote(container_tmp)} {shlex.quote(target_path)}"
            result = await self.exec(handle, script, user="root")
            if result.return_code != 0:
                raise RuntimeError(f"apptainer upload to {target_path!r} failed: {result.stderr}")
        finally:
            host_tmp.unlink(missing_ok=True)

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one sandbox file to the host.

        Fast path (source under the bind mount): read directly from the host side
        of the shared folder. Fallback (arbitrary path): cp inside the container
        into the shared folder, then read the host side.
        """
        inst = handle.raw
        target_path.parent.mkdir(parents=True, exist_ok=True)

        rel = _path_under_mount(inst.mount_point, source_path)
        if rel is not None:
            target_path.write_bytes((inst.staging_dir / rel).read_bytes())
            return

        tmp_name = uuid.uuid4().hex
        host_tmp = inst.staging_dir / tmp_name
        try:
            container_tmp = f"{inst.mount_point.rstrip('/')}/{tmp_name}"
            script = f"cp {shlex.quote(source_path)} {shlex.quote(container_tmp)}"
            result = await self.exec(handle, script, user="root")
            if result.return_code != 0:
                raise RuntimeError(f"apptainer download from {source_path!r} failed: {result.stderr}")
            target_path.write_bytes(host_tmp.read_bytes())
        finally:
            host_tmp.unlink(missing_ok=True)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Return the instance's lifecycle status by querying ``apptainer instance list``.
        Runs apptainer instance list --json
        On timeout, non-zero, unparseable JSON ---> UNKNOWN
        Look for the instance name of this sandbox. If it is found --> RUNNING. If it's gone --> STOPPED
        """
        inst = handle.raw

        try:
            code, out, _err = await self._run(
                [self._binary, "instance", "list", "--json"],
                timeout_s=self._exec_config.default_timeout_s,
            )
        except TimeoutError:
            return SandboxStatus.UNKNOWN

        if code != 0:
            return SandboxStatus.UNKNOWN

        try:
            instances = json.loads(out).get("instances", [])
        except (json.JSONDecodeError, AttributeError):
            return SandboxStatus.UNKNOWN

        for entry in instances:
            if entry.get("instance") == inst.name:
                return _to_sandbox_status(entry.get("state") or "running")

        # Not listed -> it has been stopped (or never existed anymore).
        return SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        """Stop the instance and clean up the host staging dir.
        Runs apptainer instance stop <name>
        If there is no instance --> SUCCESS
        Removes the host staging dir afterward
        """
        inst = handle.raw

        stop_error: Exception | None = None
        try:
            code, _out, err = await self._run(
                [self._binary, "instance", "stop", inst.name],
                timeout_s=self._exec_config.default_timeout_s,
            )
            if code != 0 and not _is_missing_instance(err):
                stop_error = RuntimeError(
                    f"apptainer instance stop failed (code={code}) for {inst.name!r}: {err.strip()}"
                )
        except TimeoutError as e:
            stop_error = e

        # Always best-effort remove the host staging dir, even if stop failed.
        try:
            shutil.rmtree(inst.staging_dir, ignore_errors=False)
        except OSError as e:
            LOGGER.warning("failed to remove staging dir %s: %s", inst.staging_dir, e)

        if stop_error is not None:
            raise stop_error

    async def aclose(self) -> None:
        """No provider-wide resources to close."""
        return None
