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

"""Docker sandbox provider: one long-lived container per sandbox, driven via the docker CLI."""

import asyncio
import contextlib
import logging
import math
import os
import posixpath
import shlex
import shutil
import signal
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

CONTAINER_NAME_PREFIX = "nemo-gym-"
SANDBOX_LABEL = "nemo-gym.sandbox"
READY_PROBE_COMMAND = "printf docker-sandbox-ready"
READY_PROBE_EXPECTED = "docker-sandbox-ready"
SANDBOX_RUNTIME_RETURN_CODE = 125
DEFAULT_KEEPALIVE_SHELL = "/bin/sh"
DEFAULT_KEEPALIVE_CMD = "while :; do sleep 2147483647; done"
DOCKER_RUNTIME_ERROR_MARKERS = (
    "no such container",
    "is not running",
    "is not paused",
    "cannot connect to the docker daemon",
    "error response from daemon",
)
DOCKER_MISSING_CONTAINER_MARKERS = ("no such container", "no such object")


class DockerCreateError(SandboxCreateError):
    """Raised when Docker cannot create a sandbox."""


class DockerCreateVerificationError(SandboxCreateVerificationError):
    """Raised when a new container fails its readiness probe."""


def _require_docker() -> str:
    path = shutil.which("docker")
    if path is None:
        raise RuntimeError(
            "The 'docker' binary is required for the docker sandbox provider. Install Docker and "
            "ensure the daemon is running before selecting env.sandbox.provider.name=docker."
        )
    return path


def _coerce_config(value: Any, config_cls: type[Any]) -> Any:
    if value is None:
        return config_cls()
    if isinstance(value, config_cls):
        return value
    if isinstance(value, Mapping):
        return config_cls(**value)
    raise TypeError(f"{config_cls.__name__} must be a mapping or {config_cls.__name__} instance")


def _coerce_str_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    raise DockerCreateError(f"provider_options[{what!r}] must be a string or list, got {type(value).__name__}")


def _redact_argv(argv: list[str]) -> list[str]:
    """argv copy with ``--env KEY=VALUE`` values masked, so timeouts/logs don't leak secrets."""
    out: list[str] = []
    mask_next = False
    for tok in argv:
        if mask_next:
            out.append(f"{tok.split('=', 1)[0]}=***" if "=" in tok else tok)
        else:
            out.append(tok)
        mask_next = tok == "--env"
    return out


@dataclass(frozen=True)
class DockerCreateConfig:
    keepalive_shell: str = DEFAULT_KEEPALIVE_SHELL
    keepalive_cmd: str = DEFAULT_KEEPALIVE_CMD
    start_timeout_s: float | None = 600
    use_init: bool = True
    network: str | None = None  # None: default bridge; "none": no network; else a named network.
    read_only: bool = False
    cap_drop: list[str] = field(default_factory=list)
    security_opt: list[str] = field(default_factory=list)
    pids_limit: int | None = None
    extra_run_args: list[str] = field(default_factory=list)
    apply_resource_limits: bool = True

    def __post_init__(self) -> None:
        if self.start_timeout_s is not None and self.start_timeout_s <= 0:
            raise ValueError("create.start_timeout_s must be > 0")
        if self.pids_limit is not None and self.pids_limit <= 0:
            raise ValueError("create.pids_limit must be > 0")


@dataclass(frozen=True)
class DockerExecConfig:
    default_timeout_s: float | None = 180
    extra_exec_args: list[str] = field(default_factory=list)
    concurrency: int = 32
    # `<shell> -c <cmd>`. None auto-detects bash (needed for conda `source`), else falls back to sh.
    exec_shell: str | None = None

    def __post_init__(self) -> None:
        if self.default_timeout_s is not None and self.default_timeout_s <= 0:
            raise ValueError("exec.default_timeout_s must be > 0")
        if self.concurrency < 1:
            raise ValueError("exec.concurrency must be >= 1")
        if self.exec_shell is not None and not self.exec_shell:
            raise ValueError("exec.exec_shell must be null (auto) or a non-empty shell name/path")


@dataclass(frozen=True)
class DockerProbeConfig:
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
class _DockerContainer:
    name: str
    image: str
    shell: str = "sh"
    env: dict[str, str] = field(default_factory=dict)


def _normalize_image(image: str) -> str:
    prefix = "docker://"
    return image[len(prefix) :] if image.startswith(prefix) else image


def _resource_limit_flags(resources: SandboxResources) -> list[str]:
    flags: list[str] = []
    if resources.cpu is not None:
        flags += ["--cpus", str(resources.cpu)]
    if resources.memory_mib is not None:
        # --memory-swap == --memory disables swap, so the memory limit is a hard cap (not 2x via swap).
        flags += ["--memory", f"{resources.memory_mib}m", "--memory-swap", f"{resources.memory_mib}m"]
    return flags


def _resource_passthrough_flags(resources: SandboxResources) -> list[str]:
    return ["--gpus", str(resources.gpu)] if resources.gpu else []


def _to_sandbox_status(state: str | None) -> SandboxStatus:
    normalized = str(state or "").lower()
    if normalized in {"running", "paused"}:
        return SandboxStatus.RUNNING
    if normalized in {"created", "restarting"}:
        return SandboxStatus.STARTING
    if normalized in {"exited", "removing", "dead"}:
        return SandboxStatus.STOPPED
    return SandboxStatus.UNKNOWN


def _is_runtime_failure(stderr: str) -> bool:
    low = stderr.lower()
    return any(marker in low for marker in DOCKER_RUNTIME_ERROR_MARKERS)


def _is_missing_container(stderr: str) -> bool:
    low = stderr.lower()
    return any(marker in low for marker in DOCKER_MISSING_CONTAINER_MARKERS)


class DockerProvider:
    """Sandbox provider backed by the local Docker CLI / daemon."""

    name = "docker"

    def __init__(
        self,
        *,
        exec: DockerExecConfig | Mapping[str, Any] | None = None,
        create: DockerCreateConfig | Mapping[str, Any] | None = None,
        probe: DockerProbeConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._exec_config = _coerce_config(exec, DockerExecConfig)
        self._create_config = _coerce_config(create, DockerCreateConfig)
        self._probe = _coerce_config(probe, DockerProbeConfig)
        self._binary = _require_docker()
        self._semaphore = asyncio.Semaphore(self._exec_config.concurrency)

    async def _run(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None
    ) -> tuple[int, str, str]:
        """Run a docker CLI command as (return_code, stdout, stderr); SIGKILL the group on timeout."""
        async with self._semaphore:
            proc = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE if stdin is not None else None,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(input=stdin), timeout=timeout_s)
            except asyncio.TimeoutError as e:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                with contextlib.suppress(Exception):
                    await proc.wait()
                raise TimeoutError(f"docker command timed out after {timeout_s:g}s: {_redact_argv(argv)}") from e

            return_code = proc.returncode if proc.returncode is not None else SANDBOX_RUNTIME_RETURN_CODE
            return return_code, stdout_b.decode(errors="replace"), stderr_b.decode(errors="replace")

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        """Start a detached keep-alive container (image ENTRYPOINT overridden) and probe readiness.

        ``spec.ttl_s`` bounds the lifetime (the keep-alive sleeps for it and ``--rm`` self-removes on
        exit). ``spec.provider_options`` may carry ``volumes`` (-> ``-v``) and ``run_args`` (extra run
        flags). A half-created container is force-removed on any failure.
        """
        if spec.image is None:
            raise DockerCreateError("spec.image is required for the docker provider")

        image = _normalize_image(spec.image)
        cfg = self._create_config
        name = CONTAINER_NAME_PREFIX + uuid.uuid4().hex
        volumes = _coerce_str_list(spec.provider_options.get("volumes"), "volumes")
        per_sandbox_args = _coerce_str_list(spec.provider_options.get("run_args"), "run_args")

        argv: list[str] = [self._binary, "run", "-d", "--name", name, "--label", f"{SANDBOX_LABEL}=1"]
        if cfg.use_init:
            argv.append("--init")
        if spec.workdir:
            argv += ["-w", spec.workdir]
        for key, value in spec.env.items():
            argv += ["--env", f"{key}={value}"]
        if cfg.apply_resource_limits:
            argv += _resource_limit_flags(spec.resources)
        argv += _resource_passthrough_flags(spec.resources)
        if cfg.network is not None:
            argv += ["--network", cfg.network]
        if cfg.read_only:
            argv.append("--read-only")
        for cap in cfg.cap_drop:
            argv += ["--cap-drop", cap]
        for opt in cfg.security_opt:
            argv += ["--security-opt", opt]
        if cfg.pids_limit is not None:
            argv += ["--pids-limit", str(cfg.pids_limit)]
        for vol in volumes:
            argv += ["-v", vol]
        argv += list(cfg.extra_run_args) + per_sandbox_args
        # ttl_s (no custom entrypoint): keep-alive sleeps ttl_s, --rm self-removes when it exits.
        enforce_ttl = spec.ttl_s is not None and not spec.entrypoint
        if enforce_ttl:
            argv.append("--rm")
        if spec.entrypoint:
            if spec.ttl_s is not None:
                LOGGER.warning("ttl_s is not enforced when spec.entrypoint is set; it will be ignored.")
            argv += ["--entrypoint", spec.entrypoint[0], image, *spec.entrypoint[1:]]
        else:
            keepalive_cmd = f"sleep {max(1, math.ceil(spec.ttl_s))}" if enforce_ttl else cfg.keepalive_cmd
            argv += ["--entrypoint", cfg.keepalive_shell, image, "-c", keepalive_cmd]

        try:
            code, _out, err = await self._run(argv, timeout_s=cfg.start_timeout_s)
        except TimeoutError as e:
            await self._force_remove(name)
            raise DockerCreateError(f"docker run timed out for image={image!r}: {e}") from e
        if code != 0:
            await self._force_remove(name)
            raise DockerCreateError(f"docker run failed (code={code}) for image={image!r}: {err.strip()}")

        handle = SandboxHandle(
            sandbox_id=name,
            provider_name=self.name,
            raw=_DockerContainer(name=name, image=image, env=dict(spec.env)),
        )
        try:
            await self._verify_created_handle(handle)  # readiness via default sh (printf works there)
            handle.raw.shell = await self._resolve_shell(name)  # resolve real shell once exec is confirmed live
        except Exception:
            await self._cleanup_failed_create_handle(handle)
            raise
        return handle

    async def _resolve_shell(self, name: str) -> str:
        """Configured exec shell, else bash when the image has it (for conda `source`), else sh."""
        if self._exec_config.exec_shell:
            return self._exec_config.exec_shell
        with contextlib.suppress(Exception):
            code, _out, _err = await self._run(
                [self._binary, "exec", name, "sh", "-c", "command -v bash"],
                timeout_s=self._exec_config.default_timeout_s,
            )
            if code == 0:
                return "bash"
        return "sh"

    async def _force_remove(self, name: str) -> None:
        with contextlib.suppress(Exception):
            await self._run([self._binary, "rm", "-f", name], timeout_s=self._exec_config.default_timeout_s)

    async def _cleanup_failed_create_handle(self, handle: SandboxHandle) -> None:
        await self._force_remove(handle.raw.name)

    async def _verify_created_handle(self, handle: SandboxHandle) -> None:
        """Poll the readiness probe until it passes ``stable_count`` times or the deadline elapses."""
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
                    raise DockerCreateVerificationError(
                        f"sandbox {handle.sandbox_id!r} failed readiness probe: {last_detail}"
                    )
            if deadline is not None and loop.time() >= deadline:
                raise DockerCreateVerificationError(
                    f"sandbox {handle.sandbox_id!r} did not pass readiness probe within {probe.deadline_s:g}s: {last_detail}"
                )
            if probe.stable_delay_s > 0:
                await asyncio.sleep(probe.stable_delay_s)

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
        """Run ``<shell> -c <command>`` via ``docker exec``; never raises for command failure.

        ``user`` maps to ``--user`` (root/0 -> 0). A timeout kills the local docker client only; the
        in-container process is reaped when the sandbox is closed.
        """
        inst = handle.raw
        flags: list[str] = []
        if stdin is not None:
            flags.append("-i")
        if cwd is not None:
            flags += ["-w", cwd]
        merged_env = dict(getattr(inst, "env", {}))
        if env:
            merged_env.update(env)
        for key, value in merged_env.items():
            flags += ["--env", f"{key}={value}"]
        if user is not None:
            flags += ["--user", "0" if user == "root" or user == 0 else str(user)]
        flags += list(self._exec_config.extra_exec_args)

        argv = [self._binary, "exec", *flags, inst.name, inst.shell, "-c", command]
        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s
        try:
            code, out, err = await self._run(argv, timeout_s=effective_timeout, stdin=stdin)
        except TimeoutError as e:
            return SandboxExecResult(
                stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
            )
        if code != 0 and _is_runtime_failure(err):
            return SandboxExecResult(
                stdout=out, stderr=err, return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="sandbox"
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=code, error_type=None)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        """Upload one host file (creates the parent dir; the file lands owned by root)."""
        inst = handle.raw
        parent = posixpath.dirname(target_path)
        if parent:
            mk = await self.exec(handle, f"mkdir -p {shlex.quote(parent)}", user="root")
            if mk.return_code != 0:
                raise RuntimeError(f"docker upload to {target_path!r} failed (mkdir parent): {mk.stderr}")
        code, _out, err = await self._run(
            [self._binary, "cp", str(source_path), f"{inst.name}:{target_path}"],
            timeout_s=self._exec_config.default_timeout_s,
        )
        if code != 0:
            raise RuntimeError(f"docker cp upload to {target_path!r} failed: {err.strip()}")

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        """Download one container file to the host."""
        inst = handle.raw
        target_path.parent.mkdir(parents=True, exist_ok=True)
        code, _out, err = await self._run(
            [self._binary, "cp", f"{inst.name}:{source_path}", str(target_path)],
            timeout_s=self._exec_config.default_timeout_s,
        )
        if code != 0:
            raise RuntimeError(f"docker cp download from {source_path!r} failed: {err.strip()}")

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        """Container status via ``docker inspect`` (missing -> STOPPED; error/timeout -> UNKNOWN)."""
        inst = handle.raw
        try:
            code, out, err = await self._run(
                [self._binary, "inspect", "-f", "{{.State.Status}}", inst.name],
                timeout_s=self._exec_config.default_timeout_s,
            )
        except TimeoutError:
            return SandboxStatus.UNKNOWN
        if code != 0:
            return SandboxStatus.STOPPED if _is_missing_container(err) else SandboxStatus.UNKNOWN
        return _to_sandbox_status(out.strip())

    async def close(self, handle: SandboxHandle) -> None:
        """Force-remove the container (already-gone counts as success)."""
        inst = handle.raw
        code, _out, err = await self._run(
            [self._binary, "rm", "-f", inst.name],
            timeout_s=self._exec_config.default_timeout_s,
        )
        if code != 0 and not _is_missing_container(err):
            raise RuntimeError(f"docker rm -f failed (code={code}) for {inst.name!r}: {err.strip()}")

    async def aclose(self) -> None:
        return None
