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
import ipaddress
import json
import logging
import math
import os
import posixpath
import re
import shlex
import shutil
import signal
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxPtyError,
    SandboxPtySession,
    SandboxPtySpec,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.docker import pty as docker_pty


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
DOCKER_CREATE_TRANSPORT_ERROR_MARKERS = (
    "cannot connect to the docker daemon",
    "error during connect",
    "connection refused",
    "connection reset by peer",
    "broken pipe",
    "unexpected eof",
    "i/o timeout",
    "client timeout exceeded",
)
DOCKER_NAME_CONFLICT_MARKERS = (
    "the container name",
    "is already in use by container",
)
DOCKER_RECONCILE_FORMAT = '{{index .Config.Labels "nemo-gym.sandbox"}}\t{{.Config.Image}}\t{{.State.Status}}'
# The named volume that holds shared storage (Compose volumes) unless ``create.shared_volume`` names another, and
# the label it is created with, so an operator can find it after a run that did not clean up.
DEFAULT_SHARED_VOLUME = "nemo-gym-shared-storage"
SHARED_STORAGE_LABEL = "nemo-gym.shared-storage"
# ``volume-subpath`` mounts arrived in Docker Engine 26.0 (API 1.45). An older daemon ignores the field and would
# mount the whole shared volume, every collection's directories with it, so shared storage refuses it.
MIN_SUBPATH_API_VERSION = (1, 45)
# Docker's own rule for volume names.
_VOLUME_NAME = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_.-]+")
# Networks on which a container has no address of its own for peers to reach.
_NETWORKS_WITHOUT_PEERS = ("none", "host")
# A name the Compose adapter admits for a service or alias: one token of an /etc/hosts line, never an option or a
# second field.
_HOSTS_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


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
    publish_host: str = "127.0.0.1"
    transport_retries: int = 2
    transport_retry_delay_s: float = 1.0
    # The named volume that backs shared storage (see ``DockerProvider.shared_volume_options``); created on first use.
    shared_volume: str = DEFAULT_SHARED_VOLUME

    def __post_init__(self) -> None:
        if self.start_timeout_s is not None and self.start_timeout_s <= 0:
            raise ValueError("create.start_timeout_s must be > 0")
        if not _VOLUME_NAME.fullmatch(self.shared_volume):
            raise ValueError("create.shared_volume must be a Docker volume name ([a-zA-Z0-9][a-zA-Z0-9_.-]+)")
        if self.pids_limit is not None and self.pids_limit <= 0:
            raise ValueError("create.pids_limit must be > 0")
        if self.transport_retries < 0:
            raise ValueError("create.transport_retries must be >= 0")
        if self.transport_retry_delay_s < 0:
            raise ValueError("create.transport_retry_delay_s must be >= 0")
        try:
            ipaddress.ip_address(self.publish_host)
        except ValueError as exc:
            raise ValueError("create.publish_host must be an IPv4 or IPv6 address") from exc


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
    published_ports: tuple[int, ...] = ()


@dataclass(frozen=True, kw_only=True)
class DockerVolumeMount:
    """A mount of a Docker named volume, or of one directory inside it, as ``docker run --mount`` takes it.

    ``DockerProvider.shared_volume_options`` returns these in ``provider_options["volumes"]``, beside any ``-v``
    strings. Docker's own copy of the image's files into an empty volume is always off (``volume-nocopy``): the
    Compose adapter seeds a volume itself, once per collection.
    """

    volume: str
    target: str
    subpath: str | None = None
    read_only: bool = False

    def mount_arg(self) -> str:
        """The ``--mount`` value."""
        fields = ["type=volume", f"source={self.volume}", f"target={self.target}"]
        if self.subpath is not None:
            fields.append(f"volume-subpath={self.subpath}")
        if self.read_only:
            fields.append("readonly")
        fields.append("volume-nocopy")
        return ",".join(fields)


def _volume_args(value: Any) -> tuple[list[str], list[str]]:
    """``provider_options["volumes"]`` as ``docker run`` flags, with the named volumes its mounts use.

    A :class:`DockerVolumeMount` becomes ``--mount``; anything else is a ``src:dst[:opts]`` string for ``-v``.
    """
    if value is None:
        entries = []
    elif isinstance(value, (str, DockerVolumeMount)):
        entries = [value]
    elif isinstance(value, (list, tuple)):
        entries = list(value)
    else:
        raise DockerCreateError(f"provider_options['volumes'] must be a string or list, got {type(value).__name__}")
    args: list[str] = []
    volumes: list[str] = []
    for entry in entries:
        if isinstance(entry, DockerVolumeMount):
            args += ["--mount", entry.mount_arg()]
            if entry.volume not in volumes:
                volumes.append(entry.volume)
        else:
            args += ["-v", str(entry)]
    return args, volumes


def _mount_path(path: str, what: str) -> str:
    """``path`` for one field of a ``--mount`` value, which the docker CLI reads as comma-separated CSV."""
    if any(character in path for character in ',"\n\r\x00'):
        raise ValueError(f"Shared storage {what} cannot contain a comma, quote, newline or NUL: {path!r}")
    return path


def _api_version(text: str) -> tuple[int, ...] | None:
    """``1.47`` as ``(1, 47)``; ``None`` for anything else."""
    parts = text.strip().split(".")
    return tuple(int(part) for part in parts) if all(part.isdecimal() for part in parts) else None


def _metadata_label_args(metadata: Mapping[str, Any]) -> list[str]:
    """``spec.metadata`` as ``--label key=value`` flags.

    The ownership label ``nemo-gym.sandbox``, which reconciliation and the reap filter trust, is never taken from
    metadata. Neither is a key the docker CLI cannot carry as one label name: it splits ``--label`` on the first
    ``=`` (``nemo-gym.sandbox=0`` as a key would replace the ownership label) and refuses an empty name or one with
    whitespace. Such keys are logged and skipped, as all metadata was before it became labels.
    """
    args: list[str] = []
    for key, value in metadata.items():
        if key == SANDBOX_LABEL or not key or "=" in key or any(character.isspace() for character in key):
            LOGGER.warning("docker: metadata key %r is not used as a container label", key)
            continue
        args += ["--label", f"{key}={value}"]
    return args


def _normalize_image(image: str) -> str:
    prefix = "docker://"
    return image[len(prefix) :] if image.startswith(prefix) else image


async def _finish_despite_cancellation[T](task: asyncio.Future[T]) -> T:
    """``task``'s result, waited for however often the caller is cancelled meanwhile.

    Once ``task`` is done, a cancellation the caller received while waiting is raised in place of the result.
    """
    cancellation: asyncio.CancelledError | None = None
    while not task.done():
        try:
            await asyncio.wait((task,))
        except asyncio.CancelledError as exc:
            cancellation = exc
    if cancellation is not None:
        if not task.cancelled():
            task.exception()  # The cancellation takes the place of the task's outcome, an error included.
        raise cancellation
    return task.result()


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


def _publish_arg(host: str, port: int) -> str:
    formatted_host = f"[{host}]" if ":" in host else host
    return f"{formatted_host}::{port}/tcp"


def _parse_port_binding(binding: str) -> tuple[str, int]:
    """Parse one ``docker port`` binding such as ``127.0.0.1:49153``."""

    value = binding.strip()
    if not value or ":" not in value:
        raise ValueError(f"invalid Docker port binding: {binding!r}")
    host, raw_port = value.rsplit(":", 1)
    host = host.strip("[]")
    if not host:
        host = "0.0.0.0"
    port = int(raw_port)
    if port < 1 or port > 65535:
        raise ValueError(f"invalid Docker host port: {port}")
    return host, port


def _http_endpoint(host: str, port: int) -> str:
    formatted_host = f"[{host}]" if ":" in host else host
    return f"http://{formatted_host}:{port}"


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


def _is_create_transport_failure(stderr: str) -> bool:
    low = stderr.lower()
    return any(marker in low for marker in DOCKER_CREATE_TRANSPORT_ERROR_MARKERS)


def _is_name_conflict(stderr: str) -> bool:
    low = stderr.lower()
    return all(marker in low for marker in DOCKER_NAME_CONFLICT_MARKERS)


def _user_flag(user: str | int) -> str:
    """``--user`` value: root/0 becomes 0, anything else is passed as written (``name``, ``uid``, ``uid:gid``)."""
    return "0" if user == "root" or user == 0 else str(user)


class DockerProvider:
    """Sandbox provider backed by the local Docker CLI / daemon."""

    name = "docker"
    # Compose service options this provider applies, each as the matching ``docker run`` flag.
    SERVICE_OPTION_FIELDS = frozenset({"tmpfs", "ulimits", "sysctls", "security_opt"})
    # ``docker exec --user`` picks each command's user from outside the container, root included, whatever user
    # the image runs as (see ``SupportsSandboxExecAsAnyUser``).
    EXEC_AS_ANY_USER = True

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
        # Live PTY sessions by container name; closed with their container and at provider shutdown.
        self._pty_sessions: dict[str, set[docker_pty.DockerPtySession]] = {}
        # Named volumes this provider has checked or created for shared-storage mounts (see _ensure_volumes).
        self._ready_volumes: set[str] = set()
        self._volume_lock = asyncio.Lock()

    def service_options(
        self,
        *,
        tmpfs: tuple[str, ...] = (),
        ulimits: Mapping[str, Any] | None = None,
        sysctls: Mapping[str, str] | None = None,
        security_opt: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Compose service options as ``run_args``: ``--tmpfs``, ``--ulimit``, ``--sysctl``, ``--security-opt``."""
        args: list[str] = []
        for mount in tmpfs:
            args += ["--tmpfs", str(mount)]
        for name, limit in (ulimits or {}).items():
            if isinstance(limit, Mapping):
                if set(limit) != {"soft", "hard"}:
                    raise ValueError(f"ulimit {name!r} needs both soft and hard values")
                args += ["--ulimit", f"{name}={limit['soft']}:{limit['hard']}"]
            else:
                args += ["--ulimit", f"{name}={limit}"]
        for key, value in (sysctls or {}).items():
            args += ["--sysctl", f"{key}={value}"]
        for option in security_opt:
            args += ["--security-opt", str(option)]
        return {"run_args": args} if args else {}

    def pull_policy_options(self, policy: str) -> dict[str, Any]:
        """Compose's ``pull_policy`` as ``docker run --pull``; only ``always``, ``missing`` and ``never`` exist."""
        if policy not in ("always", "missing", "never"):
            raise ValueError(f"pull policy must be always, missing or never, got {policy!r}")
        return {"run_args": ["--pull", policy]}

    async def _run(
        self,
        argv: list[str],
        *,
        timeout_s: float | None,
        stdin: bytes | None = None,
        spawned: asyncio.Event | None = None,
    ) -> tuple[int, str, str]:
        """Run a docker CLI command as (return_code, stdout, stderr); SIGKILL the group on timeout.

        ``spawned``, when given, is set once the command holds a command slot and is about to start. From then on a
        cancellation propagates only after the command has answered or timed out, because a command such as
        ``docker run`` can still act after its caller stopped waiting. A wait for a slot is cancelled at once.
        """
        async with self._semaphore:
            command = self._run_in_slot(argv, timeout_s=timeout_s, stdin=stdin)
            if spawned is None:
                return await command
            spawned.set()
            return await _finish_despite_cancellation(asyncio.ensure_future(command))

    async def _run_in_slot(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None
    ) -> tuple[int, str, str]:
        """``_run``'s command, once it holds a command slot."""
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
        exit). ``spec.provider_options`` may carry ``volumes`` (strings -> ``-v``, shared-storage mounts
        -> ``--mount``) and ``run_args`` (extra run flags). ``spec.metadata`` entries become container labels
        beside ``nemo-gym.sandbox=1``. A half-created container is force-removed on any failure. A cancelled create
        leaves no container either: once its ``docker run`` started, that run answers first (up to
        ``create.start_timeout_s``), and the container is then removed, once inspect shows it is this provider's,
        before the cancellation propagates, however often it repeats. A create cancelled while it waits for a command
        slot ends at once, without a Docker call.
        """
        if spec.image is None:
            raise DockerCreateError("spec.image is required for the docker provider")

        image = _normalize_image(spec.image)
        cfg = self._create_config
        name = CONTAINER_NAME_PREFIX + uuid.uuid4().hex
        volume_args, named_volumes = _volume_args(spec.provider_options.get("volumes"))
        per_sandbox_args = _coerce_str_list(spec.provider_options.get("run_args"), "run_args")
        if named_volumes:
            await self._ensure_volumes(named_volumes)

        argv: list[str] = [self._binary, "run", "-d", "--name", name, "--label", f"{SANDBOX_LABEL}=1"]
        argv += _metadata_label_args(spec.metadata)
        if cfg.use_init:
            argv.append("--init")
        if spec.workdir:
            argv += ["-w", spec.workdir]
        # spec.user is not passed to `docker run`: in overlay mode the user may not exist in the base image
        # yet. The sandbox facade applies it to every exec instead (see AsyncSandbox.default_user).
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
        for port in spec.ports:
            argv += ["--publish", _publish_arg(cfg.publish_host, port)]
        argv += volume_args
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

        reached_docker = asyncio.Event()
        try:
            await self._run_create_with_reconciliation(argv, name=name, image=image, spawned=reached_docker)
        except asyncio.CancelledError:
            # A caller's deadline (a Compose collection's startup timeout) expired while the container was being
            # created. No caller will hold its handle, and a Compose service's keep-alive has no TTL to end it. A
            # create cancelled before its first `docker run` started has nothing to remove.
            if reached_docker.is_set():
                await _finish_despite_cancellation(
                    asyncio.ensure_future(self._remove_cancelled_create(name=name, image=image))
                )
            raise

        handle = SandboxHandle(
            sandbox_id=name,
            provider_name=self.name,
            raw=_DockerContainer(
                name=name,
                image=image,
                env=dict(spec.env),
                published_ports=tuple(spec.ports),
            ),
        )
        try:
            await self._verify_created_handle(handle)  # readiness via default sh (printf works there)
            handle.raw.shell = await self._resolve_shell(name)  # resolve real shell once exec is confirmed live
        except BaseException:
            # Also on cancellation (a caller's deadline expiring during the probe): the container is running and no
            # caller holds its handle yet, so nothing else would remove it. Further cancellations wait for the
            # removal before they propagate.
            await _finish_despite_cancellation(asyncio.ensure_future(self._cleanup_failed_create_handle(handle)))
            raise
        return handle

    async def _remove_cancelled_create(self, *, name: str, image: str) -> None:
        """Remove the container a cancelled create left under ``name``, if inspect shows it is this provider's.

        The ownership and image check is reconciliation's: a same-name container without them is left alone. An
        error is logged rather than raised, so that the cancellation, not the error, reaches the caller.
        """
        try:
            state = await self._inspect_owned_container(name=name, image=image)
        except Exception as exc:
            LOGGER.warning("Leaving Docker container %s after a cancelled create: %s", name, exc)
            return
        if state is not None:
            await self._force_remove(name)

    async def _run_create_with_reconciliation(
        self, argv: list[str], *, name: str, image: str, spawned: asyncio.Event
    ) -> None:
        """Run ``docker run`` without leaking a container when its response is lost.

        A fixed, random name makes retry idempotent. After a timeout, transport
        error, or same-name conflict, inspect that exact name before retrying.
        Reuse it only when both the Sandbox ownership label and requested image
        match; never remove an unrelated collision.

        ``spawned`` is set once the first ``docker run`` starts. A cancellation that arrives after that, during an
        attempt's ``docker run``, propagates only once the client has answered or timed out: until then it could still
        create the container, after any removal the caller issued. A wait for a command slot is cancelled at once.
        """
        cfg = self._create_config
        last_detail = "docker run did not complete"
        last_cause: BaseException | None = None
        for attempt in range(cfg.transport_retries + 1):
            try:
                code, _out, err = await self._run(argv, timeout_s=cfg.start_timeout_s, spawned=spawned)
            except TimeoutError as exc:
                last_cause = exc
                last_detail = f"docker run timed out for image={image!r}: {exc}"
            else:
                if code == 0:
                    return
                last_detail = f"docker run failed (code={code}) for image={image!r}: {err.strip()}"
                if not (_is_create_transport_failure(err) or _is_name_conflict(err)):
                    await self._force_remove(name)
                    raise DockerCreateError(last_detail)

            if await self._reconcile_created_container(name=name, image=image):
                LOGGER.warning(
                    "Recovered Docker sandbox %s after an ambiguous create response (attempt %d/%d).",
                    name,
                    attempt + 1,
                    cfg.transport_retries + 1,
                )
                return

            if attempt >= cfg.transport_retries:
                await self._force_remove(name)
                error = DockerCreateError(
                    f"{last_detail}; exact-name reconciliation did not find an owned container "
                    f"after {cfg.transport_retries + 1} attempt(s)"
                )
                if last_cause is not None:
                    raise error from last_cause
                raise error

            delay = cfg.transport_retry_delay_s * (2**attempt)
            LOGGER.warning(
                "Retrying Docker sandbox create after an ambiguous transport result: "
                "name=%s attempt=%d/%d delay=%.3fs",
                name,
                attempt + 1,
                cfg.transport_retries + 1,
                delay,
            )
            if delay > 0:
                await asyncio.sleep(delay)

    async def _reconcile_created_container(self, *, name: str, image: str) -> bool:
        """Recover the exact Sandbox-owned container after an ambiguous ``docker run``."""
        state = await self._inspect_owned_container(name=name, image=image)
        if state is None:
            return False
        if state in {"running", "restarting"}:
            LOGGER.info("Reconciled Docker sandbox name=%s image=%s state=%s", name, image, state)
            return True
        if state == "created":
            if await self._start_reconciled_container(name=name, image=image):
                return True
            # Ownership was authenticated above. Remove the unusable partial
            # create so the fixed-name retry can make a fresh attempt.
            await self._force_remove(name)
            return False

        LOGGER.warning(
            "Removing owned Docker sandbox %s after reconciliation found unusable state=%s",
            name,
            state,
        )
        await self._force_remove(name)
        return False

    async def _inspect_owned_container(self, *, name: str, image: str) -> str | None:
        """Return state for an exact owned container, ``None`` when it cannot be resolved."""
        try:
            code, out, err = await self._run(
                [self._binary, "inspect", "--format", DOCKER_RECONCILE_FORMAT, name],
                timeout_s=self._exec_config.default_timeout_s,
            )
        except TimeoutError:
            return None
        if code != 0:
            if not (_is_missing_container(err) or _is_create_transport_failure(err)):
                LOGGER.warning("Docker reconciliation inspect failed for %s: %s", name, err.strip())
            return None

        fields = out.strip().split("\t")
        if len(fields) != 3:
            raise DockerCreateError(
                f"refusing to reconcile Docker container {name!r}: malformed inspect output {out!r}"
            )
        label, actual_image, state = fields
        if label != "1" or actual_image != image:
            raise DockerCreateError(
                f"refusing to reconcile Docker container {name!r}: ownership/image mismatch "
                f"(label={label!r}, image={actual_image!r}, expected_image={image!r})"
            )
        return state

    async def _start_reconciled_container(self, *, name: str, image: str) -> bool:
        """Finish a ``docker run`` that created its container but lost the start response."""
        cfg = self._create_config
        for attempt in range(cfg.transport_retries + 1):
            start_error: str | None = None
            try:
                code, _out, err = await self._run(
                    [self._binary, "start", name],
                    timeout_s=cfg.start_timeout_s,
                )
            except TimeoutError as exc:
                code = None
                err = ""
                start_error = f"docker start timed out: {exc}"
            else:
                if code != 0:
                    start_error = f"docker start failed (code={code}): {err.strip()}"

            state = await self._inspect_owned_container(name=name, image=image)
            if state in {"running", "restarting"}:
                LOGGER.warning(
                    "Recovered Docker sandbox %s by finishing its start (attempt %d/%d).",
                    name,
                    attempt + 1,
                    cfg.transport_retries + 1,
                )
                return True
            if state not in {None, "created"}:
                LOGGER.warning(
                    "Docker sandbox %s entered unusable state=%s while reconciling start",
                    name,
                    state,
                )
                return False
            if code not in {None, 0} and not _is_create_transport_failure(err):
                LOGGER.warning("%s", start_error)
                return False
            if attempt >= cfg.transport_retries:
                if start_error is not None:
                    LOGGER.warning("%s", start_error)
                return False

            delay = cfg.transport_retry_delay_s * (2**attempt)
            LOGGER.warning(
                "Retrying Docker sandbox start after an ambiguous transport result: name=%s attempt=%d/%d delay=%.3fs",
                name,
                attempt + 1,
                cfg.transport_retries + 1,
                delay,
            )
            if delay > 0:
                await asyncio.sleep(delay)
        return False

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Return a descriptor for reattaching to this container by id.

        Docker recovers image and shell from the running container on connect,
        and ``docker exec`` runs inside the container's own environment, so the
        id alone is enough; ``AsyncSandbox.serialize()`` layers the declared
        ports on top. ``scope`` is unused by the Docker provider.
        """
        return {"sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Create a handle for an existing running container, rediscovering its
        image and shell via the Docker CLI.

        ``sandbox_id`` is required. ``AsyncSandbox.serialize()`` adds a
        top-level ``ports`` key from the original spec; read it back here so
        ``endpoint()`` still works after reconnecting.
        """
        name = descriptor.get("sandbox_id")
        if not name:
            raise DockerCreateError("docker connect requires a sandbox_id")

        code, out, err = await self._run(
            [self._binary, "inspect", "--format", "{{.State.Running}}\t{{.Config.Image}}", str(name)],
            timeout_s=self._exec_config.default_timeout_s,
        )
        if code != 0:
            raise DockerCreateError(f"cannot connect to container {name!r}: {err.strip() or out.strip()}")
        running, _, image = out.strip().partition("\t")
        if running != "true":
            raise DockerCreateError(f"cannot connect to container {name!r}: it is not running")

        return SandboxHandle(
            sandbox_id=str(name),
            provider_name=self.name,
            raw=_DockerContainer(
                name=str(name),
                image=image,
                shell=await self._resolve_shell(str(name)),
                published_ports=tuple(descriptor.get("ports") or ()),
            ),
        )

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
        argv = self._exec_argv(handle, command, mode=["-i"] if stdin is not None else [], cwd=cwd, env=env, user=user)
        effective_timeout = timeout_s if timeout_s is not None else self._exec_config.default_timeout_s
        return await self._run_exec(argv, timeout_s=effective_timeout, stdin=stdin)

    async def exec_detached(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        """Start ``<shell> -c <command>`` with ``docker exec -d`` and return once Docker has started it.

        Only that short ``docker exec -d`` call takes an ``exec.concurrency`` slot (bounded by
        ``exec.default_timeout_s``); the command then runs on in the container until it ends or the container is
        removed, and its output is discarded unless it redirects it. ``cwd``, ``env`` and ``user`` map as in
        :meth:`exec`. A command Docker cannot start (a missing user, a stopped container) fails the result.
        """
        argv = self._exec_argv(handle, command, mode=["-d"], cwd=cwd, env=env, user=user)
        return await self._run_exec(argv, timeout_s=self._exec_config.default_timeout_s)

    def _exec_argv(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        mode: list[str],
        cwd: str | None,
        env: dict[str, str] | None,
        user: str | int | None,
    ) -> list[str]:
        """The ``docker exec`` command line of ``command``: ``mode`` flags (``-i``, ``-d``), then the working
        directory, the sandbox's and the call's environment, the user and ``exec.extra_exec_args``."""
        inst = handle.raw
        flags: list[str] = list(mode)
        if cwd is not None:
            flags += ["-w", cwd]
        merged_env = dict(getattr(inst, "env", {}))
        if env:
            merged_env.update(env)
        for key, value in merged_env.items():
            flags += ["--env", f"{key}={value}"]
        if user is not None:
            flags += ["--user", _user_flag(user)]
        flags += list(self._exec_config.extra_exec_args)
        return [self._binary, "exec", *flags, inst.name, inst.shell, "-c", command]

    async def _run_exec(
        self, argv: list[str], *, timeout_s: float | None, stdin: bytes | None = None
    ) -> SandboxExecResult:
        """Run a ``docker exec`` command line as a result: a timeout or a daemon-side failure is reported, not
        raised."""
        try:
            code, out, err = await self._run(argv, timeout_s=timeout_s, stdin=stdin)
        except TimeoutError as e:
            return SandboxExecResult(
                stdout=None, stderr=str(e), return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="timeout"
            )
        if code != 0 and _is_runtime_failure(err):
            return SandboxExecResult(
                stdout=out, stderr=err, return_code=SANDBOX_RUNTIME_RETURN_CODE, error_type="sandbox"
            )
        return SandboxExecResult(stdout=out, stderr=err, return_code=code, error_type=None)

    def _pty_argv(self, handle: SandboxHandle, spec: SandboxPtySpec, session_id: str) -> list[str]:
        """The ``docker exec`` command line of one PTY session: interactive, with a terminal in PTY mode.

        The session's marker (``NEMO_GYM_PTY_SESSION=<session id>``) goes last among the ``--env`` flags so no
        spec env can override it: it is how the recorded pid is known to be this session's process.
        """
        inst = handle.raw
        flags: list[str] = ["-i"]
        if spec.pty:
            flags.append("-t")
        if spec.cwd is not None:
            flags += ["-w", spec.cwd]
        merged_env = dict(getattr(inst, "env", {}))
        if spec.env:
            merged_env.update(spec.env)
        merged_env.pop(docker_pty.SESSION_ENV, None)
        for key, value in merged_env.items():
            flags += ["--env", f"{key}={value}"]
        flags += ["--env", docker_pty.session_marker(session_id)]
        if spec.user is not None:
            flags += ["--user", _user_flag(spec.user)]
        flags += list(self._exec_config.extra_exec_args)
        launcher = docker_pty.launcher_argv(
            inst.shell, docker_pty.pid_file_for(session_id), docker_pty.fallback_pid_file_for(session_id), spec.command
        )
        return [self._binary, "exec", *flags, inst.name, *launcher]

    async def create_pty(self, handle: SandboxHandle, spec: SandboxPtySpec) -> SandboxPtySession:
        """Open an interactive terminal in the container: one ``docker exec -it`` client under a host pty.

        ``cwd``, ``env`` and ``user`` map to the exec flags, ``rows``/``cols`` size the terminal before the
        client starts, ``pty=False`` runs the client on pipes. The client process is not counted against
        ``exec.concurrency``: it lives as long as the session. The session's pid is recorded under ``/tmp`` in the
        container (under ``/dev/shm`` when ``/tmp`` refuses the write, as under ``create.read_only: true``; a session
        that can record it nowhere ends at once with a message instead of running unsignallable), read as the
        session's own user and signalled as root (no ``CAP_SYS_PTRACE`` needed). This returns as soon as the client
        is spawned: ``send_signal()``/``close()`` wait up to two seconds for the record when they come first, refuse
        at once a record that is not a regular file, and read it under a timeout of their own
        (``PID_READ_TIMEOUT_S``) rather than ``exec.default_timeout_s``. A client that exits without having started
        the program (the daemon refused the exec: the container paused or gone, the daemon unreachable, told by the
        client's stderr as ``exec()`` tells a runtime failure) ends the session with ``SandboxPtyError`` from
        ``wait_exit()`` and ``read()`` rather than as the program's exit; a container closed while the session was
        being opened has it abandoned and ``SandboxPtyError`` raised here. See :mod:`nemo_gym.sandbox.providers.docker.pty`.
        """
        if isinstance(spec.user, bool):
            raise ValueError("Docker PTY sessions take a user name or uid, not a bool")
        session_id = uuid.uuid4().hex

        async def run_in_container(
            command: str, *, as_session_user: bool = False, timeout_s: float | None = None
        ) -> SandboxExecResult:
            # The pid read runs as the session's user (``spec.user``, or the container's default user like the
            # session itself): ``/proc/<pid>/environ`` is readable by the process's own uid, not by a root without
            # ``CAP_SYS_PTRACE``, which Docker's default capabilities omit. ``kill`` and the cleanup run as root so
            # they reach the process whatever user the session runs as. ``timeout_s`` caps the exec's timeout below
            # ``exec.default_timeout_s`` (the pid read's own bound), never above it.
            user = spec.user if as_session_user else "root"
            limit = self._exec_config.default_timeout_s
            if timeout_s is not None:
                limit = timeout_s if limit is None else min(limit, timeout_s)
            return await self.exec(handle, command, user=user, timeout_s=limit)

        name = handle.raw.name
        # The container's session set, taken before the awaits below: ``close(handle)`` pops it, so the set doubles as
        # the token that tells, once the spawn returns, whether the container is still there (see below).
        sessions = self._pty_sessions.setdefault(name, set())
        # Retire the sessions of this container whose process exited, and release them: ``closed`` is true once the
        # process exited, but the host pty master (and the pid record in the container) stays until ``close()``,
        # which a caller that only read the session to EOF never issued; discarded unreleased they would hold their
        # descriptors until the container's close or ``aclose()``, out of reach of both. Discarded before the releases
        # await: ``close(handle)`` may iterate the set meanwhile.
        stale_sessions = [s for s in sessions if s.closed]
        sessions.difference_update(stale_sessions)
        for stale in stale_sessions:
            await self._release_pty_session(stale, name, abandon=False)
        session = await self._spawn_pty(
            self._pty_argv(handle, spec, session_id),
            use_pty=spec.pty,
            rows=spec.rows,
            cols=spec.cols,
            session_id=session_id,
            run_in_container=run_in_container,
            pid_file=docker_pty.pid_file_for(session_id),
            fallback_pid_file=docker_pty.fallback_pid_file_for(session_id),
            # A client that exits with the daemon's refusal on its stderr (the lines ``exec()`` classifies as a
            # runtime failure) never ran the program: the session ends with ``SandboxPtyError``, not that code.
            runtime_failure=_is_runtime_failure,
        )
        # ``close(handle)`` while this awaited (the sandbox torn down as a harness opens a session) popped the set and
        # abandoned the sessions it held, and ``aclose()`` may have followed at once, as the facade's ``stop()`` runs
        # the two back to back: a session added to the popped set would be out of ``aclose()``'s reach, one added to
        # the provider now would sit in a provider nothing closes again, its master and client leaking either way.
        # The container is gone, so this session goes the way of its siblings and the caller is told.
        if self._pty_sessions.get(name) is not sessions:
            await session.abandon()
            raise SandboxPtyError(f"sandbox {name!r} closed while its PTY session was being opened")
        sessions.add(session)
        return session

    # The seam tests replace to run a local program under the same terminal plumbing.
    _spawn_pty = staticmethod(docker_pty.spawn)

    async def _release_pty_sessions(self, name: str, *, abandon: bool) -> None:
        """Close (or, when the container goes away, abandon) the live sessions of container ``name``."""
        for session in self._pty_sessions.pop(name, set()):
            await self._release_pty_session(session, name, abandon=abandon)

    async def _release_pty_session(self, session: SandboxPtySession, name: str, *, abandon: bool) -> None:
        """One session of :meth:`_release_pty_sessions`; a failure is logged, never raised into the caller."""
        try:
            if abandon:
                await session.abandon()
            else:
                await session.close()
        except Exception:
            LOGGER.warning("Failed to close PTY session %s of %r", session.session_id, name, exc_info=True)

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

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve a dynamically-published TCP port through the Docker daemon."""

        inst = handle.raw
        if port not in inst.published_ports:
            raise ValueError(
                f"Sandbox port {port} was not declared in SandboxSpec.ports; "
                f"declared ports: {list(inst.published_ports)!r}"
            )
        try:
            code, out, err = await self._run(
                [self._binary, "port", inst.name, f"{port}/tcp"],
                timeout_s=self._exec_config.default_timeout_s,
            )
        except TimeoutError as exc:
            raise TimeoutError(f"Timed out resolving Docker sandbox port {port}") from exc
        if code != 0:
            raise RuntimeError(f"docker port failed (code={code}) for {inst.name!r} port {port}: {err.strip()}")

        bindings = [line.strip() for line in out.splitlines() if line.strip()]
        if not bindings:
            raise RuntimeError(f"Docker returned no host binding for sandbox port {port}")
        preferred = next((binding for binding in bindings if binding.startswith("127.0.0.1:")), bindings[0])
        try:
            host, host_port = _parse_port_binding(preferred)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"Docker returned an invalid host binding for sandbox port {port}") from exc
        if host in {"0.0.0.0", "::"}:
            host = self._create_config.publish_host
        return SandboxEndpoint(endpoint=_http_endpoint(host, host_port))

    def shared_volume_metadata(self) -> dict[str, str]:
        """No placement settings: every sandbox runs on this provider's one daemon, next to the volume."""
        return {}

    def shared_volume_options(self, source: str | None, target: str, *, read_only: bool = False) -> dict[str, Any]:
        """Mount directory ``source`` of the shared volume (``create.shared_volume``) at ``target``.

        ``source`` is relative to the volume's root and must exist when the sandbox is created (Docker's
        ``volume-subpath``); ``None`` mounts the whole volume, as the Compose adapter does for the helper that
        creates and removes those directories. The volume is created on first use, labeled
        ``nemo-gym.shared-storage=1``, and outlives the provider: it is the root, collections own directories in it.
        """
        if source is not None:
            path = PurePosixPath(source)
            if not path.parts or path.is_absolute() or ".." in source.split("/") or "\\" in source:
                raise ValueError("Shared storage source must be a safe path relative to the shared volume")
            source = _mount_path(str(path), "source")
        if not PurePosixPath(target).is_absolute() or ".." in target.split("/"):
            raise ValueError("Shared storage target must be an absolute container path without traversal")
        mount = DockerVolumeMount(
            volume=self._create_config.shared_volume,
            target=_mount_path(target, "target"),
            subpath=source,
            read_only=read_only,
        )
        return {"volumes": [mount]}

    async def _ensure_volumes(self, volumes: list[str]) -> None:
        """Before the first create that mounts them: check that the daemon can mount a volume's subdirectory
        (Engine 26.0, API 1.45), then create each named volume that does not exist yet, labeled
        ``nemo-gym.shared-storage=1``. Both are done once per provider."""
        async with self._volume_lock:
            missing = [volume for volume in volumes if volume not in self._ready_volumes]
            if not missing:
                return
            timeout_s = self._exec_config.default_timeout_s
            if not self._ready_volumes:
                code, out, err = await self._run(
                    [self._binary, "version", "--format", "{{.Server.APIVersion}}"], timeout_s=timeout_s
                )
                version = _api_version(out) if code == 0 else None
                if version is None:
                    raise DockerCreateError(
                        f"cannot read the Docker daemon's API version for shared storage: {(err or out).strip()}"
                    )
                if version < MIN_SUBPATH_API_VERSION:
                    raise DockerCreateError(
                        "Docker shared storage (Compose volumes) needs Docker Engine 26.0 or newer, whose API 1.45 "
                        f"mounts a volume's subdirectory (--mount volume-subpath); the daemon reports API {out.strip()}"
                    )
            for volume in missing:
                code, _out, _err = await self._run(
                    [self._binary, "volume", "inspect", "--format", "{{.Name}}", volume], timeout_s=timeout_s
                )
                if code != 0:
                    code, _out, err = await self._run(
                        [self._binary, "volume", "create", "--label", f"{SHARED_STORAGE_LABEL}=1", volume],
                        timeout_s=timeout_s,
                    )
                    if code != 0:
                        raise DockerCreateError(f"cannot create the shared-storage volume {volume!r}: {err.strip()}")
                self._ready_volumes.add(volume)

    def validate_networking(self) -> None:
        """Refuse a ``create.network`` that gives a sandbox no address of its own for its peers to reach: ``none``,
        ``host`` and ``container:<name>``. On the default bridge or a named network, sandboxes reach each other by IP.
        """
        network = self._create_config.network
        if network in _NETWORKS_WITHOUT_PEERS or (network or "").startswith("container:"):
            raise NotImplementedError(
                "Docker networking between sandboxes needs a network that gives each container its own address; "
                f"create.network is {network!r}"
            )

    async def network_address(self, handle: SandboxHandle) -> str:
        """The container's IP address from ``docker inspect``: on ``create.network`` when the container is on it,
        else on the first network it is on (the default bridge, normally its only one)."""
        self.validate_networking()
        name = handle.raw.name
        code, out, err = await self._run(
            [self._binary, "inspect", "--format", "{{json .NetworkSettings.Networks}}", name],
            timeout_s=self._exec_config.default_timeout_s,
        )
        if code != 0:
            raise RuntimeError(f"docker inspect failed (code={code}) for {name!r}: {err.strip()}")
        try:
            networks = json.loads(out) or {}
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"docker inspect returned unreadable networks for {name!r}: {out.strip()!r}") from exc
        if not isinstance(networks, Mapping):
            raise RuntimeError(f"docker inspect returned unreadable networks for {name!r}: {out.strip()!r}")
        configured = self._create_config.network
        candidates = [networks[configured]] if configured in networks else list(networks.values())
        for entry in candidates:
            address = (
                (entry.get("IPAddress") or entry.get("GlobalIPv6Address")) if isinstance(entry, Mapping) else None
            )
            if address:
                return str(ipaddress.ip_address(address))
        raise RuntimeError(
            f"Docker sandbox {name!r} has no IP address on its networks {sorted(networks)}; peers cannot reach it"
        )

    async def set_hosts(self, handle: SandboxHandle, hosts: Mapping[str, str]) -> None:
        """Append an ``<address> <name>`` line per entry of ``hosts`` to the container's ``/etc/hosts``, as root.

        Docker bind-mounts that file into every container, read-only under ``create.read_only``, which then
        fails this call."""
        self.validate_networking()
        lines = []
        for hostname, address in hosts.items():
            if not _HOSTS_NAME.fullmatch(hostname):
                raise ValueError(f"Invalid sandbox hostname: {hostname!r}")
            lines.append(shlex.quote(f"{ipaddress.ip_address(address)} {hostname}\n"))
        if not lines:
            return
        result = await self.exec(handle, f"printf '%s' {' '.join(lines)} >> /etc/hosts", user="root")
        if result.return_code != 0 or result.error_type is not None:
            raise RuntimeError(
                f"Could not add host names to Docker sandbox {handle.sandbox_id!r}: "
                f"{(result.stderr or result.stdout or '').strip()}"
            )

    async def close(self, handle: SandboxHandle) -> None:
        """Force-remove the container (already-gone counts as success); its PTY clients are released first."""
        inst = handle.raw
        await self._release_pty_sessions(inst.name, abandon=True)
        code, _out, err = await self._run(
            [self._binary, "rm", "-f", inst.name],
            timeout_s=self._exec_config.default_timeout_s,
        )
        if code != 0 and not _is_missing_container(err):
            raise RuntimeError(f"docker rm -f failed (code={code}) for {inst.name!r}: {err.strip()}")

    async def aclose(self) -> None:
        """Close the PTY sessions still open; the containers themselves belong to their handles."""
        for name in list(self._pty_sessions):
            await self._release_pty_sessions(name, abandon=False)
