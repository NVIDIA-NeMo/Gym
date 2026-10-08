# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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

"""NeMo Gym sandbox provider backed by `Modal Sandboxes <https://modal.com/docs/guide/sandbox>`_.

Modal is image-based like the Docker provider (any OCI registry reference starts a
sandbox via ``modal.Image.from_registry``), and serverless like E2B (no daemon, no
cluster, sandboxes are billed while alive and hard-killed at ``timeout``). The
provider-neutral :class:`SandboxSpec` maps onto it almost one-to-one:

======================  =====================================================
``SandboxSpec`` field   Modal
======================  =====================================================
``image``               ``Image.from_registry(image, secret=...)``
``entrypoint``          ``Sandbox.create(*entrypoint)`` (CMD override)
``ttl_s``               ``timeout=`` (hard lifetime)
``env``                 ``env=``
``workdir``             ``workdir=``
``resources.cpu``       ``cpu=``
``resources.memory_mib``  ``memory=`` (MiB)
``resources.gpu[_type]``  ``gpu="TYPE:COUNT"``
``resources.disk_gib``  **not per-sandbox** -- warned or rejected
``metadata``            ``tags=``
``ports``               ``encrypted_ports=`` (or unencrypted/h2 via config),
                        resolved to a Modal tunnel URL by :meth:`endpoint`
``files``               applied by the Gym facade through :meth:`upload_file`
======================  =====================================================

Two things are deliberately explicit rather than silent:

**Keep-alive.** With no ``entrypoint`` the sandbox runs a POSIX-shell keep-alive
loop so that ``exec`` has a live container to attach to, exactly like the Docker
provider. Modal's own ``timeout`` still ends the sandbox at ``ttl_s``.

**Readiness.** ``create()`` returns only after a configurable exec probe passes,
so ``upload_file``/``exec`` never race the container start. A sandbox that
fails the probe is terminated before the error is raised, so nothing billable
leaks.

Authentication is the Modal SDK's own: ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET``
or ``~/.modal.toml`` from ``modal token new``. Nothing secret lives in the YAML.
"""

from __future__ import annotations

import asyncio
import logging
import math
import shlex
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxCreateVerificationError,
    SandboxEndpoint,
    SandboxExecResult,
    SandboxHandle,
    SandboxResources,
    SandboxSpec,
    SandboxStatus,
)
from nemo_gym.sandbox.providers.modal._sdk import require_modal_sdk


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# Per-sandbox knobs accepted in ``SandboxSpec.provider_options``.
_PROVIDER_OPTION_KEYS = frozenset(
    {
        "gpu",  # Modal GPU string, e.g. "A100" or "A100:2"; overrides resources.gpu/gpu_type
        "cloud",  # "aws" | "gcp" | "oci" | ...
        "region",  # str or list[str]
        "secrets",  # list[str] of modal.Secret names injected as env
        "volumes",  # {mount_path: modal.Volume name}
        "block_network",  # bool
        "idle_timeout_s",  # number
        "image_secret",  # modal.Secret name holding registry credentials
        "name",  # Modal sandbox name (unique within the app)
        "tags",  # extra tags merged over spec.metadata
    }
)

_PORT_MODES = ("unencrypted", "encrypted", "h2")

# Modal 1.5.5 catches ExecTimeoutError in wait() and returns -1. The provider
# also accepts a future SDK that raises ExecTimeoutError directly.
_MODAL_EXEC_SENTINEL_RC = -1


class ModalCreateError(SandboxCreateError):
    """Raised when a Modal sandbox cannot be created."""


class ModalCreateVerificationError(ModalCreateError, SandboxCreateVerificationError):
    """Raised when a created Modal sandbox never passes the readiness probe."""


def _require_modal() -> Any:
    """Local seam so tests can swap the SDK without touching ``_sdk``."""
    return require_modal_sdk("The modal sandbox provider")


# ----------------------------------------------------------------------------
# config
# ----------------------------------------------------------------------------


def _config_from_mapping(cls: type[T], value: Any) -> T:
    """Build a config dataclass from a mapping, rejecting unknown keys (same rule as e2b)."""
    if value is None:
        return cls()
    if isinstance(value, cls):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"{cls.__name__} expects a mapping, got {type(value).__name__}")
    allowed = {f.name for f in fields(cls)}
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} keys: {', '.join(sorted(unknown))}. Expected: {', '.join(sorted(allowed))}"
        )
    return cls(**dict(value))


def _is_finite_number(value: Any) -> bool:
    return not isinstance(value, bool) and (
        isinstance(value, int) or (isinstance(value, float) and math.isfinite(value))
    )


def _validate_optional_number(name: str, value: Any, *, positive: bool) -> None:
    operator = "> 0" if positive else ">= 0"
    if value is not None and (not _is_finite_number(value) or (value <= 0 if positive else value < 0)):
        raise ValueError(f"{name} must be {operator}")


def _validate_nonnegative_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be >= 0")


def _validate_positive_int(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be > 0")


def _coerce_str_list(value: Any, what: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if isinstance(value, (list, tuple)):
        out: list[str] = []
        for item in value:
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{what} entries must be non-empty strings")
            out.append(item)
        return out
    raise TypeError(f"{what} must be a string or list of strings")


def _validate_exec_user(user: str | int | None, *, allow_user_rewrite: bool) -> None:
    if user is None or user == "root" or (type(user) is int and user == 0):
        return
    if not isinstance(user, str) or not user.strip() or user.startswith("-") or user.isdecimal():
        raise ValueError("modal user rewrite requires a non-option username; non-root numeric UIDs are unsupported")
    if not allow_user_rewrite:
        raise ValueError("Non-root modal exec users require exec.allow_user_rewrite: true and su in the image")


@dataclass(frozen=True)
class ModalConnectionConfig:
    """Which Modal App and environment sandboxes are created under.

    Credentials are never configured here: the SDK reads ``MODAL_TOKEN_ID`` /
    ``MODAL_TOKEN_SECRET`` or ``~/.modal.toml``. ``MODAL_PROFILE`` selects a profile.
    """

    # Sandboxes must belong to an App. Looked up (and created when missing) once
    # per provider instance.
    app_name: str = "nemo-gym-sandboxes"
    # Modal environment (workspace sub-namespace). ``None`` uses the profile default.
    environment_name: str | None = None
    create_app_if_missing: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.app_name, str) or not self.app_name.strip():
            raise ValueError("connection.app_name must be a non-empty string")


@dataclass(frozen=True)
class ModalCreateConfig:
    """Sandbox creation settings."""

    # Hard sandbox lifetime (Modal ``timeout``). ``SandboxSpec.ttl_s`` overrides it.
    timeout_s: float | None = 3600.0
    # Modal idle activity: active exec, stdin writes, or open tunnel connections.
    idle_timeout_s: float | None = None
    # Shell + command that keeps the container alive when the spec has no
    # entrypoint. The loop form works under dash/busybox, which lack
    # ``sleep infinity``.
    keepalive_shell: list[str] = field(default_factory=lambda: ["/bin/sh", "-c"])
    keepalive_cmd: str = "while :; do sleep 3600; done"
    # modal.Secret name holding registry credentials for private images
    # (REGISTRY_USERNAME / REGISTRY_PASSWORD per Modal's docs).
    image_secret: str | None = None
    # modal.Secret names injected into every sandbox as environment variables.
    secrets: list[str] = field(default_factory=list)
    cloud: str | None = None
    region: str | list[str] | None = None
    block_network: bool = False
    # Modal terminates TLS in front of a plaintext service for encrypted/h2.
    # Unencrypted explicitly requests an additional public raw TCP socket.
    port_mode: str = "encrypted"
    # Tags stamped on every sandbox before spec.metadata / provider_options.tags.
    default_tags: dict[str, str] = field(default_factory=lambda: {"nemo-gym": "sandbox"})
    # disk_gib cannot be set per sandbox on Modal. False warns once per image;
    # True rejects the create.
    strict_resources: bool = False
    # GPU string used when resources.gpu is set but gpu_type is not.
    default_gpu_type: str = "any"

    def __post_init__(self) -> None:
        _validate_optional_number("create.timeout_s", self.timeout_s, positive=True)
        _validate_optional_number("create.idle_timeout_s", self.idle_timeout_s, positive=True)
        if (
            not isinstance(self.keepalive_shell, list)
            or not self.keepalive_shell
            or not all(isinstance(p, str) and p.strip() for p in self.keepalive_shell)
        ):
            raise ValueError("create.keepalive_shell must be a non-empty list of strings")
        if not isinstance(self.keepalive_cmd, str) or not self.keepalive_cmd.strip():
            raise ValueError("create.keepalive_cmd must be a non-empty string")
        if self.port_mode not in _PORT_MODES:
            raise ValueError(f"create.port_mode must be one of {', '.join(_PORT_MODES)}")
        object.__setattr__(self, "secrets", _coerce_str_list(self.secrets, "create.secrets"))
        if not isinstance(self.default_tags, Mapping):
            raise TypeError("create.default_tags must be a mapping")


@dataclass(frozen=True)
class ModalProbeConfig:
    """Readiness probe run through ``exec`` after the sandbox is created.

    ``command: null`` disables probing entirely (create returns as soon as
    Modal hands back the sandbox object).
    """

    command: str | None = "printf ok"
    expected_stdout: str | None = "ok"
    timeout_s: float = 30.0
    # Budget for App lookup, allocation and probing. ``SandboxSpec.ready_timeout_s``
    # overrides it. ``None`` disables the overall deadline; a failed probe raises.
    deadline_s: float | None = 180.0
    stable_count: int = 1
    stable_delay_s: float = 0.5

    def __post_init__(self) -> None:
        if self.command is not None and (not isinstance(self.command, str) or not self.command.strip()):
            raise ValueError("probe.command must be a non-empty string or null")
        _validate_optional_number("probe.timeout_s", self.timeout_s, positive=True)
        _validate_optional_number("probe.deadline_s", self.deadline_s, positive=True)
        _validate_positive_int("probe.stable_count", self.stable_count)
        _validate_optional_number("probe.stable_delay_s", self.stable_delay_s, positive=False)


@dataclass(frozen=True)
class ModalExecConfig:
    """Command execution settings."""

    # Used when the caller passes ``timeout_s=None``. The Gym facade normally
    # supplies its own 180s default.
    default_timeout_s: float | None = 180.0
    # ``exec`` receives a shell string; this is the shell that runs it.
    shell: list[str] = field(default_factory=lambda: ["/bin/sh", "-c"])
    # Default user for every exec. Modal has no native per-exec user, so a user
    # other than root requires ``su`` and ``allow_user_rewrite``.
    user: str | int | None = None
    allow_user_rewrite: bool = False

    def __post_init__(self) -> None:
        _validate_optional_number("exec.default_timeout_s", self.default_timeout_s, positive=False)
        if (
            not isinstance(self.shell, list)
            or not self.shell
            or not all(isinstance(p, str) and p.strip() for p in self.shell)
        ):
            raise ValueError("exec.shell must be a non-empty list of strings")
        _validate_exec_user(self.user, allow_user_rewrite=self.allow_user_rewrite)


@dataclass(frozen=True)
class ModalFilesConfig:
    """The SDK streams file copies; no legacy FileIO or shell fallback is needed."""

    transfer_timeout_s: float | None = 600.0

    def __post_init__(self) -> None:
        _validate_optional_number("files.transfer_timeout_s", self.transfer_timeout_s, positive=True)


@dataclass(frozen=True)
class ModalOperationConfig:
    """Retry policy for transient SDK/transport failures."""

    retries: int = 2
    retry_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0
    # Time allowed for Modal to hand back tunnel metadata in :meth:`endpoint`.
    tunnel_timeout_s: float = 50.0
    close_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        _validate_nonnegative_int("operations.retries", self.retries)
        _validate_optional_number("operations.retry_delay_s", self.retry_delay_s, positive=False)
        _validate_optional_number("operations.retry_max_delay_s", self.retry_max_delay_s, positive=False)
        _validate_optional_number("operations.tunnel_timeout_s", self.tunnel_timeout_s, positive=True)
        _validate_optional_number("operations.close_timeout_s", self.close_timeout_s, positive=True)


# ----------------------------------------------------------------------------
# handle payload
# ----------------------------------------------------------------------------


@dataclass
class _ModalSandbox:
    """Provider-owned state carried in ``SandboxHandle.raw``."""

    sandbox: Any
    image: str | None
    declared_ports: tuple[int, ...]
    app_name: str
    port_mode: str = "encrypted"


# ----------------------------------------------------------------------------
# provider
# ----------------------------------------------------------------------------


class ModalProvider:
    """Sandbox provider backed by ``modal.Sandbox``.

    Implements the core :class:`~nemo_gym.sandbox.providers.base.SandboxProvider`
    protocol plus the optional ``SupportsSandboxEndpoint`` (declared ports ->
    Modal tunnel URL) and ``ConnectableProvider`` (``Sandbox.from_id``) capabilities.
    """

    name = "modal"

    def __init__(
        self,
        *,
        connection: ModalConnectionConfig | Mapping[str, Any] | None = None,
        create: ModalCreateConfig | Mapping[str, Any] | None = None,
        probe: ModalProbeConfig | Mapping[str, Any] | None = None,
        exec: ModalExecConfig | Mapping[str, Any] | None = None,
        files: ModalFilesConfig | Mapping[str, Any] | None = None,
        operations: ModalOperationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _config_from_mapping(ModalConnectionConfig, connection)
        self._create = _config_from_mapping(ModalCreateConfig, create)
        self._probe = _config_from_mapping(ModalProbeConfig, probe)
        self._exec = _config_from_mapping(ModalExecConfig, exec)
        self._files = _config_from_mapping(ModalFilesConfig, files)
        self._operations = _config_from_mapping(ModalOperationConfig, operations)
        self._app: Any | None = None
        self._app_lock = asyncio.Lock()
        self._warned_resource_images: set[str] = set()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _sandbox(handle: SandboxHandle) -> Any:
        raw = handle.raw
        if raw is None:
            raise RuntimeError(f"Sandbox handle {handle.sandbox_id} has been closed")
        if isinstance(raw, _ModalSandbox):
            return raw.sandbox
        return raw

    @staticmethod
    def _payload(handle: SandboxHandle) -> _ModalSandbox:
        raw = handle.raw
        if not isinstance(raw, _ModalSandbox):
            raise RuntimeError(f"Sandbox handle {handle.sandbox_id} carries no modal sandbox state")
        return raw

    def _exception_types(self, modal: Any) -> dict[str, tuple[type, ...]]:
        """Group Modal exception classes by how the provider should treat them."""
        exc = getattr(modal, "exception", None)

        def pick(*names: str) -> tuple[type, ...]:
            return tuple(t for t in (getattr(exc, n, None) for n in names) if isinstance(t, type))

        return {
            "not_found": pick("NotFoundError", "SandboxFilesystemNotFoundError"),
            # Deterministic: retrying only adds latency or amplifies a rate limit.
            "non_retryable": pick(
                "NotFoundError",
                "AuthError",
                "InvalidError",
                "PermissionDeniedError",
                "ResourceExhaustedError",
                "ExecTimeoutError",
                "SandboxTimeoutError",
                "SandboxTerminatedError",
                "SandboxFilesystemError",
            ),
            "exec_timeout": pick("ExecTimeoutError"),
            "sandbox_gone": pick("SandboxTimeoutError", "SandboxTerminatedError"),
            "exec_sandbox": pick("NotFoundError", "ConflictError", "ClientClosed", "InvalidError"),
            "retryable": pick("ConnectionError", "InternalError", "ServiceError"),
        }

    async def _with_retries(
        self,
        factory: Callable[[], Awaitable[T]],
        *,
        operation: str,
    ) -> T:
        """Retry transient failures with exponential backoff."""
        modal = _require_modal()
        types = self._exception_types(modal)
        non_retryable = types["non_retryable"]
        attempts = self._operations.retries + 1
        delay = self._operations.retry_delay_s
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return await factory()
            except non_retryable:
                raise
            except types["retryable"] as exc:
                last_exc = exc
                if attempt == attempts - 1:
                    break
                LOGGER.debug("modal %s failed (attempt %d/%d): %s", operation, attempt + 1, attempts, exc)
                await asyncio.sleep(min(delay, self._operations.retry_max_delay_s))
                delay *= 2
        assert last_exc is not None
        raise last_exc

    async def _get_app(self) -> Any:
        """Resolve the Modal App once per provider instance."""
        if self._app is not None:
            return self._app
        async with self._app_lock:
            if self._app is None:
                modal = _require_modal()
                cfg = self._connection
                try:
                    self._app = await self._with_retries(
                        lambda: modal.App.lookup.aio(
                            cfg.app_name,
                            environment_name=cfg.environment_name,
                            create_if_missing=False,
                        ),
                        operation="app_lookup",
                    )
                except modal.exception.NotFoundError:
                    if not cfg.create_app_if_missing:
                        raise
                    self._app = await self._with_retries(
                        lambda: modal.App.lookup.aio(
                            cfg.app_name,
                            environment_name=cfg.environment_name,
                            create_if_missing=True,
                        ),
                        operation="app_create",
                    )
        return self._app

    def _secret(self, modal: Any, name: str) -> Any:
        return modal.Secret.from_name(name, environment_name=self._connection.environment_name)

    def _volume(self, modal: Any, name: str) -> Any:
        return modal.Volume.from_name(name, environment_name=self._connection.environment_name)

    def _build_image(self, modal: Any, image: str, image_secret: str | None) -> Any:
        kwargs: dict[str, Any] = {}
        if image_secret:
            kwargs["secret"] = self._secret(modal, image_secret)
        return modal.Image.from_registry(image, **kwargs)

    def _validate_provider_options(self, options: Any) -> dict[str, Any]:
        if options is None:
            return {}
        if not isinstance(options, Mapping):
            raise TypeError("modal provider_options must be a mapping")
        unknown = set(options) - _PROVIDER_OPTION_KEYS
        if unknown:
            raise ValueError(
                f"Unknown modal provider option(s): {', '.join(sorted(unknown))}. "
                f"Supported: {', '.join(sorted(_PROVIDER_OPTION_KEYS))}"
            )
        return dict(options)

    def _gpu_string(self, resources: SandboxResources, options: Mapping[str, Any]) -> str | None:
        explicit = options.get("gpu")
        if explicit is not None:
            if not isinstance(explicit, str) or not explicit.strip():
                raise ValueError("modal provider option 'gpu' must be a non-empty string like 'A100' or 'A100:2'")
            return explicit
        if resources.gpu is None and resources.gpu_type is None:
            return None
        count = resources.gpu if resources.gpu is not None else 1
        if count <= 0:
            return None
        gpu_type = resources.gpu_type or self._create.default_gpu_type
        return f"{gpu_type}:{count}" if count > 1 else gpu_type

    def _check_resources(self, spec: SandboxSpec) -> None:
        """Surface the one resource request Modal cannot honour per sandbox."""
        if spec.resources.disk_gib is None:
            return
        image = spec.image or "<no image>"
        message = (
            f"Modal sandboxes have no per-sandbox disk size; disk_gib={spec.resources.disk_gib} requested for "
            f"image {image!r} is ignored. Use a modal.Volume (provider_options.volumes) for large working sets."
        )
        if self._create.strict_resources:
            raise ModalCreateError(message)
        if image not in self._warned_resource_images:
            self._warned_resource_images.add(image)
            LOGGER.warning("%s", message)

    def _shell_argv(self, command: str, *, user: str | int | None) -> list[str]:
        """Wrap a shell command string in the configured shell, honouring ``user`` via ``su``."""
        effective_user = user if user is not None else self._exec.user
        _validate_exec_user(effective_user, allow_user_rewrite=self._exec.allow_user_rewrite)
        argv = [*self._exec.shell, command]
        if effective_user in (None, "root", 0):
            return argv
        # Run the complete shell argv inside su, preserving flags such as bash -lc.
        return ["su", "-s", "/bin/sh", "-c", shlex.join(argv), "--", effective_user]

    # ---------------------------------------------------------------- lifecycle

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        modal = _require_modal()

        if not spec.image:
            raise ModalCreateError("SandboxSpec.image is required for the modal provider")
        options = self._validate_provider_options(spec.provider_options)

        ttl_s = spec.ttl_s if spec.ttl_s is not None else self._create.timeout_s
        if ttl_s is not None and (not _is_finite_number(ttl_s) or not 0 < ttl_s <= 86400):
            raise ModalCreateError("modal sandbox ttl_s must be > 0 and <= 86400 seconds")
        if spec.ready_timeout_s is not None and (
            not _is_finite_number(spec.ready_timeout_s) or spec.ready_timeout_s <= 0
        ):
            raise ModalCreateError("modal sandbox ready_timeout_s must be > 0")
        self._check_resources(spec)

        image_secret = options.get("image_secret", self._create.image_secret)
        if image_secret is not None and (not isinstance(image_secret, str) or not image_secret.strip()):
            raise ModalCreateError("modal image_secret must be a non-empty modal.Secret name")

        kwargs: dict[str, Any] = {}
        if ttl_s is not None:
            kwargs["timeout"] = max(1, math.ceil(ttl_s))
        idle = options.get("idle_timeout_s", self._create.idle_timeout_s)
        if idle is not None:
            _validate_optional_number("idle_timeout_s", idle, positive=True)
            kwargs["idle_timeout"] = max(1, math.ceil(idle))
        if spec.workdir:
            kwargs["workdir"] = spec.workdir
        if spec.env:
            kwargs["env"] = {str(k): str(v) for k, v in spec.env.items()}

        resources = spec.resources
        if resources.cpu is not None:
            kwargs["cpu"] = float(resources.cpu)
        if resources.memory_mib is not None:
            kwargs["memory"] = int(resources.memory_mib)
        gpu = self._gpu_string(resources, options)
        if gpu is not None:
            kwargs["gpu"] = gpu

        cloud = options.get("cloud", self._create.cloud)
        if cloud is not None:
            kwargs["cloud"] = str(cloud)
        region = options.get("region", self._create.region)
        if region is not None:
            kwargs["region"] = region
        block_network = options.get("block_network", self._create.block_network)
        if not isinstance(block_network, bool):
            raise ModalCreateError("modal block_network must be a boolean")
        if block_network and spec.ports:
            raise ModalCreateError("Modal does not allow block_network with exposed sandbox ports")
        kwargs["block_network"] = block_network

        secret_names = [*self._create.secrets, *_coerce_str_list(options.get("secrets"), "provider_options.secrets")]

        volumes = options.get("volumes")
        if volumes is not None:
            if not isinstance(volumes, Mapping):
                raise ModalCreateError("modal provider option 'volumes' must map mount paths to Volume names")
            for mount, vol in volumes.items():
                if (
                    not isinstance(mount, str)
                    or not mount.startswith("/")
                    or not isinstance(vol, str)
                    or not vol.strip()
                ):
                    raise ModalCreateError("modal volumes require absolute mount paths and non-empty Volume names")

        tags: dict[str, str] = {str(k): str(v) for k, v in self._create.default_tags.items()}
        tags.update({str(k): str(v) for k, v in spec.metadata.items()})
        extra_tags = options.get("tags")
        if extra_tags is not None:
            if not isinstance(extra_tags, Mapping):
                raise ModalCreateError("modal provider option 'tags' must be a mapping")
            tags.update({str(k): str(v) for k, v in extra_tags.items()})
        # Verified against both the live service and its explicit InvalidError.
        if len(tags) > 10:
            raise ModalCreateError("Modal sandboxes allow at most 10 tags, including default tags and spec.metadata")
        if tags:
            kwargs["tags"] = tags

        name = options.get("name")
        if name is not None:
            if not isinstance(name, str) or not name.strip():
                raise ModalCreateError("modal provider option 'name' must be a non-empty string")
            kwargs["name"] = name

        if spec.ports:
            port_key = {"unencrypted": "unencrypted_ports", "encrypted": "encrypted_ports", "h2": "h2_ports"}[
                self._create.port_mode
            ]
            kwargs[port_key] = list(spec.ports)

        if spec.entrypoint:
            entrypoint = [str(p) for p in spec.entrypoint]
        else:
            entrypoint = [*self._create.keepalive_shell, self._create.keepalive_cmd]

        deadline_s = spec.ready_timeout_s if spec.ready_timeout_s is not None else self._probe.deadline_s
        deadline = asyncio.get_running_loop().time() + deadline_s if deadline_s is not None else None
        create_task = None
        try:
            async with asyncio.timeout_at(deadline):
                # All local validation precedes App lookup, which can allocate an App.
                kwargs["app"] = await self._get_app()
                kwargs["image"] = self._build_image(modal, spec.image, image_secret)
                if secret_names:
                    kwargs["secrets"] = [self._secret(modal, n) for n in secret_names]
                if volumes:
                    kwargs["volumes"] = {mount: self._volume(modal, vol) for mount, vol in volumes.items()}
                # The SDK owns request idempotency/retries; the provider adds no
                # retry that could allocate a second billable sandbox.
                create_task = asyncio.create_task(modal.Sandbox.create.aio(*entrypoint, **kwargs))
                sandbox = await asyncio.shield(create_task)
        except (TimeoutError, asyncio.CancelledError) as exc:
            if create_task is not None:
                await asyncio.shield(self._reconcile_interrupted_create(create_task))
            if isinstance(exc, asyncio.CancelledError):
                raise
            raise ModalCreateError(
                f"modal sandbox creation exceeded ready timeout of {deadline_s}s for image {spec.image!r}"
            ) from exc
        except Exception as exc:
            raise ModalCreateError(f"Failed to create modal sandbox from image {spec.image!r}: {exc}") from exc

        handle = SandboxHandle(
            sandbox_id=str(sandbox.object_id),
            provider_name=self.name,
            raw=_ModalSandbox(
                sandbox=sandbox,
                image=spec.image,
                declared_ports=tuple(spec.ports),
                app_name=self._connection.app_name,
                port_mode=self._create.port_mode,
            ),
        )
        try:
            await self._verify_created_handle(handle, ready_timeout_s=deadline_s, deadline=deadline)
        except BaseException:
            await asyncio.shield(self._cleanup_after_error(handle))
            raise
        return handle

    async def _reconcile_interrupted_create(self, create_task: asyncio.Task) -> None:
        # Allocation can finish after the caller's deadline. Bound both waiting
        # for its id and cleanup; the remote TTL is the backstop for an unknown id.
        try:
            async with asyncio.timeout(self._operations.close_timeout_s):
                sandbox = await create_task
                await sandbox.terminate.aio(wait=True)
                await sandbox.detach.aio()
        except Exception:
            LOGGER.exception(
                "Could not reconcile interrupted Modal create; app=%s; remote TTL remains the cleanup backstop",
                self._connection.app_name,
            )

    async def _cleanup_after_error(self, handle: SandboxHandle) -> None:
        try:
            await self.close(handle)
        except Exception as exc:  # noqa: BLE001 - preserve the original operation error
            LOGGER.warning("modal sandbox %s cleanup after failure raised: %s", handle.sandbox_id, exc)

    async def _verify_created_handle(
        self, handle: SandboxHandle, *, ready_timeout_s: float | None, deadline: float | None = None
    ) -> None:
        deadline_s = ready_timeout_s if ready_timeout_s is not None else self._probe.deadline_s
        if deadline is None and deadline_s is not None:
            deadline = asyncio.get_running_loop().time() + deadline_s
        # Per-create state: concurrent creates must not overwrite each other's diagnostics.
        detail = ["no probe attempt completed"]
        try:
            async with asyncio.timeout_at(deadline):
                await self._run_readiness_probe(handle, deadline_s=deadline_s, detail=detail)
        except TimeoutError as exc:
            raise ModalCreateVerificationError(
                f"modal sandbox {handle.sandbox_id!r} did not pass readiness probe within {deadline_s}s: {detail[0]}"
            ) from exc

    async def _run_readiness_probe(
        self, handle: SandboxHandle, *, deadline_s: float | None, detail: list[str]
    ) -> None:
        """Poll until stable success; the caller owns the single overall deadline."""
        probe = self._probe
        if probe.command is None:
            return
        consecutive = 0
        while True:
            try:
                result = await self.exec(handle, probe.command, timeout_s=probe.timeout_s)
            except (ValueError, TypeError):
                raise
            except Exception as exc:  # noqa: BLE001 - container may not be attachable yet
                result = SandboxExecResult(stdout=None, stderr=str(exc), return_code=-1, error_type="probe_exec")
            passed = result.return_code == 0 and (
                probe.expected_stdout is None or probe.expected_stdout in (result.stdout or "")
            )
            detail[0] = (
                f"return_code={result.return_code}, stdout={(result.stdout or '').strip()!r}, "
                f"stderr={(result.stderr or '').strip()!r}"
            )
            if passed:
                consecutive += 1
                if consecutive >= probe.stable_count:
                    return
            else:
                consecutive = 0
                if deadline_s is None:
                    raise ModalCreateVerificationError(
                        f"modal sandbox {handle.sandbox_id!r} failed readiness probe: {detail[0]}"
                    )
            # Yield even for zero delay so a stream of immediate failures cannot
            # starve the deadline or caller cancellation.
            await asyncio.sleep(probe.stable_delay_s)

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict[str, Any]:
        """Descriptor for :meth:`connect` in another process (``ConnectableProvider``)."""
        payload = handle.raw
        descriptor: dict[str, Any] = {"sandbox_id": handle.sandbox_id}
        if isinstance(payload, _ModalSandbox):
            descriptor["ports"] = list(payload.declared_ports)
            descriptor["port_mode"] = payload.port_mode
            if payload.image:
                descriptor["image"] = payload.image
        return descriptor

    async def connect(self, descriptor: Mapping[str, Any]) -> SandboxHandle:
        """Rebuild a live handle from ``Sandbox.from_id``."""
        modal = _require_modal()
        sandbox_id = str(descriptor["sandbox_id"])
        sandbox = await self._with_retries(lambda: modal.Sandbox.from_id.aio(sandbox_id), operation="from_id")
        ports = tuple(int(p) for p in descriptor.get("ports", ()) or ())
        return SandboxHandle(
            sandbox_id=sandbox_id,
            provider_name=self.name,
            raw=_ModalSandbox(
                sandbox=sandbox,
                image=descriptor.get("image"),
                declared_ports=ports,
                app_name=self._connection.app_name,
                port_mode=descriptor.get("port_mode", self._create.port_mode),
            ),
        )

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        modal = _require_modal()
        types = self._exception_types(modal)
        try:
            sandbox = self._sandbox(handle)
        except RuntimeError:
            return SandboxStatus.STOPPED
        try:
            # ``poll`` is None while the sandbox is alive, else its exit code.
            code = await sandbox.poll.aio()
        except types["not_found"]:
            return SandboxStatus.STOPPED
        except Exception:  # noqa: BLE001 - status must not raise on transient issues
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if code is None else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        modal = _require_modal()
        types = self._exception_types(modal)
        raw = handle.raw
        if raw is None:
            return
        sandbox = raw.sandbox if isinstance(raw, _ModalSandbox) else raw
        async with asyncio.timeout(self._operations.close_timeout_s):
            try:
                await self._with_retries(lambda: sandbox.terminate.aio(wait=True), operation="terminate")
            except types["not_found"] + types["sandbox_gone"]:
                LOGGER.debug("modal sandbox %s already gone on close", handle.sandbox_id)
            await sandbox.detach.aio()
        handle.raw = None

    async def aclose(self) -> None:
        """No provider-scoped client to close; the Modal client is process-global."""
        self._app = None

    # ----------------------------------------------------------------- commands

    async def exec(
        self,
        handle: SandboxHandle,
        command: str,
        *,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_s: float | None = None,
        user: str | int | None = None,
    ) -> SandboxExecResult:
        modal = _require_modal()
        types = self._exception_types(modal)
        sandbox = self._sandbox(handle)

        effective_timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        if effective_timeout is not None and (not _is_finite_number(effective_timeout) or effective_timeout < 0):
            raise ValueError("modal command timeout_s must be >= 0")
        if cwd is not None and not str(cwd).startswith("/"):
            raise ValueError("modal exec cwd must be an absolute path")

        argv = self._shell_argv(command, user=user)
        kwargs: dict[str, Any] = {"text": False}
        if cwd is not None:
            kwargs["workdir"] = cwd
        if env:
            kwargs["env"] = {str(k): str(v) for k, v in env.items()}
        # Modal treats 0/None as "no deadline"; keep None explicit.
        modal_timeout = max(1, math.ceil(effective_timeout)) if effective_timeout else None
        if modal_timeout is not None:
            kwargs["timeout"] = modal_timeout

        # Modal validates the total argv length before dispatch. Report the
        # provider limit through Gym's exec result contract, without an SDK error.
        if sum(len(arg) for arg in argv) > 65536:
            return SandboxExecResult(
                stdout="",
                stderr="Modal exec argv exceeds 65,536 characters; upload a script and execute its path",
                return_code=125,
                error_type="sandbox",
            )

        output = [bytearray(), bytearray()]

        async def collect(stream: Any, buffer: bytearray) -> None:
            async for chunk in stream:
                buffer.extend(chunk.encode("utf-8") if isinstance(chunk, str) else chunk)

        error_type = None
        error_detail = None
        try:
            process = await sandbox.exec.aio(*argv, **kwargs)
            # The provider has no stdin input parameter. Send EOF so commands
            # such as cat/read do not wait until their exec deadline.
            process.stdin.write_eof()
            tasks = [
                asyncio.create_task(coro)
                for coro in (
                    collect(process.stdout, output[0]),
                    collect(process.stderr, output[1]),
                    process.wait.aio(),
                    process.stdin.drain.aio(),
                )
            ]
            try:
                _, _, return_code, _ = await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
        except asyncio.CancelledError:
            # No public per-exec kill API. Cleanup must not replace cancellation,
            # including when asyncio.timeout() relies on it to raise TimeoutError.
            await asyncio.shield(self._cleanup_after_error(handle))
            raise
        except types["exec_timeout"] as exc:
            return_code, error_type, error_detail = 125, "timeout", str(exc)
        except types["exec_sandbox"] + types["sandbox_gone"] as exc:
            return_code, error_type, error_detail = 125, "sandbox", str(exc)

        if return_code == _MODAL_EXEC_SENTINEL_RC:
            return_code, error_type = 125, "timeout"
        stdout, stderr = (bytes(part).decode("utf-8", "replace") for part in output)
        if error_detail:
            stderr = f"{stderr}\n{error_detail}" if stderr else error_detail
        return SandboxExecResult(stdout=stdout, stderr=stderr, return_code=int(return_code), error_type=error_type)

    # -------------------------------------------------------------------- files

    async def _file_operation(self, handle: SandboxHandle, operation: Callable[[], Awaitable[T]]) -> T:
        try:
            async with asyncio.timeout(self._files.transfer_timeout_s):
                return await operation()
        except (TimeoutError, asyncio.CancelledError):
            # The SDK currently cannot kill an individual interrupted file writer.
            # Discard this sandbox so it cannot keep modifying a later evaluation.
            await asyncio.shield(self._cleanup_after_error(handle))
            raise

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        sandbox = self._sandbox(handle)
        payload = data.encode("utf-8") if isinstance(data, str) else data
        await self._file_operation(handle, lambda: sandbox.filesystem.write_bytes.aio(payload, target_path))

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        sandbox = self._sandbox(handle)
        try:
            return await self._file_operation(handle, lambda: sandbox.filesystem.read_bytes.aio(source_path))
        except _require_modal().exception.SandboxFilesystemNotFoundError as exc:
            raise FileNotFoundError(f"Sandbox file not found: {source_path}") from exc

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        sandbox = self._sandbox(handle)
        await self._file_operation(handle, lambda: sandbox.filesystem.copy_from_local.aio(source_path, target_path))

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        sandbox = self._sandbox(handle)
        try:
            await self._file_operation(handle, lambda: sandbox.filesystem.copy_to_local.aio(source_path, target_path))
        except _require_modal().exception.SandboxFilesystemNotFoundError as exc:
            raise FileNotFoundError(f"Sandbox file not found: {source_path}") from exc

    # ----------------------------------------------------------------- endpoint

    async def endpoint(self, handle: SandboxHandle, port: int) -> SandboxEndpoint:
        """Resolve a declared port to its Modal tunnel URL (``SupportsSandboxEndpoint``)."""
        payload = self._payload(handle)
        if port not in payload.declared_ports:
            raise ValueError(
                f"Sandbox port {port} was not declared in SandboxSpec.ports; declared ports: {list(payload.declared_ports)!r}"
            )
        timeout = max(1, math.ceil(self._operations.tunnel_timeout_s))
        try:
            tunnels = await self._with_retries(
                lambda: payload.sandbox.tunnels.aio(timeout=timeout), operation="tunnels"
            )
        except Exception as exc:
            raise RuntimeError(f"modal did not return tunnel metadata for sandbox {handle.sandbox_id}: {exc}") from exc
        tunnel = tunnels.get(port) if isinstance(tunnels, Mapping) else None
        if tunnel is None:
            raise RuntimeError(f"modal returned no tunnel for sandbox port {port}")
        if payload.port_mode == "unencrypted":
            host, host_port = tunnel.tcp_socket
            url = f"http://{host}:{host_port}"
        else:
            url = tunnel.url
        return SandboxEndpoint(endpoint=str(url))
