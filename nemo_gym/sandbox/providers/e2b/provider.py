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

"""Sandbox provider backed by the E2B Python SDK.

Works against e2b.dev itself and against any e2b-compatible gateway (point
``connection.api_url``/``connection.sandbox_url`` at it).

Two E2B concepts differ from the provider-neutral :class:`SandboxSpec` and are
handled explicitly rather than silently:

**Templates, not images.** E2B starts sandboxes from a pre-built *template*
alias, not from an arbitrary registry reference, and rejects anything outside
``[A-Za-z0-9_-]``. ``SandboxSpec.image`` is therefore resolved to a template
via :meth:`E2BProvider._resolve_template`; supply ``create.template_map`` when
your image references are not already valid aliases.

**Resources are fixed at template build time.** ``cpu_count``/``memory_mb`` are
arguments to the template *build*, so a per-sandbox
``SandboxSpec.resources`` cannot be honoured at create time. Requests are
reported once per provider instance (or raise, with ``create.strict_resources``)
instead of being dropped quietly.
"""

import asyncio
import hashlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Awaitable, Callable, TypeVar

from nemo_gym.sandbox.providers.base import (
    SandboxCreateError,
    SandboxExecResult,
    SandboxHandle,
    SandboxSpec,
    SandboxStatus,
)


LOGGER = logging.getLogger(__name__)

T = TypeVar("T")

# E2B template aliases accept ASCII letters, digits, hyphens and underscores.
_TEMPLATE_ALIAS_RE = re.compile(r"^[A-Za-z0-9_-]+$")

# Passed straight through to the SDK (``ApiParams``) on every call.
_API_PARAM_KEYS = (
    "api_key",
    "access_token",
    "api_url",
    "sandbox_url",
    "domain",
    "debug",
    "validate_api_key",
    "headers",
    "api_headers",
    "proxy",
)


class E2BCreateError(SandboxCreateError):
    """Raised when a sandbox cannot be created."""


def _require_e2b_sdk() -> Any:
    """Import the optional ``e2b`` dependency with an actionable error."""
    try:
        import e2b
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise ImportError(
            "The e2b sandbox provider requires the 'e2b' package. Install it with `pip install 'e2b>=2.25.0'`."
        ) from exc
    return e2b


def _config_from_mapping(cls: type[T], value: Any) -> T:
    """Build a config dataclass from a mapping, rejecting unknown keys."""
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


@dataclass(frozen=True)
class E2BConnectionConfig:
    """Connection settings forwarded to the SDK.

    Any field left ``None`` falls back to the SDK's own environment variables
    (``E2B_API_KEY``, ``E2B_API_URL``, ``E2B_SANDBOX_URL``, ``E2B_DOMAIN``, ...).
    """

    api_key: str | None = None
    access_token: str | None = None
    api_url: str | None = None
    sandbox_url: str | None = None
    domain: str | None = None
    debug: bool | None = None
    validate_api_key: bool | None = None
    headers: dict[str, str] | None = None
    api_headers: dict[str, str] | None = None
    proxy: str | None = None
    request_timeout_s: float | None = None


@dataclass(frozen=True)
class E2BCreateConfig:
    """Sandbox creation settings."""

    # Template alias used when the spec does not resolve to one.
    template: str | None = None
    # Explicit ``SandboxSpec.image`` -> template alias mapping. Needed whenever
    # image references are not themselves valid aliases (registry refs contain
    # '/' and ':', which E2B rejects).
    template_map: dict[str, str] = field(default_factory=dict)
    # Sandbox lifetime in seconds; E2B kills the sandbox when it elapses.
    # ``SandboxSpec.ttl_s`` overrides it per sandbox.
    timeout_s: float = 3600.0
    allow_internet_access: bool = True
    secure: bool = True
    # Raise instead of warning when a spec requests resources E2B cannot apply
    # per sandbox (they are fixed when the template is built).
    strict_resources: bool = False

    # --- on-demand template building -------------------------------------
    # E2B cannot start a sandbox from an OCI reference; it only starts from a
    # pre-built template alias. With this enabled the provider builds a
    # template from ``SandboxSpec.image`` on first use and reuses it after,
    # so callers can supply ordinary image references.
    auto_build_from_image: bool = False
    # Resources for auto-built templates. E2B fixes cpu/memory at build time,
    # so a spec's own resources are applied here (and take precedence).
    build_cpu_count: int = 2
    build_memory_mb: int = 1024
    build_timeout_s: float = 3600.0
    # Private registry credentials for ``from_image``.
    registry_username: str | None = None
    registry_password: str | None = None


@dataclass(frozen=True)
class E2BExecConfig:
    """Command execution settings."""

    # Applied when the caller passes no ``timeout_s``. E2B's own default is 60s,
    # which silently truncates long-running commands (builds, test suites).
    default_timeout_s: float | None = 1800.0
    user: str | None = None
    request_timeout_s: float | None = None


@dataclass(frozen=True)
class E2BOperationConfig:
    """Retry policy for transient SDK/transport failures."""

    retries: int = 2
    retry_delay_s: float = 0.5
    retry_max_delay_s: float = 8.0


class E2BProvider:
    """Provider backed by the E2B Python SDK."""

    name = "e2b"

    def __init__(
        self,
        *,
        connection: E2BConnectionConfig | Mapping[str, Any] | None = None,
        create: E2BCreateConfig | Mapping[str, Any] | None = None,
        exec: E2BExecConfig | Mapping[str, Any] | None = None,
        operations: E2BOperationConfig | Mapping[str, Any] | None = None,
    ) -> None:
        self._connection = _config_from_mapping(E2BConnectionConfig, connection)
        self._create = _config_from_mapping(E2BCreateConfig, create)
        self._exec = _config_from_mapping(E2BExecConfig, exec)
        self._operations = _config_from_mapping(E2BOperationConfig, operations)
        self._warned_resource_specs: set[str] = set()
        # Aliases this process has already built or confirmed, plus a per-alias
        # lock so N concurrent sandboxes on the same image build it once.
        self._built_aliases: set[str] = set()
        self._build_locks: dict[str, asyncio.Lock] = {}

    # ---------------------------------------------------------------- helpers

    def _api_params(self) -> dict[str, Any]:
        """SDK ``ApiParams`` for every call; omitted keys fall back to env."""
        params = {key: getattr(self._connection, key) for key in _API_PARAM_KEYS}
        params = {key: value for key, value in params.items() if value is not None}
        if self._connection.request_timeout_s is not None:
            params["request_timeout"] = self._connection.request_timeout_s
        return params

    def _build_resources(self, spec: SandboxSpec) -> tuple[int, int]:
        """cpu/memory for an auto-built template; the spec wins over config."""
        resources = spec.resources
        cpu = resources.cpu if getattr(resources, "cpu", None) else None
        memory_mib = resources.memory_mib if getattr(resources, "memory_mib", None) else None
        cpu_count = max(1, int(cpu)) if cpu else self._create.build_cpu_count
        memory_mb = int(memory_mib) if memory_mib else self._create.build_memory_mb
        return cpu_count, memory_mb

    def _derive_alias(self, image: str, cpu_count: int, memory_mb: int) -> str:
        """Deterministic, charset-safe alias for an image + its build resources.

        The digest covers the resources because E2B bakes them into the
        template: two sandboxes asking for different cpu/memory must not share
        one template, or the second silently gets the first one's sizing.
        """
        digest = hashlib.sha256(f"{image}|cpu={cpu_count}|mem={memory_mb}".encode()).hexdigest()[:12]
        stem = image.rsplit("/", 1)[-1].split("@", 1)[0].replace(":", "-")
        stem = re.sub(r"[^A-Za-z0-9_-]", "-", stem).strip("-") or "image"
        return f"{stem[:48]}__{digest}"

    async def _ensure_template_for_image(self, image: str, spec: SandboxSpec) -> str:
        """Build (once) and return a template alias for an OCI image."""
        e2b = _require_e2b_sdk()
        cpu_count, memory_mb = self._build_resources(spec)
        alias = self._derive_alias(image, cpu_count, memory_mb)
        if alias in self._built_aliases:
            return alias

        lock = self._build_locks.setdefault(alias, asyncio.Lock())
        async with lock:
            if alias in self._built_aliases:
                return alias
            api_params = self._api_params()
            try:
                if await e2b.AsyncTemplate.alias_exists(alias, **api_params):
                    self._built_aliases.add(alias)
                    return alias
            except Exception as exc:  # noqa: BLE001 - existence check is best-effort
                LOGGER.debug("e2b alias_exists(%s) failed, attempting build: %s", alias, exc)

            LOGGER.info(
                "Building e2b template %s from image %s (cpu_count=%d, memory_mb=%d)",
                alias,
                image,
                cpu_count,
                memory_mb,
            )
            builder = e2b.AsyncTemplate().from_image(
                image,
                username=self._create.registry_username,
                password=self._create.registry_password,
            )
            try:
                await asyncio.wait_for(
                    e2b.AsyncTemplate.build(
                        builder,
                        alias=alias,
                        cpu_count=cpu_count,
                        memory_mb=memory_mb,
                        **api_params,
                    ),
                    timeout=self._create.build_timeout_s,
                )
            except asyncio.TimeoutError as exc:
                raise E2BCreateError(
                    f"Timed out after {self._create.build_timeout_s}s building e2b template {alias!r} "
                    f"from image {image!r}."
                ) from exc
            except Exception as exc:
                raise E2BCreateError(f"Failed to build e2b template {alias!r} from image {image!r}: {exc}") from exc
            self._built_aliases.add(alias)
            return alias

    async def _resolve_template(self, spec: SandboxSpec) -> str:
        """Map a spec onto an E2B template alias.

        Precedence: ``provider_options.template`` -> ``create.template_map`` ->
        ``spec.image`` when it is already a valid alias -> build from the image
        (``create.auto_build_from_image``) -> ``create.template``.
        """
        option = (spec.provider_options or {}).get("template")
        if option:
            return str(option)

        if spec.image:
            mapped = self._create.template_map.get(spec.image)
            if mapped:
                return str(mapped)
            if _TEMPLATE_ALIAS_RE.match(spec.image):
                return spec.image
            if self._create.auto_build_from_image:
                return await self._ensure_template_for_image(spec.image, spec)
            raise E2BCreateError(
                f"E2B starts sandboxes from a template alias, but SandboxSpec.image={spec.image!r} is not a valid "
                "alias (allowed: letters, digits, '-', '_'). Map it with create.template_map, set "
                "provider_options.template, or enable create.auto_build_from_image to build one from the image."
            )

        if self._create.template:
            return self._create.template

        raise E2BCreateError(
            "No E2B template to start from: set SandboxSpec.image, provider_options.template, or create.template."
        )

    def _check_resources(self, spec: SandboxSpec, template: str, *, applied_at_build: bool = False) -> None:
        """Surface resource requests E2B cannot honour per sandbox.

        ``applied_at_build`` suppresses the report: when the provider built the
        template itself, the request *was* satisfied (as build arguments).
        """
        if applied_at_build:
            return
        resources = spec.resources
        requested = {
            name: getattr(resources, name)
            for name in ("cpu", "memory_mib", "disk_gib", "gpu")
            if getattr(resources, name, None)
        }
        if not requested:
            return
        detail = ", ".join(f"{key}={value}" for key, value in sorted(requested.items()))
        message = (
            f"E2B fixes sandbox resources when the template is built, so {detail} requested for template "
            f"{template!r} cannot be applied at create time. Build a template with the desired "
            "cpu_count/memory_mb instead."
        )
        if self._create.strict_resources:
            raise E2BCreateError(message)
        if template not in self._warned_resource_specs:
            self._warned_resource_specs.add(template)
            LOGGER.warning("%s", message)

    async def _with_retries(self, factory: Callable[[], Awaitable[T]], *, operation: str) -> T:
        """Retry transient failures with exponential backoff."""
        e2b = _require_e2b_sdk()
        # Never retry these: they are deterministic and retrying only adds latency.
        non_retryable = tuple(
            exc
            for exc in (
                getattr(e2b, "NotFoundException", None),
                getattr(e2b, "SandboxNotFoundException", None),
                getattr(e2b, "AuthenticationException", None),
                getattr(e2b, "InvalidArgumentException", None),
                getattr(e2b, "TimeoutException", None),
            )
            if isinstance(exc, type)
        )
        attempts = max(0, self._operations.retries) + 1
        delay = self._operations.retry_delay_s
        last_exc: BaseException | None = None
        for attempt in range(attempts):
            try:
                return await factory()
            except non_retryable:
                raise
            except Exception as exc:  # noqa: BLE001 - transport/5xx errors are provider-specific
                last_exc = exc
                if attempt == attempts - 1:
                    break
                LOGGER.debug("e2b %s failed (attempt %d/%d): %s", operation, attempt + 1, attempts, exc)
                await asyncio.sleep(min(delay, self._operations.retry_max_delay_s))
                delay *= 2
        assert last_exc is not None
        raise last_exc

    @staticmethod
    def _sandbox(handle: SandboxHandle) -> Any:
        sandbox = handle.raw
        if sandbox is None:
            raise RuntimeError(f"Sandbox handle {handle.sandbox_id} carries no e2b sandbox object")
        return sandbox

    # ------------------------------------------------------------- lifecycle

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        e2b = _require_e2b_sdk()
        template = await self._resolve_template(spec)
        # When we built the template ourselves the spec's resources became the
        # build's cpu_count/memory_mb, so there is nothing to report.
        self._check_resources(spec, template, applied_at_build=template in self._built_aliases)

        timeout_s = spec.ttl_s if spec.ttl_s is not None else self._create.timeout_s
        kwargs: dict[str, Any] = {
            "template": template,
            "timeout": int(timeout_s) if timeout_s is not None else None,
            "allow_internet_access": self._create.allow_internet_access,
            "secure": self._create.secure,
            **self._api_params(),
        }
        if spec.env:
            kwargs["envs"] = {str(k): str(v) for k, v in spec.env.items()}
        if spec.metadata:
            kwargs["metadata"] = {str(k): str(v) for k, v in spec.metadata.items()}

        try:
            sandbox = await self._with_retries(
                lambda: e2b.AsyncSandbox.create(**kwargs),
                operation="create",
            )
        except Exception as exc:
            raise E2BCreateError(f"Failed to create e2b sandbox from template {template!r}: {exc}") from exc

        handle = SandboxHandle(sandbox_id=sandbox.sandbox_id, provider_name=self.name, raw=sandbox)
        if spec.files:
            for target_path, contents in spec.files.items():
                await self.write_file(handle, target_path, contents)
        return handle

    async def connect(self, sandbox_id: str) -> SandboxHandle:
        """Attach to an already-running sandbox."""
        e2b = _require_e2b_sdk()
        sandbox = await self._with_retries(
            lambda: e2b.AsyncSandbox.connect(sandbox_id, **self._api_params()),
            operation="connect",
        )
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=sandbox)

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        e2b = _require_e2b_sdk()
        sandbox = self._sandbox(handle)
        not_found = tuple(
            exc
            for exc in (getattr(e2b, "SandboxNotFoundException", None), getattr(e2b, "NotFoundException", None))
            if isinstance(exc, type)
        )
        try:
            running = await sandbox.is_running(**self._api_params())
        except not_found:
            return SandboxStatus.STOPPED
        except Exception:  # noqa: BLE001 - status must not raise for transient issues
            return SandboxStatus.UNKNOWN
        return SandboxStatus.RUNNING if running else SandboxStatus.STOPPED

    async def close(self, handle: SandboxHandle) -> None:
        e2b = _require_e2b_sdk()
        sandbox = handle.raw
        if sandbox is None:
            return
        not_found = tuple(
            exc
            for exc in (getattr(e2b, "SandboxNotFoundException", None), getattr(e2b, "NotFoundException", None))
            if isinstance(exc, type)
        )
        try:
            await sandbox.kill(**self._api_params())
        except not_found:
            # Already gone (expired TTL or killed elsewhere) - closing is idempotent.
            LOGGER.debug("e2b sandbox %s already gone on close", handle.sandbox_id)
        finally:
            handle.raw = None

    async def aclose(self) -> None:
        """No provider-scoped client to close; sandboxes own their connections."""
        return None

    # -------------------------------------------------------------- commands

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
        e2b = _require_e2b_sdk()
        sandbox = self._sandbox(handle)

        effective_timeout = timeout_s if timeout_s is not None else self._exec.default_timeout_s
        effective_user = user if user is not None else self._exec.user
        kwargs: dict[str, Any] = {"cmd": command, **self._api_params()}
        if cwd is not None:
            kwargs["cwd"] = cwd
        if env:
            kwargs["envs"] = {str(k): str(v) for k, v in env.items()}
        if effective_user is not None:
            kwargs["user"] = str(effective_user)
        # E2B treats ``timeout`` as "no timeout" when falsy; keep None explicit.
        kwargs["timeout"] = float(effective_timeout) if effective_timeout is not None else None
        if self._exec.request_timeout_s is not None:
            kwargs["request_timeout"] = self._exec.request_timeout_s

        timeout_exc = getattr(e2b, "TimeoutException", None)
        exit_exc = getattr(e2b, "CommandExitException", None)
        try:
            result = await sandbox.commands.run(**kwargs)
        except Exception as exc:
            # A non-zero exit is a normal outcome, not a provider failure.
            if isinstance(exit_exc, type) and isinstance(exc, exit_exc):
                return SandboxExecResult(
                    stdout=getattr(exc, "stdout", None),
                    stderr=getattr(exc, "stderr", None),
                    return_code=int(getattr(exc, "exit_code", 1) or 1),
                )
            # Surface timeouts as TimeoutError so callers can treat a wedged
            # command as a command timeout rather than an infra error.
            if isinstance(timeout_exc, type) and isinstance(exc, timeout_exc):
                raise TimeoutError(f"e2b command timed out after {effective_timeout}s: {exc}") from exc
            raise

        return SandboxExecResult(
            stdout=getattr(result, "stdout", None),
            stderr=getattr(result, "stderr", None),
            return_code=int(getattr(result, "exit_code", 0) or 0),
        )

    # ----------------------------------------------------------------- files

    async def write_file(self, handle: SandboxHandle, target_path: str, data: str | bytes) -> None:
        sandbox = self._sandbox(handle)
        await self._with_retries(
            lambda: sandbox.files.write(target_path, data, **self._api_params()),
            operation="write_file",
        )

    async def read_file(self, handle: SandboxHandle, source_path: str) -> bytes:
        sandbox = self._sandbox(handle)
        data = await self._with_retries(
            lambda: sandbox.files.read(source_path, format="bytes", **self._api_params()),
            operation="read_file",
        )
        return data if isinstance(data, bytes) else bytes(data)

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        source = Path(source_path)
        if not source.is_file():
            raise FileNotFoundError(f"Source file not found: {source}")
        await self.write_file(handle, target_path, source.read_bytes())

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        target = Path(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(await self.read_file(handle, source_path))
